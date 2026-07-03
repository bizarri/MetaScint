from __future__ import annotations
from pathlib import Path
import copy
import json
import numpy as np
from typing import Any, Dict, Optional

from config.schema import (
    BenchmarkConfig, SimulationConfig, IrradiationConfig,
)

from config.merge import resolve_run_config_dict
from config.loader import resolved_dict_to_benchmark_config
from geometry.composite_pixel import CompositeGeometry
from simulation.composite_benchmark import run_composite_benchmark
from geometry.manufacturing import ManufacturingConstraint, solve_constraint
from signal.timing import ScintillationPulse
from signal.analysis import compute_markers


def _set_nested(d: dict, path: str, value: Any) -> dict:
    keys = path.split('.')
    cur = d
    for k in keys[:-1]:
        if k not in cur:
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value
    return d


def _resolve_material_ref(base_cfg: dict, role: str, material_name: str) -> dict:
    materials_dir = Path(base_cfg.get('_config_dir', 'configs')) / 'materials'
    material_file = materials_dir / f'{material_name.lower()}.json'
    if material_file.exists():
        return json.loads(material_file.read_text(encoding='utf-8'))
    for ref_path in base_cfg.get('material_refs', {}).values():
        p = Path(ref_path)
        if p.stem.lower() == material_name.lower():
            return json.loads(p.read_text(encoding='utf-8'))
    raise FileNotFoundError(f'No material file found for "{material_name}"')


class OptimizationEvaluator:
    def __init__(
        self,
        base_config_path: str | Path,
        n_events: int = 5000,
        seed: int = 42,
        verbose: bool = False,
    ):
        self.base_path = Path(base_config_path)
        self.n_events = n_events
        self.seed = seed
        self.verbose = verbose
        raw = json.loads(self.base_path.read_text(encoding='utf-8'))
        raw['_config_dir'] = str(self.base_path.parent)
        self.base_raw = raw

    def _build_benchmark_config(
        self,
        candidate: dict[str, float | str],
    ) -> tuple[BenchmarkConfig, CompositeGeometry]:
        raw = copy.deepcopy(self.base_raw)
        config_dir = raw.pop('_config_dir', 'configs')

        if 'material_refs' in raw and 'geometry_ref' in raw:
            resolved = resolve_run_config_dict(raw)
            resolved['default_material_role'] = raw.get('default_material_role', '')
        else:
            resolved = dict(raw)
            resolved.setdefault('default_material_role', '')
            resolved.setdefault('geometry', {})

        geometry_raw = resolved.get('geometry', {})

        for key, value in candidate.items():
            if key.startswith('pixel_'):
                _set_nested(geometry_raw, key, float(value))
            elif key.startswith('manufacturing.'):
                sub_key = key.split('.', 1)[1]
                _set_nested(geometry_raw, f'manufacturing.{sub_key}', value)
            elif key.startswith('material.'):
                parts = key.split('.')
                role = parts[1]
                material_name = str(value)
                mat_data = _resolve_material_ref(
                    {**raw, '_config_dir': config_dir}, role, material_name
                )
                resolved.setdefault('materials', {})[role] = {
                    'density_g_cm3': mat_data['density_g_cm3'],
                    'formula': mat_data['formula'],
                    'dominant_photo_element': mat_data.get('dominant_photo_element'),
                    'scintillation': mat_data.get('scintillation'),
                }

        resolved['geometry'] = geometry_raw
        resolved.setdefault('irradiation', {})
        resolved.setdefault('simulation', {
            'gamma_energy_keV': 511.0,
            'n_events': self.n_events,
            'seed': self.seed,
        })
        resolved['simulation']['n_events'] = self.n_events

        cfg = self._resolve_manufacturing(resolved)
        geometry = CompositeGeometry(
            pixel_x_mm=cfg.geometry.pixel_x_mm,
            pixel_y_mm=cfg.geometry.pixel_y_mm,
            pixel_z_mm=cfg.geometry.pixel_z_mm,
            regions=cfg.geometry.regions,
            default_material_role=cfg.default_material_role,
        )
        return cfg, geometry

    _FALLBACK_STRATEGIES = [
        'adjust_walls', 'adjust_grooves', 'adjust_matrix_edges', 'minimize_change',
    ]

    def _resolve_manufacturing(self, resolved: dict) -> BenchmarkConfig:
        geo = resolved['geometry']
        if 'manufacturing' not in geo:
            return resolved_dict_to_benchmark_config(resolved)
        mfg = geo['manufacturing']
        preferred = mfg.get('strategy', 'adjust_walls')
        strategies = [preferred] + [s for s in self._FALLBACK_STRATEGIES if s != preferred]

        last_exc: Exception | None = None
        for strategy in strategies:
            try:
                mc = ManufacturingConstraint(
                    axis=mfg.get('axis', 'x'),
                    n_channels=int(mfg['n_channels']),
                    groove_width_mm=float(mfg['groove_width_mm']),
                    groove_material_role=str(mfg.get('groove_material_role', '')),
                    wall_width_mm=float(mfg['wall_width_mm']),
                    matrix_material_role=str(mfg.get('matrix_material_role', '')),
                    matrix_edge_width_mm=float(mfg['matrix_edge_width_mm']),
                    strategy=strategy,
                    tilt_deg=float(mfg.get('tilt_deg', 0.0)),
                )
                pixel_w = geo.get('pixel_x_mm' if mc.axis == 'x' else 'pixel_y_mm', 0)
                solved = solve_constraint(mc, float(pixel_w))
                from geometry.manufacturing import generate_regions
                pixel_h = geo.get('pixel_y_mm' if mc.axis == 'x' else 'pixel_x_mm', 0)
                pixel_d = geo.get('pixel_z_mm', 0)
                geo['regions'] = generate_regions(solved, float(pixel_h), float(pixel_d), float(pixel_w))
                break
            except (ValueError, KeyError) as e:
                last_exc = e
                continue
        else:
            raise ValueError(f'Manufacturing resolution failed (tried {strategies}): {last_exc}')
        return resolved_dict_to_benchmark_config(resolved)

    def evaluate(self, candidate: dict[str, float | str]) -> dict:
        try:
            cfg, geometry = self._build_benchmark_config(candidate)
        except (ValueError, KeyError) as e:
            if self.verbose:
                print(f'  invalid: {e}')
            return {
                'n_events': 0,
                'full_count': 0,
                'full_fraction': 0.0,
                'total_deposited_keV': 0.0,
                'deposition_by_material_keV': {},
                'deposition_by_material_fraction': {},
                'infeasible': True,
            }
        result = run_composite_benchmark(
            cfg.materials, geometry, cfg, seed_offset=0,
            collect_per_event_data=True,
        )
        pulses = {}
        for role, mat in cfg.materials.items():
            if mat.scintillation is not None:
                pulses[role] = ScintillationPulse(mat.scintillation)
        markers = compute_markers(result, pulses, cfg.materials, threshold_photons=10)
        result['figure_of_merit'] = markers['figure_of_merit']
        result['avg_detection_time_ns'] = markers['avg_detection_time_ns']
        result['median_detection_time_ns'] = markers['median_detection_time_ns']
        if self.verbose:
            n = result['n_events']
            print(
                f'  full={result["full_count"]}({result["full_count"]/n:.3f}) '
                f'total_dep={result["total_deposited_keV"]:.0f} '
                f'detect={markers["avg_detection_time_ns"]:.4f}ns '
                f'FOM={markers["figure_of_merit"]:.4f} '
                f'candidate={candidate}'
            )
        return result



