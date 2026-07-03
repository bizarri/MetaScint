from __future__ import annotations
from pathlib import Path
import warnings
from typing import Any, Dict

from config.schema import (
    RunConfig,
    MaterialConfig,
    ScintillationConfig,
    ScintillationComponent,
    RegionConfig,
    GeometryConfig,
    IrradiationConfig,
    SimulationConfig,
    BenchmarkConfig,
)
from config.merge import resolve_run_config_dict, load_json
from config.validation import validate_benchmark_config
from geometry.manufacturing import apply_manufacturing


def load_run_config(path: str | Path) -> RunConfig:
    data = load_json(path)
    return RunConfig(
        material_refs=data["material_refs"],
        geometry_ref=data["geometry_ref"],
        photonics_ref=data.get("photonics_ref"),
        optimization_ref=data.get("optimization_ref"),
        irradiation=data.get("irradiation"),
        simulation=data.get("simulation"),
        material_overrides=data.get("material_overrides"),
        geometry_overrides=data.get("geometry_overrides"),
        photonics_overrides=data.get("photonics_overrides"),
        optimization_overrides=data.get("optimization_overrides"),
    )


def _regions_from_geometry_dict(geometry_raw: dict) -> tuple[RegionConfig, ...]:
    has_manufacturing = "manufacturing" in geometry_raw
    if has_manufacturing:
        geometry_raw = apply_manufacturing(geometry_raw)
    raw_regions = geometry_raw.get("regions", [])
    return tuple(RegionConfig(**r) for r in raw_regions)


def _scintillation_from_dict(d: dict) -> ScintillationConfig | None:
    sc = d.get("scintillation")
    if sc is None:
        return None
    components = tuple(
        ScintillationComponent(
            c["amplitude_fraction"],
            c["decay_time_ns"],
            c.get("pulse_model", "standard"),
        )
        for c in sc["decay_components"]
    )
    return ScintillationConfig(
        light_yield_per_keV=sc["light_yield_per_keV"],
        rise_time_ns=sc["rise_time_ns"],
        decay_components=components,
    )


def resolved_dict_to_benchmark_config(data: Dict[str, Any]) -> BenchmarkConfig:
    materials = {}
    for k, v in data["materials"].items():
        runtime_material = {
            "density_g_cm3": v["density_g_cm3"],
            "formula": v["formula"],
            "dominant_photo_element": v.get("dominant_photo_element"),
            "scintillation": _scintillation_from_dict(v),
        }
        materials[k] = MaterialConfig(**runtime_material)

    geometry_raw = data["geometry"]
    runtime_geometry = {
        "pixel_x_mm": geometry_raw["pixel_x_mm"],
        "pixel_y_mm": geometry_raw["pixel_y_mm"],
        "pixel_z_mm": geometry_raw["pixel_z_mm"],
        "regions": _regions_from_geometry_dict(geometry_raw),
    }
    geometry = GeometryConfig(**runtime_geometry)

    irradiation = IrradiationConfig(**data.get("irradiation", {}))
    try:
        simulation = SimulationConfig(**data["simulation"])
    except KeyError:
        raise ValueError(
            "Config is missing the required 'simulation' block "
            "(needs at least gamma_energy_keV, n_events, seed)."
        )
    except TypeError as e:
        raise ValueError(
            f"Invalid 'simulation' block — {e}. "
            "Required fields: gamma_energy_keV, n_events, seed."
        ) from e

    cfg = BenchmarkConfig(
        materials=materials,
        geometry=geometry,
        irradiation=irradiation,
        simulation=simulation,
        default_material_role=data.get("default_material_role", ""),
    )

    issues = validate_benchmark_config(cfg)
    if issues:
        raise ValueError("Invalid resolved benchmark config: " + "; ".join(issues))

    return cfg


def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    raw = load_json(path)

    if "material_refs" in raw and "geometry_ref" in raw:
        if "materials" in raw or "geometry" in raw:
            warnings.warn(
                "Config contains both reference-based keys (material_refs/geometry_ref) "
                "and inline keys (materials/geometry). The reference-based path takes "
                "precedence; inline keys are ignored.",
                stacklevel=2,
            )
        resolved = resolve_run_config_dict(raw)
        resolved["default_material_role"] = raw.get("default_material_role", "")
        return resolved_dict_to_benchmark_config(resolved)

    if "materials" in raw and "geometry" in raw:
        resolved = dict(raw)
        resolved.setdefault("default_material_role", "")
        return resolved_dict_to_benchmark_config(resolved)

    raise ValueError(
        "Unsupported config format. Expected either "
        "reference-based keys (material_refs, geometry_ref) or "
        "resolved keys (materials, geometry)."
    )