from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import json
from typing import Any, Dict


# -----------------------------------------------------------------------------
# Generic deep-merge utility
# -----------------------------------------------------------------------------

def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dictionaries.

    Rules:
    - nested dicts are merged recursively
    - non-dict values in `override` replace values in `base`
    - inputs are not modified
    """
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


# -----------------------------------------------------------------------------
# JSON helpers
# -----------------------------------------------------------------------------

def load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding='utf-8'))


# -----------------------------------------------------------------------------
# Reference resolution helpers for MetaScint run configs
# -----------------------------------------------------------------------------

def resolve_material_refs(material_refs: Dict[str, str], material_overrides: Dict[str, Dict[str, Any]] | None = None) -> Dict[str, Dict[str, Any]]:
    """Load canonical material files and apply optional per-material overrides.

    Example:
        material_refs = {
            "BGO": "configs/materials/bgo.json",
            "LSO": "configs/materials/lso.json",
        }
        material_overrides = {
            "BGO": {"density_g_cm3": 7.15}
        }
    """
    material_overrides = material_overrides or {}
    resolved: Dict[str, Dict[str, Any]] = {}
    for role, ref_path in material_refs.items():
        base = load_json(ref_path)
        override = material_overrides.get(role, {})
        resolved[role] = deep_merge(base, override)
    return resolved



def resolve_single_ref(ref_path: str | None, overrides: Dict[str, Any] | None = None) -> Dict[str, Any] | None:
    """Load a single referenced JSON file and apply optional overrides.

    Returns None if `ref_path` is None.
    """
    if ref_path is None:
        return None
    base = load_json(ref_path)
    return deep_merge(base, overrides or {})



def resolve_run_config_dict(run_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve a lightweight reference-based run config into a fully expanded dict.

    Expected run-config keys:
    - material_refs
    - geometry_ref
    - photonics_ref (optional)
    - optimization_ref (optional)
    - irradiation (optional inline overrides / runtime settings)
    - simulation (optional inline overrides / runtime settings)
    - material_overrides (optional)
    - geometry_overrides (optional)
    - photonics_overrides (optional)
    - optimization_overrides (optional)

    Output structure matches the resolved runtime configuration style used by
    the current benchmark code.
    """
    resolved_materials = resolve_material_refs(
        run_cfg.get('material_refs', {}),
        run_cfg.get('material_overrides'),
    )

    resolved_geometry = resolve_single_ref(
        run_cfg.get('geometry_ref'),
        run_cfg.get('geometry_overrides'),
    )

    resolved_photonics = resolve_single_ref(
        run_cfg.get('photonics_ref'),
        run_cfg.get('photonics_overrides'),
    )

    resolved_optimization = resolve_single_ref(
        run_cfg.get('optimization_ref'),
        run_cfg.get('optimization_overrides'),
    )

    resolved = {
        'materials': resolved_materials,
        'geometry': resolved_geometry,
        'irradiation': deepcopy(run_cfg.get('irradiation', {})),
        'simulation': deepcopy(run_cfg.get('simulation', {})),
    }

    if resolved_photonics is not None:
        resolved['photonics'] = resolved_photonics
    if resolved_optimization is not None:
        resolved['optimization'] = resolved_optimization

    return resolved
