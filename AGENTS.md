# MetaScint — AGENTS.md

## Setup

- **Python ≥ 3.10** only. No pip install needed — no `requirements.txt`, no package install.
- All scripts use `workflows/_bootstrap.py` to add `src/` to `sys.path` at runtime.
- External dependency: **numpy** (used in physics and geometry modules). Install if missing: `pip install numpy`.

## Run

```bash
# Main benchmark (BGO + LSO, 3×3×15mm, 511 keV, 50k events)
python workflows/run_uniform_benchmark.py

# Composite multi-material benchmark (BGO matrix + BaF2 channels, 50k events)
python workflows/run_composite_benchmark.py

# Design optimization (groove/wall width scan, 5k events per eval)
python workflows/run_design_optimization.py

# Validate geometry config files
python workflows/run_geometry_validation.py

# Ray-trace diagnostic for composite geometry
python workflows/run_ray_trace.py

# Run all non-optimization workflows sequentially
python workflows/run_all.py
```

Always run from repository root.

## Test

```bash
python -m pytest tests/
```

6 test files (44 tests) — no fixtures, no integration services needed. Tests duplicate the `_bootstrap.py` sys.path hack inline (normal for this repo).

## Config system

- JSON configs under `configs/` (benchmarks/, materials/, geometry/, optimization/, photonics/).
- Two formats accepted by `load_benchmark_config`:
  - **Reference-based** (preferred): `material_refs` + `geometry_ref` pointing to other JSON files.
  - **Fully resolved**: inlined `materials` + `geometry` dicts.
- Config merging pipeline: `loader.py` → `merge.py` → `schema.py` (frozen dataclasses) → `validation.py`.
- Output: `outputs/` directory, JSON files.

## Composite geometry (region-based)

Geometry configs define axis-aligned box regions, each with a `material_role` referencing a key in the benchmark's `materials` dict. Points not in any region fall back to `default_material_role`.

```json
{
  "pixel_x_mm": 3.0,
  "pixel_y_mm": 3.0,
  "pixel_z_mm": 15.0,
  "regions": [
    {"name": "channel", "material_role": "BaF2",
     "xmin": -0.5, "xmax": 0.5, "ymin": -1.5, "ymax": 1.5, "zmin": 0, "zmax": 15}
  ]
}
```

Regions support an optional `tilt_deg` parameter for rotation around the pixel y-axis (tilting the groove in the xz-plane relative to the z-axis):

```json
{"name": "groove", "material_role": "BaF2", "tilt_deg": 45.0,
 "xmin": -2, "xmax": 2, "ymin": -0.1, "ymax": 0.1, "zmin": 0, "zmax": 15}
```

Benchmark configs for composite geometries must set `default_material_role`. See `configs/geometry/composite_channel_reference.json`.

## Manufacturing constraints

Geometry configs can use a `manufacturing` block instead of explicit `regions` to auto-generate a groove-and-wall pattern that satisfies the 1D sum constraint (total width = pixel width). The pattern is: `matrix_left + [groove + wall] × (n-1) + groove + matrix_right`.

```json
{
  "pixel_x_mm": 3.0,
  "pixel_y_mm": 3.0,
  "pixel_z_mm": 15.0,
  "manufacturing": {
    "axis": "x",
    "n_channels": 12,
    "groove_width_mm": 0.2,
    "groove_material_role": "BaF2",
    "wall_width_mm": 0.05,
    "matrix_material_role": "BGO",
    "matrix_edge_width_mm": 0.025,
    "strategy": "adjust_walls"
  }
}
```

Four correction strategies when total ≠ pixel width:
- `adjust_walls` — fix groove and matrix_edge, solve for wall
- `adjust_grooves` — fix wall and matrix_edge, solve for groove
- `adjust_matrix_edges` — fix groove and wall, solve for matrix_edge
- `minimize_change` — least-squares adjustment of all three

The loader (`src/config/loader.py`) automatically resolves `manufacturing` → `regions` at config load time. The validator (`src/geometry/manufacturing.py:check_manufacturing_rules`) reports unsatisfiable constraints.

Region bounds use half-open intervals `[lower, upper)` to resolve boundary ambiguity. See `configs/geometry/composite_channel_manufacturing.json` and `configs/benchmarks/composite_bgo_baf2_mfg_511keV.json`.

## Optimization

Full optimization pipeline in `src/optimization/`. Supports:
- **Parameter types:** numeric (min/max/step) and categorical (choices list)
- **Objective functions:** full_absorption, total_deposition, material_deposition, interacted_fraction, material_fraction
- **Constraints:** bounds, manufacturing_width, and (logical AND)
- **Search methods:** grid (deterministic), random (sampling), adaptive (coarse grid + local refinement)
- **Evaluator:** `OptimizationEvaluator` takes a base benchmark config JSON, applies candidate overrides (manufacturing params, pixel dims, material roles), resolves manufacturing constraints with fallback strategies, runs composite benchmark (`n_events` from config, default 5000)

Config file in `configs/optimization/design_search_template.json`. Run:
```bash
python workflows/run_design_optimization.py [config_path]
```
Output written to `outputs/optimization_<study_name>.json`.

## Architecture

| Directory | Purpose |
|-----------|---------|
| `src/config/` | Config loading, merge, schema, validation |
| `src/simulation/` | Event loop (Compton tracking), tallies, uniform + composite benchmark runners |
| `src/physics/` | Attenuation, photon kinematics, Tabata CSDA electron transport, relaxation (fluorescence/Auger) |
| `src/geometry/` | Uniform pixel, composite pixel (region-based, active), ray-box queries, validator, manufacturing |
| `src/optimization/` | Full optimization: interfaces, constraints, objectives, evaluator, search |
| `src/photonics/` | **Placeholder** — LDOS, coupling, corrections stubs |
| `src/io_utils/` | JSON read/write, export, report |
| `workflows/` | Runnable entry points |
| `docs/` | Architecture, config, geometry, physics, workflow docs |

## Key simulation parameters (uniform benchmark)

- 3×3×15mm uniform pixel (BGO/LSO)
- 511 keV gamma, front-face random irradiation (95% area fraction by default)
- Compton scattering (Klein-Nishina) + photoelectric (K-shell fluorescence/Auger, minimal)
- Tabata 1994 CSDA electron transport (straight-line)
- 50,000 events default, seed configurable in JSON
- Output: full/partial/none absorption tallies + example event histories

## Simulation internals (material transitions)

The composite event loop (`src/simulation/composite_benchmark.py`) determines the current material from position at each tracking step. When a sampled free path would cross a region boundary, the particle stops at the boundary and continues with the new material's cross section and range model.

**Electron transport tracks material transitions:** `transport_electron_csda` segments the straight-line CSDA path at each material boundary, using each material's own range model for that segment. Deposited energy is attributed to the material the electron is passing through at that moment, not the emission material.

Output includes `deposition_by_material_keV` (summed per-material deposited energy across all events) and `deposition_by_material_fraction` (per-material fraction of total incident gamma energy).

## State

Early-stage research framework.
- `optimization/` is fully functional (interfaces, objectives, constraints, evaluator, grid/random/adaptive search).
- `src/geometry/manufacturing.py` has the manufacturing constraint solver with 4 strategies.
- `geometry/composite_pixel.py` is functional (region-based axis-aligned boxes + tilted regions, half-open bounds, ray tracing).
- `photonics/` is a non-functional placeholder.
- No linter, formatter, or type checker configured.
- No CI pipeline.
- No `.gitignore`.
