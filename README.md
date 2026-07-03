# MetaScint

Config-driven scintillator and meta-scintillator simulation framework for PET detector design. Python 3.10+, numpy only.

## Current capabilities

- **Uniform benchmark** — BGO/LSO 3×3×15mm, 511 keV gamma, 50k events
- **Composite geometry** — multi-material pixels with region-based composition (axis-aligned and tilted structures)
- **Manufacturing constraints** — auto-generate groove-and-wall patterns from a 1D sum constraint with 4 correction strategies (`adjust_walls`, `adjust_grooves`, `adjust_matrix_edges`, `minimize_change`)
- **Per-material energy deposition** — energy attributed to the material the particle is passing through, not the emission material
- **Cross-material electron transport** — CSDA straight-line path segmented at material boundaries using per-material range models
- **Physics stack** — Compton scattering (Klein-Nishina), photoelectric effect with K-shell fluorescence/Auger, Tabata 1994 CSDA electron ranges
- **Ray tracing** — `trace_ray()` diagnostic for geometry verification
- **Geometry validation** — checks region bounds, material references, and manufacturing constraint satisfiability

## Quick start

```bash
# Uniform benchmark
python workflows/run_uniform_benchmark.py

# Composite multi-material benchmark
python workflows/run_composite_benchmark.py

# Geometry ray-trace diagnostic
python workflows/run_ray_trace.py

# Validate geometry configs
python workflows/run_geometry_validation.py

# All non-optimization workflows
python workflows/run_all.py
```

```bash
# Tests
python -m pytest tests/
```

Outputs go to `outputs/`. Run from repository root.

## Documentation

| File | Content |
|---|---|
| `docs/ARCHITECTURE.md` | Module map, data flow, design principles |
| `docs/GEOMETRY.md` | `CompositeGeometry`, manufacturing constraints, half-open bounds, ray tracing |
| `docs/CONFIGS.md` | All config files, formats, and field descriptions |
| `docs/PHYSICS.md` | Photon transport, CSDA electron transport, per-material deposition |
| `docs/WORKFLOWS.md` | Available entry points |
| `AGENTS.md` | Agent instructions (for LLM-assisted development) |

## Project status

Early-stage research framework. `src/optimization/` and `src/photonics/` are non-functional placeholders.
