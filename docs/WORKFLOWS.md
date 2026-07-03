# Workflows

| Script | Description |
|---|---|
| `run_uniform_benchmark.py` | Validated uniform benchmark (BGO + LSO, 3×3×15mm, 511 keV, 50k events). Outputs to `outputs/`. |
| `run_composite_benchmark.py` | Composite benchmark (BGO matrix + BaF2 channels, 50k events). Includes per-material deposition breakdown. |
| `run_geometry_validation.py` | Validates all geometry config files in `configs/geometry/`. |
| `run_ray_trace.py` | Ray-trace diagnostic: prints material segments along vertical and horizontal rays through the composite geometry. Useful for verifying region layout. |
| `run_design_optimization.py` | **Placeholder** — prints config, no optimizer yet. |
| `run_all.py` | Convenience launcher — runs all non-optimization workflows sequentially. |

All scripts must be run from repository root. They use `_bootstrap.py` to add `src/` to `sys.path`.
