# MetaScint Architecture

## Design principles

1. **Configuration first**: geometry, materials, irradiation, and simulation parameters live in external JSON config files. No hardcoded values in simulation code.
2. **Physics modularity**: attenuation, photon kinematics, electron transport, and relaxation are independent modules with well-defined interfaces.
3. **Geometry modularity**: `CompositeGeometry` handles uniform and composite geometries through the same query interface (`material_role_at`, `distance_to_box_mm`, `distance_to_next_material_change`).
4. **Manufacturing-aware**: geometry configs can specify manufacturing constraints (groove+wall patterns) that auto-resolve to regions at load time, with automated dimension correction.
5. **Workflow separation**: runnable studies (`run_uniform_benchmark.py`, `run_composite_benchmark.py`) and utilities (`run_ray_trace.py`, `run_geometry_validation.py`) live in `workflows/`.

## Module map

| Directory | Purpose | Status |
|---|---|---|
| `src/config/` | JSON loading, reference resolution, schema (frozen dataclasses), validation | Functional |
| `src/geometry/` | `CompositeGeometry`, region queries, manufacturing solver, validator | Functional |
| `src/simulation/` | Event loops (uniform + composite), Compton tracking, cross-material electron transport, per-material tallies | Functional |
| `src/physics/` | Attenuation coefficients, Klein-Nishina sampling, Tabata CSDA ranges, fluorescence/Auger | Functional |
| `src/io_utils/` | JSON read/write, export helpers | Functional |
| `src/optimization/` | Design objectives, constraints, search stubs | Placeholder |
| `src/photonics/` | LDOS, coupling, corrections stubs | Placeholder |

## Data flow

```
Geometry JSON  ──┐
Material JSON  ──┤
Benchmark JSON ──┼──→ loader.load_benchmark_config() → BenchmarkConfig → CompositeGeometry
Simulation JSON ─┘                                                  ↓
                                                            run_composite_benchmark()
                                                                      ↓
                                                              Output JSON (tallies, histories)
```

The loader pipeline:
1. Resolves material/geometry/photonics references via `merge.py`.
2. If geometry has `manufacturing`, solves constraints and generates `regions` via `manufacturing.py`.
3. Validates the resolved config via `validation.py`.
4. Returns a frozen `BenchmarkConfig` dataclass.

## Key classes

- `CompositeGeometry` — geometry queries (point lookup, boundary distance, ray tracing).
- `ManufacturingConstraint` — 1D sum constraint with solvable strategies.
- `BenchmarkConfig` — frozen runtime config consumed by simulation.
- `TabataRangeModel` — CSDA electron range per material.
