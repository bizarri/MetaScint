# Geometry Modules

## Uniform Pixel (`geometry/composite_pixel.py` — `CompositeGeometry`)

The key class is `CompositeGeometry` which handles both uniform and composite geometries:

```python
geo = CompositeGeometry(pixel_x_mm, pixel_y_mm, pixel_z_mm, regions, default_material_role)
```

- **Uniform**: pass an empty `regions=()`. `material_role_at` always returns `default_material_role`.
- **Composite**: pass a tuple of `RegionConfig` objects. Points are tested in order; first match wins. Points matching no region get `default_material_role`.

### Queries

- `material_role_at(pos)` — returns material role string at a 3D point.
- `distance_to_box_mm(pos, dir)` — distance along direction to exit the pixel bounding box.
- `distance_to_next_material_change(pos, dir)` — distance to the nearest region boundary where the material changes. Returns `(distance, next_material_role)`. If no change before box exit, returns `(box_dist, None)`.
- `sample_entry_xy(rng, irradiation)` — samples a random entry point on the front face within an irradiation area fraction.

### Boundary convention

Region bounds use **half-open intervals** `[lower, upper)` to eliminate boundary ambiguity. A point at exactly `xmax` of region A is NOT in region A; if region B starts at the same coordinate, it IS in region B (lower bound inclusive). This ensures every point inside the pixel belongs to exactly one region.

### Ray tracing

`trace_ray(origin, direction, max_segments=200)` — steps through all material domains along a ray, returning an ordered list of segments:

```python
[
    {'material_role': 'BGO',  'start_mm': [...], 'end_mm': [...], 'length_mm': 0.5,  'exits_pixel': False},
    {'material_role': 'BaF2', 'start_mm': [...], 'end_mm': [...], 'length_mm': 1.0,  'exits_pixel': False},
    {'material_role': 'BGO',  'start_mm': [...], 'end_mm': [...], 'length_mm': 1.0,  'exits_pixel': True},
]
```

Handles origins anywhere (inside or outside the pixel). Rays that miss the box return an empty list.

### Tilted regions

Regions support `tilt_deg` for rotation around the pixel y-axis (tilting in the xz-plane relative to the z-axis). The `xmin/xmax/ymin/ymax` define the box in local coordinates; `tilt_deg` rotates around the box center. Point containment and ray intersection both handle rotation.

## Manufacturing Constraints (`geometry/manufacturing.py`)

Auto-generates a groove-and-wall region pattern that satisfies the 1D sum constraint:

```
matrix_left + [groove + wall] × (n_channels - 1) + groove + matrix_right = pixel_width
```

### Correction strategies

| Strategy | Fixed | Solved |
|---|---|---|
| `adjust_walls` | matrix_edge, groove | wall |
| `adjust_grooves` | matrix_edge, wall | groove |
| `adjust_matrix_edges` | groove, wall | matrix_edge |
| `minimize_change` | — | all three (least-squares) |

### Key functions

- `solve_constraint(mc, pixel_width_mm)` — returns a new `ManufacturingConstraint` with corrected dimensions.
- `generate_regions(mc, height, depth, width)` — produces a list of region dicts (half-open bounds, contiguous).
- `check_manufacturing_rules(config)` — validates satisfiability, reports issues only for unsolvable constraints.
- `apply_manufacturing(config)` — converts a geometry dict from `manufacturing` → `regions` in place.

### Integration

The loader (`config/loader.py`) calls `apply_manufacturing` automatically when a geometry dict contains a `manufacturing` key. The user need not change any code; configs with `manufacturing` are transparently resolved to regions.

## Validator (`geometry/validator.py`)

- `validate_geometry_config(path)` — loads a geometry JSON file and returns `{'path', 'issues', 'valid'}`.
- Checks pixel dimensions, calls `check_manufacturing_rules`.
