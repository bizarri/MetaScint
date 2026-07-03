# Configuration Guide

## Benchmark configs

| File | Description |
|---|---|
| `benchmarks/uniform_bgo_lso_511keV.json` | Uniform BGO + LSO, 3×3×15mm, 511 keV, 50k events |
| `benchmarks/composite_bgo_baf2_511keV.json` | BGO matrix + 12 explicit BaF2 channels, 3×3×15mm |
| `benchmarks/composite_bgo_baf2_mfg_511keV.json` | Same geometry via manufacturing constraint |

### Format

Two formats accepted by `load_benchmark_config`:

**Reference-based** (preferred):
```json
{
  "material_refs": {"BGO": "configs/materials/bgo.json"},
  "geometry_ref": "configs/geometry/composite_channel_manufacturing.json",
  "default_material_role": "BGO",
  "irradiation": {"front_face_area_fraction": 0.95},
  "simulation": {"gamma_energy_keV": 511.0, "n_events": 50000, "seed": 20260622}
}
```

**Fully resolved** (inline):
```json
{
  "materials": {"BGO": {"density_g_cm3": 7.13, "formula": {"Bi": 4, "Ge": 3, "O": 12}}},
  "geometry": {"pixel_x_mm": 3.0, "pixel_y_mm": 3.0, "pixel_z_mm": 15.0},
  "default_material_role": "",
  "simulation": {...}
}
```

Composite geometries (with regions or manufacturing) **must** set `default_material_role`.

## Material library

| File | Material |
|---|---|
| `materials/bgo.json` | Bi₄Ge₃O₁₂ (BGO) |
| `materials/lso.json` | Lu₂SiO₅ (LSO) |
| `materials/baf2.json` | BaF₂ |
| `materials/template_compound.json` | Template for adding new materials |

Each material JSON:
```json
{
  "density_g_cm3": 7.13,
  "formula": {"Bi": 4, "Ge": 3, "O": 12},
  "dominant_photo_element": "Bi"
}
```

- `formula` maps element symbols to stoichiometric counts.
- `dominant_photo_element` is used for fluorescence/Auger post-processing.

## Geometry library

| File | Description |
|---|---|
| `geometry/uniform_pixel_3x3x15mm.json` | Uniform 3×3×15mm pixel (no regions) |
| `geometry/composite_channel_reference.json` | 12 explicit BaF2 channels in BGO matrix |
| `geometry/composite_channel_manufacturing.json` | Same geometry via manufacturing constraint |
| `geometry/template_geometry.json` | Template for new geometry configs |

### Manual regions
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

Regions support `tilt_deg` for rotation around the pixel y-axis (tilting grooves in the xz-plane relative to the z-axis).

### Manufacturing constraint
```json
{
  "pixel_x_mm": 3.0,
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

The loader auto-resolves `manufacturing` → `regions`. See `docs/GEOMETRY.md` for strategy details.

## Optimization templates (placeholders)
- `optimization/design_search_template.json`
- `optimization/objective_fullabsorption.json`
- `optimization/constraints_manufacturing_template.json`

## Photonics templates (placeholders)
- `photonics/template_photonics.json`
- `photonics/ldos_disabled.json`
