from _bootstrap import ROOT
from config.loader import load_benchmark_config
from geometry.composite_pixel import CompositeGeometry
import numpy as np

cfg = load_benchmark_config(
    ROOT / 'configs' / 'benchmarks' / 'composite_bgo_baf2_511keV.json'
)

geometry = CompositeGeometry(
    pixel_x_mm=cfg.geometry.pixel_x_mm,
    pixel_y_mm=cfg.geometry.pixel_y_mm,
    pixel_z_mm=cfg.geometry.pixel_z_mm,
    regions=cfg.geometry.regions,
    default_material_role=cfg.default_material_role,
)

print("Ray-trace: vertical entry at (0, 0, 0)")
segments = geometry.trace_ray(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]))
for s in segments:
    print(f"  {s['length_mm']:8.3f} mm  {s['material_role']:5s}  → exit={s['exits_pixel']}")

print()

print("Ray-trace: horizontal crossing at z=7.5, y=-1.4 → +y")
segments = geometry.trace_ray(np.array([0.0, -1.4, 7.5]), np.array([0.0, 1.0, 0.0]))
for s in segments:
    mat = s['material_role']
    start = f"({s['start_mm'][0]:.2f}, {s['start_mm'][1]:.2f}, {s['start_mm'][2]:.2f})"
    seg_info = f"  {s['length_mm']:8.3f} mm  {mat:5s}  {start}"
    if s['exits_pixel']:
        seg_info += "  → exits pixel"
    print(seg_info)
