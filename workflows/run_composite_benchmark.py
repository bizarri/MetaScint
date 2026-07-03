from _bootstrap import ROOT
from config.loader import load_benchmark_config
from geometry.composite_pixel import CompositeGeometry
from simulation.composite_benchmark import run_composite_benchmark
from io_utils.jsonio import dump_json


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

result = run_composite_benchmark(cfg.materials, geometry, cfg, seed_offset=0)

out = ROOT / 'outputs' / 'composite_bgo_baf2_benchmark_results.json'
dump_json(result, out)
print(f'Wrote {out}')

r = result
print(f"  irradiation_area_fraction = {r['irradiation_area_fraction']:.3f}")
print(f"  full_count      = {r['full_count']} ({r['full_fraction']:.5f})")
print(f"  partial_count   = {r['partial_count']} ({r['partial_fraction']:.5f})")
print(f"  none_count      = {r['none_count']} ({r['none_fraction']:.5f})")
print(f"  interacted_frac = {r['interacted_fraction']:.5f}")
print(f"  total_deposited_keV = {r['total_deposited_keV']:.1f}")
print(f"  deposition_by_material:")
for role in cfg.materials:
    keV = r['deposition_by_material_keV'][role]
    frac = r['deposition_by_material_fraction'][role]
    print(f"    {role}: {keV:.1f} keV ({frac:.5f})")
