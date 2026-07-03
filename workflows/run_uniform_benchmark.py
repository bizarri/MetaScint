from _bootstrap import ROOT
from config.loader import load_benchmark_config
from simulation.uniform_benchmark import run_uniform_material
from io_utils.jsonio import dump_json


cfg = load_benchmark_config(
    ROOT / 'configs' / 'benchmarks' / 'uniform_bgo_lso_511keV.json'
)

results = []
for i, name in enumerate(['BGO', 'LSO']):
    results.append(
        run_uniform_material(name, cfg.materials[name], cfg, seed_offset=1000 * i)
    )

out = ROOT / 'outputs' / 'uniform_bgo_lso_benchmark_results.json'
dump_json(results, out)
print(f'Wrote {out}')

for r in results:
    print(f"Material: {r['material']}")
    print(f"  irradiation_area_fraction = {r['irradiation_area_fraction']:.3f}")
    print(f"  full_count      = {r['full_count']} ({r['full_fraction']:.5f})")
    print(f"  partial_count   = {r['partial_count']} ({r['partial_fraction']:.5f})")
    print(f"  none_count      = {r['none_count']} ({r['none_fraction']:.5f})")
    print(f"  interacted_frac = {r['interacted_fraction']:.5f}")
    print()
