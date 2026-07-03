from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from config.loader import load_benchmark_config
from simulation.uniform_benchmark import run_uniform_material


def test_smoke_runs_small_sample():
    cfg = load_benchmark_config(ROOT / 'configs' / 'benchmarks' / 'uniform_bgo_lso_511keV.json')
    sim = cfg.simulation
    small_cfg = type(cfg)(materials=cfg.materials, geometry=cfg.geometry, irradiation=cfg.irradiation, simulation=type(sim)(**{**sim.__dict__, 'n_events': 50}))
    r = run_uniform_material('BGO', small_cfg.materials['BGO'], small_cfg, seed_offset=0)
    assert r['n_events'] == 50
    assert (r['full_count'] + r['partial_count'] + r['none_count']) == 50
