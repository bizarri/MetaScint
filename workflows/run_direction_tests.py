from _bootstrap import ROOT
from config.loader import load_benchmark_config
from geometry.uniform_pixel import sample_entry_xy
import numpy as np

cfg = load_benchmark_config(ROOT / 'configs' / 'benchmarks' / 'uniform_bgo_lso_511keV.json')
rng = np.random.default_rng(1234)
for i in range(5):
    x, y = sample_entry_xy(rng, cfg.geometry, cfg.irradiation)
    print({'index': i, 'entry_xy_mm': [x, y], 'direction': [0.0, 0.0, 1.0]})
