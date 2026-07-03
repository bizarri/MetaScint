from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from physics.electron_tabata import TabataRangeModel


def test_tabata_ranges_increase_with_energy():
    model = TabataRangeModel({'Bi':4,'Ge':3,'O':12}, 7.13)
    assert model.range_mm(50.0) < model.range_mm(100.0) < model.range_mm(200.0)
