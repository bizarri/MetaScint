from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from geometry.validator import validate_geometry_config


def test_uniform_geometry_validates():
    res = validate_geometry_config(ROOT / 'configs' / 'geometry' / 'uniform_pixel_3x3x15mm.json')
    assert res['valid'] is True
