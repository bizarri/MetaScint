from _bootstrap import ROOT
from geometry.validator import validate_geometry_config

for rel in [
    ROOT / 'configs' / 'geometry' / 'uniform_pixel_3x3x15mm.json',
    ROOT / 'configs' / 'geometry' / 'composite_channel_reference.json',
]:
    result = validate_geometry_config(rel)
    print(result)
