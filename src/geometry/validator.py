from pathlib import Path
import json
from geometry.manufacturing import check_manufacturing_rules


def validate_geometry_config(path) -> dict:
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    issues = []

    px = data.get('pixel_x_mm', 0)
    py = data.get('pixel_y_mm', 0)
    pz = data.get('pixel_z_mm', 0)

    for key, val in [('pixel_x_mm', px), ('pixel_y_mm', py), ('pixel_z_mm', pz)]:
        if key in data and val <= 0:
            issues.append(f'{key} must be positive')

    issues.extend(check_manufacturing_rules(data))

    # Region-level checks (only meaningful if pixel dimensions are valid)
    regions = data.get('regions', [])
    if regions and px > 0 and py > 0 and pz > 0:
        hx, hy = px / 2.0, py / 2.0
        seen_names: set[str] = set()
        for r in regions:
            name = r.get('name', '<unnamed>')
            if name in seen_names:
                issues.append(f'region name "{name}" is not unique')
            seen_names.add(name)

            mat_role = r.get('material_role', '')
            if not mat_role:
                issues.append(f'region "{name}" has an empty material_role')

            tilt = r.get('tilt_deg', 0.0)
            if abs(tilt) < 1e-8:
                xmin, xmax = r.get('xmin', 0), r.get('xmax', 0)
                ymin, ymax = r.get('ymin', 0), r.get('ymax', 0)
                if xmin < -hx - 1e-9 or xmax > hx + 1e-9:
                    issues.append(f'region "{name}" x-bounds [{xmin}, {xmax}] exceed pixel x [{-hx}, {hx}]')
                if ymin < -hy - 1e-9 or ymax > hy + 1e-9:
                    issues.append(f'region "{name}" y-bounds [{ymin}, {ymax}] exceed pixel y [{-hy}, {hy}]')

            zmin, zmax = r.get('zmin', 0), r.get('zmax', 0)
            if zmin < -1e-9 or zmax > pz + 1e-9:
                issues.append(f'region "{name}" z-bounds [{zmin}, {zmax}] exceed pixel z [0, {pz}]')

    return {'path': str(path), 'issues': issues, 'valid': len(issues) == 0}
