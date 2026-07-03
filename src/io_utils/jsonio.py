from pathlib import Path
import json


def dump_json(data, path):
    Path(path).write_text(json.dumps(data, indent=2), encoding='utf-8')
