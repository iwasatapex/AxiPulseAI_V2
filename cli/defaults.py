import json
from pathlib import Path

DEFAULTS_PATH = Path(__file__).resolve().parents[1] / "defaults.json"


def load_defaults(path=DEFAULTS_PATH):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def save_defaults(defaults, path=DEFAULTS_PATH):
    target = Path(path)
    target.write_text(json.dumps(defaults, indent=2) + "\n", encoding="utf-8")
    return target


def display_defaults(defaults=None):
    values = load_defaults() if defaults is None else defaults
    for key, value in values.items():
        print(f"{key}: {value}")
    return values


def update_defaults_from_args(defaults, args):
    values = dict(defaults)
    for key in values:
        value = getattr(args, key, None)
        if value is not None:
            values[key] = value
    return values
