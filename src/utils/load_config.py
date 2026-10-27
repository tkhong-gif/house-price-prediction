"""Load config/config.yaml, the single place where the project's paths and settings live."""

import pathlib
import yaml

PROJECT_PATH = pathlib.Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_PATH / "config" / "config.yaml"

def load_config():
    """Return config/config.yaml as a dict, with paths.root set to this project's
    absolute path (found from this file's location), so every relative path in the
    config works no matter which directory the code is run from."""
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    config["paths"]["root"] = str(PROJECT_PATH) + "/"
    return config

if __name__ == "__main__":
    config = load_config()
    print(config)
