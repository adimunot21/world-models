"""
Configuration loading and validation.

Provides a single entry point for loading YAML configs and merging
with CLI overrides. All config access goes through this module —
no scattered yaml.safe_load() calls elsewhere.
"""

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file and return as a nested dict.

    Args:
        config_path: Path to the YAML file (relative or absolute).

    Returns:
        Parsed config as a dictionary.

    Raises:
        FileNotFoundError: If config_path does not exist.
        yaml.YAMLError: If the YAML is malformed.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path.resolve()}")
    with open(path) as f:
        config = yaml.safe_load(f)
    if config is None:
        return {}
    return config


def get_nested(config: dict, key_path: str, default: Any = None) -> Any:
    """Access a nested config value using dot notation.

    Example:
        get_nested(config, "model.latent_dim", 32)
        is equivalent to config["model"]["latent_dim"] with a default fallback.

    Args:
        config: The config dictionary.
        key_path: Dot-separated path to the value (e.g., "training.batch_size").
        default: Value to return if the key path doesn't exist.

    Returns:
        The value at key_path, or default if not found.
    """
    keys = key_path.split(".")
    current = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


if __name__ == "__main__":
    config_dir = Path("config")
    if not config_dir.exists():
        print("Run from project root (~/projects/world-models/)")
    else:
        for yaml_file in sorted(config_dir.glob("*.yaml")):
            cfg = load_config(yaml_file)
            print(f"{yaml_file.name}: {list(cfg.keys())}")
