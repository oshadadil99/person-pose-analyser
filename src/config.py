"""Loads the YAML config and the .env file.

The config is kept as a plain nested dict on purpose. Every module reads the
section it needs (e.g. cfg["frame_rules"]["walking_speed"]), which is easy to
follow and means tuning never touches code. Secrets only come from the
environment, never from the YAML.
"""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

REQUIRED_SECTIONS = [
    "sampling", "perception", "camera", "scene", "patient", "features", "frame_rules", "state_machine",
    "triggers", "agent", "vlm", "events", "alerts", "evaluation",
]
VLM_PROVIDERS = ("vertex", "aistudio", "none")


def load_config(path: str | Path = "configs/default.yaml") -> dict:
    """Read the YAML config, load .env, and fail early if something is missing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    missing = [s for s in REQUIRED_SECTIONS if s not in cfg]
    if missing:
        raise ValueError(f"Config {path} is missing sections: {missing}")

    if cfg["vlm"]["provider"] not in VLM_PROVIDERS:
        raise ValueError(f"vlm.provider must be one of {VLM_PROVIDERS}")

    # .env is optional: with vlm.provider "none" the pipeline needs no secrets.
    load_dotenv()
    return cfg


def get_env(name: str) -> str:
    """Return an environment variable, with a clear error if it isn't set."""
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set. Add it to your .env file (see .env.example).")
    return value
