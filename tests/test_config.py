import pytest
import yaml

from src.config import REQUIRED_SECTIONS, get_env, load_config


def test_default_config_loads():
    cfg = load_config("configs/default.yaml")
    for section in REQUIRED_SECTIONS:
        assert section in cfg
    assert cfg["agent"]["max_tool_calls"] == 4


def test_every_state_has_a_dwell_time():
    cfg = load_config("configs/default.yaml")
    dwell = cfg["state_machine"]["min_dwell_sec"]
    for state in ["lying_in_bed", "sitting_on_bed", "sitting_outside_bed",
                  "standing", "walking", "lying_outside_bed", "unknown"]:
        assert dwell[state] > 0


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_config("configs/does_not_exist.yaml")


def test_missing_section_raises(tmp_path):
    cfg = load_config("configs/default.yaml")
    del cfg["alerts"]
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="alerts"):
        load_config(bad)


def test_bad_vlm_provider_raises(tmp_path):
    cfg = load_config("configs/default.yaml")
    cfg["vlm"]["provider"] = "openai"
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="provider"):
        load_config(bad)


def test_get_env_missing(monkeypatch):
    monkeypatch.delenv("SOME_UNSET_VAR", raising=False)
    with pytest.raises(RuntimeError, match="SOME_UNSET_VAR"):
        get_env("SOME_UNSET_VAR")
