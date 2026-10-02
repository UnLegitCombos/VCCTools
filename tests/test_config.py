"""Tests for teamMaker.core.config (synthetic config files only)."""

import os
import warnings

import pytest

from teamMaker.core.config import (
    DEFAULTS,
    TIGHTER_PROFILE,
    deep_merge,
    load_team_config,
    load_yaml,
    resolve_seed,
)


def _write(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return str(path)


def test_deep_merge_child_wins_and_does_not_mutate():
    base = {"a": 1, "nested": {"x": 1, "y": 2}, "lst": [1, 2]}
    override = {"nested": {"y": 3}, "lst": [9]}
    merged = deep_merge(base, override)
    assert merged == {"a": 1, "nested": {"x": 1, "y": 3}, "lst": [9]}
    assert base["nested"]["y"] == 2


def test_load_yaml_reads_utf8(tmp_path):
    path = _write(tmp_path / "c.yaml", "title: café Λ\n")
    assert load_yaml(path)["title"] == "café Λ"


def test_extends_chain_merges_child_over_parent(tmp_path):
    _write(tmp_path / "base.yaml", "weight_current: 0.5\nweight_peak: 0.5\ncurrent_act: 2\n")
    _write(tmp_path / "mid.yaml", "extends: base.yaml\nweight_current: 0.6\n")
    top = _write(tmp_path / "top.yaml", "extends: mid.yaml\ncurrent_act: 4\n")
    cfg = load_team_config(top)
    assert cfg["weight_current"] == 0.6
    assert cfg["weight_peak"] == 0.5
    assert cfg["current_act"] == 4
    assert [os.path.basename(p) for p in cfg["_sources"]] == [
        "base.yaml",
        "mid.yaml",
        "top.yaml",
    ]
    assert "extends" not in cfg
    # Untouched keys come from DEFAULTS
    assert cfg["acts_per_season"] == DEFAULTS["acts_per_season"]


def test_extends_cycle_is_detected(tmp_path):
    _write(tmp_path / "a.yaml", "extends: b.yaml\n")
    _write(tmp_path / "b.yaml", "extends: a.yaml\n")
    with pytest.raises(ValueError, match="cycle"):
        load_team_config(str(tmp_path / "a.yaml"))


def test_missing_extends_target_raises(tmp_path):
    path = _write(tmp_path / "a.yaml", "extends: nope.yaml\n")
    with pytest.raises(FileNotFoundError):
        load_team_config(path)


def test_falls_back_to_example_when_no_local_config(tmp_path):
    _write(tmp_path / "config.example.yaml", "current_act: 3\n")
    cfg = load_team_config(config_dir=str(tmp_path))
    assert cfg["current_act"] == 3
    assert os.path.basename(cfg["_sources"][0]) == "config.example.yaml"


def test_local_config_wins_over_example(tmp_path):
    _write(tmp_path / "config.example.yaml", "current_act: 3\n")
    _write(tmp_path / "config.yaml", "current_act: 1\n")
    cfg = load_team_config(config_dir=str(tmp_path))
    assert cfg["current_act"] == 1


def test_no_config_at_all_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_team_config(config_dir=str(tmp_path))


def test_unknown_key_warns(tmp_path):
    path = _write(tmp_path / "c.yaml", "definitely_not_a_key: 1\n")
    with pytest.warns(UserWarning, match="Unknown config key 'definitely_not_a_key'"):
        cfg = load_team_config(path)
    assert any("definitely_not_a_key" in w for w in cfg["_warnings"])


def test_unknown_nested_optimizer_key_warns(tmp_path):
    path = _write(tmp_path / "c.yaml", "optimizer:\n  bogus: 1\n")
    with pytest.warns(UserWarning, match="optimizer.bogus"):
        load_team_config(path)


def test_deprecated_debuff_and_compression_keys_warn(tmp_path):
    path = _write(
        tmp_path / "c.yaml",
        "use_rating_scaled_debuffs: true\ndebuff_max_penalty: 0.3\n"
        "use_top_tier_compression: true\n",
    )
    with pytest.warns(UserWarning) as record:
        cfg = load_team_config(path)
    text = " ".join(str(r.message) for r in record)
    assert "use_rating_scaled_debuffs" in text
    assert "debuff_max_penalty" in text
    assert "use_top_tier_compression" in text
    # The old flag must not turn the new compression on
    assert cfg["top_tier_compression"]["enabled"] is False


def test_legacy_optimizer_keys_are_mapped(tmp_path):
    path = _write(
        tmp_path / "c.yaml",
        "role_balance_weight: 5\nmax_time: 90\nearly_termination_threshold: 0.25\n"
        "max_restarts: 7\ncooling_rate: 0.99\ntabu_tenure: 10\n",
    )
    with pytest.warns(UserWarning) as record:
        cfg = load_team_config(path)
    opt = cfg["optimizer"]
    assert opt["role_weight"] == pytest.approx(0.5)
    assert opt["time_limit_s"] == 90
    assert opt["target_range"] == 0.25
    assert opt["restarts"] == 7
    text = " ".join(str(r.message) for r in record)
    assert "cooling_rate" in text and "ignored" in text


def test_legacy_key_does_not_override_explicit_optimizer_value(tmp_path):
    path = _write(
        tmp_path / "c.yaml", "max_restarts: 7\noptimizer:\n  restarts: 3\n"
    )
    with pytest.warns(UserWarning):
        cfg = load_team_config(path)
    assert cfg["optimizer"]["restarts"] == 3


def test_groups_block_gets_defaults_only_when_configured(tmp_path):
    path = _write(tmp_path / "c.yaml", "current_act: 1\n")
    assert load_team_config(path)["groups"] is None
    path = _write(tmp_path / "g.yaml", "groups:\n  count: 2\n")
    cfg = load_team_config(path)
    assert cfg["groups"]["count"] == 2
    assert cfg["groups"]["server"] == "Frankfurt"
    assert cfg["groups"]["servers"] is None


def test_resolve_seed_uses_configured_seed():
    cfg = {"random_seed": 13}
    assert resolve_seed(cfg) == 13
    assert cfg["_seed_generated"] is False


def test_resolve_seed_draws_and_records_when_null(capsys):
    cfg = {"random_seed": None}
    seed = resolve_seed(cfg)
    assert isinstance(seed, int)
    assert cfg["random_seed"] == seed
    assert cfg["_seed_generated"] is True
    assert str(seed) in capsys.readouterr().out


def test_shipped_configs_load_without_warnings():
    from teamMaker.core.config import CONFIG_DIR

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        example = load_team_config(os.path.join(CONFIG_DIR, "config.example.yaml"))
        tighter = load_team_config(
            os.path.join(CONFIG_DIR, "config.example.yaml"), profile=TIGHTER_PROFILE
        )
    assert example["rank_values"]["Immortal 3"] == 27
    assert example["top_tier_compression"]["enabled"] is False
    assert example["latest_season_available"] == "auto"
    # The profile only changes optimizer settings; seed and scoring stay.
    assert tighter["optimizer"]["target_range"] == 0.05
    assert tighter["optimizer"]["time_limit_s"] == 300
    assert tighter["optimizer"]["iterations"] == example["optimizer"]["iterations"]
    assert tighter["random_seed"] == example["random_seed"]
    assert tighter["weight_current"] == example["weight_current"]
    assert [os.path.basename(p) for p in tighter["_sources"]] == [
        "config.example.yaml", TIGHTER_PROFILE,
    ]
