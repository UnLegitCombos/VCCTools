"""Tests for teamMaker.core.scoring (synthetic players only)."""

import warnings

import pytest

from teamMaker.core import scoring
from teamMaker.core.config import DEFAULT_RANK_VALUES, DEFAULTS, deep_merge
from teamMaker.core.scoring import (
    REGION_ALIASES,
    apply_top_tier_compression,
    compute_player_score_detailed,
    normalize_season,
    parse_peak_act,
    rank_to_numeric_with_rr,
    score_players,
)

FAKE_DIST = [0.5 + 0.02 * i for i in range(40)]


@pytest.fixture(autouse=True)
def _clean_caches():
    scoring._reset_caches()
    yield
    scoring._reset_caches()


def make_config(**overrides):
    cfg = deep_merge(DEFAULTS, {})
    cfg["season_rating_distributions"] = {"S12": FAKE_DIST, "S13": FAKE_DIST}
    cfg.update(overrides)
    return cfg


def make_player(**kw):
    info = {
        "current_rank": "Diamond 1",
        "peak_rank": "Diamond 1",
        "region": "EU",
        "is_returning_player": False,
    }
    info.update(kw)
    return info


# --- peak acts ------------------------------------------------------------


def test_parse_peak_act_season_uses_six_acts():
    # S25A1 -> S26A5 is 6 + 4 = 10 acts
    assert parse_peak_act("S25A1", 26, 5, acts_per_season=6) == 10
    assert parse_peak_act("S26A3", 26, 5) == 2


def test_parse_peak_act_accepts_v_alias_and_high_acts():
    assert parse_peak_act("V26A4", 26, 5) == 1
    assert parse_peak_act("S26A5", 26, 5) == 0
    assert parse_peak_act("S26A6", 26, 5) == 0
    assert parse_peak_act("S26A7", 26, 5) == 0  # invalid act


def test_parse_peak_act_episode_acts_stay_three():
    # E9A3 is the transition (27 acts); current S26A5 = 27 + 6 + 5
    assert parse_peak_act("E9A3", 26, 5, acts_per_season=6) == 11
    assert parse_peak_act("E8A1", 26, 5, acts_per_season=6) == 11 + 5
    assert parse_peak_act("E10A1", 26, 5) == 0


def test_peak_act_age_uses_acts_per_season():
    cfg = make_config(use_returning_player_stats=False, use_ping_adjustment=False)
    player = make_player(peak_rank_act="E7A3")
    detail = compute_player_score_detailed(player, cfg)
    assert detail["advanced_components"]["peak_act"]["acts_ago"] == 11 + 6


def test_old_peak_fades_toward_current_rank():
    cfg = make_config(
        use_returning_player_stats=False, use_ping_adjustment=False, use_tracker=False,
        current_season=26, current_act=5, acts_per_season=6, peak_act_decay_rate=0.9,
    )
    def peak_value(**kw):
        d = compute_player_score_detailed(make_player(**kw), cfg)
        return d["rank_components"]["peak_rank_value"], d["final_score"]
    cur = cfg["rank_values"]["Diamond 3"]
    peak = cfg["rank_values"]["Immortal 3"]
    fresh, fresh_score = peak_value(current_rank="Diamond 3", peak_rank="Immortal 3", peak_rank_act="S26A5")
    old, old_score = peak_value(current_rank="Diamond 3", peak_rank="Immortal 3", peak_rank_act="S25A5")
    no_act, _ = peak_value(current_rank="Diamond 3", peak_rank="Immortal 3")
    assert fresh == peak and no_act == peak  # this act / unknown act: full peak
    assert old == pytest.approx(cur + (peak - cur) * 0.9 ** 6)  # 6 acts ago
    assert old_score < fresh_score  # older peak counts less, never a bonus
    # A peak at or below the current rank is untouched.
    same, _ = peak_value(current_rank="Immortal 3", peak_rank="Immortal 3", peak_rank_act="E6A3")
    assert same == peak


def test_removed_peak_act_keys_warn():
    from teamMaker.core.config import DEPRECATED_KEYS
    for key in ("peak_act_max_episode", "peak_act_max_act", "weight_peak_act",
                "mode", "use_region_debuff", "non_eu_debuff"):
        assert key in DEPRECATED_KEYS


# --- regions --------------------------------------------------------------


def test_region_alias_me_to_mena():
    assert REGION_ALIASES == {"ME": "MENA"}


def test_mena_gets_ping_estimate_penalty():
    cfg = make_config(use_new_player_debuff=False)
    eu = compute_player_score_detailed(make_player(region="EU"), cfg)
    mena = compute_player_score_detailed(make_player(region="MENA"), cfg)
    me = compute_player_score_detailed(make_player(region="ME"), cfg)
    adj = mena["advanced_components"]["ping_adjustment"]
    assert adj["ping"] == 80
    assert adj["multiplier"] == pytest.approx(0.95)
    assert mena["final_score"] < eu["final_score"]
    assert me["final_score"] == pytest.approx(mena["final_score"])


def test_legacy_me_key_in_estimates_still_works():
    cfg = make_config(
        use_new_player_debuff=False,
        region_ping_estimates={"EU": 30, "ME": 110},
    )
    detail = compute_player_score_detailed(make_player(region="MENA"), cfg)
    assert detail["advanced_components"]["ping_adjustment"]["ping"] == 110


def test_default_breakpoints_do_not_penalise_eu():
    cfg = make_config(use_new_player_debuff=False)
    eu = compute_player_score_detailed(make_player(region="EU"), cfg)
    assert eu["advanced_components"]["ping_adjustment"]["multiplier"] == 1.0


# --- RR granularity -------------------------------------------------------


def _v25_rr(rank, base, rr):
    """The v2.5 hardcoded formula (Immortal 3 = 25, Radiant = 30)."""
    if rank == "Immortal 3":
        return 25.0 + min(2.0, rr / 100.0)
    if rank == "Radiant":
        return 30.0 + max(0.0, (rr - 550) / 100.0)
    return base


def test_rr_parity_with_v25_when_imm3_is_25():
    values = dict(DEFAULT_RANK_VALUES, **{"Immortal 3": 25})
    cfg = {"rr_granularity_ranks": ["Immortal 3", "Radiant"]}
    for rank in ("Immortal 2", "Immortal 3", "Radiant"):
        for rr in (0, 50, 199, 200, 350, 550, 600, 800):
            expected = _v25_rr(rank, values[rank], rr)
            assert rank_to_numeric_with_rr(rank, values, rr, cfg) == pytest.approx(
                expected
            )


def test_rr_is_relative_to_rank_values():
    values = dict(DEFAULT_RANK_VALUES)  # Immortal 3 = 27
    cfg = {"rr_bonus_cap": 2.0}
    assert rank_to_numeric_with_rr("Immortal 3", values, 150, cfg) == pytest.approx(28.5)
    assert rank_to_numeric_with_rr("Immortal 3", values, 900, cfg) == pytest.approx(29.0)
    cfg = {"rr_bonus_cap": 1.0}
    assert rank_to_numeric_with_rr("Immortal 3", values, 900, cfg) == pytest.approx(28.0)


def test_rr_only_for_listed_ranks_and_threshold_is_live():
    values = dict(DEFAULT_RANK_VALUES)
    cfg = {"rr_granularity_ranks": ["Radiant"], "radiant_rr_threshold": 400}
    assert rank_to_numeric_with_rr("Immortal 3", values, 150, cfg) == 27
    assert rank_to_numeric_with_rr("Radiant", values, 500, cfg) == pytest.approx(31.0)
    assert rank_to_numeric_with_rr("Radiant", values, None, cfg) == 30


# --- seasons --------------------------------------------------------------


def test_normalize_season():
    assert normalize_season("14") == "S14"
    assert normalize_season(14) == "S14"
    assert normalize_season("s14") == "S14"
    assert normalize_season("V14") == "S14"
    assert normalize_season(" S9 ") == "S9"


def test_season_14_alias_scores_like_s14():
    cfg = make_config(
        season_rating_distributions={"S13": FAKE_DIST, "S14": FAKE_DIST}
    )
    a = make_player(
        is_returning_player=True,
        previous_season_stats=[{"season": "14", "adjusted_rating": 1.0}],
    )
    b = make_player(
        is_returning_player=True,
        previous_season_stats=[{"season": "S14", "adjusted_rating": 1.0}],
    )
    sa = compute_player_score_detailed(a, cfg)
    sb = compute_player_score_detailed(b, cfg)
    assert sa["final_score"] == pytest.approx(sb["final_score"])
    assert sa["advanced_components"]["previous_season"]["previous_season_score"] > 0


def test_unknown_season_warns_once():
    cfg = make_config()
    player = make_player(
        is_returning_player=True,
        previous_season_stats=[{"season": "S99", "adjusted_rating": 1.0}],
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compute_player_score_detailed(player, cfg)
        compute_player_score_detailed(player, cfg)
    season_warnings = [w for w in caught if "S99" in str(w.message)]
    assert len(season_warnings) == 1


def test_latest_season_auto_uses_newest_distribution():
    cfg = make_config(latest_season_available="auto")
    assert scoring._latest_season_available(cfg) == 13
    cfg = make_config(latest_season_available=14)
    assert scoring._latest_season_available(cfg) == 14


def test_recent_vs_older_blend_follows_auto_latest():
    cfg = make_config(use_new_player_debuff=False, use_ping_adjustment=False)
    recent = make_player(
        is_returning_player=True,
        previous_season_stats=[{"season": "S13", "adjusted_rating": 1.0}],
    )
    older = make_player(
        is_returning_player=True,
        previous_season_stats=[{"season": "S12", "adjusted_rating": 1.0}],
    )
    r = compute_player_score_detailed(recent, cfg)["advanced_components"]["previous_season"]
    o = compute_player_score_detailed(older, cfg)["advanced_components"]["previous_season"]
    assert r["previous_weight"] == cfg["recent_data_previous_weight"]
    assert o["previous_weight"] == cfg["older_data_previous_weight"]


# --- compression ----------------------------------------------------------


def test_soft_knee_is_monotone_and_never_raises():
    scores = {f"p{i}": 10.0 + i * 0.7 for i in range(30)}
    cfg = {"top_tier_compression": {"knee_percentile": 80, "slope": 0.5}}
    out = apply_top_tier_compression(scores, cfg)
    for name, value in scores.items():
        assert out[name] <= value + 1e-12
    ordered = sorted(scores, key=lambda n: scores[n])
    compressed = [out[n] for n in ordered]
    assert compressed == sorted(compressed)
    assert len(set(compressed)) == len(compressed)  # strictly increasing
    # Below the knee nothing moves
    assert out["p0"] == scores["p0"]
    # The top player is lowered
    assert out["p29"] < scores["p29"]


def test_soft_knee_slope_one_is_identity():
    scores = {"a": 10.0, "b": 20.0, "c": 30.0}
    out = apply_top_tier_compression(scores, {"top_tier_compression": {"slope": 1.0}})
    assert out == scores


def test_compression_disabled_by_default_in_score_players():
    cfg = make_config()
    players = {"a": make_player(), "b": make_player(current_rank="Radiant", peak_rank="Radiant")}
    scores, breakdowns = score_players(players, cfg)
    assert "compressed_final_score" not in breakdowns["b"]
    assert scores["b"] == round(breakdowns["b"]["final_score"], 1)


# --- score_players --------------------------------------------------------


def test_score_players_returns_rounded_scores_and_breakdowns():
    cfg = make_config()
    players = {
        "solo": make_player(current_rank="Gold 1", peak_rank="Gold 2"),
        "top": make_player(current_rank="Radiant", peak_rank="Radiant", current_rr=650),
    }
    scores, breakdowns = score_players(players, cfg)
    assert list(scores) == ["solo", "top"]
    assert set(breakdowns) == {"solo", "top"}
    assert scores["top"] > scores["solo"]
    for name, value in scores.items():
        assert value == round(value, 1)
        assert breakdowns[name]["final_score"] == pytest.approx(value, abs=0.05)


def test_score_players_applies_enabled_compression():
    cfg = make_config(top_tier_compression={"enabled": True, "knee_percentile": 50, "slope": 0.5})
    players = {
        "a": make_player(current_rank="Gold 1", peak_rank="Gold 1"),
        "b": make_player(current_rank="Diamond 1", peak_rank="Diamond 1"),
        "c": make_player(current_rank="Radiant", peak_rank="Radiant"),
    }
    plain, _ = score_players(players, make_config())
    scores, breakdowns = score_players(players, cfg)
    assert scores["c"] < plain["c"]
    assert scores["a"] == plain["a"]
    assert breakdowns["c"]["original_final_score"] > breakdowns["c"]["final_score"]


# --- missing peak ---------------------------------------------------------


def test_missing_peak_rank_counts_as_current():
    cfg = make_config()
    with_peak = compute_player_score_detailed(make_player(), cfg)
    no_peak = compute_player_score_detailed(make_player(peak_rank=None), cfg)
    assert no_peak["rank_components"]["peak_rank_value"] == (
        with_peak["rank_components"]["current_rank_value"]
    )
    assert no_peak["final_score"] == pytest.approx(with_peak["final_score"])


def test_rank_lookup_ignores_case_and_spaces():
    values = {"Diamond 1": 16, "Diamond 2": 17, "Immortal 3": 27, "Radiant": 30}
    assert scoring.rank_to_numeric("diamond 1", values) == 16
    assert scoring.rank_to_numeric("  DIAMOND   1 ", values) == 16
    assert scoring.rank_to_numeric("diamond", values) == 17
    cfg = {"rr_granularity_ranks": ["Immortal 3", "Radiant"], "rr_bonus_cap": 2.0}
    assert rank_to_numeric_with_rr("immortal 3", values, 150, cfg) == 28.5
    with pytest.warns(UserWarning):
        assert scoring.rank_to_numeric("Plat 2", values) == 0


# --- vcc_mode -------------------------------------------------------------


def _returning(adjusted_rating, **kw):
    return make_player(
        is_returning_player=True,
        previous_season_stats=[{"season": "S13", "adjusted_rating": adjusted_rating}],
        **kw,
    )


def _vcc_cfg(**kw):
    return make_config(use_ping_adjustment=False, use_tracker=False, **kw)


def test_default_vcc_mode_is_the_old_blend():
    cfg = _vcc_cfg()
    assert cfg["vcc_mode"] == "blend"
    d = compute_player_score_detailed(_returning(1.0), cfg)
    prev = d["advanced_components"]["previous_season"]
    assert "vcc_adjustment" not in prev
    assert prev["post_blend_score"] == pytest.approx(
        prev["pre_blend_score"] * 0.65 + prev["previous_season_score"] * 0.35
    )


def test_ranked_baseline_no_change_when_vcc_matches_rank():
    cfg = _vcc_cfg(vcc_mode="ranked_baseline")
    base = compute_player_score_detailed(_returning(1.0), _vcc_cfg(use_returning_player_stats=False, use_new_player_debuff=False))
    ranked = base["final_score"]
    # Put the player exactly at their expected VCC level.
    expected_z = (ranked - cfg["vcc_center"]) / cfg["vcc_points_per_sd"]
    target_score = 18.0 + 8.0 * expected_z
    # Find the adjusted rating whose season score is target_score (FAKE_DIST is linear).
    mean = sum(FAKE_DIST) / len(FAKE_DIST)
    std = (sum((x - mean) ** 2 for x in FAKE_DIST) / len(FAKE_DIST)) ** 0.5
    rating = mean + std * (target_score - 18.0) / 8.0
    d = compute_player_score_detailed(_returning(rating), cfg)
    assert d["advanced_components"]["previous_season"]["vcc_adjustment"] == pytest.approx(0.0, abs=1e-6)
    assert d["final_score"] == pytest.approx(ranked)


def test_ranked_baseline_adjustment_follows_vcc_and_is_capped():
    cfg = _vcc_cfg(vcc_mode="ranked_baseline")
    low_rank = dict(current_rank="Gold 1", peak_rank="Gold 1")
    high_rank = dict(current_rank="Immortal 2", peak_rank="Immortal 2")
    good = compute_player_score_detailed(_returning(1.3, **low_rank), cfg)["advanced_components"]["previous_season"]
    bad = compute_player_score_detailed(_returning(0.55, **high_rank), cfg)["advanced_components"]["previous_season"]
    assert good["vcc_adjustment"] > 0 > bad["vcc_adjustment"]
    assert abs(good["vcc_adjustment"]) <= cfg["vcc_max_adjustment"]
    tight = _vcc_cfg(vcc_mode="ranked_baseline", vcc_max_adjustment=0.1)
    capped = compute_player_score_detailed(_returning(1.3, **low_rank), tight)["advanced_components"]["previous_season"]
    assert capped["vcc_adjustment"] == pytest.approx(0.1)


def test_ranked_baseline_higher_weight_only_amplifies_the_gap():
    low = _vcc_cfg(vcc_mode="ranked_baseline", recent_data_previous_weight=0.1, older_data_previous_weight=0.1)
    high = _vcc_cfg(vcc_mode="ranked_baseline", recent_data_previous_weight=0.5, older_data_previous_weight=0.5)
    p = _returning(1.3)
    a = compute_player_score_detailed(p, low)["advanced_components"]["previous_season"]["vcc_adjustment"]
    b = compute_player_score_detailed(p, high)["advanced_components"]["previous_season"]["vcc_adjustment"]
    assert 0 < a < b


def test_missing_tracker_is_flagged():
    cfg = make_config(use_ping_adjustment=False)
    d = compute_player_score_detailed(make_player(), cfg)
    assert d["tracker_components"] == {"enabled": False, "tracker_missing": True}
    d = compute_player_score_detailed(make_player(tracker_current=800, tracker_peak=900), cfg)
    assert d["tracker_components"]["enabled"] is True


def test_calibrate_recovers_a_known_line():
    cfg = _vcc_cfg()
    players = {}
    ranks = ["Gold 1", "Gold 3", "Platinum 2", "Diamond 1", "Diamond 3", "Ascendant 2", "Immortal 1", "Immortal 2"]
    mean = sum(FAKE_DIST) / len(FAKE_DIST)
    std = (sum((x - mean) ** 2 for x in FAKE_DIST) / len(FAKE_DIST)) ** 0.5
    for i, rank in enumerate(ranks):
        ranked = compute_player_score_detailed(
            make_player(current_rank=rank, peak_rank=rank), _vcc_cfg(use_returning_player_stats=False, use_new_player_debuff=False)
        )["final_score"]
        z = (ranked - 25.0) / 10.0  # exact line: center 25, 10 points per sd
        players[f"p{i}"] = _returning(mean + std * z, current_rank=rank, peak_rank=rank)
    fit = scoring.calibrate(players, cfg)
    assert fit is not None
    assert fit["center"] == pytest.approx(25.0, abs=0.2)
    assert fit["points_per_sd"] == pytest.approx(10.0, abs=0.2)
    assert fit["r"] == pytest.approx(1.0, abs=0.01)
    assert scoring.calibrate({"a": players["p0"]}, cfg) is None
