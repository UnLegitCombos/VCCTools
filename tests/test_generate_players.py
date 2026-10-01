"""Tests for generate_players (synthetic data only)."""

from teamMaker.core import generate_players as gp
from teamMaker.core.utils.vcc import SeasonStats

UUID_A = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
UUID_B = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"

CONFIG = {
    "min_rounds": 100,
    "region_map": {"EU": "EU", "NA": "NA", "MENA": "MENA"},
    "season_map": {"EMEA Season 1": "S_1", "EMEA Season 2": "S_2"},
    "seasons": {
        "S_1": {"code": "S1", "link": "https://vcc.example/s1"},
        "S_2": {"code": "S2", "link": "https://vcc.example/s2"},
    },
    "vcc": {
        "user_agent": "test-agent",
        "request_delay": 0.0,
        "stats_stage": "all",
        "ar_decimals": 2,
    },
}


def make_row(sid, name, status="Approved", stack="Solo", server="EU", **extra):
    row = {
        "tally_submission_id": sid,
        "stack": stack,
        "discord": f"{name.lower()}_dc",
        "current_rank": "Gold 2",
        "peak_rank": "Platinum 1",
        "peak_rank_act": "S25A6",
        "tracker_current": "500",
        "tracker_peak": "600",
        "roles": "Duelist, Initiator",
        "server": server,
        "returning": "FALSE",
        "vcc_profile": "",
        "previous_seasons": "",
        "display_name": name,
        "status": status,
    }
    row.update(extra)
    return row


def sample_rows():
    return [
        make_row("s1", "Alpha"),
        make_row("s2", "Bravo", stack="Duo"),
        make_row("s2", "Charlie", stack="Duo"),
        make_row("s3", "Delta", stack="Trio"),
        make_row("s3", "Echo", stack="Trio"),
        make_row("s3", "Foxtrot", stack="Trio"),
        make_row("s4", "Golf", status="Denied"),
        make_row("s5", "Hotel", status="Substitute"),
        make_row("s6", "India", status="Duplicate"),
        make_row("s7", "Juliet", status=""),
    ]


# --- CSV stage ---------------------------------------------------------


def test_build_entries_groups_and_status():
    result = gp.build_player_entries(sample_rows(), CONFIG)
    e = result.entries
    assert set(e) == {"Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot", "Hotel"}
    assert e["Alpha"]["group_id"] == 1
    assert e["Bravo"]["group_id"] == e["Charlie"]["group_id"] == 2
    assert e["Delta"]["group_id"] == e["Echo"]["group_id"] == e["Foxtrot"]["group_id"] == 3
    assert e["Hotel"]["group_id"] is None
    assert e["Hotel"]["status"] == "substitute"
    assert e["Alpha"]["status"] == "active"
    assert all(isinstance(v["group_id"], (int, type(None))) for v in e.values())
    assert result.skipped == {"Denied": 1, "Duplicate": 1, "(blank)": 1}
    assert result.warnings == []


def test_group_id_is_int_for_alphanumeric_submission_ids():
    rows = [make_row("wA1b2C3", "Alpha"), make_row("zZ9", "Bravo")]
    result = gp.build_player_entries(rows, CONFIG)
    assert [e["group_id"] for e in result.entries.values()] == [1, 2]


def test_entry_fields_emitted():
    result = gp.build_player_entries(sample_rows(), CONFIG)
    entry = result.entries["Bravo"]
    assert entry["discord"] == "bravo_dc"
    assert entry["server"] == "EU"
    assert entry["stack"] == "Duo"
    assert entry["submission_id"] == "s2"
    assert entry["signup_index"] == 2
    assert entry["role"] == ["duelist", "initiator"]
    assert entry["region"] == "EU"


def test_group_size_counts():
    result = gp.build_player_entries(sample_rows(), CONFIG)
    assert dict(gp.group_size_counts(result.entries)) == {1: 1, 2: 1, 3: 1}


def test_group_override_column_merges_submissions():
    rows = [
        make_row("s1", "Alpha", stack="Duo", group_override="team-x"),
        make_row("s2", "Bravo", stack="Duo", group_override="team-x"),
        make_row("s3", "Charlie"),
    ]
    result = gp.build_player_entries(rows, CONFIG)
    assert result.entries["Alpha"]["group_id"] == result.entries["Bravo"]["group_id"]
    assert result.entries["Charlie"]["group_id"] != result.entries["Alpha"]["group_id"]
    assert result.warnings == []


def test_stack_mismatch_warns_and_names_missing_member():
    rows = [
        make_row("s1", "Alpha", stack="Duo"),
        make_row("s1", "Bravo", stack="Duo", status="Denied"),
    ]
    result = gp.build_player_entries(rows, CONFIG)
    assert len(result.warnings) == 1
    assert "declared 2 but 1" in result.warnings[0]
    assert "Bravo (Denied)" in result.warnings[0]
    assert result.entries["Alpha"]["group_id"] == 1


def test_four_stack_is_invalid_and_warned():
    rows = [make_row("s1", n, stack="Trio") for n in ("Alpha", "Bravo", "Charlie", "Delta")]
    result = gp.build_player_entries(rows, CONFIG)
    assert any("not allowed" in w and "4" in w for w in result.warnings)


def test_declared_four_stack_label_is_unknown():
    rows = [make_row("s1", "Alpha", stack="4 Stack")]
    result = gp.build_player_entries(rows, CONFIG)
    assert any("unknown stack value" in w for w in result.warnings)


def test_five_stack_is_valid():
    rows = [make_row("s1", f"P{i}", stack="5 Stack") for i in range(5)]
    result = gp.build_player_entries(rows, CONFIG)
    assert result.warnings == []
    assert {e["group_id"] for e in result.entries.values()} == {1}


def test_name_collision_uses_discord_suffix():
    rows = [
        make_row("s1", "Alpha", discord="one#1"),
        make_row("s2", "alpha", discord="two#2"),
    ]
    result = gp.build_player_entries(rows, CONFIG)
    assert set(result.entries) == {"Alpha (one#1)", "Alpha (two#2)"}
    assert any("collision" in w for w in result.warnings)


def test_vcc_slug_preferred_for_name():
    rows = [make_row("s1", "Alpha", vcc_profile=f"https://vcc.example/player/{UUID_A}/slugname")]
    result = gp.build_player_entries(rows, CONFIG)
    assert list(result.entries) == ["Slugname"]


def test_unknown_server_warns_and_defaults_to_eu():
    result = gp.build_player_entries([make_row("s1", "Alpha", server="XX")], CONFIG)
    assert result.entries["Alpha"]["region"] == "EU"
    assert any("unknown server" in w for w in result.warnings)


def test_region_mapping():
    result = gp.build_player_entries([make_row("s1", "Alpha", server="NA")], CONFIG)
    assert result.entries["Alpha"]["region"] == "NA"


# --- hand-entered ranks and tracker scores ------------------------------


def test_csv_ratings_are_parsed():
    row = make_row(
        "s1", "Alpha", current_rank=" immortal  3 ", current_rr="1,234",
        peak_rank="radiant", peak_rank_act="E26: A5",
        tracker_current="812", tracker_peak="950.5",
    )
    result = gp.build_player_entries([row], CONFIG)
    e = result.entries["Alpha"]
    assert e["current_rank"] == "Immortal 3"
    assert e["current_rr"] == 1234
    assert e["peak_rank"] == "Radiant"
    assert e["peak_rank_act"] == "S26A5"
    assert e["tracker_current"] == 812 and e["tracker_peak"] == 950.5
    assert result.warnings == [] and result.issues == {}


def test_rank_column_is_fallback_and_tier_only_warns():
    row = make_row("s1", "Alpha", current_rank="", rank="Diamond", peak_rank="Diamond 3")
    result = gp.build_player_entries([row], CONFIG)
    assert result.entries["Alpha"]["current_rank"] == "Diamond 2"
    assert any("no division" in w for w in result.warnings)


def test_missing_values_become_issues():
    row = make_row("s1", "Alpha", current_rank="", peak_rank="", tracker_current="",
                   tracker_peak="")
    result = gp.build_player_entries([row], CONFIG)
    reasons = [r for _, r in result.issues["Alpha"]]
    assert "no current rank" in reasons
    assert any("no peak rank" in r for r in reasons)
    assert any("no tracker score" in r for r in reasons)
    assert set(gp.build_missing(result)["Alpha"]["kinds"]) == {"rank", "tracker"}


def test_unknown_rank_is_an_issue():
    result = gp.build_player_entries([make_row("s1", "Alpha", current_rank="Plat 2")], CONFIG)
    assert result.entries["Alpha"]["current_rank"] is None
    assert "unknown rank 'Plat 2'" in result.issues["Alpha"][0][1]


def test_missing_peak_tracker_uses_current():
    result = gp.build_player_entries([make_row("s1", "Alpha", tracker_peak="")], CONFIG)
    assert result.entries["Alpha"]["tracker_peak"] == 500
    assert result.issues["Alpha"][0][0] == "tracker"


def test_bad_values_warn():
    row = make_row(
        "s1", "Alpha", current_rank="Gold 3", peak_rank="Gold 1", peak_rank_act="last year",
        tracker_current="abc", tracker_peak="5000", current_rr="lots",
    )
    result = gp.build_player_entries([row], CONFIG)
    w = " | ".join(result.warnings)
    assert "below current rank" in w
    assert "peak_rank_act 'last year'" in w
    assert "tracker_current 'abc'" in w
    assert "outside 0-1000" in w
    assert "current_rr 'lots'" in w


def test_parse_act_formats():
    assert gp.parse_act("s25a6") == ("S25A6", True)
    assert gp.parse_act("E8A1") == ("E8A1", True)
    assert gp.parse_act("E26: A5") == ("S26A5", True)
    assert gp.parse_act("") == (None, True)
    assert gp.parse_act("V25: ACT VI") == (None, False)


# --- VCC consistency check ----------------------------------------------


def vcc_team_config():
    from teamMaker.core.config import DEFAULTS, deep_merge

    cfg = deep_merge(DEFAULTS, {})
    cfg["use_returning_player_stats"] = True
    cfg["season_rating_distributions"] = {"S13": [0.8 + 0.01 * i for i in range(41)]}
    return cfg


def test_vcc_check_flags_rank_far_from_history():
    ranks = ["Gold 2", "Platinum 2", "Diamond 2", "Ascendant 2", "Immortal 1", "Radiant"]
    rows = [make_row(f"s{i}", f"P{i}", current_rank=r, peak_rank=r) for i, r in enumerate(ranks)]
    result = gp.build_player_entries(rows, CONFIG)
    # P5 is Radiant (top of the pool) but bottom of their VCC season; P2 matches.
    result.entries["P5"]["is_returning_player"] = True
    result.entries["P5"]["previous_season_stats"] = [{"season": "S13", "adjusted_rating": 0.8}]
    result.entries["P2"]["is_returning_player"] = True
    result.entries["P2"]["previous_season_stats"] = [{"season": "S13", "adjusted_rating": 1.0}]
    flagged = gp.check_vcc_consistency(result, vcc_team_config(), threshold=1.5)
    assert [name for name, _, _ in flagged] == ["P5"]
    assert any("P5: rank Radiant" in w and "much higher" in w for w in result.warnings)


def test_vcc_check_needs_a_pool():
    result = gp.build_player_entries([make_row("s1", "Alpha")], CONFIG)
    assert gp.check_vcc_consistency(result, vcc_team_config()) == []


# --- VCC previous seasons ------------------------------------------------


def make_stats(code, players, truncated=False):
    return SeasonStats(
        code=code, url="u", stage="all", source="payload", players=players,
        truncated=truncated,
    )


def test_previous_seasons_lookup_and_min_rounds():
    calls = []
    data = {
        "S1": make_stats("S1", {UUID_A: (1.10, 300), UUID_B: (0.90, 50)}),
        "S2": make_stats("S2", {UUID_A: (1.20, 250)}),
    }

    def loader(code, link, ua, stage="all", decimals=2):
        calls.append(code)
        return data[code]

    rows = [
        make_row(
            "s1", "Alpha", returning="TRUE",
            vcc_profile=f"https://vcc.example/player/{UUID_A}/alpha",
            previous_seasons="EMEA Season 1, EMEA Season 2",
        ),
        make_row(
            "s2", "Bravo", returning="TRUE",
            vcc_profile=f"https://vcc.example/player/{UUID_B}/bravo",
            previous_seasons="EMEA Season 1, EMEA Season 2",
        ),
    ]
    result = gp.build_player_entries(rows, CONFIG)
    gp.apply_previous_seasons(result, CONFIG, loader=loader, sleep=lambda s: None)
    assert calls == ["S1", "S2"]  # one load per season
    assert result.entries["Alpha"]["previous_season_stats"] == [
        {"season": "S1", "adjusted_rating": 1.10},
        {"season": "S2", "adjusted_rating": 1.20},
    ]
    # Bravo: S1 below min_rounds (skipped), S2 not found
    assert "previous_season_stats" not in result.entries["Bravo"]
    assert result.issues["Bravo"] == [("vcc", "not found in VCC S2 stats")]


def test_unknown_previous_season_warns_once():
    rows = [
        make_row(
            f"s{i}", f"P{i}", returning="TRUE",
            vcc_profile=f"https://vcc.example/player/{UUID_A}/p{i}",
            previous_seasons="EMEA Season 99",
        )
        for i in range(2)
    ]
    result = gp.build_player_entries(rows, CONFIG)
    gp.apply_previous_seasons(result, CONFIG, loader=lambda *a, **k: None)
    assert sum("EMEA Season 99" in w for w in result.warnings) == 1


def test_previous_season_load_failure_is_recorded():
    rows = [
        make_row(
            "s1", "Alpha", returning="TRUE",
            vcc_profile=f"https://vcc.example/player/{UUID_A}/alpha",
            previous_seasons="EMEA Season 1",
        )
    ]
    result = gp.build_player_entries(rows, CONFIG)
    gp.apply_previous_seasons(result, CONFIG, loader=lambda *a, **k: None)
    assert result.issues["Alpha"] == [("vcc", "VCC S1 stats unavailable")]


def test_returning_without_profile_is_recorded():
    rows = [make_row("s1", "Alpha", returning="TRUE", previous_seasons="EMEA Season 1")]
    result = gp.build_player_entries(rows, CONFIG)
    gp.apply_previous_seasons(
        result, CONFIG, loader=lambda *a, **k: make_stats("S1", {}), sleep=lambda s: None
    )
    assert "VCC profile" in result.issues["Alpha"][0][1]


# --- overrides ---------------------------------------------------------


def test_overrides_merge():
    rows = [make_row("s1", "Alpha"), make_row("s2", "Bravo"), make_row("s3", "Charlie")]
    result = gp.build_player_entries(rows, CONFIG)
    overrides = {
        "alpha": {"ping": 42, "region": "NA", "role": "smokes, sentinel"},
        "bravo_dc": {"status": "substitute"},
        "Charlie": {"status": "skip"},
        "Nobody": {"ping": 1},
        "Alpha ": {"bogus": 1},
    }
    gp.apply_overrides(result, overrides)
    assert result.entries["Alpha"]["ping"] == 42
    assert result.entries["Alpha"]["region"] == "NA"
    assert result.entries["Alpha"]["role"] == ["smokes", "sentinel"]
    assert result.entries["Bravo"]["status"] == "substitute"
    assert result.entries["Bravo"]["group_id"] is None
    assert "Charlie" not in result.entries
    assert any("matches no player" in w for w in result.warnings)


def test_overrides_bad_field_warns():
    result = gp.build_player_entries([make_row("s1", "Alpha")], CONFIG)
    gp.apply_overrides(result, {"Alpha": {"bogus": 1, "status": "maybe"}})
    assert sum("Alpha" in w for w in result.warnings) == 2


def test_load_overrides_missing_file(tmp_path):
    assert gp.load_overrides(str(tmp_path / "nope.yaml")) == {}


def test_load_overrides_file(tmp_path):
    path = tmp_path / "o.yaml"
    path.write_text("Alpha:\n  ping: 50\n", encoding="utf-8")
    assert gp.load_overrides(str(path)) == {"Alpha": {"ping": 50}}


def test_example_overrides_file_parses():
    import os

    path = os.path.join(gp.BASE_DIR, "data", "overrides.example.yaml")
    data = gp.load_overrides(path)
    assert data and all(isinstance(v, dict) for v in data.values())
    for fields in data.values():
        assert set(fields) <= set(gp.OVERRIDE_FIELDS)


def test_missing_only_lists_players_still_present():
    result = gp.build_player_entries([make_row("s1", "Alpha", tracker_current="")], CONFIG)
    assert "Alpha" in result.issues
    gp.apply_overrides(result, {"Alpha": {"status": "skip"}})
    assert gp.build_missing(result) == {}
