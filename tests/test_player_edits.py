"""Tests for player_edits (synthetic data only)."""

import csv

import pytest

from teamMaker.core import player_edits as pe
from teamMaker.core.config import DEFAULT_RANK_VALUES
from teamMaker.core.utils.vcc import SeasonStats

RV = DEFAULT_RANK_VALUES
REGIONS = {"EU", "NA"}
GOOD_URL = "https://tracker.gg/valorant/profile/riot/Name%23EU/overview"


def cells(**kw):
    new = {
        "current_rank": "Gold 1", "current_rr": None, "peak_rank": "Gold 2", "peak_rr": None,
        "peak_rank_act": None, "tracker_current": 500, "tracker_peak": 600, "ping": 40,
        "region": "eu", "role": ["duelist"], "status": "active", "is_returning_player": False,
    }
    new.update(kw)
    return new


def row(player, **kw):
    """What the data editor hands back for an untouched player, with some cells changed."""
    return {**{col: pe.clean(player.get(col), col) for col in pe.EDIT_COLS}, **kw}


def errors(**kw):
    return pe.validate_edit("P", cells(**kw), RV, REGIONS)


def players():
    return {
        "Solo": {"status": "active", "group_id": 1, "tracker_current": 100, "role": ["duelist"]},
        "Sub": {"status": "substitute", "group_id": None, "role": ["flex"]},
        "DuoA": {"status": "active", "group_id": 2},
        "DuoB": {"status": "active", "group_id": 2},
    }


# --- validation -------------------------------------------------------------


def test_valid_edit_is_normalised():
    new = cells(current_rank="gold 1", peak_rank_act="e26: a5")
    assert pe.validate_edit("P", new, RV, REGIONS) == []
    assert (new["current_rank"], new["peak_rank_act"], new["region"]) == ("Gold 1", "S26A5", "EU")


@pytest.mark.parametrize(
    ("change", "text"),
    [
        ({"peak_rank": "Silver 1"}, "below current_rank"),
        ({"current_rank": "Nope"}, "not a rank"),
        ({"peak_rank_act": "act six"}, "not an act"),
        ({"tracker_current": 1001}, "outside"),
        ({"ping": 0}, "outside"),
        ({"ping": 900}, "outside"),
        ({"peak_rr": -1}, "negative"),
        ({"region": "XX"}, "region"),
        ({"role": ["duelist", "healer"]}, "unknown role"),
        ({"status": "gone"}, "status"),
    ],
)
def test_validation_rejects(change, text):
    assert any(text in e for e in errors(**change))


def test_equal_peak_and_current_is_valid():
    assert errors(current_rank="Gold 2", peak_rank="Gold 2") == []


def test_clean_and_display():
    assert pe.clean("Duelist, flex ;duelist", "role") == ["duelist", "flex"]
    assert pe.clean(float("nan"), "ping") is None
    assert pe.clean(42.0, "ping") == 42
    assert pe.display_value({"role": ["a", "b"]}, "role") == "a, b"


# --- applying edits ---------------------------------------------------------


def test_apply_edits_writes_only_changed_keys():
    before = players()
    updated, warnings = pe.apply_edits(before, {"Solo": row(before["Solo"], tracker_current=200)})
    assert updated["Solo"]["tracker_current"] == 200
    assert "ping" not in updated["Solo"]
    assert before["Solo"]["tracker_current"] == 100
    assert warnings == []


def test_substitute_made_active_gets_their_own_group():
    updated, _ = pe.apply_edits(players(), {"Sub": row(players()["Sub"], status="active")})
    assert updated["Sub"]["status"] == "active"
    assert updated["Sub"]["group_id"] == 3


def test_stack_member_made_substitute_warns_about_partner():
    updated, warnings = pe.apply_edits(players(), {"DuoA": row(players()["DuoA"], status="substitute")})
    assert updated["DuoA"]["group_id"] == 2
    assert len(warnings) == 1
    assert "DuoB" in warnings[0]


# --- edit log and undo ------------------------------------------------------


def test_log_and_undo_round_trip(tmp_path):
    log = str(tmp_path / "log.jsonl")
    before = players()
    after, _ = pe.apply_edits(before, {"Solo": row(before["Solo"], tracker_current=200)})
    entry = pe.record_changes(log, before, after, "test")
    assert entry is not None
    assert pe.describe(entry)[0].startswith("Solo: ")
    assert "tracker_current 100 → 200" in pe.describe(entry)[0]
    restored, undone = pe.undo_last(log, after)
    assert restored == before
    assert undone["id"] == entry["id"]
    assert pe.last_undoable(log) is None
    with pytest.raises(ValueError, match="no saved edit"):
        pe.undo_last(log, restored)


def test_no_log_entry_when_nothing_changed(tmp_path):
    log = str(tmp_path / "log.jsonl")
    assert pe.record_changes(log, players(), players()) is None
    assert pe.recent_changes(log) == []


def test_undo_refuses_when_changed_since(tmp_path):
    log = str(tmp_path / "log.jsonl")
    before = players()
    after, _ = pe.apply_edits(before, {"Solo": row(before["Solo"], tracker_current=200)})
    pe.record_changes(log, before, after)
    after["Solo"]["tracker_current"] = 999
    with pytest.raises(ValueError, match=r"Solo\.tracker_current"):
        pe.undo_last(log, after)
    assert pe.last_undoable(log) is not None


def test_undo_goes_back_one_save_at_a_time(tmp_path):
    log = str(tmp_path / "log.jsonl")
    p0 = players()
    p1, _ = pe.apply_edits(p0, {"Solo": row(p0["Solo"], tracker_current=200)})
    pe.record_changes(log, p0, p1)
    p2, _ = pe.apply_edits(p1, {"Solo": row(p1["Solo"], tracker_current=300)})
    pe.record_changes(log, p1, p2)
    p, _ = pe.undo_last(log, p2)
    assert p["Solo"]["tracker_current"] == 200
    p, _ = pe.undo_last(log, p)
    assert p["Solo"]["tracker_current"] == 100


# --- adding one player ------------------------------------------------------

UUID = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
RATINGS_CFG = {
    "region_map": {"EU": "EU", "NA": "NA"},
    "season_map": {"EMEA Season 1": "S_1"},
    "seasons": {"S_1": {"code": "S1", "link": "https://vcc.example/s1"}},
    "vcc": {"user_agent": "test", "request_delay": 0.0},
}


def signup(**kw):
    out = {"display_name": "Late", "discord": "late_dc", "server": "EU", "status": "Approved",
           "stack": "Solo", "roles": "duelist", "returning": "FALSE", "current_rank": "Gold 1",
           "peak_rank": "Gold 2", "tracker_current": "500", "tracker_peak": "", "ping": ""}
    out.update(kw)
    return out


def test_new_player_gets_its_own_group_and_the_next_signup_index():
    pool = {**players(), "Old": {"status": "active", "group_id": 7, "signup_index": 40, "discord": "old"}}
    name, entry, notes = pe.new_player(signup(), pool, RATINGS_CFG, RV, REGIONS)
    assert name == "Late"
    assert (entry["group_id"], entry["signup_index"], entry["status"]) == (8, 41, "active")
    assert entry["tracker_peak"] == 500  # blank peak uses the current one, as the CSV import does
    assert any("peak tracker" in n for n in notes)


def test_new_substitute_has_no_group():
    _, entry, _ = pe.new_player(signup(status="Substitute"), players(), RATINGS_CFG, RV, REGIONS)
    assert (entry["status"], entry["group_id"]) == ("substitute", None)


@pytest.mark.parametrize(
    ("change", "text"),
    [
        ({"display_name": "solo", "discord": "x"}, "already in the player list"),
        ({"discord": "OLD"}, "already belongs"),
        ({"current_rank": ""}, "not a rank"),
        ({"display_name": "", "discord": ""}, "Enter a display name"),
    ],
)
def test_new_player_rejects(change, text):
    pool = {**players(), "Old": {"discord": "old"}}
    with pytest.raises(ValueError, match=text):
        pe.new_player(signup(**change), pool, RATINGS_CFG, RV, REGIONS)


def test_returning_new_player_loads_vcc_history():
    stats = SeasonStats(code="S1", url="u", stage="all", source="payload", players={UUID: (1.1, 300)})
    row = signup(returning="TRUE", vcc_profile=f"https://vcc.example/player/{UUID}/late",
                 previous_seasons="EMEA Season 1")
    name, entry, _ = pe.new_player(row, players(), RATINGS_CFG, RV, REGIONS, loader=lambda *a, **k: stats)
    assert name == "Late"  # the VCC profile slug, capitalised
    assert entry["previous_season_stats"] == [{"season": "S1", "adjusted_rating": 1.1}]


def test_undo_removes_an_added_player(tmp_path):
    log = str(tmp_path / "log.jsonl")
    before = players()
    name, entry, _ = pe.new_player(signup(), before, RATINGS_CFG, RV, REGIONS)
    after = {**before, name: entry}
    entry_logged = pe.record_changes(log, before, after)
    assert pe.describe(entry_logged) == ["Late: added"]
    restored, _ = pe.undo_last(log, after)
    assert restored == before


def test_changes_for_one_player(tmp_path):
    log = str(tmp_path / "log.jsonl")
    p0 = players()
    p1, _ = pe.apply_edits(p0, {"Solo": row(p0["Solo"], tracker_current=200)})
    pe.record_changes(log, p0, p1)
    p2, _ = pe.apply_edits(p1, {"Sub": row(p1["Sub"], role=["duelist"])})
    pe.record_changes(log, p1, p2)
    assert [list(e["changes"]) for e in pe.changes_for(log, "Solo")] == [["Solo"]]
    assert pe.changes_for(log, "Nobody") == []


# --- tracker links ----------------------------------------------------------


def write_csv(folder, rows):
    with open(folder / "signups.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["discord", "tracker", "status"])
        w.writeheader()
        w.writerows(rows)


def test_tracker_links_prefer_the_stored_link_then_the_csv(tmp_path):
    write_csv(tmp_path, [
        {"discord": "Beta_DC", "tracker": GOOD_URL, "status": "Approved"},
        {"discord": "gamma_dc", "tracker": GOOD_URL, "status": "Denied"},
        {"discord": "delta_dc", "tracker": "http://tracker.gg/valorant/profile/riot/x", "status": ""},
    ])
    stored = "https://tracker.gg/valorant/profile/riot/Stored%23EU"
    found = pe.tracker_links(
        {
            "Alpha": {"tracker_url": stored, "discord": "alpha_dc"},
            "Beta": {"discord": "beta_dc"},
            "Gamma": {"discord": "gamma_dc"},
            "Delta": {"discord": "delta_dc"},
            "Epsilon": {},
        },
        str(tmp_path),
    )
    assert found == {"Alpha": stored, "Beta": GOOD_URL}


def test_tracker_links_without_a_csv(tmp_path):
    assert pe.tracker_links({"A": {"discord": "x"}}, str(tmp_path)) == {}


# --- stale missing list -----------------------------------------------------


def test_still_missing_drops_what_was_fixed():
    missing = {
        "Fixed": {"reason": "no tracker score", "kinds": ["tracker"]},
        "Half": {"reason": "no rank; no tracker score", "kinds": ["rank", "tracker"]},
        "Open": {"reason": "no current rank", "kinds": ["rank"]},
        "Gone": {"reason": "renamed", "kinds": ["rank"]},
    }
    people = {
        "Fixed": {"tracker_current": 300},
        "Half": {"current_rank": "Gold 1", "peak_rank": "Gold 1"},
        "Open": {"current_rank": None},
    }
    assert set(pe.still_missing(missing, people, RV)) == {"Half", "Open", "Gone"}
