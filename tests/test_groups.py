"""Tests for make_groups (synthetic teams only)."""

import json
import random

import pytest

from teamMaker.core import make_groups as mg
from teamMaker.core.config import GROUPS_DEFAULTS


def make_teams_doc(n_teams, na_counts=None, seed=0, spread=1.0, generated_at="t0"):
    """Build a minimal synthetic teams.json document.

    Args:
        n_teams: Number of teams.
        na_counts: {team id: number of NA players}.
        seed: Seed for the scores.
        spread: Half-width of the random team score spread around 150.
        generated_at: Value of meta.generated_at.
    """
    rng = random.Random(seed)
    na_counts = na_counts or {}
    teams = []
    for tid in range(1, n_teams + 1):
        na = na_counts.get(tid, 0)
        regions = {"EU": 5 - na}
        if na:
            regions["NA"] = na
        teams.append(
            {
                "id": tid,
                "name": f"Team {tid}",
                "team_score": round(150 + rng.uniform(-spread, spread), 1),
                "regions": regions,
                "players": [],
            }
        )
    return {"schema_version": 1, "meta": {"generated_at": generated_at}, "teams": teams}


# Fixed servers (na_server off): the layout most tests below were written for.
FIXED = {"na_server": None, "servers": ["London", "Frankfurt", "Frankfurt"]}
# The default: every group on Frankfurt, a mostly-NA group may move to London.
DYNAMIC = {"na_server": "London", "servers": ["Frankfurt", "Frankfurt", "Frankfurt"]}


def make_settings(**overrides):
    """GroupSettings with test-friendly defaults (fast optimizer)."""
    config = {"groups": dict(GROUPS_DEFAULTS, iterations=20000, restarts=2, **{**FIXED, **overrides})}
    return mg.GroupSettings.from_config(config)


def write_teams(tmp_path, doc):
    path = tmp_path / "teams.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return str(path)


def run_groups(tmp_path, doc, **overrides):
    groups = dict(GROUPS_DEFAULTS, iterations=20000, restarts=2, **{**FIXED, **overrides})
    config = {"groups": groups, "random_seed": 5}
    return mg.run(
        config=config, teams_path=write_teams(tmp_path, doc), out_dir=str(tmp_path), render=False
    )


def test_four_na_teams_all_in_london(tmp_path):
    doc = make_teams_doc(18, na_counts={2: 1, 7: 2, 11: 1, 15: 1})
    out, _ = run_groups(tmp_path, doc)
    sizes = [len(g["team_ids"]) for g in out["groups"]]
    assert sizes == [6, 6, 6]
    london = set(out["groups"][0]["team_ids"])
    assert {2, 7, 11, 15} <= london
    assert out["summary"]["region_cost"] == 0
    assert out["summary"]["costly_teams"] == 0


def test_seven_na_teams_keep_most_na_in_london(tmp_path):
    na = {1: 1, 2: 1, 3: 2, 4: 3, 5: 1, 6: 2, 7: 1}
    doc = make_teams_doc(18, na_counts=na)
    out, text = run_groups(tmp_path, doc)
    london = set(out["groups"][0]["team_ids"])
    assert len(london) == 6
    # Exactly one NA team is outside London and it has a single NA player.
    outside = {t for t in na if t not in london}
    assert len(outside) == 1
    assert na[outside.pop()] == 1
    assert out["summary"]["region_cost"] == 10
    assert "NA teams outside London-type groups: 1" in text


def test_auto_sizes_extra_slots_go_to_cheapest_group():
    settings = make_settings()
    assert mg.resolve_sizes(20, settings) == [7, 7, 6]
    assert mg.resolve_sizes(19, settings) == [7, 6, 6]
    assert mg.resolve_sizes(18, settings) == [6, 6, 6]


def test_pinned_respected(tmp_path):
    doc = make_teams_doc(18, na_counts={2: 1, 7: 1})
    out, _ = run_groups(tmp_path, doc, pinned={2: 3, 9: 1})
    by_team = {t: g["index"] for g in out["groups"] for t in g["team_ids"]}
    assert by_team[2] == 3
    assert by_team[9] == 1
    assert by_team[7] == 1  # the other NA team still goes to London


def test_pin_over_capacity_fails():
    settings = make_settings(pinned={1: 1, 2: 1, 3: 1}, sizes=[2, 2, 2])
    with pytest.raises(mg.GroupsError, match="capacity"):
        mg.validate_pins(settings.pinned, [1, 2, 3, 4, 5, 6], [2, 2, 2])


def test_all_pinned(tmp_path):
    doc = make_teams_doc(6)
    pins = {1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3}
    out, _ = run_groups(tmp_path, doc, pinned=pins)
    assert [g["team_ids"] for g in out["groups"]] == [[1, 2], [3, 4], [5, 6]]


def test_single_group(tmp_path):
    doc = make_teams_doc(5)
    out, _ = run_groups(tmp_path, doc, count=1, servers=["London"])
    assert out["groups"][0]["team_ids"] == [1, 2, 3, 4, 5]
    assert out["summary"]["range"] == 0


def test_validation_errors(tmp_path):
    doc = make_teams_doc(4)
    with pytest.raises(mg.GroupsError, match="larger than the number of teams"):
        run_groups(tmp_path, doc, count=5, servers=["London"] * 5)
    with pytest.raises(mg.GroupsError, match="servers"):
        run_groups(tmp_path, doc, count=2, servers=["London"])
    with pytest.raises(mg.GroupsError, match="sums to"):
        run_groups(tmp_path, doc, count=2, servers=["London", "Frankfurt"], sizes=[3, 3])


def test_manual_duplicate_fails(tmp_path):
    doc = make_teams_doc(6)
    with pytest.raises(mg.GroupsError, match="more than once"):
        run_groups(tmp_path, doc, mode="manual", manual=[[1, 2], [2, 3], [4, 5, 6]])


def test_manual_missing_and_empty_fail():
    ids = [1, 2, 3, 4]
    with pytest.raises(mg.GroupsError, match="not assigned"):
        mg.manual_groups([[1], [2], [3]], ids, 3)
    with pytest.raises(mg.GroupsError, match="empty"):
        mg.manual_groups([[1, 2, 3, 4], [], []], ids, 3)
    with pytest.raises(mg.GroupsError, match="unknown"):
        mg.manual_groups([[1, 2], [3, 4, 9], [5]][:2] + [[7]], ids, 3)


def test_manual_ok(tmp_path):
    doc = make_teams_doc(6, na_counts={1: 1})
    out, _ = run_groups(tmp_path, doc, mode="manual", manual=[[1, 2], [3, 4], [5, 6]])
    assert out["meta"]["mode"] == "manual"
    assert out["summary"]["region_cost"] == 0


def test_reuse_roundtrip_and_mismatch(tmp_path):
    doc = make_teams_doc(9, na_counts={1: 1}, generated_at="A")
    first, _ = run_groups(tmp_path, doc)
    again, _ = run_groups(tmp_path, doc, mode="reuse")
    assert [g["team_ids"] for g in again["groups"]] == [g["team_ids"] for g in first["groups"]]

    newer = make_teams_doc(9, na_counts={1: 1}, generated_at="B")
    with pytest.raises(mg.GroupsError, match="generated at"):
        run_groups(tmp_path, newer, mode="reuse")


def test_reuse_without_file_fails(tmp_path):
    with pytest.raises(mg.GroupsError, match="does not exist"):
        run_groups(tmp_path, make_teams_doc(6), mode="reuse")


def test_reuse_new_team_ids_fail(tmp_path):
    doc = make_teams_doc(6, generated_at="A")
    run_groups(tmp_path, doc)
    bigger = make_teams_doc(9, generated_at="A")
    with pytest.raises(mg.GroupsError, match="does not match"):
        run_groups(tmp_path, bigger, mode="reuse")


def test_balance_term_reduces_range():
    rng = random.Random(3)
    scores = [150 + rng.uniform(-10, 10) for _ in range(18)]
    settings = make_settings(region_server_cost={})
    costs = mg.team_server_costs(
        [{"regions": {"EU": 5}} for _ in scores], settings
    )
    sizes = [6, 6, 6]

    def range_of(assign):
        sums = [0.0] * 3
        for t, g in enumerate(assign):
            sums[g] += scores[t]
        means = [s / 6 for s in sums]
        return max(means) - min(means)

    randoms = []
    for k in range(20):
        problem = mg.GroupProblem(scores, costs, sizes, settings, {}, random.Random(k))
        randoms.append(range_of(problem.assign))
    assign, energy = mg.optimize_groups(scores, costs, sizes, settings, {}, seed=1)
    assert range_of(assign) < min(randoms)
    assert range_of(assign) < 0.2


def test_incremental_energy_matches_exact():
    rng = random.Random(9)
    scores = [150 + rng.uniform(-5, 5) for _ in range(12)]
    settings = make_settings()
    teams = [{"regions": {"EU": 4, "NA": i % 3}} for i in range(12)]
    costs = mg.team_server_costs(teams, settings)
    problem = mg.GroupProblem(scores, costs, [4, 4, 4], settings, {3: 1}, rng)
    for _ in range(2000):
        new = problem.propose(rng)
        if new is None:
            continue
        assert new == pytest.approx(problem.exact_energy())
        if rng.random() < 0.5:
            problem.accept()
        else:
            problem.reject()
            assert problem.energy == pytest.approx(problem.exact_energy())
    assert problem.assign[3] == 1


def test_seed_reproducible():
    rng = random.Random(4)
    scores = [150 + rng.uniform(-5, 5) for _ in range(18)]
    settings = make_settings()
    teams = [{"regions": {"EU": 5 - (i % 2), "NA": i % 2}} for i in range(18)]
    costs = mg.team_server_costs(teams, settings)
    a = mg.optimize_groups(scores, costs, [6, 6, 6], settings, {}, seed=7)
    b = mg.optimize_groups(scores, costs, [6, 6, 6], settings, {}, seed=7)
    assert a == b


# --- default: Frankfurt, a mostly-NA group may move to London --------------


def test_defaults_are_frankfurt_with_london_switch():
    assert GROUPS_DEFAULTS["servers"] == ["Frankfurt", "Frankfurt", "Frankfurt"]
    assert GROUPS_DEFAULTS["na_server"] == "London"


def test_all_eu_stays_on_frankfurt(tmp_path):
    out, _ = run_groups(tmp_path, make_teams_doc(18, spread=8.0), **DYNAMIC)
    assert [g["server"] for g in out["groups"]] == ["Frankfurt"] * 3
    assert out["summary"]["range"] < 0.5


def test_few_na_players_stay_on_frankfurt(tmp_path):
    # Two NA players in 90 cannot make any group mostly NA.
    out, _ = run_groups(tmp_path, make_teams_doc(18, na_counts={2: 1, 9: 1}), **DYNAMIC)
    assert [g["server"] for g in out["groups"]] == ["Frankfurt"] * 3


def test_enough_na_teams_get_a_london_group(tmp_path):
    na = {1: 5, 5: 5, 9: 5, 13: 4}  # 19 NA players: enough for a 30-player group
    out, text = run_groups(tmp_path, make_teams_doc(18, na_counts=na, spread=1.0), **DYNAMIC)
    servers = [g["server"] for g in out["groups"]]
    assert servers.count("London") == 1
    london = out["groups"][servers.index("London")]
    assert set(na) <= set(london["team_ids"])
    assert london["regions"]["NA"] * 2 >= sum(london["regions"].values())
    assert out["summary"]["range"] <= 2.0 + 1.0  # balanced spread + tolerance


def test_london_never_costs_more_balance_than_the_tolerance(tmp_path):
    # Both NA teams (3 NA players each) are far stronger than the rest. Only
    # putting both in one group makes it mostly NA, which would wreck the
    # balance, so everyone stays on Frankfurt with the balanced split.
    doc = make_teams_doc(6, na_counts={1: 3, 2: 3})
    for t in doc["teams"]:
        t["team_score"] = 200.0 if t["id"] in (1, 2) else 100.0
    out, _ = run_groups(tmp_path, doc, **DYNAMIC)
    assert [g["server"] for g in out["groups"]] == ["Frankfurt"] * 3
    by_team = {t: g["index"] for g in out["groups"] for t in g["team_ids"]}
    assert by_team[1] != by_team[2]


def test_manual_groups_use_the_majority_rule(tmp_path):
    doc = make_teams_doc(6, na_counts={3: 5, 4: 3})
    manual = [[1, 2], [3, 4], [5, 6]]
    out, _ = run_groups(tmp_path, doc, mode="manual", manual=manual, **DYNAMIC)
    assert [g["server"] for g in out["groups"]] == ["Frankfurt", "London", "Frankfurt"]


def test_choose_servers_needs_the_minimum_share():
    settings = make_settings(**DYNAMIC, na_server_min_share=0.6)
    teams = {1: {"regions": {"NA": 3, "EU": 2}}, 2: {"regions": {"EU": 5}}}
    assert mg.choose_servers([[1], [2]], teams, settings)[:2] == ["London", "Frankfurt"]
    teams[1]["regions"] = {"NA": 2, "EU": 3}
    assert mg.choose_servers([[1], [2]], teams, settings)[:2] == ["Frankfurt", "Frankfurt"]
