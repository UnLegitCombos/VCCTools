"""Tests for the optional OR-Tools solver (skipped when OR-Tools is missing)."""

import pytest

from teamMaker.core import build_teams, cpsat
from teamMaker.core import optimizer as opt
from teamMaker.core.config import DEFAULTS, deep_merge
from tests.synthetic_players import make_players

pytestmark = pytest.mark.skipif(not cpsat.available(), reason="OR-Tools not installed")

SIZES = [3, 2, 1, 1, 2, 3, 1, 2, 1, 1, 3, 2, 1, 1, 2, 1, 3]  # 30 players, 6 teams


def _prepare(sizes=SIZES, seed=4, **optimizer):
    players, scores = make_players(sizes, seed=seed)
    config = deep_merge(DEFAULTS, {})
    config["optimizer"] = dict(config["optimizer"], **optimizer)
    build = opt.build_units(players, scores, config)
    kept, _ = opt.select_units(build.units)
    return players, config, kept


def _check_valid(result, kept):
    placed = [u.uid for team in result.teams for u in team]
    assert sorted(placed) == sorted(u.uid for u in kept)  # every unit exactly once
    assert all(sum(u.size for u in team) == opt.TEAM_SIZE for team in result.teams)


def test_ortools_finds_valid_teams_and_reports_real_energy():
    _, config, kept = _prepare()
    result = cpsat.solve_teams(kept, config, 7, time_limit_s=4, workers=2)
    _check_valid(result, kept)
    pos = {u.uid: i for i, u in enumerate(kept)}
    again = opt.TeamProblem(
        kept, [[pos[u.uid] for u in t] for t in result.teams], opt.optimizer_weights(config)
    )
    assert result.energy == pytest.approx(again.energy)
    assert result.runs[0]["restart"] == "OR"


@pytest.mark.parametrize("provers", cpsat.PROVERS)
def test_ortools_never_worse_than_its_starting_hint(provers):
    _, config, kept = _prepare(time_limit_s=2, restarts=1, iterations=5000)
    start = opt.optimize_teams(kept, config, 3)
    result = cpsat.solve_teams(
        kept, config, 3, hint=start.teams, time_limit_s=2, provers=provers, workers=2
    )
    _check_valid(result, kept)
    # The objective leaves out the small std term, so allow for it.
    assert result.energy <= start.energy + 0.2 * start.metrics["std"] + 1e-6


def test_unknown_provers_value_fails():
    _, config, kept = _prepare()
    with pytest.raises(ValueError, match="provers"):
        cpsat.solve_teams(kept, config, 1, provers="maybe")


def test_missing_ortools_gives_a_clear_error(monkeypatch):
    _, config, kept = _prepare()
    monkeypatch.setattr(cpsat, "cp_model", None)
    assert not cpsat.available()
    with pytest.raises(cpsat.OrToolsUnavailable, match="requirements-ortools"):
        cpsat.solve_teams(kept, config, 1)


def test_cluster_cap_is_respected():
    players, config, kept = _prepare()
    na = [u for u in kept if u.cluster_n]
    assert na, "synthetic data should contain NA players"
    result = cpsat.solve_teams(kept, config, 5, cluster_cap=len(na), time_limit_s=4, workers=2)
    assert result.metrics["cluster_teams"] <= len(na)


def test_search_both_keeps_the_better_result_and_all_runs():
    _, config, kept = _prepare(
        solver="both", time_limit_s=2, restarts=2, iterations=5000,
        ortools_time_limit_s=3, ortools_workers=2,
    )
    first = opt.optimize_teams(kept, config, 9)
    result = build_teams.search(kept, config, 9, None)
    _check_valid(result, kept)
    assert [r["restart"] for r in result.runs] == [0, 1, "OR"]
    assert result.energy <= first.energy + 0.2 * first.metrics["std"] + 1e-6
    assert "OR-Tools" in result.stopped_by


def test_search_rejects_an_unknown_solver():
    _, config, kept = _prepare(solver="magic")
    with pytest.raises(ValueError, match="solver"):
        build_teams.search(kept, config, 1, None)


def test_cli_flags_set_the_solver_options(monkeypatch):
    seen = {}
    monkeypatch.setattr(build_teams, "run", lambda config: (seen.setdefault("cfg", config), "report"))
    build_teams.main(["--solver", "both", "--provers", "none", "--ortools-time", "42", "--ortools-workers", "3"])
    o = seen["cfg"]["optimizer"]
    assert (o["solver"], o["ortools_provers"], o["ortools_time_limit_s"], o["ortools_workers"]) == (
        "both", "none", 42.0, 3,
    )
