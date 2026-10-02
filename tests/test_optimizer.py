import random

import pytest

from teamMaker.core import anneal
from teamMaker.core import optimizer as opt
from teamMaker.core.config import DEFAULTS, deep_merge
from tests.synthetic_players import make_players

SIZES = [3, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1]  # 30 players


def _config(**optimizer):
    config = deep_merge(DEFAULTS, {})
    config["optimizer"] = dict(config["optimizer"], **optimizer)
    return config


def _prepare(sizes=SIZES, subs=2, seed=1, **optimizer):
    players, scores = make_players(sizes, seed=seed, subs=subs)
    config = _config(**optimizer)
    build = opt.build_units(players, scores, config)
    kept, dropped = opt.select_units(build.units)
    return players, scores, config, build, kept, dropped


def _team_key(result):
    return [sorted(n for u in t for n in u.names) for t in result.teams]


def test_packing_infeasible_four_trios_three_solos():
    assert not opt.is_packable([3, 3, 3, 3, 1, 1, 1])
    assert opt.packing_patterns(3, 0, 4, 3) == []


def test_packing_feasible_and_infeasible_cases():
    assert opt.is_packable([3, 2])
    assert opt.is_packable([3, 1, 1, 2, 2, 1])
    assert opt.is_packable([1] * 5)
    assert not opt.is_packable([3, 3])
    assert not opt.is_packable([1, 1, 1])
    assert not opt.is_packable([4, 1])
    # Five duos: 10 players but no way to make two teams of exactly 5.
    assert not opt.is_packable([2] * 5)


def test_random_packing_is_feasible_and_seeded():
    players, scores, config, build, kept, _ = _prepare()
    teams = sum(u.size for u in kept) // 5
    a = opt.random_packing(kept, teams, random.Random(3))
    b = opt.random_packing(kept, teams, random.Random(3))
    assert a == b
    assert sorted(i for t in a for i in t) == list(range(len(kept)))
    assert all(sum(kept[i].size for i in t) == 5 for t in a)


def test_random_packing_raises_when_infeasible():
    players, scores = make_players([3, 3, 3, 3, 1, 1, 1])
    build = opt.build_units(players, scores, _config())
    with pytest.raises(ValueError):
        opt.random_packing(build.units, 3, random.Random(0))


def test_four_stack_excluded_with_reason():
    players, scores = make_players([4, 2, 1, 1, 1], subs=1)
    build = opt.build_units(players, scores, _config())
    assert len(build.excluded) == 1
    names, reason = build.excluded[0]
    assert len(names) == 4
    assert "4" in reason and "not allowed" in reason
    assert any("stack of 4" in w for w in build.warnings)
    assert all(u.size <= 3 for u in build.units)


def test_five_stack_is_fixed_team_and_oversize_excluded():
    players, scores = make_players([5, 6, 2, 3])
    build = opt.build_units(players, scores, _config())
    assert len(build.fixed) == 1 and len(build.fixed[0].names) == 5
    assert len(build.excluded) == 1 and len(build.excluded[0][0]) == 6
    assert sorted(u.size for u in build.units) == [2, 3]


def test_missing_group_id_is_solo_with_warning_and_null_is_sub():
    players, scores = make_players([2, 1])
    del players["P002"]["group_id"]
    players["P003"] = {"group_id": None, "role": ["duelist"], "region": "EU"}
    scores["P003"] = 10.0
    build = opt.build_units(players, scores, _config())
    assert any("P002" in w for w in build.warnings)
    assert sorted(u.size for u in build.units) == [1, 2]
    assert [n for n, _ in build.pool_subs] == ["P003"]


def test_select_units_multiple_of_five_and_packable():
    players, scores, config, build, kept, dropped = _prepare()
    total = sum(u.size for u in build.units)
    assert sum(u.size for u in kept) % 5 == 0
    assert (sum(u.size for u in dropped) - total) % 5 == 0
    assert opt.is_packable([u.size for u in kept])


def test_select_units_drops_latest_non_returning_first():
    players, scores = make_players([1] * 7)
    for i, info in enumerate(players.values()):
        info["is_returning_player"] = i not in (2, 5)
    build = opt.build_units(players, scores, _config())
    kept, dropped = opt.select_units(build.units)
    assert [u.names[0] for u in dropped] == ["P002", "P005"]
    for info in players.values():
        info["is_returning_player"] = False
    build = opt.build_units(players, scores, _config())
    kept, dropped = opt.select_units(build.units)
    assert {u.names[0] for u in dropped} == {"P005", "P006"}


def test_select_units_grows_removal_when_remainder_unpackable():
    # 3 trios + 2 solos = 11 players. Removing 1 leaves 3 trios in 2 teams,
    # which is impossible, so more has to go.
    players, scores = make_players([3, 3, 3, 1, 1])
    build = opt.build_units(players, scores, _config())
    kept, dropped = opt.select_units(build.units)
    assert opt.is_packable([u.size for u in kept])
    assert sum(u.size for u in dropped) in (1, 6)
    assert sum(u.size for u in kept) in (5, 10)


def _full_result(seed=7, **optimizer):
    players, scores, config, build, kept, dropped = _prepare(
        iterations=6000, restarts=2, **optimizer
    )
    result = opt.optimize_teams(kept, config, seed)
    return players, config, build, kept, dropped, result


def test_solution_invariants():
    players, config, build, kept, dropped, result = _full_result()
    subs = [n for n, _ in build.pool_subs] + [n for u in dropped for n in u.names]
    excluded = [n for names, _ in build.excluded for n in names]
    opt.validate_solution(players, result.teams, build.fixed, subs, excluded)
    assert all(sum(u.size for u in t) == 5 for t in result.teams)
    names = [n for t in result.teams for u in t for n in u.names]
    assert len(names) == len(set(names))
    stacks = {}
    for team_index, team in enumerate(result.teams):
        for u in team:
            for n in u.names:
                gid = players[n]["group_id"]
                stacks.setdefault(gid, set()).add(team_index)
    assert all(len(v) == 1 for v in stacks.values())


def test_validate_solution_detects_problems():
    players, config, build, kept, dropped, result = _full_result()
    subs = [n for n, _ in build.pool_subs] + [n for u in dropped for n in u.names]
    bad_teams = [list(t) for t in result.teams]
    bad_teams[0] = bad_teams[0][:-1]
    with pytest.raises(ValueError):
        opt.validate_solution(players, bad_teams, build.fixed, subs, [])
    with pytest.raises(ValueError):
        opt.validate_solution(players, result.teams, build.fixed, subs + [subs[0]], [])


def _solo(name, uid):
    return opt.Unit(
        uid=uid, names=(name,), size=1, tenths=100, sig_ids=(0,), cluster_n=0,
        returning=0, signup=uid,
    )


def test_split_stack_is_detected():
    players, scores = make_players([2] + [1] * 8)
    names = list(players)
    team_a = [_solo(n, i) for i, n in enumerate([names[0]] + names[2:6])]
    team_b = [_solo(n, 10 + i) for i, n in enumerate([names[1]] + names[6:10])]
    with pytest.raises(ValueError, match="split"):
        opt.validate_solution(players, [team_a, team_b], [], [], [])


def test_seed_reproducibility():
    r1 = _full_result(seed=11)[-1]
    r2 = _full_result(seed=11)[-1]
    r3 = _full_result(seed=12)[-1]
    assert _team_key(r1) == _team_key(r2)
    assert r1.energy == r2.energy
    assert _team_key(r1) != _team_key(r3) or r1.energy == r3.energy


def test_incremental_energy_matches_full_after_10k_moves():
    players, scores, config, build, kept, _ = _prepare(cluster_max_teams=3)
    teams = sum(u.size for u in kept) // 5
    rng = random.Random(5)
    weights = opt.optimizer_weights(config)
    problem = opt.TeamProblem(
        kept, opt.random_packing(kept, teams, rng), weights, cluster_cap=3
    )
    moved = 0
    for i in range(10000):
        new = problem.propose(rng)
        if new is None:
            continue
        if rng.random() < 0.5:
            problem.accept()
            moved += 1
        else:
            problem.reject()
        if i % 500 == 0:
            assert problem.energy == pytest.approx(problem.exact_energy(), abs=1e-9)
    assert moved > 100
    assert problem.energy == pytest.approx(problem.exact_energy(), abs=1e-9)
    assert all(sum(kept[u].size for u in t) == 5 for t in problem.teams)
    assert sorted(u for t in problem.teams for u in t) == list(range(len(kept)))
    fresh = opt.TeamProblem(kept, problem.teams, weights, cluster_cap=3)
    assert fresh.energy == pytest.approx(problem.energy, abs=1e-9)


def test_moves_change_team_compositions():
    players, scores, config, build, kept, _ = _prepare()
    teams = sum(u.size for u in kept) // 5
    rng = random.Random(9)
    problem = opt.TeamProblem(
        kept, opt.random_packing(kept, teams, rng), opt.optimizer_weights(config)
    )
    seen = set()
    for _ in range(3000):
        if problem.propose(rng) is not None:
            problem.accept()
            shape = tuple(sorted(tuple(sorted(kept[u].size for u in t)) for t in problem.teams))
            seen.add(shape)
    assert len(seen) > 3


def test_best_energy_equals_recomputed():
    players, scores, config, build, kept, _ = _prepare()
    teams = sum(u.size for u in kept) // 5
    rng = random.Random(2)
    weights = opt.optimizer_weights(config)
    problem = opt.TeamProblem(kept, opt.random_packing(kept, teams, rng), weights)
    t0, t_end = anneal.calibrate_temperatures(problem, rng)
    assert t0 > t_end > 0
    res = anneal.anneal(problem, rng, 5000, t0, t_end)
    fresh = opt.TeamProblem(kept, [list(t) for t in res.best_state], weights)
    assert res.best_energy == pytest.approx(fresh.exact_energy(), abs=1e-9)
    config["optimizer"].update(iterations=3000, restarts=2)
    result = opt.optimize_teams(kept, config, 3)
    recomputed = opt.TeamProblem(
        kept,
        [[kept.index(u) for u in team] for team in result.teams],
        weights,
    ).exact_energy()
    assert result.energy == pytest.approx(recomputed, abs=1e-9)
    assert result.energy == pytest.approx(min(r["energy"] for r in result.runs), abs=1e-5)


def test_annealing_improves_on_random_start():
    players, scores, config, build, kept, _ = _prepare(iterations=20000, restarts=1)
    teams = sum(u.size for u in kept) // 5
    start = opt.TeamProblem(
        kept,
        opt.random_packing(kept, teams, random.Random(3000)),
        opt.optimizer_weights(config),
    )
    result = opt.optimize_teams(kept, config, 3)
    assert result.energy < start.energy


def test_target_early_stop():
    players, scores, config, build, kept, _ = _prepare(
        iterations=200000, restarts=3, target_range=50.0, target_role_penalty=100.0
    )
    result = opt.optimize_teams(kept, config, 1)
    assert result.stopped_by == "target"
    assert len(result.runs) == 1
    assert result.runs[0]["iterations"] < 200000


def test_cluster_cap_is_respected_when_feasible():
    players, scores = make_players([1] * 30, seed=4)
    for i, info in enumerate(players.values()):
        info["region"] = "NA" if i < 6 else "EU"
    config = _config(iterations=30000, restarts=2, cluster_max_teams=2, cluster_weight=5.0)
    build = opt.build_units(players, scores, config)
    cap = opt.auto_cluster_cap(config, 6)
    assert cap == 2
    result = opt.optimize_teams(build.units, config, 1, cap)
    assert result.metrics["cluster_teams"] <= 2


def test_auto_cluster_cap_from_groups_and_disabled():
    config = _config()
    assert opt.auto_cluster_cap(config, 18) is None
    config["groups"] = {
        "count": 3,
        "servers": ["London", "Frankfurt", "Frankfurt"],
        "sizes": "auto",
    }
    assert opt.auto_cluster_cap(config, 18) == 6
    assert opt.auto_cluster_cap(config, 18, fixed_cluster_teams=1) == 5


def test_fixed_teams_are_not_optimized():
    players, scores = make_players([5, 3, 2, 1, 1, 1, 1, 1], subs=0)
    config = _config(iterations=2000, restarts=1)
    build = opt.build_units(players, scores, config)
    kept, dropped = opt.select_units(build.units)
    result = opt.optimize_teams(kept, config, 1)
    assert len(result.teams) == 2
    fixed_names = {n for f in build.fixed for n in f.names}
    assert not fixed_names & {n for t in result.teams for u in t for n in u.names}


def test_auto_cluster_cap_with_na_server_is_one_group():
    config = _config()
    config["groups"] = {
        "count": 3,
        "servers": ["Frankfurt", "Frankfurt", "Frankfurt"],
        "sizes": "auto",
        "na_server": "London",
    }
    assert opt.auto_cluster_cap(config, 20) == 7  # sizes 7, 7, 6
