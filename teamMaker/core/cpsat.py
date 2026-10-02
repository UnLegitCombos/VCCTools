"""Team search with Google OR-Tools CP-SAT (optional: ``pip install ortools``).

Used by ``build_teams --solver ortools`` (on its own) or ``--solver both``
(starting from the annealing result and improving it).

The model places every unit in one team of exactly 5 players and minimises

    range_weight * range + role_weight * role_penalty + cluster_weight * extra_cluster_teams

on the same integer scales as ``optimizer.TeamProblem`` (tenths of a point,
halves of a role penalty). The spread term (``std_weight * std``) is left out
because it is not linear; the returned teams are always scored with
``TeamProblem``, so the reported energy is the real one.

``provers`` controls how CP-SAT spends its workers:

- ``all``: its default mix, where most workers try to prove a lower bound;
- ``one``: one proving worker, the rest search for better teams;
- ``none``: only workers that search for better teams (no proof at all).
"""

import time

from teamMaker.core import roles as role_mod
from teamMaker.core.optimizer import TEAM_SIZE, OptimizationResult, TeamProblem, optimizer_weights

PROVERS = ("all", "one", "none")
# Objective scale: energy * 100 as integers (weights may be fractional).
_SCALE = 100

try:
    from ortools.sat.python import cp_model
except ImportError:  # optional dependency
    cp_model = None


class OrToolsUnavailable(RuntimeError):
    """Raised when OR-Tools is requested but not installed."""


def available():
    """True when OR-Tools can be imported."""
    return cp_model is not None


def solve_teams(
    units,
    config,
    seed,
    cluster_cap=None,
    hint=None,
    time_limit_s=300.0,
    provers="one",
    workers=8,
    progress=None,
):
    """Search for the best split of ``units`` into teams with CP-SAT.

    Args:
        units: Units to place (as for optimizer.optimize_teams).
        config: Config dict (objective weights).
        seed: Integer seed for CP-SAT (runs with several workers can still
            differ slightly).
        cluster_cap: Cluster cap (None disables the cluster term).
        hint: Optional starting teams (list of lists of Unit), e.g. the
            annealing result.
        time_limit_s: Wall-clock limit in seconds.
        provers: "all", "one" or "none" (see module docstring).
        workers: Number of CP-SAT worker threads.
        progress: Optional callable(seconds, metrics) for every better split.

    Returns:
        OptimizationResult with a single run labelled "OR".

    Raises:
        OrToolsUnavailable: If OR-Tools is not installed.
        ValueError: On an unknown ``provers`` value.
    """
    if cp_model is None:
        raise OrToolsUnavailable(
            "OR-Tools is not installed: pip install -r requirements-ortools.txt"
        )
    if provers not in PROVERS:
        raise ValueError(f"provers must be one of {PROVERS}, got {provers!r}")
    started = time.time()
    weights = optimizer_weights(config)
    n_teams = sum(u.size for u in units) // TEAM_SIZE
    if n_teams < 2:
        from teamMaker.core.optimizer import optimize_teams

        return optimize_teams(units, config, seed, cluster_cap)

    idx = range(len(units))
    total10 = sum(u.tenths for u in units)
    m = cp_model.CpModel()
    x = {(u, t): m.new_bool_var(f"x{u}_{t}") for u in idx for t in range(n_teams)}
    for u in idx:
        m.add_exactly_one(x[u, t] for t in range(n_teams))
    sums = []
    for t in range(n_teams):
        m.add(sum(units[u].size * x[u, t] for u in idx) == TEAM_SIZE)
        s = m.new_int_var(0, total10, f"sum{t}")
        m.add(s == sum(units[u].tenths * x[u, t] for u in idx))
        sums.append(s)
    for a, b in zip(sums, sums[1:]):  # teams are interchangeable: order them
        m.add(a <= b)
    range10 = m.new_int_var(0, total10, "range10")
    m.add(range10 == sums[-1] - sums[0])

    # Role cover: four distinct players per team take the four main roles at
    # the cheapest cost (minimising picks the best assignment).
    players = [(u, sig) for u in idx for sig in units[u].sig_ids]
    pen_terms, per_player = [], [[] for _ in players]
    if weights["role_weight"]:
        for t in range(n_teams):
            for r in range(len(role_mod.MAIN_ROLES)):
                cover = []
                for p, (u, sig) in enumerate(players):
                    y = m.new_bool_var(f"y{p}_{r}_{t}")
                    m.add_implication(y, x[u, t])
                    cover.append(y)
                    per_player[p].append(y)
                    cost = role_mod.signature_of(sig)[r]
                    if cost:
                        pen_terms.append(cost * y)
                m.add_exactly_one(cover)
        for ys in per_player:
            m.add_at_most_one(ys)
    pen2 = m.new_int_var(0, 2 * len(role_mod.MAIN_ROLES) * n_teams, "pen2")
    m.add(pen2 == sum(pen_terms))

    extra = None
    clustered = [u for u in idx if units[u].cluster_n]
    if cluster_cap is not None and clustered:
        has = []
        for t in range(n_teams):
            z = m.new_bool_var(f"cluster{t}")
            for u in clustered:
                m.add_implication(x[u, t], z)
            has.append(z)
        extra = m.new_int_var(0, n_teams, "extra_cluster")
        m.add(extra >= sum(has) - cluster_cap)

    objective = round(weights["range_weight"] * _SCALE / 10) * range10
    objective += round(weights["role_weight"] * _SCALE / 2) * pen2
    if extra is not None:
        objective += round(weights["cluster_weight"] * _SCALE) * extra
    m.minimize(objective)

    if hint:
        uid_pos = {u.uid: i for i, u in enumerate(units)}
        ordered = sorted(hint, key=lambda team: sum(u.tenths for u in team))
        for t, team in enumerate(ordered):
            inside = {uid_pos[u.uid] for u in team}
            for u in idx:
                m.add_hint(x[u, t], u in inside)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.parameters.num_workers = max(1, int(workers))
    solver.parameters.random_seed = int(seed) % (2**31)
    if provers == "none":
        solver.parameters.use_lns_only = True
    elif provers == "one":
        solver.parameters.num_full_subsolvers = 1

    def teams_of(value):
        return [[units[u] for u in idx if value(x[u, t])] for t in range(n_teams)]

    class _Progress(cp_model.CpSolverSolutionCallback):
        def __init__(self):
            super().__init__()
            self.solutions = 0

        def on_solution_callback(self):
            self.solutions += 1
            if progress:
                found = teams_of(self.value)
                pos = {u.uid: i for i, u in enumerate(units)}
                problem = TeamProblem(
                    units, [[pos[u.uid] for u in team] for team in found], weights, cluster_cap
                )
                progress(self.wall_time, problem.metrics())

    callback = _Progress()
    status = solver.solve(m, callback)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"OR-Tools found no team split ({solver.status_name(status)})")

    teams = teams_of(solver.value)
    pos = {u.uid: i for i, u in enumerate(units)}
    problem = TeamProblem(units, [[pos[u.uid] for u in team] for team in teams], weights, cluster_cap)
    metrics = problem.metrics()
    stopped_by = "optimal" if status == cp_model.OPTIMAL else "time"
    run = {
        "restart": "OR",
        "seed": seed,
        "energy": round(problem.energy, 6),
        "range": metrics["range"],
        "std": round(metrics["std"], 4),
        "role_penalty": metrics["role_penalty"],
        "cluster_teams": metrics["cluster_teams"],
        "iterations": callback.solutions,
        "provers": provers,
        "elapsed_s": round(time.time() - started, 2),
        "stopped_by": stopped_by,
    }
    return OptimizationResult(
        teams=teams,
        energy=metrics["energy"],
        metrics=metrics,
        runs=[run],
        stopped_by=stopped_by,
        elapsed=time.time() - started,
    )
