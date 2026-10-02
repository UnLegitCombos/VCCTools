"""Team optimizer: units, exact packing, sub selection and annealing.

Players are grouped into units (stacks of 1-3 players) that must stay
together. Stacks of 5 are fixed teams and are not optimized. Stacks of 4 are
not allowed (nor anything above 5) and are excluded with a reason.

Every team has exactly 5 players, so a team's composition is one of the five
patterns {3,2}, {3,1,1}, {2,2,1}, {2,1,1,1} and 1x5. The move exchanges two
equal-size sets of units between two teams, which lets compositions change.

The objective (one function used to accept moves, to keep the best solution and
to report) is computed on integer tenths of score::

    range_weight * range + std_weight * std + role_weight * sum(role penalty)
    + cluster_weight * max(0, cluster teams - cap)
"""

import itertools
import math
import random
import time
from dataclasses import dataclass, field

from teamMaker.core import anneal
from teamMaker.core import roles as role_mod

TEAM_SIZE = 5
MAX_UNIT_SIZE = 3
STACK_LABELS = {1: "solo", 2: "duo", 3: "trio", 5: "5-stack"}

# The only ways to fill a 5-player team with units of size 1, 2 or 3, in the
# order used by packing_patterns: {3,2}, {3,1,1}, {2,2,1}, {2,1,1,1}, 1x5.
PATTERNS = (
    (3, 2),
    (3, 1, 1),
    (2, 2, 1),
    (2, 1, 1, 1),
    (1, 1, 1, 1, 1),
)


@dataclass
class Unit:
    """A set of players that must be placed in the same team."""

    uid: int
    names: tuple
    size: int
    tenths: int
    sig_ids: tuple
    cluster_n: int
    returning: int
    signup: int
    group_id: object = None

    @property
    def label(self):
        """Human label: solo, duo or trio."""
        return STACK_LABELS.get(self.size, f"{self.size}-stack")


@dataclass
class FixedTeam:
    """A 5-stack: a complete team that is not optimized."""

    names: tuple
    group_id: object


@dataclass
class BuildResult:
    """Output of build_units."""

    units: list
    fixed: list
    pool_subs: list  # [(name, reason)]
    excluded: list  # [(names tuple, reason)]
    warnings: list = field(default_factory=list)


def build_units(players, scores, config=None):
    """Group players into units, fixed teams, substitutes and exclusions.

    Args:
        players: Dict {name: player_info} (input order = signup order unless
            ``signup_index`` is present).
        scores: Dict {name: score rounded to 1 decimal}.
        config: Config dict (``optimizer.cluster_regions`` is read).

    Returns:
        BuildResult. Players with a null group_id or status "substitute"
        go to the substitute pool; a missing group_id key makes a solo (with
        a warning); stacks of 1-3 become units; stacks of 5 become fixed
        teams; stacks of 4 or more than 5 are excluded with a reason.
    """
    opt = (config or {}).get("optimizer") or {}
    cluster = {str(r).upper() for r in (opt.get("cluster_regions") or [])}

    result = BuildResult(units=[], fixed=[], pool_subs=[], excluded=[])
    groups = {}
    order = []
    missing = []
    for position, (name, info) in enumerate(players.items()):
        signup = info.get("signup_index", position)
        if info.get("status") == "substitute":
            result.pool_subs.append((name, "substitute signup"))
            continue
        if "group_id" not in info:
            missing.append(name)
            key = ("solo", name)
        elif info["group_id"] is None:
            result.pool_subs.append((name, "substitute signup (no group_id)"))
            continue
        else:
            key = ("group", info["group_id"])
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append((name, info, signup))
    if missing:
        result.warnings.append(
            "no group_id for " + ", ".join(missing) + ": treated as solo players"
        )

    for key in order:
        members = groups[key]
        size = len(members)
        names = tuple(m[0] for m in members)
        group_id = key[1] if key[0] == "group" else None
        if size == TEAM_SIZE:
            result.fixed.append(FixedTeam(names=names, group_id=group_id))
        elif size == 4:
            reason = "stack of 4 is not allowed (stacks are 1, 2, 3 or 5)"
            result.excluded.append((names, reason))
            result.warnings.append(f"{reason}: {', '.join(names)}")
        elif size > TEAM_SIZE:
            reason = f"stack of {size} exceeds the team size of {TEAM_SIZE}"
            result.excluded.append((names, reason))
            result.warnings.append(f"{reason}: {', '.join(names)}")
        else:
            sig_ids = tuple(
                role_mod.signature_id(role_mod.role_signature(info.get("role")))
                for _, info, _ in members
            )
            result.units.append(
                Unit(
                    uid=len(result.units),
                    names=names,
                    size=size,
                    tenths=sum(int(round(scores[n] * 10)) for n in names),
                    sig_ids=sig_ids,
                    cluster_n=sum(
                        1
                        for _, info, _ in members
                        if str(info.get("region", "")).upper() in cluster
                    ),
                    returning=sum(
                        1 for _, info, _ in members if info.get("is_returning_player")
                    ),
                    signup=max(m[2] for m in members),
                    group_id=group_id if size > 1 else None,
                )
            )
    return result


# ---------------------------------------------------------------------------
# Exact packing
# ---------------------------------------------------------------------------


def packing_patterns(n1, n2, n3, teams):
    """List every way to fill ``teams`` teams from units of size 1, 2 and 3.

    Args:
        n1: Number of solo units.
        n2: Number of duo units.
        n3: Number of trio units.
        teams: Number of teams.

    Returns:
        List of tuples (a, b, c, d, e): how many teams use {3,2}, {3,1,1},
        {2,2,1}, {2,1,1,1} and 1x5 respectively.
    """
    solutions = []
    if n1 + 2 * n2 + 3 * n3 != TEAM_SIZE * teams:
        return solutions
    for a in range(0, min(n3, n2) + 1):
        b = n3 - a
        left2 = n2 - a
        for c in range(0, left2 // 2 + 1):
            d = left2 - 2 * c
            e = teams - a - b - c - d
            if e < 0:
                continue
            if 2 * b + c + 3 * d + 5 * e == n1:
                solutions.append((a, b, c, d, e))
    return solutions


def is_packable(sizes):
    """Return True when unit sizes can be packed into teams of exactly 5.

    Args:
        sizes: Iterable of unit sizes (each 1, 2 or 3).

    Returns:
        True if a packing exists.
    """
    sizes = list(sizes)
    if any(s not in (1, 2, 3) for s in sizes):
        return False
    total = sum(sizes)
    if total % TEAM_SIZE:
        return False
    counts = (sizes.count(1), sizes.count(2), sizes.count(3))
    return bool(packing_patterns(*counts, total // TEAM_SIZE))


def random_packing(units, teams, rng):
    """Return a random feasible assignment of units to teams.

    A pattern mix is drawn uniformly among the feasible mixes, then units of
    each size are shuffled into the slots.

    Args:
        units: List of Unit (sizes 1-3).
        teams: Number of teams.
        rng: random.Random instance.

    Returns:
        List of ``teams`` lists of unit indices (positions in ``units``).

    Raises:
        ValueError: If the units cannot be packed.
    """
    by_size = {1: [], 2: [], 3: []}
    for index, unit in enumerate(units):
        by_size[unit.size].append(index)
    mixes = packing_patterns(len(by_size[1]), len(by_size[2]), len(by_size[3]), teams)
    if not mixes:
        raise ValueError("units cannot be packed into teams of 5")
    mix = mixes[rng.randrange(len(mixes))]
    for pool in by_size.values():
        rng.shuffle(pool)
    bins = []
    for pattern, count in zip(PATTERNS, mix):
        bins.extend([pattern] * count)
    rng.shuffle(bins)
    return [[by_size[size].pop() for size in pattern] for pattern in bins]


# ---------------------------------------------------------------------------
# Substitute selection
# ---------------------------------------------------------------------------


def select_units(units):
    """Choose which units become substitutes so the rest fill whole teams.

    The removed set sums to ``total % 5`` players (then +5, +10 ... only when
    no removal of that size leaves a packable remainder). Among feasible
    removals, the one dropping the fewest returning players wins, then the
    fewest units, then the latest signups.

    Args:
        units: List of Unit (sizes 1-3).

    Returns:
        Tuple (kept, dropped) of unit lists (input order preserved).

    Raises:
        ValueError: If no selection leaves a packable remainder.
    """
    total = sum(u.size for u in units)
    by_size = {s: [u for u in units if u.size == s] for s in (1, 2, 3)}
    # Drop preference inside a size: non-returning first, latest signup first.
    for pool in by_size.values():
        pool.sort(key=lambda u: (u.returning > 0, -u.signup))
    counts = {s: len(p) for s, p in by_size.items()}

    target = total % TEAM_SIZE
    while target <= total:
        best = None
        for c3 in range(0, min(counts[3], target // 3) + 1):
            for c2 in range(0, min(counts[2], (target - 3 * c3) // 2) + 1):
                c1 = target - 3 * c3 - 2 * c2
                if c1 > counts[1]:
                    continue
                remaining = (counts[1] - c1, counts[2] - c2, counts[3] - c3)
                left = remaining[0] + 2 * remaining[1] + 3 * remaining[2]
                if not packing_patterns(*remaining, left // TEAM_SIZE):
                    continue
                dropped = by_size[1][:c1] + by_size[2][:c2] + by_size[3][:c3]
                cost = (
                    sum(u.returning for u in dropped),
                    len(dropped),
                    -sum(u.signup for u in dropped),
                    c3,
                    c2,
                )
                if best is None or cost < best[0]:
                    best = (cost, dropped)
        if best is not None:
            dropped_ids = {u.uid for u in best[1]}
            kept = [u for u in units if u.uid not in dropped_ids]
            dropped = sorted(best[1], key=lambda u: u.uid)
            return kept, dropped
        target += TEAM_SIZE
    raise ValueError("no selection of units can be packed into teams of 5")


# ---------------------------------------------------------------------------
# Optimization problem
# ---------------------------------------------------------------------------

_SUBSET_CACHE = {}


def _subsets(sizes, total):
    """Return position tuples (1-3 units) of ``sizes`` that sum to ``total``."""
    key = (sizes, total)
    found = _SUBSET_CACHE.get(key)
    if found is None:
        found = []
        for r in (1, 2, 3):
            for combo in itertools.combinations(range(len(sizes)), r):
                if sum(sizes[i] for i in combo) == total:
                    found.append(combo)
        _SUBSET_CACHE[key] = found
    return found


class TeamProblem:
    """Annealing problem: assign units to teams of exactly 5 players.

    Energies are computed from integer aggregates (tenths of score and halves
    of role penalty), so incremental and full evaluation agree exactly.
    """

    def __init__(
        self, units, teams, weights, cluster_cap=None, target_range=0.1, target_role=0.0
    ):
        """Create the problem.

        Args:
            units: List of Unit.
            teams: Initial assignment, list of lists of indices into ``units``.
            weights: Dict with range_weight, std_weight, role_weight and
                cluster_weight.
            cluster_cap: Max teams containing a cluster-region player before
                the cluster term applies (None disables the term).
            target_range: Range (points) at or below which the run may stop.
            target_role: Role penalty (points, all teams) at or below which the
                run may stop.
        """
        self.units = units
        self.size = [u.size for u in units]
        self.tenths = [u.tenths for u in units]
        self.sig_ids = [u.sig_ids for u in units]
        self.cluster_n = [u.cluster_n for u in units]
        self.w_range = float(weights["range_weight"])
        self.w_std = float(weights["std_weight"])
        self.w_role = float(weights["role_weight"])
        self.w_cluster = float(weights["cluster_weight"])
        self.cluster_cap = cluster_cap
        self.target_range10 = int(round(target_range * 10))
        self.target_role2 = int(round(target_role * 2))
        self.teams = [list(t) for t in teams]
        self.k = len(self.teams)
        self._rebuild()
        self._undo = None
        self._new_energy = self.energy

    # -- aggregates -------------------------------------------------------

    def _team_pen(self, members):
        """Role penalty (halves) of a team given its unit indices."""
        ids = []
        for u in members:
            ids.extend(self.sig_ids[u])
        ids.sort()
        return role_mod.team_cost_halves(tuple(ids))

    def _rebuild(self):
        """Recompute all aggregates from ``self.teams`` and the energy."""
        self.sums = [sum(self.tenths[u] for u in t) for t in self.teams]
        self.pens = [self._team_pen(t) if self.w_role else 0 for t in self.teams]
        self.clus = [sum(self.cluster_n[u] for u in t) for t in self.teams]
        self.tot = sum(self.sums)
        self.sumsq = sum(s * s for s in self.sums)
        self.pen_total = sum(self.pens)
        self.ncl = sum(1 for c in self.clus if c > 0)
        self.energy = self._energy_from(self.sums, self.sumsq, self.pen_total, self.ncl)

    def _energy_from(self, sums, sumsq, pen_total, ncl):
        """Energy from integer aggregates."""
        k = self.k
        rng10 = max(sums) - min(sums)
        var = sumsq / k - (self.tot / k) ** 2
        std10 = math.sqrt(var) if var > 0 else 0.0
        energy = self.w_range * rng10 / 10.0 + self.w_std * std10 / 10.0
        energy += self.w_role * pen_total / 2.0
        if self.cluster_cap is not None and ncl > self.cluster_cap:
            energy += self.w_cluster * (ncl - self.cluster_cap)
        return energy

    # -- annealing API ----------------------------------------------------

    def propose(self, rng):
        """Exchange equal-size unit sets between two teams (applied in place)."""
        k = self.k
        if k < 2:
            return None
        a = rng.randrange(k)
        b = rng.randrange(k - 1)
        if b >= a:
            b += 1
        teams = self.teams
        team_a = teams[a]
        team_b = teams[b]
        size = self.size
        need = rng.randrange(1, 4)
        subs_a = _subsets(tuple([size[u] for u in team_a]), need)
        if not subs_a:
            return None
        subs_b = _subsets(tuple([size[u] for u in team_b]), need)
        if not subs_b:
            return None
        pos_a = subs_a[rng.randrange(len(subs_a))]
        pos_b = subs_b[rng.randrange(len(subs_b))]
        out_a = [team_a[i] for i in pos_a]
        out_b = [team_b[i] for i in pos_b]
        tenths = self.tenths
        ta = sum([tenths[u] for u in out_a])
        tb = sum([tenths[u] for u in out_b])
        new_a = [u for i, u in enumerate(team_a) if i not in pos_a] + out_b
        new_b = [u for i, u in enumerate(team_b) if i not in pos_b] + out_a

        sums = self.sums
        self._undo = (
            a,
            b,
            team_a,
            team_b,
            sums[a],
            sums[b],
            self.pens[a],
            self.pens[b],
            self.clus[a],
            self.clus[b],
            self.sumsq,
            self.pen_total,
            self.ncl,
            self.energy,
        )
        sa_old = sums[a]
        sb_old = sums[b]
        sa_new = sa_old - ta + tb
        sb_new = sb_old - tb + ta
        sums[a] = sa_new
        sums[b] = sb_new
        sumsq = (
            self.sumsq + sa_new * sa_new + sb_new * sb_new - sa_old * sa_old - sb_old * sb_old
        )

        pen_total = self.pen_total
        if self.w_role:
            pa = self._team_pen(new_a)
            pb = self._team_pen(new_b)
            pen_total += pa + pb - self.pens[a] - self.pens[b]
            self.pens[a] = pa
            self.pens[b] = pb

        ncl = self.ncl
        if self.cluster_cap is not None:
            cn = self.cluster_n
            ca_old = self.clus[a]
            cb_old = self.clus[b]
            d = sum([cn[u] for u in out_b]) - sum([cn[u] for u in out_a])
            ca_new = ca_old + d
            cb_new = cb_old - d
            ncl += (ca_new > 0) + (cb_new > 0) - (ca_old > 0) - (cb_old > 0)
            self.clus[a] = ca_new
            self.clus[b] = cb_new

        teams[a] = new_a
        teams[b] = new_b
        self.sumsq = sumsq
        self.pen_total = pen_total
        self.ncl = ncl
        self._new_energy = self._energy_from(sums, sumsq, pen_total, ncl)
        return self._new_energy

    def accept(self):
        """Keep the last proposed move."""
        self.energy = self._new_energy
        self._undo = None

    def reject(self):
        """Undo the last proposed move."""
        assert self._undo is not None, "reject() called without a proposed move"
        (a, b, team_a, team_b, sa, sb, pa, pb, ca, cb, sumsq, pen_total, ncl, energy) = (
            self._undo
        )
        self.teams[a] = team_a
        self.teams[b] = team_b
        self.sums[a] = sa
        self.sums[b] = sb
        self.pens[a] = pa
        self.pens[b] = pb
        self.clus[a] = ca
        self.clus[b] = cb
        self.sumsq = sumsq
        self.pen_total = pen_total
        self.ncl = ncl
        self.energy = energy
        self._undo = None

    def snapshot(self):
        """Return an independent copy of the assignment."""
        return tuple(tuple(t) for t in self.teams)

    def restore(self, snapshot):
        """Load an assignment produced by snapshot()."""
        self.teams = [list(t) for t in snapshot]
        self._rebuild()

    def metrics(self):
        """Recompute every quantity from scratch (ignores the aggregates).

        Returns:
            Dict with team sums (tenths), range, std, role penalty (points),
            cluster teams, cluster excess and energy.
        """
        sums = [sum(self.tenths[u] for u in t) for t in self.teams]
        pens = [self._team_pen(t) for t in self.teams]
        clus = [sum(self.cluster_n[u] for u in t) for t in self.teams]
        k = len(sums)
        mean = sum(sums) / k
        std10 = math.sqrt(sum((s - mean) ** 2 for s in sums) / k)
        ncl = sum(1 for c in clus if c > 0)
        excess = 0
        if self.cluster_cap is not None:
            excess = max(0, ncl - self.cluster_cap)
        rng10 = max(sums) - min(sums)
        role_pts = sum(pens) / 2.0
        energy = self.w_range * rng10 / 10.0 + self.w_std * std10 / 10.0
        energy += self.w_role * role_pts + self.w_cluster * excess
        return {
            "sums10": sums,
            "range": rng10 / 10.0,
            "std": std10 / 10.0,
            "role_penalty": role_pts,
            "cluster_teams": ncl,
            "cluster_excess": excess,
            "energy": energy,
        }

    def exact_energy(self):
        """Energy recomputed from scratch."""
        return self.metrics()["energy"]

    def at_target(self):
        """True when range, role penalty and cluster excess meet the targets."""
        if max(self.sums) - min(self.sums) > self.target_range10:
            return False
        if self.pen_total > self.target_role2:
            return False
        return not (self.cluster_cap is not None and self.ncl > self.cluster_cap)


# ---------------------------------------------------------------------------
# Multi-start driver
# ---------------------------------------------------------------------------


@dataclass
class OptimizationResult:
    """Best solution over all restarts."""

    teams: list  # list of lists of Unit
    energy: float
    metrics: dict
    runs: list
    stopped_by: str
    elapsed: float


def optimizer_weights(config):
    """Return the objective weights from a config (role weight 0 if disabled)."""
    opt = config.get("optimizer") or {}
    roles_on = config.get("use_role_balancing", True)
    return {
        "range_weight": opt.get("range_weight", 1.0),
        "std_weight": opt.get("std_weight", 0.2),
        "role_weight": opt.get("role_weight", 0.5) if roles_on else 0.0,
        "cluster_weight": opt.get("cluster_weight", 2.0),
    }


def optimize_teams(units, config, seed, cluster_cap=None, progress=None):
    """Multi-start simulated annealing over unit-to-team assignments.

    Restart ``k`` uses ``random.Random(seed * 1000 + k)`` and a fresh random
    feasible start. Runs bounded by iterations are reproducible; the time
    limit (a safety net) is not.

    Args:
        units: Units to place (sizes 1-3, total divisible by 5, packable).
        config: Config dict (``optimizer`` block).
        seed: Integer seed.
        cluster_cap: Cluster cap (None disables the cluster term).
        progress: Optional callable(run_dict) called after each restart.

    Returns:
        OptimizationResult.
    """
    opt = config.get("optimizer") or {}
    total = sum(u.size for u in units)
    teams = total // TEAM_SIZE
    weights = optimizer_weights(config)
    iterations = int(opt.get("iterations", 50000000))
    restarts = max(1, int(opt.get("restarts", 8)))
    limit = opt.get("time_limit_s")
    target_range = float(opt.get("target_range") or 0.0)
    target_role = float(opt.get("target_role_penalty") or 0.0)

    started = time.time()
    if teams == 0:
        return OptimizationResult([], 0.0, {}, [], "iterations", 0.0)
    if teams == 1:
        problem = TeamProblem(units, [list(range(len(units)))], weights, cluster_cap)
        return OptimizationResult(
            [list(units)], problem.energy, problem.metrics(), [], "iterations", 0.0
        )

    runs = []
    best = None
    stopped_by = anneal.STOP_ITERATIONS
    for k in range(restarts):
        remaining = None if limit is None else limit - (time.time() - started)
        if remaining is not None and remaining <= 0 and runs:
            stopped_by = anneal.STOP_TIME
            break
        rng = random.Random(seed * 1000 + k)
        start = random_packing(units, teams, rng)
        problem = TeamProblem(units, start, weights, cluster_cap, target_range, target_role)
        t0 = opt.get("t0")
        t_end = opt.get("t_end")
        if t0 is None or t_end is None:
            c0, c_end = anneal.calibrate_temperatures(problem, rng)
            t0 = c0 if t0 is None else t0
            t_end = c_end if t_end is None else t_end
        run_limit = None if remaining is None else max(0.0, remaining) / (restarts - k)
        res = anneal.anneal(problem, rng, iterations, t0, t_end, time_limit_s=run_limit)
        problem.restore(res.best_state)
        m = problem.metrics()
        run = {
            "restart": k,
            "seed": seed * 1000 + k,
            "energy": round(res.best_energy, 6),
            "range": m["range"],
            "std": round(m["std"], 4),
            "role_penalty": m["role_penalty"],
            "cluster_teams": m["cluster_teams"],
            "iterations": res.iterations,
            "accepted": res.accepted,
            "t0": round(res.t0, 4),
            "t_end": round(res.t_end, 5),
            "elapsed_s": round(res.elapsed, 2),
            "stopped_by": res.stopped_by,
        }
        runs.append(run)
        if progress:
            progress(run)
        if best is None or res.best_energy < best[0] - 1e-12:
            best = (res.best_energy, res.best_state)
        if res.stopped_by == anneal.STOP_TARGET:
            stopped_by = anneal.STOP_TARGET
            break
        if res.stopped_by == anneal.STOP_TIME:
            stopped_by = anneal.STOP_TIME

    assert best is not None, "at least one restart always runs"
    problem = TeamProblem(units, [list(t) for t in best[1]], weights, cluster_cap)
    metrics = problem.metrics()
    solution = [[units[u] for u in team] for team in best[1]]
    return OptimizationResult(
        teams=solution,
        energy=metrics["energy"],
        metrics=metrics,
        runs=runs,
        stopped_by=stopped_by,
        elapsed=time.time() - started,
    )


# ---------------------------------------------------------------------------
# Cluster cap
# ---------------------------------------------------------------------------


def group_sizes(total_teams, groups_cfg):
    """Return the group sizes for ``total_teams`` teams from a groups block."""
    count = int(groups_cfg.get("count", 3))
    sizes = groups_cfg.get("sizes", "auto")
    if isinstance(sizes, (list, tuple)):
        return [int(s) for s in sizes]
    base, extra = divmod(total_teams, count)
    return [base + (1 if i < extra else 0) for i in range(count)]


def auto_cluster_cap(config, total_teams, fixed_cluster_teams=0):
    """Compute the cluster cap from the config.

    Args:
        config: Config dict.
        total_teams: Optimized plus fixed teams.
        fixed_cluster_teams: Fixed teams already containing a cluster player.

    Returns:
        Integer cap for the optimized teams, or None when disabled (no
        ``groups`` block and no ``cluster_max_teams``).
    """
    opt = config.get("optimizer") or {}
    explicit = opt.get("cluster_max_teams")
    if explicit is not None:
        return max(0, int(explicit) - fixed_cluster_teams)
    groups = config.get("groups")
    if not groups:
        return None
    sizes = group_sizes(total_teams, groups)
    if groups.get("na_server"):
        # NA teams can only gather in one group (the one that may move server).
        capacity = max(sizes)
    else:
        servers = groups.get("servers") or []
        home = servers[0] if servers else "London"
        capacity = sum(s for s, srv in zip(sizes, servers) if srv == home)
    return max(0, capacity - fixed_cluster_teams)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_solution(players, teams, fixed, subs, excluded):
    """Raise if any invariant of a solution is violated.

    Checks: every team has exactly 5 players; each player appears exactly once
    across teams, fixed teams, substitutes and exclusions; stacks of 2-3 are in
    a single team; no 4-stack plays.

    Args:
        players: Dict {name: player_info} (all input players).
        teams: List of lists of Unit (optimized teams).
        fixed: List of FixedTeam.
        subs: Iterable of substitute names.
        excluded: Iterable of excluded names.

    Raises:
        ValueError: Listing every violation found.
    """
    problems = []
    seen = {}

    def note(name, where):
        if name in seen:
            problems.append(f"{name} appears twice ({seen[name]} and {where})")
        seen[name] = where

    team_of = {}
    for i, team in enumerate(teams):
        names = [n for u in team for n in u.names]
        if len(names) != TEAM_SIZE:
            problems.append(f"team {i} has {len(names)} players")
        for unit in team:
            if unit.size > MAX_UNIT_SIZE:
                problems.append(f"team {i} contains a unit of size {unit.size}")
        for n in names:
            note(n, f"team {i}")
            team_of[n] = i
    for j, ft in enumerate(fixed):
        if len(ft.names) != TEAM_SIZE:
            problems.append(f"fixed team {j} has {len(ft.names)} players")
        for n in ft.names:
            note(n, f"fixed team {j}")
    for n in subs:
        note(n, "subs")
    for n in excluded:
        note(n, "excluded")

    missing = [n for n in players if n not in seen]
    if missing:
        problems.append("players not placed anywhere: " + ", ".join(missing))
    unknown = [n for n in seen if n not in players]
    if unknown:
        problems.append("unknown players placed: " + ", ".join(unknown))

    stacks = {}
    for n, info in players.items():
        gid = info.get("group_id")
        if gid is not None and info.get("status") != "substitute":
            stacks.setdefault(gid, []).append(n)
    for gid, members in stacks.items():
        placed = [team_of.get(n) for n in members if n in team_of]
        if len(members) == 4 and placed:
            problems.append(f"4-stack {gid} is playing")
        if 2 <= len(members) <= 3:
            if len(placed) not in (0, len(members)) or len(set(placed)) > 1:
                problems.append(f"stack {gid} is split")
    if problems:
        raise ValueError("invalid solution: " + "; ".join(problems))
