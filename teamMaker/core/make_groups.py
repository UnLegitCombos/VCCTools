"""Split the teams of ``output/teams.json`` into groups (pools).

Goal: put as many region-costly teams (for example NA players) into the group
whose server suits them (for example London), then balance strength across the
groups. The energy minimised is::

    sum(region cost of each team on its group's server)
    + balance_weight * range(group mean team score)
    + std_weight * std(group mean team score)

Modes:

- ``auto``: simulated annealing (reuses ``anneal.py``), swapping two unpinned
  teams that sit in different groups.
- ``manual``: groups are listed in the config (``groups.manual``).
- ``reuse``: keep the groups of an existing ``output/groups.json``; fails
  loudly if the teams changed since it was written.

Run from the repo root with ``python -m teamMaker.core.make_groups``.
"""

import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field

from teamMaker.core import anneal
from teamMaker.core.config import BASE_DIR, GROUPS_DEFAULTS, load_team_config, resolve_seed
from teamMaker.core.teams_output import load_teams_json
from teamMaker.core.utils.console import setup_console

OUTPUT_DIR = os.path.join(BASE_DIR, "output")
GROUPS_SCHEMA_VERSION = 1
MODES = ("auto", "manual", "reuse")


class GroupsError(ValueError):
    """Raised for invalid group configuration or inputs."""


@dataclass
class GroupSettings:
    """Settings of the ``groups:`` config block."""

    count: int = 3
    servers: list = field(default_factory=lambda: ["London", "Frankfurt", "Frankfurt"])
    sizes: object = "auto"
    region_server_cost: dict = field(default_factory=dict)
    balance_weight: float = 1.0
    std_weight: float = 0.2
    mode: str = "auto"
    manual: object = None
    pinned: dict = field(default_factory=dict)
    iterations: int = 200000
    restarts: int = 4
    title: str = "VCC Groups"
    priority_region: str = "NA"

    @classmethod
    def from_config(cls, config):
        """Build the settings from a config dict.

        Args:
            config: Config dict; ``groups`` may be None (defaults are used)
                and ``optimizer.cluster_regions[0]`` is the priority region.

        Returns:
            GroupSettings instance.

        Raises:
            GroupsError: If a value has the wrong type or the mode is unknown.
        """
        block = dict(GROUPS_DEFAULTS)
        block.update(config.get("groups") or {})
        cluster = (config.get("optimizer") or {}).get("cluster_regions") or ["NA"]
        try:
            pinned = {int(k): int(v) for k, v in (block.get("pinned") or {}).items()}
            settings = cls(
                count=int(block["count"]),
                servers=[str(s) for s in block["servers"]],
                sizes=block["sizes"],
                region_server_cost=block.get("region_server_cost") or {},
                balance_weight=float(block["balance_weight"]),
                std_weight=float(block["std_weight"]),
                mode=str(block["mode"]).lower(),
                manual=block.get("manual"),
                pinned=pinned,
                iterations=int(block["iterations"]),
                restarts=int(block["restarts"]),
                title=str(block["title"]),
                priority_region=str(cluster[0]).upper(),
            )
        except (TypeError, ValueError, KeyError) as exc:
            raise GroupsError(f"Invalid groups config: {exc}") from exc
        if settings.mode not in MODES:
            raise GroupsError(f"groups.mode must be one of {MODES}, got {settings.mode!r}")
        return settings


# ---------------------------------------------------------------------------
# Costs and sizes
# ---------------------------------------------------------------------------


def region_cost_table(settings):
    """Return {server_lower: {REGION_UPPER: cost}} from the settings."""
    table = {}
    for server, costs in (settings.region_server_cost or {}).items():
        table[str(server).lower()] = {
            str(region).upper(): float(cost) for region, cost in (costs or {}).items()
        }
    return table


def team_server_costs(teams, settings):
    """Return the cost of every team on every group's server.

    The cost of a team in a group is the sum over its players of
    ``region_server_cost[server][player region]``.

    Args:
        teams: List of team dicts from teams.json (``regions`` counts are used).
        settings: GroupSettings.

    Returns:
        List indexed like ``teams``; each item is a list with one cost per
        group (indexed by group), so ``costs[t][g]``.
    """
    table = region_cost_table(settings)
    costs = []
    for team in teams:
        row = []
        for server in settings.servers:
            per_region = table.get(str(server).lower(), {})
            row.append(
                sum(
                    per_region.get(str(region).upper(), 0.0) * n
                    for region, n in (team.get("regions") or {}).items()
                )
            )
        costs.append(row)
    return costs


def resolve_sizes(n_teams, settings):
    """Return the size of every group.

    With ``sizes: auto`` the teams are split evenly and the extra slots go to
    the groups cheapest for the priority region (first in the list on ties).

    Args:
        n_teams: Number of teams.
        settings: GroupSettings.

    Returns:
        List of group sizes.

    Raises:
        GroupsError: If the count, servers or sizes are inconsistent.
    """
    count = settings.count
    if count < 1:
        raise GroupsError("groups.count must be at least 1")
    if count > n_teams:
        raise GroupsError(f"groups.count ({count}) is larger than the number of teams ({n_teams})")
    if len(settings.servers) != count:
        raise GroupsError(
            f"groups.servers has {len(settings.servers)} entries but groups.count is {count}"
        )
    if isinstance(settings.sizes, (list, tuple)):
        sizes = [int(s) for s in settings.sizes]
        if len(sizes) != count:
            raise GroupsError(f"groups.sizes has {len(sizes)} entries but groups.count is {count}")
        if sum(sizes) != n_teams:
            raise GroupsError(f"groups.sizes sums to {sum(sizes)} but there are {n_teams} teams")
        if min(sizes) < 1:
            raise GroupsError("groups.sizes must all be at least 1")
        return sizes
    if str(settings.sizes).lower() != "auto":
        raise GroupsError(f"groups.sizes must be 'auto' or a list, got {settings.sizes!r}")
    base, extra = divmod(n_teams, count)
    table = region_cost_table(settings)
    priority = settings.priority_region

    def cost_for_priority(g):
        return table.get(settings.servers[g].lower(), {}).get(priority, 0.0)

    order = sorted(range(count), key=lambda g: (cost_for_priority(g), g))
    sizes = [base] * count
    for g in order[:extra]:
        sizes[g] += 1
    return sizes


def validate_pins(pinned, team_ids, sizes):
    """Check the pins against the team ids and the group capacities.

    Args:
        pinned: {team_id: group_number (1-based)}.
        team_ids: List of valid team ids.
        sizes: Group sizes.

    Raises:
        GroupsError: On an unknown team, bad group number or full group.
    """
    known = set(team_ids)
    used = [0] * len(sizes)
    for team_id, group in pinned.items():
        if team_id not in known:
            raise GroupsError(f"groups.pinned: unknown team id {team_id}")
        if not 1 <= group <= len(sizes):
            raise GroupsError(
                f"groups.pinned: team {team_id} -> group {group} (valid: 1-{len(sizes)})"
            )
        used[group - 1] += 1
    for g, n in enumerate(used):
        if n > sizes[g]:
            raise GroupsError(f"groups.pinned: {n} teams pinned to group {g + 1} (capacity {sizes[g]})")


# ---------------------------------------------------------------------------
# Energy and optimization
# ---------------------------------------------------------------------------


def _balance(means, balance_weight, std_weight):
    """Return the balance part of the energy for a list of group means."""
    g = len(means)
    if g < 2:
        return 0.0
    mean = sum(means) / g
    std = math.sqrt(sum((m - mean) ** 2 for m in means) / g)
    return balance_weight * (max(means) - min(means)) + std_weight * std


def assignment_energy(assign, scores, costs, sizes, settings):
    """Compute the energy of an assignment from scratch.

    Args:
        assign: List, group index of each team.
        scores: Team scores.
        costs: Output of team_server_costs.
        sizes: Group sizes.
        settings: GroupSettings.

    Returns:
        Float energy.
    """
    sums = [0.0] * len(sizes)
    cost = 0.0
    for t, g in enumerate(assign):
        sums[g] += scores[t]
        cost += costs[t][g]
    means = [sums[g] / sizes[g] for g in range(len(sizes))]
    return cost + _balance(means, settings.balance_weight, settings.std_weight)


class GroupProblem:
    """Annealing problem: swap two unpinned teams that sit in different groups.

    Group sizes never change. Each move is evaluated in O(groups).
    """

    def __init__(self, scores, costs, sizes, settings, pins, rng):
        """Create a random feasible start.

        Args:
            scores: Team scores indexed by team position.
            costs: Output of team_server_costs.
            sizes: Group sizes.
            settings: GroupSettings.
            pins: {team position: group index (0-based)}.
            rng: random.Random used for the start.
        """
        self.scores = scores
        self.costs = costs
        self.sizes = sizes
        self.settings = settings
        self.pins = pins
        n = len(scores)
        self.movable = [t for t in range(n) if t not in pins]
        self.assign = [0] * n
        slots = []
        left = list(sizes)
        for t, g in pins.items():
            self.assign[t] = g
            left[g] -= 1
        for g, k in enumerate(left):
            slots.extend([g] * k)
        rng.shuffle(slots)
        for t, g in zip(self.movable, slots):
            self.assign[t] = g
        self._recompute()
        self._undo = None

    def _recompute(self):
        """Rebuild sums, cost and energy from the assignment."""
        self.sums = [0.0] * len(self.sizes)
        self.cost = 0.0
        for t, g in enumerate(self.assign):
            self.sums[g] += self.scores[t]
            self.cost += self.costs[t][g]
        self.energy = self._energy(self.sums, self.cost)

    def _energy(self, sums, cost):
        means = [sums[g] / self.sizes[g] for g in range(len(sums))]
        return cost + _balance(means, self.settings.balance_weight, self.settings.std_weight)

    def can_move(self):
        """Return True if at least one swap is possible."""
        return len(self.sizes) > 1 and len(self.movable) > 1

    def propose(self, rng):
        """Swap two random movable teams from different groups (in place)."""
        if not self.can_move():
            return None
        movable = self.movable
        for _ in range(30):
            a = movable[rng.randrange(len(movable))]
            b = movable[rng.randrange(len(movable))]
            ga, gb = self.assign[a], self.assign[b]
            if ga != gb:
                break
        else:
            return None
        sa, sb = self.scores[a], self.scores[b]
        self._undo = (a, b, self.energy, self.cost, self.sums[ga], self.sums[gb])
        self.sums[ga] += sb - sa
        self.sums[gb] += sa - sb
        self.cost += (
            self.costs[a][gb] + self.costs[b][ga] - self.costs[a][ga] - self.costs[b][gb]
        )
        self.assign[a], self.assign[b] = gb, ga
        self.energy = self._energy(self.sums, self.cost)
        return self.energy

    def accept(self):
        """Keep the proposed swap."""
        self._undo = None

    def reject(self):
        """Undo the proposed swap."""
        assert self._undo is not None, "reject() called without a proposed swap"
        a, b, energy, cost, sum_a, sum_b = self._undo
        ga, gb = self.assign[a], self.assign[b]
        self.assign[a], self.assign[b] = gb, ga
        self.sums[gb] = sum_a
        self.sums[ga] = sum_b
        self.cost = cost
        self.energy = energy
        self._undo = None

    def snapshot(self):
        """Return a copy of the assignment."""
        return list(self.assign)

    def exact_energy(self):
        """Recompute the energy of the current assignment from scratch."""
        return assignment_energy(self.assign, self.scores, self.costs, self.sizes, self.settings)

    def at_target(self):
        """Groups have no early-stop target."""
        return False


def optimize_groups(scores, costs, sizes, settings, pins, seed, iterations=None, restarts=None):
    """Find the assignment with the lowest energy (multi-start, seeded).

    Args:
        scores: Team scores indexed by team position.
        costs: Output of team_server_costs.
        sizes: Group sizes.
        settings: GroupSettings.
        pins: {team position: group index (0-based)}.
        seed: Integer seed; restart k uses ``random.Random(seed * 1000 + k)``.
        iterations: Iterations per restart (default from settings).
        restarts: Number of restarts (default from settings).

    Returns:
        Tuple (assign, energy): group index of each team and its energy.
    """
    iterations = settings.iterations if iterations is None else iterations
    restarts = max(1, settings.restarts if restarts is None else restarts)
    best_assign, best_energy = None, None
    for k in range(restarts):
        rng = random.Random(seed * 1000 + k)
        problem = GroupProblem(scores, costs, sizes, settings, pins, rng)
        if problem.can_move():
            t0, t_end = anneal.calibrate_temperatures(problem, rng, samples=500)
            result = anneal.anneal(problem, rng, iterations, t0, t_end)
            assign, energy = result.best_state, result.best_energy
        else:
            assign, energy = problem.snapshot(), problem.exact_energy()
        if best_energy is None or energy < best_energy - 1e-9:
            best_assign, best_energy = assign, energy
        if not problem.can_move():
            break
    return best_assign, best_energy


# ---------------------------------------------------------------------------
# Manual and reuse modes
# ---------------------------------------------------------------------------


def manual_groups(manual, team_ids, count):
    """Validate a manual grouping and return it as lists of team ids.

    Args:
        manual: List of lists of team ids from the config.
        team_ids: All team ids.
        count: Expected number of groups.

    Returns:
        List of lists of int team ids.

    Raises:
        GroupsError: On duplicates, missing or unknown ids, or empty groups.
    """
    if not isinstance(manual, (list, tuple)) or not manual:
        raise GroupsError("groups.mode is 'manual' but groups.manual is missing")
    try:
        groups = [[int(t) for t in g] for g in manual]
    except (TypeError, ValueError) as exc:
        raise GroupsError(f"groups.manual must be a list of lists of team ids: {exc}") from exc
    if len(groups) != count:
        raise GroupsError(f"groups.manual has {len(groups)} groups but groups.count is {count}")
    for i, g in enumerate(groups, start=1):
        if not g:
            raise GroupsError(f"groups.manual: group {i} is empty")
    seen = {}
    for i, g in enumerate(groups, start=1):
        for t in g:
            if t in seen:
                raise GroupsError(
                    f"groups.manual: team {t} appears more than once "
                    f"(groups {seen[t]} and {i})"
                )
            seen[t] = i
    known = set(team_ids)
    unknown = sorted(set(seen) - known)
    if unknown:
        raise GroupsError(f"groups.manual: unknown team ids {unknown}")
    missing = sorted(known - set(seen))
    if missing:
        raise GroupsError(f"groups.manual: teams not assigned to any group: {missing}")
    return groups


def reuse_groups(groups_doc, teams_doc, count):
    """Return the groups of an existing groups.json after checking it.

    Args:
        groups_doc: Parsed groups.json (or None if it does not exist).
        teams_doc: Parsed teams.json.
        count: Expected number of groups.

    Returns:
        List of lists of team ids.

    Raises:
        GroupsError: If groups.json is missing, was made from other teams, or
            its team ids do not match teams.json.
    """
    if not groups_doc:
        raise GroupsError("groups.mode is 'reuse' but output/groups.json does not exist")
    generated = (groups_doc.get("meta") or {}).get("teams_generated_at")
    current = (teams_doc.get("meta") or {}).get("generated_at")
    if generated != current:
        raise GroupsError(
            f"groups.json was made from teams generated at {generated}, but teams.json is "
            f"from {current}; use mode auto or manual to regroup the new teams"
        )
    groups = [[int(t) for t in g["team_ids"]] for g in groups_doc.get("groups", [])]
    known = {int(t["id"]) for t in teams_doc["teams"]}
    used = [t for g in groups for t in g]
    missing = sorted(known - set(used))
    extra = sorted(set(used) - known)
    if missing or extra or len(used) != len(set(used)):
        raise GroupsError(
            f"groups.json does not match teams.json (new teams: {missing}, "
            f"unknown teams: {extra}, duplicates: {len(used) - len(set(used))})"
        )
    if len(groups) != count:
        raise GroupsError(f"groups.json has {len(groups)} groups but groups.count is {count}")
    return groups


# ---------------------------------------------------------------------------
# Document, report, entry point
# ---------------------------------------------------------------------------


def build_groups_doc(teams_doc, groups, settings, seed, mode, energy=None):
    """Assemble the groups.json document.

    Args:
        teams_doc: Parsed teams.json.
        groups: List of lists of team ids, one list per group.
        settings: GroupSettings.
        seed: Integer seed.
        mode: Mode used (auto/manual/reuse).
        energy: Optimizer energy, when known.

    Returns:
        JSON-serializable dict.
    """
    teams = {int(t["id"]): t for t in teams_doc["teams"]}
    ordered = [sorted(g) for g in groups]
    positions = list(teams)
    pos_of = {tid: i for i, tid in enumerate(positions)}
    costs = team_server_costs([teams[t] for t in positions], settings)

    priority = settings.priority_region
    table = region_cost_table(settings)
    group_entries = []
    means = []
    total_cost = 0.0
    for gi, ids in enumerate(ordered):
        member_scores = [teams[t]["team_score"] for t in ids]
        mean = sum(member_scores) / len(ids)
        means.append(mean)
        regions = {}
        for t in ids:
            for region, n in (teams[t].get("regions") or {}).items():
                regions[region] = regions.get(region, 0) + n
        cost = sum(costs[pos_of[t]][gi] for t in ids)
        total_cost += cost
        group_entries.append(
            {
                "index": gi + 1,
                "name": f"Group {gi + 1}",
                "server": settings.servers[gi],
                "team_ids": ids,
                "mean_team_score": round(mean, 2),
                "regions": dict(sorted(regions.items(), key=lambda kv: (-kv[1], str(kv[0])))),
                "na_teams": sum(1 for t in ids if priority in {str(r).upper() for r in teams[t].get("regions") or {}}),
                "region_cost": cost,
                "costly_teams": sum(1 for t in ids if costs[pos_of[t]][gi] > 0),
                "priority_outside": (
                    sum(1 for t in ids if priority in {str(r).upper() for r in teams[t].get("regions") or {}})
                    if table.get(str(settings.servers[gi]).lower(), {}).get(priority, 0.0) > 0
                    else 0
                ),
            }
        )
    summary = {
        "region_cost": total_cost,
        "costly_teams": sum(g["costly_teams"] for g in group_entries),
        "priority_outside": sum(g["priority_outside"] for g in group_entries),
        "priority_region": priority,
        "group_means": [round(m, 2) for m in means],
        "range": round(max(means) - min(means), 2),
        "energy": None if energy is None else round(energy, 4),
    }
    meta = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "teams_generated_at": (teams_doc.get("meta") or {}).get("generated_at"),
        "seed": seed,
        "mode": mode,
        "title": settings.title,
        "servers": list(settings.servers),
        "balance_weight": settings.balance_weight,
        "std_weight": settings.std_weight,
        "region_server_cost": settings.region_server_cost,
        "pinned": {str(k): v for k, v in settings.pinned.items()},
    }
    return {
        "schema_version": GROUPS_SCHEMA_VERSION,
        "meta": meta,
        "summary": summary,
        "groups": group_entries,
    }


def format_groups(doc, teams_doc):
    """Return the console listing of the groups as text."""
    teams = {int(t["id"]): t for t in teams_doc["teams"]}
    lines = [f"=== {doc['meta']['title']} ==="]
    for g in doc["groups"]:
        lines.append(
            f"Group {g['index']} ({g['server']}): {len(g['team_ids'])} teams, "
            f"mean {g['mean_team_score']:.2f}, NA teams {g['na_teams']}"
        )
        for tid in g["team_ids"]:
            t = teams[tid]
            regions = ", ".join(f"{r} {n}" for r, n in (t.get("regions") or {}).items())
            lines.append(f"  Team {tid:>2}  {t['team_score']:6.1f}  {regions}")
    s = doc["summary"]
    lines.append(f"Group mean range: {s['range']:.2f}  (means: {s['group_means']})")
    lines.append(f"Total region cost: {s['region_cost']:g}")
    lines.append(
        f"{s['priority_region']} teams outside London-type groups: {s['priority_outside']}"
    )
    return "\n".join(lines)


def _load_json(path):
    """Load a JSON file, or return None if it does not exist."""
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run(config=None, teams_path=None, out_dir=None, render=True):
    """Group the teams and write groups.json (and groups.png when possible).

    Args:
        config: Optional preloaded config (defaults to load_team_config()).
        teams_path: teams.json path (default ``output/teams.json``).
        out_dir: Output directory (default ``output``).
        render: Whether to try rendering groups.png.

    Returns:
        Tuple (groups_doc, report_text).

    Raises:
        GroupsError: On invalid configuration or inputs.
    """
    config = config or load_team_config()
    out_dir = out_dir or OUTPUT_DIR
    teams_path = teams_path or os.path.join(out_dir, "teams.json")
    if not os.path.isfile(teams_path):
        raise GroupsError(f"{teams_path} not found; run build_teams first")
    teams_doc = load_teams_json(teams_path)
    settings = GroupSettings.from_config(config)
    seed = resolve_seed(config)

    teams = teams_doc["teams"]
    team_ids = [int(t["id"]) for t in teams]
    pos_of = {tid: i for i, tid in enumerate(team_ids)}
    groups_path = os.path.join(out_dir, "groups.json")

    energy = None
    if settings.mode == "manual":
        if not settings.servers or len(settings.servers) != settings.count:
            raise GroupsError(
                f"groups.servers has {len(settings.servers)} entries but groups.count is {settings.count}"
            )
        groups = manual_groups(settings.manual, team_ids, settings.count)
        for tid, g in settings.pinned.items():
            if not 1 <= g <= len(groups) or tid not in groups[g - 1]:
                raise GroupsError(f"groups.pinned: team {tid} is not in group {g} of groups.manual")
    elif settings.mode == "reuse":
        groups = reuse_groups(_load_json(groups_path), teams_doc, settings.count)
        if len(settings.servers) != settings.count:
            raise GroupsError(
                f"groups.servers has {len(settings.servers)} entries but groups.count is {settings.count}"
            )
    else:
        sizes = resolve_sizes(len(teams), settings)
        validate_pins(settings.pinned, team_ids, sizes)
        pins = {pos_of[t]: g - 1 for t, g in settings.pinned.items()}
        scores = [float(t["team_score"]) for t in teams]
        costs = team_server_costs(teams, settings)
        assign, energy = optimize_groups(scores, costs, sizes, settings, pins, seed)
        assert assign is not None
        groups = [[] for _ in sizes]
        for pos, g in enumerate(assign):
            groups[g].append(team_ids[pos])

    doc = build_groups_doc(teams_doc, groups, settings, seed, settings.mode, energy)
    text = format_groups(doc, teams_doc)

    os.makedirs(out_dir, exist_ok=True)
    with open(groups_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
        f.write("\n")
    if render:
        _render_png(doc, teams_doc, os.path.join(out_dir, "groups.png"), settings.title)
    return doc, text


def _render_png(doc, teams_doc, path, title):
    """Render groups.png when the renderer (and Pillow) are available."""
    try:
        from teamMaker.core.render import render_groups_sheet
    except ImportError:
        return None
    try:
        render_groups_sheet(doc, teams_doc, path, title, per_row=6)
    except Exception as exc:  # rendering must never lose groups.json
        print(f"Warning: could not render {os.path.basename(path)}: {exc}")
        return None
    return path


def main():
    """Command line entry point."""
    setup_console()
    try:
        doc, text = run()
    except (GroupsError, ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    print(text)
    print(f"Wrote {os.path.join(OUTPUT_DIR, 'groups.json')}")


if __name__ == "__main__":
    main()
