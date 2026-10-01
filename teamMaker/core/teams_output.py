"""Build, write and load the ``teams.json`` output (schema_version 1).

Layout::

    {
      "schema_version": 1,
      "meta": {generated_at, seed, seed_generated, config_sources,
               players_file, optimizer, stopped_by, restarts},
      "summary": {teams, optimized, all, role_penalty, cluster},
      "teams": [{id, name, fixed, formation, team_score, avg_score,
                 role_penalty, regions, players: [{name, discord, score, role,
                 assigned_role, region, server, formation, stack_id,
                 is_returning, ranks: {current, peak}}]}],
      "subs": [{name, discord, score, role, region, server, is_returning,
                ranks, reason}],
      "excluded": [{name, discord, score, stack_id, reason}]
    }

Timings are deliberately not stored so a rerun with the same seed gives an
identical file (apart from ``generated_at``).

Team ids are shuffled with the seed so numbering does not reveal strength;
fixed teams (5-stacks) are appended after the optimized ones. ``formation`` on
a team is its stack composition (for example "3+2"); on a player it is
solo/duo/trio/5-stack.
"""

import json
import math
import os
import random
import time

from teamMaker.core import roles as role_mod
from teamMaker.core.optimizer import STACK_LABELS

SCHEMA_VERSION = 1


def _stats(sums10):
    """Return min/max/range/mean/std of team totals given in tenths."""
    if not sums10:
        return {"n": 0, "min": None, "max": None, "range": None, "mean": None, "std": None}
    k = len(sums10)
    mean = sum(sums10) / k
    std = math.sqrt(sum((s - mean) ** 2 for s in sums10) / k)
    return {
        "n": k,
        "min": min(sums10) / 10.0,
        "max": max(sums10) / 10.0,
        "range": (max(sums10) - min(sums10)) / 10.0,
        "mean": round(mean / 10.0, 2),
        "std": round(std / 10.0, 3),
    }


def _player_base(name, info, scores):
    """Fields shared by team, sub and excluded entries."""
    return {
        "name": name,
        "discord": info.get("discord", ""),
        "score": scores[name],
    }


def _ranks(info):
    """Return the current/peak rank pair of a player."""
    return {"current": info.get("current_rank"), "peak": info.get("peak_rank")}


def _team_entry(team_id, fixed, groups, players, scores):
    """Build one team entry from a list of (names tuple, stack_id) units."""
    names = [n for unit_names, _ in groups for n in unit_names]
    penalty, assigned = role_mod.best_role_assignment([players[n].get("role") for n in names])
    assigned_by_name = dict(zip(names, assigned))
    entries = []
    regions = {}
    for unit_names, stack_id in groups:
        label = STACK_LABELS.get(len(unit_names), f"{len(unit_names)}-stack")
        for n in unit_names:
            info = players[n]
            entry = _player_base(n, info, scores)
            entry.update(
                {
                    "role": role_mod.normalize_roles(info.get("role")),
                    "assigned_role": assigned_by_name[n],
                    "region": info.get("region"),
                    "server": info.get("server", ""),
                    "formation": label,
                    "stack_id": stack_id,
                    "is_returning": bool(info.get("is_returning_player")),
                    "ranks": _ranks(info),
                }
            )
            entries.append(entry)
            regions[info.get("region")] = regions.get(info.get("region"), 0) + 1
    entries.sort(key=lambda e: (-e["score"], e["name"]))
    total10 = sum(int(round(e["score"] * 10)) for e in entries)
    formation = "+".join(
        str(s) for s in sorted((len(g[0]) for g in groups), reverse=True)
    )
    return {
        "id": team_id,
        "name": f"Team {team_id}",
        "fixed": fixed,
        "formation": formation,
        "team_score": total10 / 10.0,
        "avg_score": round(total10 / 10.0 / len(entries), 2),
        "role_penalty": penalty,
        "regions": dict(sorted(regions.items(), key=lambda kv: (-kv[1], str(kv[0])))),
        "players": entries,
    }


def build_teams_doc(
    players, scores, build, dropped_units, result, config, seed, cluster_cap=None
):
    """Assemble the teams.json document.

    Args:
        players: Dict {name: player_info}.
        scores: Dict {name: score rounded to 1 decimal}.
        build: optimizer.BuildResult (fixed teams, pool subs, exclusions).
        dropped_units: Units dropped by select_units (become subs).
        result: optimizer.OptimizationResult.
        config: Config dict.
        seed: Integer seed used.
        cluster_cap: Cluster cap used by the optimizer (or None).

    Returns:
        JSON-serializable dict.
    """
    output_cfg = config.get("output") or {}
    ordered = list(result.teams)
    if output_cfg.get("team_order", "shuffled") == "shuffled":
        random.Random(seed * 1000 + 999).shuffle(ordered)
    else:
        ordered.sort(key=lambda team: -sum(u.tenths for u in team))

    teams = []
    for team in ordered:
        groups = [(u.names, u.group_id) for u in team]
        teams.append(_team_entry(len(teams) + 1, False, groups, players, scores))
    optimized_count = len(teams)
    for ft in build.fixed:
        groups = [(ft.names, ft.group_id)]
        teams.append(_team_entry(len(teams) + 1, True, groups, players, scores))

    def tenths(team):
        return sum(int(round(p["score"] * 10)) for p in team["players"])

    opt_teams = teams[:optimized_count]
    cluster_regions = [str(r).upper() for r in (config.get("optimizer") or {}).get("cluster_regions") or []]
    cluster_teams = sum(
        1 for t in teams if any(str(r).upper() in cluster_regions for r in t["regions"])
    )
    opt_cluster = sum(
        1 for t in opt_teams if any(str(r).upper() in cluster_regions for r in t["regions"])
    )
    summary = {
        "teams": len(teams),
        "players_in_teams": sum(len(t["players"]) for t in teams),
        "optimized": _stats([tenths(t) for t in opt_teams]),
        "all": _stats([tenths(t) for t in teams]),
        "role_penalty": {
            "optimized": sum(t["role_penalty"] for t in opt_teams),
            "fixed": sum(t["role_penalty"] for t in teams[optimized_count:]),
            "total": sum(t["role_penalty"] for t in teams),
        },
        "cluster": {
            "regions": cluster_regions,
            "cap": cluster_cap,
            "teams_optimized": opt_cluster,
            "teams_all": cluster_teams,
        },
    }

    subs = []
    for name, reason in build.pool_subs:
        subs.append((name, reason))
    for unit in dropped_units:
        reason = (
            f"not needed to fill whole teams ({unit.label}, signup #{unit.signup}"
            + (", returning" if unit.returning else "")
            + ")"
        )
        for n in unit.names:
            subs.append((n, reason))
    sub_entries = []
    for name, reason in subs:
        info = players[name]
        entry = _player_base(name, info, scores)
        entry.update(
            {
                "role": role_mod.normalize_roles(info.get("role")),
                "region": info.get("region"),
                "server": info.get("server", ""),
                "is_returning": bool(info.get("is_returning_player")),
                "ranks": _ranks(info),
                "reason": reason,
            }
        )
        sub_entries.append(entry)
    sub_entries.sort(key=lambda e: (-e["score"], e["name"]))

    excluded = []
    for names, reason in build.excluded:
        for n in names:
            entry = _player_base(n, players[n], scores)
            entry["stack_id"] = players[n].get("group_id")
            entry["reason"] = reason
            excluded.append(entry)

    optimizer_meta = dict(config.get("optimizer") or {})
    meta = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed": seed,
        "seed_generated": bool(config.get("_seed_generated")),
        "config_sources": [os.path.basename(p) for p in config.get("_sources", [])],
        "players_file": config.get("players_file"),
        "optimizer": optimizer_meta,
        "stopped_by": result.stopped_by,
        "restarts": [
            {k: v for k, v in run.items() if k != "elapsed_s"} for run in result.runs
        ],
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "meta": meta,
        "summary": summary,
        "teams": teams,
        "subs": sub_entries,
        "excluded": excluded,
    }


def write_teams_json(doc, path):
    """Write the document as UTF-8 JSON (ensure_ascii off)."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
        f.write("\n")


def load_teams_json(path):
    """Load teams.json and validate its schema version.

    Args:
        path: Path of the file.

    Returns:
        The parsed document.

    Raises:
        ValueError: If the schema version is missing or not supported.
    """
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    version = doc.get("schema_version") if isinstance(doc, dict) else None
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported teams.json schema_version {version!r} "
            f"(expected {SCHEMA_VERSION}): {path}"
        )
    return doc
