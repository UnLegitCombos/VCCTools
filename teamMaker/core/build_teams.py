"""Team Maker entry point.

Pipeline: config -> players -> scores -> units -> sub selection ->
optimizer -> validation -> teams.json -> report -> teams.png.

Run from the repo root with ``python -m teamMaker.core.build_teams``; add
``--tighter`` to apply the tighter-teams optimizer profile and ``--no-time-limit``
to let every restart run its full iteration budget. ``--solver`` picks the
search: annealing (default), ortools or both (needs requirements-ortools.txt).
"""

import argparse
import os
import sys

from teamMaker.core import cpsat
from teamMaker.core import optimizer as opt
from teamMaker.core import report, teams_output
from teamMaker.core.config import BASE_DIR, TIGHTER_PROFILE, load_team_config, resolve_seed
from teamMaker.core.scoring import _export_scores, _load_players
from teamMaker.core.utils.console import setup_console

DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")


def _players_path(config):
    """Resolve the players file: data/ first, then data/input/."""
    name = config.get("players_file", "players.json")
    if os.path.isabs(name):
        return name
    for folder in (DATA_DIR, os.path.join(DATA_DIR, "input")):
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            return path
    return os.path.join(DATA_DIR, name)


def _render_png(doc, path, config):
    """Render teams.png when enabled and the renderer is available."""
    if not (config.get("output") or {}).get("teams_png", True):
        return None
    try:
        from teamMaker.core.render import render_teams_sheet
    except ImportError:
        return None
    title = (config.get("groups") or {}).get("title", "VCC Teams")
    try:
        render_teams_sheet(doc, path, title)
    except Exception as exc:  # rendering must never lose the teams
        print(f"Warning: could not render {os.path.basename(path)}: {exc}")
        return None
    return path


SOLVERS = ("annealing", "ortools", "both")


def search(units, config, seed, cluster_cap, progress=None):
    """Split the units into teams with the configured solver.

    ``optimizer.solver``: ``annealing`` (simulated annealing), ``ortools``
    (OR-Tools CP-SAT) or ``both`` (annealing, then OR-Tools starts from its
    result; the better of the two is kept).

    Returns:
        OptimizationResult; with ``both`` its runs list holds the annealing
        restarts followed by the OR-Tools run.
    """
    o = config["optimizer"]
    solver = str(o.get("solver", "annealing")).lower()
    if solver not in SOLVERS:
        raise ValueError(f"optimizer.solver must be one of {SOLVERS}, got {solver!r}")
    if solver == "annealing":
        return opt.optimize_teams(units, config, seed, cluster_cap, progress)

    def or_progress(seconds, m):
        print(
            f"  OR-Tools {seconds:6.1f}s: range {m['range']:.2f}, "
            f"roles {m['role_penalty']:.1f}, energy {m['energy']:.3f}"
        )

    options = {
        "time_limit_s": o.get("ortools_time_limit_s", 300),
        "provers": str(o.get("ortools_provers", "one")).lower(),
        "workers": o.get("ortools_workers", 8),
        "progress": or_progress,
    }
    if solver == "ortools":
        print(f"Searching with OR-Tools ({options['time_limit_s']}s, provers: {options['provers']})...")
        return cpsat.solve_teams(units, config, seed, cluster_cap, **options)

    first = opt.optimize_teams(units, config, seed, cluster_cap, progress)
    print(
        f"Improving the annealing result with OR-Tools "
        f"({options['time_limit_s']}s, provers: {options['provers']})..."
    )
    second = cpsat.solve_teams(units, config, seed, cluster_cap, hint=first.teams, **options)
    best = second if second.energy < first.energy - 1e-9 else first
    return opt.OptimizationResult(
        teams=best.teams,
        energy=best.energy,
        metrics=best.metrics,
        runs=first.runs + second.runs,
        stopped_by=f"{first.stopped_by}, then OR-Tools {second.stopped_by}",
        elapsed=first.elapsed + second.elapsed,
    )


def run(config=None):
    """Build the teams and write the outputs.

    Args:
        config: Optional preloaded config (defaults to load_team_config()).

    Returns:
        Tuple (doc, report_text) with the teams.json document and the report.
    """
    config = config or load_team_config()
    seed = resolve_seed(config)
    warnings = list(config.get("_warnings", []))

    players_path = _players_path(config)
    players = _load_players(players_path)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    _, scores = _export_scores(
        players,
        config,
        os.path.join(OUTPUT_DIR, "player_scores.json"),
        os.path.join(OUTPUT_DIR, "player_scores_minimal.json"),
    )

    build = opt.build_units(players, scores, config)
    warnings += build.warnings
    kept, dropped = opt.select_units(build.units)
    team_count = sum(u.size for u in kept) // opt.TEAM_SIZE
    fixed_cluster = 0
    cluster_regions = {str(r).upper() for r in (config["optimizer"].get("cluster_regions") or [])}
    for ft in build.fixed:
        if any(str(players[n].get("region", "")).upper() in cluster_regions for n in ft.names):
            fixed_cluster += 1
    cap = opt.auto_cluster_cap(config, team_count + len(build.fixed), fixed_cluster)

    def progress(run_info):
        print(
            f"  restart {run_info['restart']}: range {run_info['range']:.2f}, "
            f"roles {run_info['role_penalty']:.1f}, {run_info['iterations']} iters, "
            f"{run_info['elapsed_s']:.1f}s ({run_info['stopped_by']})"
        )

    print(f"Optimizing {team_count} teams from {len(kept)} units (seed {seed})...")
    result = search(kept, config, seed, cap, progress)

    excluded_names = [n for names, _ in build.excluded for n in names]
    sub_names = [n for n, _ in build.pool_subs] + [n for u in dropped for n in u.names]
    opt.validate_solution(players, result.teams, build.fixed, sub_names, excluded_names)

    doc = teams_output.build_teams_doc(
        players, scores, build, dropped, result, config, seed, cap
    )
    paths = {"teams": os.path.join(OUTPUT_DIR, "teams.json")}
    teams_output.write_teams_json(doc, paths["teams"])
    png = _render_png(doc, os.path.join(OUTPUT_DIR, "teams.png"), config)
    if png:
        paths["teams png"] = png
    paths["scores"] = os.path.join(OUTPUT_DIR, "player_scores_minimal.json")

    info = {
        "players": len(players),
        "units": len(build.units),
        "fixed": len(build.fixed),
        "pool_subs": len(build.pool_subs),
        "dropped_units": len(dropped),
        "excluded": len(excluded_names),
        "elapsed_s": result.elapsed,
        "run_times": [r["elapsed_s"] for r in result.runs],
    }
    text = report.format_report(doc, info, paths, warnings)
    return doc, text


def main(argv=None):
    """Command line entry point."""
    setup_console()
    parser = argparse.ArgumentParser(description="Build balanced teams from players.json")
    parser.add_argument(
        "--tighter",
        action="store_true",
        help=f"apply the optimizer settings in config/{TIGHTER_PROFILE} "
        "(closer team totals, slower)",
    )
    parser.add_argument(
        "--no-time-limit",
        action="store_true",
        help="ignore optimizer.time_limit_s: every restart runs its full iteration "
        "budget (stops early only at the target); slow, but reproducible",
    )
    parser.add_argument(
        "--solver", choices=SOLVERS,
        help="search method (default: optimizer.solver, normally annealing); "
        "ortools and both need requirements-ortools.txt",
    )
    parser.add_argument(
        "--provers", choices=cpsat.PROVERS,
        help="OR-Tools workers spent proving a bound: all, one or none (default one)",
    )
    parser.add_argument("--ortools-time", type=float, metavar="SECONDS",
                        help="how long OR-Tools searches (default 300)")
    parser.add_argument("--ortools-workers", type=int, metavar="N",
                        help="CPU threads for OR-Tools (default 8)")
    args = parser.parse_args(argv)
    config = load_team_config(profile=TIGHTER_PROFILE if args.tighter else None)
    opt_cfg = config["optimizer"]
    for flag, key in (
        ("solver", "solver"),
        ("provers", "ortools_provers"),
        ("ortools_time", "ortools_time_limit_s"),
        ("ortools_workers", "ortools_workers"),
    ):
        if getattr(args, flag) is not None:
            opt_cfg[key] = getattr(args, flag)
    if args.tighter:
        print(f"Using the tighter-teams profile ({TIGHTER_PROFILE})")
    if args.no_time_limit:
        opt_cfg["time_limit_s"] = None
        print(
            f"No time limit: {opt_cfg['restarts']} restarts of up to "
            f"{opt_cfg['iterations']:,} iterations each; this can take tens of minutes"
        )
    try:
        _, text = run(config)
    except (cpsat.OrToolsUnavailable, ValueError) as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    print(text)


if __name__ == "__main__":
    main()
