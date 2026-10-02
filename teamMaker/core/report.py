"""Plain ASCII console report for a teams.json document."""

import os

WIDTH = 78
BAR_WIDTH = 20  # characters per side of the team totals chart
BAR_STEPS = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0)  # points per character, smallest that fits


def _rule(char="="):
    return char * WIDTH


def _section(title):
    return ["", _rule(), title, _rule()]


def bar_step(diffs):
    """Points per bar character: the smallest step that fits every difference."""
    biggest = max((abs(d) for d in diffs), default=0.0)
    for step in BAR_STEPS:
        if biggest / step <= BAR_WIDTH:
            return step
    return BAR_STEPS[-1]


def diff_bar(diff, step):
    """Bar for a difference from the average, left of '|' below it, right above.

    Each character is ``step`` points; a bar longer than BAR_WIDTH ends in
    '<' or '>' to show it was cut.
    """
    n = int(round(abs(diff) / step))
    cut = n > BAR_WIDTH
    n = min(n, BAR_WIDTH)
    if diff < 0:
        left = ("<" + "#" * (n - 1)) if cut else "#" * n
        return f"{left:>{BAR_WIDTH}}|"
    right = ("#" * (n - 1) + ">") if cut else "#" * n
    return f"{'':>{BAR_WIDTH}}|{right}"


def _team_tags(team):
    """Stack tags of a team, for example '3+2' or '5-STACK'."""
    return "5-STACK" if team.get("fixed") else team["formation"]


def format_report(doc, info=None, paths=None, warnings=None):
    """Render the report for a teams.json document.

    Args:
        doc: Document from teams_output.build_teams_doc.
        info: Optional dict with input counts (players, units, dropped_units,
            fixed, excluded, pool_subs).
        paths: Optional dict {label: path} of written files.
        warnings: Optional list of warning strings.

    Returns:
        The report as one string.
    """
    meta = doc["meta"]
    summary = doc["summary"]
    teams = doc["teams"]
    lines = [_rule(), "VCC TEAM MAKER - TEAMS REPORT", _rule()]
    sources = ", ".join(meta.get("config_sources") or []) or "defaults"
    seed_note = " (generated)" if meta.get("seed_generated") else ""
    lines.append(f"Config: {sources}")
    lines.append(f"Seed: {meta['seed']}{seed_note}   Players file: {meta.get('players_file')}")

    lines += _section("INPUT AND SELECTION")
    if info:
        lines.append(f"Players in file:        {info.get('players', '?')}")
        lines.append(f"Stack units (1-3):      {info.get('units', '?')}")
        lines.append(f"Fixed teams (5-stacks): {info.get('fixed', 0)}")
        lines.append(f"Substitute signups:     {info.get('pool_subs', 0)}")
        lines.append(f"Units left out (fit):   {info.get('dropped_units', 0)}")
        lines.append(f"Excluded players:       {info.get('excluded', 0)}")
    opt_teams = summary["optimized"]["n"]
    lines.append(
        f"Teams: {summary['teams']} ({opt_teams} optimized, "
        f"{summary['teams'] - opt_teams} fixed), {summary['players_in_teams']} players"
    )

    lines += _section("OPTIMIZATION")
    lines.append(
        f"{'run':>3} {'iters':>9} {'range':>6} {'std':>6} {'roles':>6} "
        f"{'NA-t':>5} {'time':>6}  stop"
    )
    run_times = (info or {}).get("run_times") or []
    for i, run in enumerate(meta.get("restarts") or []):
        seconds = run_times[i] if i < len(run_times) else 0.0
        lines.append(
            f"{run['restart']:>3} {run['iterations']:>9} {run['range']:>6.2f} "
            f"{run['std']:>6.3f} {run['role_penalty']:>6.1f} {run['cluster_teams']:>5} "
            f"{seconds:>5.1f}s  {run['stopped_by']}"
        )
    total = (info or {}).get("elapsed_s")
    total_text = "n/a" if total is None else f"{total:.1f}s"
    lines.append(f"Stopped by: {meta['stopped_by']}   Total time: {total_text}")

    lines += _section("BALANCE")
    for label, key in (("Optimized teams", "optimized"), ("All teams", "all")):
        s = summary[key]
        if s["n"]:
            lines.append(
                f"{label:<16} n={s['n']:<3} min {s['min']:.1f}  max {s['max']:.1f}  "
                f"range {s['range']:.1f}  mean {s['mean']:.2f}  std {s['std']:.3f}"
            )
    rp = summary["role_penalty"]
    lines.append(
        f"Role penalty: {rp['total']:.1f} total ({rp['optimized']:.1f} optimized, "
        f"{rp['fixed']:.1f} fixed)"
    )
    cl = summary["cluster"]
    if cl["regions"]:
        cap = "off" if cl["cap"] is None else cl["cap"]
        lines.append(
            f"Teams with {'/'.join(cl['regions'])} players: {cl['teams_all']} "
            f"(optimized {cl['teams_optimized']}, cap {cap})"
        )

    lines += _section("TEAMS")
    all_stats = summary["all"]
    mean = all_stats["mean"] or 0.0
    for team in teams:
        regions = " ".join(f"{r}:{n}" for r, n in team["regions"].items())
        lines.append(
            f"{team['name']:<8} score {team['team_score']:>6.1f} ({team['team_score'] - mean:+.1f})"
            f"  roles {team['role_penalty']:.1f}  [{_team_tags(team)}]  {regions}"
        )
        for p in team["players"]:
            stack = f" s{p['stack_id']}" if p["stack_id"] is not None else ""
            fixed = "*" if p["is_returning"] else " "
            lines.append(
                f"    {p['name'][:22]:<22}{fixed}{p['score']:>5.1f}  "
                f"{p['assigned_role']:<10} {str(p['region']):<5}{stack}"
            )
    if teams:
        lines.append("")
        mean = sum(t["team_score"] for t in teams) / len(teams)
        diffs = [t["team_score"] - mean for t in teams]
        step = bar_step(diffs)
        lines.append(f"Team totals vs the average {mean:.1f}  (# = {step:g} point)")
        for team, diff in zip(teams, diffs):
            lines.append(
                f"  {team['name']:<8} {team['team_score']:>6.1f} {diff:>+5.1f}  "
                f"{diff_bar(diff, step)}".rstrip()
            )
        lines.append("  (* = returning player)")

    subs = doc.get("subs") or []
    lines += _section(f"SUBSTITUTES ({len(subs)})")
    for s in subs:
        flag = "*" if s["is_returning"] else " "
        role = "/".join(s["role"]) if s["role"] else "?"
        lines.append(
            f"  {s['name'][:22]:<22}{flag}{s['score']:>5.1f}  {str(s['region']):<5} "
            f"{role[:24]:<24} {s['reason']}"
        )

    excluded = doc.get("excluded") or []
    if excluded:
        lines += _section(f"EXCLUDED ({len(excluded)})")
        for e in excluded:
            lines.append(f"  {e['name'][:22]:<22} {e['score']:>5.1f}  {e['reason']}")

    if warnings:
        lines += _section("WARNINGS")
        for w in warnings:
            lines.append(f"  - {w}")

    if paths:
        lines += _section("OUTPUT FILES")
        for label, path in paths.items():
            lines.append(f"  {label:<10} {os.path.normpath(path)}")
    lines.append("")
    return "\n".join(lines)
