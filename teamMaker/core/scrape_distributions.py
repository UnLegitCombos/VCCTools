#!/usr/bin/env python3
"""Build teamMaker/archive/season_distributions.json from the VCC stats pages.

Seasons already in the archive are frozen (never re-scraped) unless
``vcc.refresh_existing`` is true or ``--refresh`` is passed. A refresh never
overwrites a season with fewer values or with data from the truncated HTML
fallback.

Usage:
    python -m teamMaker.core.scrape_distributions [--refresh] [--dry-run]
"""
import argparse
import json
import math
import os
import time

import yaml

from teamMaker.core.utils.vcc import fetch_season_stats

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFERENCE_AR = 1.00
MIN_STD = 0.001


def load_config():
    with open(
        os.path.join(BASE_DIR, "config", "player_ratings_config.yaml"),
        encoding="utf-8",
    ) as f:
        return yaml.safe_load(f)


def summarize(values):
    """Return (count, mean, population std) of a list of ratings."""
    n = len(values)
    if n == 0:
        return 0, 0.0, 0.0
    mean = sum(values) / n
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / n)
    return n, mean, std


def zscore(value, values):
    """Z-score of value within values (0.0 if the spread is degenerate)."""
    _, mean, std = summarize(values)
    return (value - mean) / std if std >= MIN_STD else 0.0


def decide_write(existing, stats, refresh):
    """Decide whether a scraped season may be written to the archive.

    Args:
        existing: Stored ratings list for the season, or None if absent.
        stats: SeasonStats just scraped.
        refresh: True if existing seasons may be replaced.

    Returns:
        Tuple (write, reason).
    """
    if stats.source != "payload":
        return False, "refused: HTML fallback data is truncated"
    if not stats.ar_values:
        return False, "refused: no values scraped"
    if existing is None:
        return True, "new season"
    if not refresh:
        return False, "frozen (already in archive)"
    if len(stats.ar_values) < len(existing):
        return False, (
            f"refused: fewer values than archive "
            f"({len(stats.ar_values)} < {len(existing)})"
        )
    return True, "refresh"


def print_diff(code, existing, new):
    """Print count, mean/std and the z-score of aR 1.00 for old vs new values."""
    n1, m1, s1 = summarize(new)
    if existing is None:
        print(
            f"  {code}: n={n1} mean={m1:.4f} std={s1:.4f} "
            f"z(aR {REFERENCE_AR:.2f})={zscore(REFERENCE_AR, new):+.3f}"
        )
        return
    n0, m0, s0 = summarize(existing)
    changed = sum(1 for a, b in zip(sorted(existing), sorted(new)) if a != b)
    print(
        f"  {code}: n {n0} -> {n1}, mean {m0:.4f} -> {m1:.4f}, "
        f"std {s0:.4f} -> {s1:.4f}, "
        f"z(aR {REFERENCE_AR:.2f}) {zscore(REFERENCE_AR, existing):+.3f} -> "
        f"{zscore(REFERENCE_AR, new):+.3f}, {changed} value(s) differ"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "--refresh", action="store_true", help="re-scrape seasons already archived"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="scrape and diff without writing"
    )
    args = parser.parse_args(argv)

    config = load_config()
    seasons = config.get("seasons", {})
    ua = config["vcc"]["user_agent"]
    vcc_cfg = config.get("vcc", {})
    stage = vcc_cfg.get("stats_stage", "all")
    delay = vcc_cfg.get("request_delay", 1.0)
    decimals = vcc_cfg.get("ar_decimals", 2)
    refresh = args.refresh or bool(vcc_cfg.get("refresh_existing", False))
    output_path = os.path.join(BASE_DIR, "archive", "season_distributions.json")

    if os.path.isfile(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            distributions = json.load(f)
    else:
        distributions = {}
    meta = distributions.get("_meta")
    if not isinstance(meta, dict):
        meta = {}

    changed = False
    fetched_any = False
    for season_key, season_info in seasons.items():
        code = season_info.get("code")
        link = season_info.get("link")
        if not code or not link:
            print(f"Skipping {season_key}: missing code or link")
            continue

        existing = distributions.get(code)
        if not isinstance(existing, list):
            existing = None
        if existing is not None and not refresh:
            print(f"{code}: frozen (already in archive, {len(existing)} values)")
            continue

        if fetched_any:
            time.sleep(delay)
        print(f"Scraping {code} (stage={stage})...")
        stats = fetch_season_stats(code, link, ua, stage=stage, decimals=decimals)
        fetched_any = True
        if stats is None:
            print(f"  {code}: scrape failed, keeping the archive as it is")
            continue

        write, reason = decide_write(existing, stats, refresh)
        print_diff(code, existing, stats.ar_values)
        print(f"  {code}: {reason} (source={stats.source})")
        if not write:
            continue
        if args.dry_run:
            print(f"  {code}: dry run, not written")
            continue
        distributions[code] = stats.ar_values
        meta[code] = {
            "source": stats.source,
            "fetched_at": stats.fetched_at,
            "n": len(stats.ar_values),
            "stage": stats.stage,
        }
        changed = True

    if not changed:
        print("\nNo changes, archive not rewritten.")
        return

    distributions["_meta"] = meta
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(distributions, f, indent=2)
    print(f"\nDistributions written to {output_path}")


if __name__ == "__main__":
    main()
