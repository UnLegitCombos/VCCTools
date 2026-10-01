"""Build players.json from the hand-filled signup CSV and VCC season stats.

Ranks and tracker.gg scores are typed into the CSV by hand. The CSV-to-entries
logic (build_player_entries) is pure and free of network access; the VCC stage
(apply_previous_seasons), the rank-vs-VCC check and the overrides merge run on
its result.
"""

import argparse
import csv
import glob
import json
import math
import os
import re
import time
from collections import Counter
from dataclasses import dataclass, field

import yaml

from teamMaker.core import scoring
from teamMaker.core.config import DEFAULT_RANK_VALUES, load_team_config
from teamMaker.core.utils import vcc
from teamMaker.core.utils.console import setup_console
from teamMaker.core.utils.helpers import (
    parse_roles,
    parse_vcc_player_id,
    parse_vcc_player_name,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STACK_SIZES = {"solo": 1, "duo": 2, "trio": 3, "5 stack": 5, "5-stack": 5, "5stack": 5}
ALLOWED_STACK_SIZES = (1, 2, 3, 5)
STATUS_MAP = {"approved": "active", "substitute": "substitute"}
OVERRIDE_FIELDS = ("ping", "group_id", "status", "role", "region")
OVERRIDE_STATUSES = ("active", "substitute", "skip")
# tracker.gg Tracker Score range; values outside it are probably typos.
TRACKER_SCORE_RANGE = (0, 1000)
# "S25A6", "E8A1", "V26A4"; spaces and colons are ignored ("E26: A5").
ACT_RE = re.compile(r"^([ESV])(\d+)A(\d+)$")


@dataclass
class BuildResult:
    """Result of the CSV stage.

    Attributes:
        entries: Player name to players.json entry.
        sources: Player name to raw CSV-derived data used by the network stages.
        skipped: Count of skipped rows by CSV status.
        warnings: Human-readable warnings.
        issues: Player name to list of (kind, reason) problems.
        total_rows: Number of CSV rows read.
    """

    entries: dict = field(default_factory=dict)
    sources: dict = field(default_factory=dict)
    skipped: Counter = field(default_factory=Counter)
    warnings: list = field(default_factory=list)
    issues: dict = field(default_factory=dict)
    total_rows: int = 0

    def warn(self, message):
        self.warnings.append(message)
        print(f"  WARNING: {message}")

    def issue(self, name, kind, reason):
        self.issues.setdefault(name, []).append((kind, reason))


def load_config():
    with open(
        os.path.join(BASE_DIR, "config", "player_ratings_config.yaml"), encoding="utf-8"
    ) as f:
        return yaml.safe_load(f)


def find_latest_csv(data_dir):
    files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    return max(files, key=os.path.getmtime)


def parse_stack_label(label):
    """Return the stack size for a CSV stack label, or None if unknown/empty."""
    return STACK_SIZES.get((label or "").strip().lower())


def _base_name(row):
    """Player key: prefer VCC profile slug, then display name, then discord."""
    name = (
        parse_vcc_player_name(row.get("vcc_profile", "").strip())
        or row.get("display_name", "").strip()
        or row.get("discord", "").strip()
    )
    return name[0].upper() + name[1:] if name else ""


def _group_key(row):
    """Grouping key of a row: group_override column, else the Tally submission."""
    override = (row.get("group_override") or "").strip()
    if override:
        return f"override:{override}"
    sid = (row.get("tally_submission_id") or "").strip()
    return f"submission:{sid}" if sid else None


def _validate_groups(result, groups, all_members):
    """Warn about stacks that do not match the stack column or are not allowed."""
    for key, members in groups.items():
        size = len(members)
        labels = {m["stack_label"] for m in members if m["stack_label"]}
        expected = {
            p for p in (parse_stack_label(label) for label in labels) if p is not None
        }
        names = ", ".join(m["name"] for m in members)

        if size not in ALLOWED_STACK_SIZES:
            result.warn(
                f"stack of {size} is not allowed (stacks are 1, 2, 3 or 5): {names}. "
                "These players will be excluded from team building"
            )
        for label in labels:
            if parse_stack_label(label) is None:
                result.warn(f"unknown stack value '{label}' for: {names}")
        for exp in sorted(expected):
            if exp != size:
                others = [
                    f"{n} ({st or 'no status'})"
                    for n, st in all_members.get(key, [])
                    if n not in {m["name"] for m in members}
                ]
                missing = (
                    "; other rows of the submission: " + ", ".join(others)
                    if others
                    else "; no other rows share this submission"
                )
                result.warn(
                    f"stack declared {exp} but {size} approved member(s): {names}"
                    f"{missing}"
                )


def parse_rank(raw, rank_lookup):
    """Normalise a hand-typed rank.

    Args:
        raw: CSV text, e.g. "diamond 2", "Radiant" or "Diamond".
        rank_lookup: Lowercase rank name to canonical name (rank_values keys).

    Returns:
        Tuple of (canonical rank or None, problem text or "").
        A tier without division ("Diamond") becomes the middle division
        ("Diamond 2"), as scoring does, with a problem note.
    """
    text = " ".join((raw or "").split())
    if not text:
        return None, ""
    rank = rank_lookup.get(text.lower())
    if rank:
        return rank, ""
    rank = rank_lookup.get(f"{text.lower()} 2")
    if rank:
        return rank, f"rank '{text}' has no division, using {rank}"
    return None, f"unknown rank '{text}'"


def parse_number(raw):
    """Parse a hand-typed number ("1,234" ok). Returns (value or None, ok)."""
    text = (raw or "").strip().replace(",", "").replace(" ", "")
    if not text:
        return None, True
    try:
        value = float(text)
    except ValueError:
        return None, False
    return (int(value) if value.is_integer() else value), True


def parse_act(raw):
    """Normalise an act like "S25A6" / "E8A1" / "E26: A5". Returns (code or None, ok).

    tracker.gg labels season acts "E26: A5"; E numbers of 25+ are seasons.
    """
    text = re.sub(r"[\s:]", "", (raw or "")).upper()
    if not text:
        return None, True
    match = ACT_RE.match(text)
    if not match:
        return None, False
    prefix, major, act = match.groups()
    if prefix == "E" and int(major) >= 25:
        prefix = "S"
    return f"{prefix}{int(major)}A{int(act)}", True


def apply_csv_ratings(result, name, row, rank_values):
    """Fill the hand-entered rank and tracker fields of one entry.

    Missing values the score depends on become issues (players_missing.json);
    malformed or odd values become warnings.
    """
    entry = result.entries[name]
    rank_lookup = {r.lower(): r for r in rank_values}

    raw_current = row.get("current_rank") or row.get("rank") or ""
    current, problem = parse_rank(raw_current, rank_lookup)
    if not raw_current.strip():
        result.issue(name, "rank", "no current rank")
    elif current is None:
        result.issue(name, "rank", f"{problem} (current_rank)")
    elif problem:
        result.warn(f"{name}: {problem} (current_rank)")
    entry["current_rank"] = current

    rr, ok = parse_number(row.get("current_rr"))
    if not ok:
        result.warn(f"{name}: current_rr '{row.get('current_rr')}' is not a number, ignored")
    entry["current_rr"] = rr

    raw_peak = row.get("peak_rank") or ""
    peak, problem = parse_rank(raw_peak, rank_lookup)
    if not raw_peak.strip():
        result.issue(name, "rank", "no peak rank (scored as the current rank)")
    elif peak is None:
        result.issue(name, "rank", f"{problem} (peak_rank, scored as the current rank)")
    elif problem:
        result.warn(f"{name}: {problem} (peak_rank)")
    if current and peak and rank_values[peak] < rank_values[current]:
        result.warn(f"{name}: peak rank {peak} is below current rank {current}")
    entry["peak_rank"] = peak

    act, ok = parse_act(row.get("peak_rank_act"))
    if not ok:
        result.warn(
            f"{name}: peak_rank_act '{row.get('peak_rank_act')}' not understood "
            "(use e.g. S25A6 or E8A1), ignored"
        )
    entry["peak_rank_act"] = act

    for col in ("tracker_current", "tracker_peak"):
        value, ok = parse_number(row.get(col))
        if not ok:
            result.warn(f"{name}: {col} '{row.get(col)}' is not a number, ignored")
        elif value is not None and not (
            TRACKER_SCORE_RANGE[0] <= value <= TRACKER_SCORE_RANGE[1]
        ):
            result.warn(
                f"{name}: {col} {value} is outside {TRACKER_SCORE_RANGE[0]}-"
                f"{TRACKER_SCORE_RANGE[1]}, check for a typo"
            )
        entry[col] = value
    if entry["tracker_current"] is None:
        result.issue(name, "tracker", "no tracker score (tracker part of the score skipped)")
    elif entry["tracker_peak"] is None:
        result.issue(name, "tracker", "no peak tracker score (using the current one)")
        entry["tracker_peak"] = entry["tracker_current"]


def build_player_entries(rows, config, rank_values=None):
    """Turn CSV rows into players.json entries (no network access).

    Approved rows become active players, substitute rows become subs with
    group_id null, every other status is skipped and counted.

    Args:
        rows: CSV rows as dicts (csv.DictReader output).
        config: player_ratings_config dict (uses region_map).
        rank_values: Rank name to value mapping used to validate ranks
            (defaults to DEFAULT_RANK_VALUES).

    Returns:
        BuildResult with entries, sources, skipped counts and warnings.
    """
    result = BuildResult(total_rows=len(rows))
    region_map = config.get("region_map", {})
    rank_values = rank_values or DEFAULT_RANK_VALUES

    # Every row per submission, for naming missing stack members.
    all_members = {}
    for row in rows:
        key = _group_key(row)
        if key:
            all_members.setdefault(key, []).append(
                (_base_name(row) or "?", row.get("status", "").strip())
            )

    kept = []
    for index, row in enumerate(rows, start=1):
        status_raw = row.get("status", "").strip()
        status = STATUS_MAP.get(status_raw.lower())
        if status is None:
            result.skipped[status_raw or "(blank)"] += 1
            continue
        name = _base_name(row)
        if not name:
            result.warn(f"row {index} has no name, skipped")
            result.skipped["(no name)"] += 1
            continue
        kept.append({"row": row, "index": index, "name": name, "status": status})

    # Name collisions -> "Name (discord)"
    counts = Counter(k["name"].lower() for k in kept)
    for k in kept:
        if counts[k["name"].lower()] > 1:
            discord = k["row"].get("discord", "").strip() or f"row {k['index']}"
            unique = f"{k['name']} ({discord})"
            result.warn(f"name collision on '{k['name']}', using '{unique}'")
            k["name"] = unique

    # group ids: ints in first-appearance order over approved rows
    group_ids = {}
    groups = {}
    for k in kept:
        row = k["row"]
        k["stack_label"] = row.get("stack", "").strip()
        if k["status"] != "active":
            k["group_id"] = None
            continue
        key = _group_key(row) or f"solo:{k['index']}"
        if key not in group_ids:
            group_ids[key] = len(group_ids) + 1
        k["group_id"] = group_ids[key]
        groups.setdefault(key, []).append(k)
    _validate_groups(result, groups, all_members)

    for k in kept:
        row = k["row"]
        if k["status"] == "substitute" and k["stack_label"].lower() not in ("", "solo"):
            result.warn(
                f"substitute {k['name']} declared '{k['stack_label']}'; "
                "treated as a solo sub"
            )
        server = row.get("server", "").strip()
        if server and server not in region_map:
            result.warn(f"unknown server '{server}' for {k['name']}, using EU")
        entry = {
            "current_rank": None,
            "current_rr": None,
            "peak_rank": None,
            "peak_rank_act": None,
            "group_id": k["group_id"],
            "tracker_peak": None,
            "tracker_current": None,
            "role": parse_roles(row.get("roles", "").strip()),
            "region": region_map.get(server, "EU"),
            "is_returning_player": row.get("returning", "").strip().upper() == "TRUE",
            "discord": row.get("discord", "").strip(),
            "server": server,
            "stack": k["stack_label"],
            "submission_id": row.get("tally_submission_id", "").strip(),
            "signup_index": k["index"],
            "status": k["status"],
        }
        result.entries[k["name"]] = entry
        result.sources[k["name"]] = {
            "vcc_profile": row.get("vcc_profile", "").strip(),
            "previous_seasons": row.get("previous_seasons", "").strip(),
        }
        apply_csv_ratings(result, k["name"], row, rank_values)
    return result


def apply_previous_seasons(result, config, loader=vcc.load_season_stats, sleep=time.sleep):
    """Attach previous VCC season adjusted ratings to returning players.

    Each season page is fetched once (via the loader's cache) and looked up
    per player, so players ranked below the HTML table cut-off are found too.

    Args:
        result: BuildResult.
        config: player_ratings_config dict.
        loader: Season loader (code, link, ua, stage=, decimals=) -> SeasonStats.
        sleep: Sleep function between season fetches.
    """
    season_map = config.get("season_map", {})
    seasons = config.get("seasons", {})
    vcc_cfg = config.get("vcc", {})
    stage = vcc_cfg.get("stats_stage", "all")
    decimals = vcc_cfg.get("ar_decimals", 2)
    delay = vcc_cfg.get("request_delay", 1.0)
    ua = vcc_cfg["user_agent"]
    min_rounds = config.get("min_rounds", 0)

    # Work out which seasons each returning player needs (unknown names warn once).
    needed = {}
    unknown = set()
    for name, entry in result.entries.items():
        src = result.sources[name]
        if not (entry["is_returning_player"] and src["previous_seasons"]):
            continue
        plan = []
        for display in src["previous_seasons"].split(","):
            display = display.strip()
            if not display:
                continue
            key = season_map.get(display)
            if not key or key not in seasons:
                unknown.add(display)
                continue
            info = seasons[key]
            code = info.get("code", key.replace("_", ""))
            if info.get("link"):
                plan.append((code, info["link"]))
        needed[name] = plan
    for display in sorted(unknown):
        result.warn(f"unknown previous season '{display}' (not in season_map), ignored")

    loaded = {}
    for plan in needed.values():
        for code, link in plan:
            if (code, link) in loaded:
                continue
            if loaded:
                sleep(delay)
            print(f"Loading VCC {code} stats...")
            stats = loader(code, link, ua, stage=stage, decimals=decimals)
            loaded[(code, link)] = stats
            if stats is None:
                result.warn(f"could not load VCC stats for {code}")
            elif stats.truncated:
                result.warn(
                    f"VCC {code} stats are truncated (HTML fallback); "
                    "lower-ranked players may be missing"
                )

    for name, plan in needed.items():
        entry = result.entries[name]
        player_id = parse_vcc_player_id(result.sources[name]["vcc_profile"])
        if not player_id:
            result.issue(name, "vcc", "returning player without a valid VCC profile link")
            continue
        prev = []
        for code, link in plan:
            stats = loaded[(code, link)]
            if stats is None:
                result.issue(name, "vcc", f"VCC {code} stats unavailable")
                continue
            ar, rounds = vcc.lookup_player(stats, player_id)
            if ar is None:
                result.issue(name, "vcc", f"not found in VCC {code} stats")
            elif rounds is None or rounds >= min_rounds:
                prev.append({"season": code, "adjusted_rating": ar})
            else:
                print(f"  Skipping {code} for {name}: {rounds} rounds < min_rounds ({min_rounds})")
        if prev:
            entry["previous_season_stats"] = prev


def check_vcc_consistency(result, team_config, threshold=1.5):
    """Warn when a hand-entered rank and the player's VCC history disagree.

    The rank is placed against this signup pool (standard deviations from the
    mean rank value); VCC history uses scoring's previous-season score, which
    maps aR to 18 + 8 per standard deviation of its season. A gap of at least
    ``threshold`` deviations usually means a typo in the CSV.

    Args:
        result: BuildResult after apply_previous_seasons.
        team_config: Team maker config (rank_values, distributions, weights).
        threshold: Minimum gap in standard deviations to warn about.

    Returns:
        List of (name, rank z, VCC z) for the flagged players.
    """
    rank_values = team_config.get("rank_values") or DEFAULT_RANK_VALUES
    values = {
        name: rank_values.get(e.get("current_rank"), 0)
        for name, e in result.entries.items()
    }
    pool = [v for v in values.values() if v > 0]
    if len(pool) < 5:
        return []
    mean = sum(pool) / len(pool)
    std = math.sqrt(sum((v - mean) ** 2 for v in pool) / len(pool))
    if std <= 0:
        return []

    flagged = []
    for name, entry in result.entries.items():
        if values[name] <= 0 or not entry.get("previous_season_stats"):
            continue
        vcc_score = scoring.calculate_previous_season_score(entry, team_config)
        if not vcc_score:
            continue
        rank_z = (values[name] - mean) / std
        vcc_z = (vcc_score - 18.0) / 8.0
        if abs(rank_z - vcc_z) >= threshold:
            direction = "higher" if rank_z > vcc_z else "lower"
            result.warn(
                f"{name}: rank {entry['current_rank']} ({rank_z:+.1f} sd vs signups) is "
                f"much {direction} than their VCC history ({vcc_z:+.1f} sd); "
                "check current_rank in the CSV"
            )
            flagged.append((name, rank_z, vcc_z))
    return flagged


def load_overrides(path):
    """Load overrides.yaml (empty dict if the file does not exist)."""
    if not path or not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping of player -> fields")
    return data


def apply_overrides(result, overrides):
    """Merge per-player overrides into the entries.

    Overrides match the player name or discord handle case-insensitively.
    Supported fields: ping, group_id, status (active/substitute/skip), role,
    region. Unknown fields and unmatched players produce warnings.

    Args:
        result: BuildResult.
        overrides: Mapping of player name or discord to a dict of fields.
    """
    lookup = {}
    for name, entry in result.entries.items():
        lookup[name.lower()] = name
        if entry.get("discord"):
            lookup.setdefault(entry["discord"].lower(), name)

    for key, fields in overrides.items():
        name = lookup.get(str(key).lower())
        if name is None:
            result.warn(f"override for '{key}' matches no player")
            continue
        if not isinstance(fields, dict):
            result.warn(f"override for '{key}' must be a mapping, ignored")
            continue
        entry = result.entries[name]
        for fname, value in fields.items():
            if fname not in OVERRIDE_FIELDS:
                result.warn(f"override for '{key}': unknown field '{fname}' ignored")
            elif fname == "status":
                if value not in OVERRIDE_STATUSES:
                    result.warn(f"override for '{key}': bad status '{value}' ignored")
                elif value == "skip":
                    del result.entries[name]
                    result.sources.pop(name, None)
                    result.issues.pop(name, None)
                    result.skipped["override skip"] += 1
                    break
                else:
                    entry["status"] = value
                    if value == "substitute":
                        entry["group_id"] = None
            elif fname == "role":
                entry["role"] = parse_roles(value) if isinstance(value, str) else [
                    str(r).strip().lower() for r in value
                ]
            else:
                entry[fname] = value
        print(f"  override applied: {name}")


def build_missing(result):
    """Return the players_missing.json content: real reasons per player."""
    missing = {}
    for name, problems in result.issues.items():
        if name not in result.entries:
            continue
        missing[name] = {
            "reason": "; ".join(reason for _, reason in problems),
            "kinds": sorted({kind for kind, _ in problems}),
            "discord": result.entries[name].get("discord", ""),
        }
    return missing


def group_size_counts(entries):
    """Count active players' stack sizes by group_id: {size: number of groups}."""
    sizes = Counter(
        e["group_id"] for e in entries.values() if e["group_id"] is not None
    )
    return Counter(sizes.values())


def print_summary(result, missing=None):
    """Print the end-of-run summary."""
    entries = result.entries
    active = sum(1 for e in entries.values() if e["status"] == "active")
    subs = sum(1 for e in entries.values() if e["status"] == "substitute")
    print("\n=== Summary ===")
    print(f"CSV rows: {result.total_rows}")
    print(f"Players: {active} active, {subs} substitute")
    if result.skipped:
        skipped = ", ".join(f"{k}: {v}" for k, v in sorted(result.skipped.items()))
        print(f"Skipped rows: {skipped}")
    sizes = group_size_counts(entries)
    names = {1: "solo", 2: "duo", 3: "trio", 4: "4-stack", 5: "5-stack"}
    print(
        "Stacks: "
        + ", ".join(f"{names.get(s, str(s))} {n}" for s, n in sorted(sizes.items()))
    )
    if missing:
        print(f"Players with missing data: {len(missing)}")
        for name, info in missing.items():
            print(f"  - {name}: {info['reason']}")
    print(f"Warnings: {len(result.warnings)}")


def main(argv=None):
    setup_console()
    parser = argparse.ArgumentParser(description="Generate players.json from the signup CSV")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="parse and check the CSV only: no VCC requests and nothing written",
    )
    args = parser.parse_args(argv)

    config = load_config()
    team_config = load_team_config()
    data_dir = os.path.join(BASE_DIR, "data", config.get("data_dir", "input"))
    output_path = os.path.join(BASE_DIR, "data", config.get("output", "players.json"))
    missing_path = output_path.replace(".json", "_missing.json")
    overrides_path = os.path.join(BASE_DIR, "data", "overrides.yaml")

    csv_path = find_latest_csv(data_dir)
    print(f"CSV: {csv_path}\n")
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    result = build_player_entries(rows, config, team_config.get("rank_values"))
    if not args.dry_run:
        apply_previous_seasons(result, config)
        check_vcc_consistency(
            result, team_config, config.get("vcc_check_threshold", 1.5)
        )
    apply_overrides(result, load_overrides(overrides_path))
    missing = build_missing(result)

    if not args.dry_run:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result.entries, f, indent=2, ensure_ascii=False)
        if missing:
            with open(missing_path, "w", encoding="utf-8") as f:
                json.dump(missing, f, indent=2, ensure_ascii=False)
        elif os.path.exists(missing_path):
            os.remove(missing_path)
        print(f"\nDone. Output: {output_path}")
        if missing:
            print(f"Missing data for {len(missing)} player(s): {missing_path}")
    print_summary(result, missing)


if __name__ == "__main__":
    main()
