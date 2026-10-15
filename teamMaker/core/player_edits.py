"""Editing players.json from the app: validation, status changes, adding a player, an edit log
with undo and tracker.gg links. Pure functions, no Streamlit.
"""

import csv
import json
import math
import os
import time
import uuid

from teamMaker.core import roles as role_mod
from teamMaker.core.generate_players import (
    EXCLUDED_STATUSES,
    PING_RANGE,
    TRACKER_SCORE_RANGE,
    apply_previous_seasons,
    build_player_entries,
    find_latest_csv,
    parse_act,
    valid_tracker_url,
)
from teamMaker.core.scoring import canonical_rank

EDIT_COLS = [
    "current_rank", "current_rr", "peak_rank", "peak_rr", "peak_rank_act",
    "tracker_current", "tracker_peak", "ping", "region", "role", "status", "is_returning_player",
]
NUMERIC = {"current_rr", "peak_rr", "tracker_current", "tracker_peak", "ping"}
STATUSES = ("active", "substitute")


def _blank(value):
    return value is None or (isinstance(value, float) and math.isnan(value)) or (
        isinstance(value, str) and not value.strip()
    )


def parse_roles(value):
    """"Duelist, flex" or a list -> a de-duplicated lowercase list."""
    items = value if isinstance(value, (list, tuple)) else str(value or "").replace(";", ",").split(",")
    return list(dict.fromkeys(str(r).strip().lower() for r in items if str(r).strip()))


def clean(value, col):
    """Bring a cell (from the data editor or players.json) to one comparable form."""
    if col == "is_returning_player":
        return bool(value) if not _blank(value) else False
    if col == "role":
        return parse_roles([] if _blank(value) else value)
    if _blank(value):
        return None
    if col in NUMERIC:
        number = float(value)
        return int(number) if number.is_integer() else number
    return str(value).strip()


def display_value(player, col):
    """What the data editor shows for one cell."""
    value = clean(player.get(col), col)
    return ", ".join(value) if isinstance(value, list) else value


def validate_edit(name, new, rank_values, regions=()):
    """Check one player's edited cells; fix their spelling in ``new`` and return the errors."""
    errors = []
    cur = canonical_rank(new["current_rank"], rank_values)
    if not cur:
        errors.append(f"{name}: current_rank '{new['current_rank']}' is not a rank")
    new["current_rank"] = cur or new["current_rank"]
    peak = None
    if new["peak_rank"]:
        peak = canonical_rank(new["peak_rank"], rank_values)
        if not peak:
            errors.append(f"{name}: peak_rank '{new['peak_rank']}' is not a rank")
        new["peak_rank"] = peak or new["peak_rank"]
    if cur and peak and rank_values[peak] < rank_values[cur]:
        errors.append(f"{name}: peak_rank {peak} is below current_rank {cur}")
    if new["peak_rank_act"]:
        act, ok = parse_act(new["peak_rank_act"])
        if ok:
            new["peak_rank_act"] = act
        else:
            errors.append(f"{name}: peak_rank_act '{new['peak_rank_act']}' is not an act like S25A6 or E8A1")
    for col in ("tracker_current", "tracker_peak"):
        low, high = TRACKER_SCORE_RANGE
        if new[col] is not None and not low <= new[col] <= high:
            errors.append(f"{name}: {col} {new[col]} is outside {low}-{high}")
    for col in ("current_rr", "peak_rr"):
        if new[col] is not None and new[col] < 0:
            errors.append(f"{name}: {col} cannot be negative")
    low, high = PING_RANGE
    if new["ping"] is not None and not low <= new["ping"] <= high:
        errors.append(f"{name}: ping {new['ping']} is outside {low}-{high}")
    if new["region"]:
        new["region"] = new["region"].upper()
        if regions and new["region"] not in regions:
            errors.append(f"{name}: region '{new['region']}' is not one of {', '.join(sorted(regions))}")
    elif regions:
        errors.append(f"{name}: region is empty")
    unknown = [r for r in new["role"] if r not in (*role_mod.MAIN_ROLES, role_mod.FLEX)]
    if unknown:
        errors.append(f"{name}: unknown role(s) {', '.join(unknown)}")
    if new["status"] not in STATUSES:
        errors.append(f"{name}: status '{new['status']}' must be active or substitute")
    return errors


def known_regions(players, config):
    """Regions the config or the existing players already use."""
    regions = {str(r).upper() for r in (config.get("region_ping_estimates") or {})}
    regions.update(str(p["region"]).upper() for p in players.values() if p.get("region"))
    return regions


def apply_edits(players, new_values):
    """Return (updated copy of players, warnings) with the edited cells applied.

    Only keys whose value changed are written. A substitute made active gets a group_id of
    their own; a player made substitute takes their whole stack out (as the optimizer does).
    """
    updated = {name: dict(info) for name, info in players.items()}
    warnings = []
    for name, new in new_values.items():
        player = updated[name]
        for col, value in new.items():
            if clean(player.get(col), col) != value:
                player[col] = value
        if clean(players[name].get("status"), "status") != new["status"]:
            _apply_status(updated, name, warnings)
    return updated, warnings


def _apply_status(players, name, warnings):
    player = players[name]
    if player["status"] == "active" and player.get("group_id") is None:
        player["group_id"] = max((p.get("group_id") or 0 for p in players.values()), default=0) + 1
    elif player["status"] == "substitute" and player.get("group_id") is not None:
        partners = [
            other for other, p in players.items()
            if other != name and p.get("group_id") == player["group_id"]
        ]
        if partners:
            warnings.append(
                f"{name} is now a substitute, so their stack partners sit out too: {', '.join(partners)}"
            )


def still_missing(missing, players, rank_values):
    """The entries of players_missing.json that the current players.json has not fixed yet.

    That file is written when the players are built, so it goes stale as ratings are edited.
    Entries for names that are not in players.json are kept as they are.
    """
    out = {}
    for name, info in missing.items():
        player = players.get(name)
        if player is None:
            out[name] = info
            continue
        resolved = set()
        if canonical_rank(player.get("current_rank"), rank_values) and canonical_rank(player.get("peak_rank"), rank_values):
            resolved.add("rank")
        if player.get("tracker_current") is not None:
            resolved.add("tracker")
        kinds = set(info.get("kinds") or [])
        if not kinds or kinds - resolved:
            out[name] = info
    return out


# --- adding one player ------------------------------------------------------


def new_player(row, players, ratings_config, rank_values, regions=(), loader=None):
    """Build one players.json entry from sign-up fields, the way the CSV import does.

    ``row`` uses the sign-up CSV column names (display_name, discord, server, roles,
    returning "TRUE"/"FALSE", vcc_profile, previous_seasons, status, current_rank, ...).
    A returning player's VCC history is loaded with ``loader`` (the real VCC site when None).

    Returns:
        (name, entry, notes): notes are the import's warnings and missing-data reasons.

    Raises:
        ValueError: no name, a name or Discord already in ``players``, or invalid ratings.
    """
    result = build_player_entries([row], ratings_config, rank_values)
    if not result.entries:
        raise ValueError("Enter a display name, a VCC profile link or a Discord name.")
    name, entry = next(iter(result.entries.items()))
    if name.casefold() in {n.casefold() for n in players}:
        raise ValueError(f"{name} is already in the player list.")
    discord = entry["discord"].casefold()
    if discord and any(str(p.get("discord") or "").casefold() == discord for p in players.values()):
        raise ValueError(f"Discord {entry['discord']} already belongs to a player.")
    errors = validate_edit(name, {c: clean(entry.get(c), c) for c in EDIT_COLS}, rank_values, regions)
    if errors:
        raise ValueError("; ".join(errors))
    if entry["is_returning_player"]:
        apply_previous_seasons(result, ratings_config, **({"loader": loader} if loader else {}))
    entry["signup_index"] = max((p.get("signup_index") or 0 for p in players.values()), default=0) + 1
    if entry["status"] == "active":
        entry["group_id"] = max((p.get("group_id") or 0 for p in players.values()), default=0) + 1
    notes = result.warnings + [reason for _, reason in result.issues.get(name, [])]
    return name, entry, notes


# --- edit log and undo ------------------------------------------------------


def _read_log(path):
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_log(path, entries):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def record_changes(path, before, after, note=""):
    """Append one log entry for the differences between two players dicts (nothing if none)."""
    changes = {}
    for name, new in after.items():
        old = before.get(name, {})
        fields = {
            col: {"old": old.get(col), "old_present": col in old, "new": new.get(col), "new_present": col in new}
            for col in sorted(set(old) | set(new))
            if old.get(col) != new.get(col) or (col in old) != (col in new)
        }
        if fields:
            changes[name] = fields
    if not changes:
        return None
    entry = {
        "id": uuid.uuid4().hex[:8],
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "note": note,
        "undone": False,
        "changes": changes,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def recent_changes(path, limit=20):
    """The newest log entries first."""
    return list(reversed(_read_log(path)[-limit:]))


def last_undoable(path):
    entries = [e for e in _read_log(path) if not e["undone"]]
    return entries[-1] if entries else None


def undo_last(path, players):
    """Return (players with the newest save reverted, the entry undone).

    Raises ValueError when there is nothing to undo or a changed field was edited since.
    """
    entries = _read_log(path)
    entry = next((e for e in reversed(entries) if not e["undone"]), None)
    if entry is None:
        raise ValueError("There is no saved edit to undo.")
    restored = {name: dict(info) for name, info in players.items()}
    conflicts = []
    for name, fields in entry["changes"].items():
        if name not in restored:
            conflicts.append(f"{name} is no longer in players.json")
            continue
        for col, change in fields.items():
            present = col in restored[name]
            if present != change["new_present"] or (present and restored[name][col] != change["new"]):
                conflicts.append(f"{name}.{col} changed after that save")
                continue
            if change["old_present"]:
                restored[name][col] = change["old"]
            else:
                restored[name].pop(col, None)
    if conflicts:
        raise ValueError("Cannot undo: " + "; ".join(conflicts))
    for name, fields in entry["changes"].items():
        if not any(change["old_present"] for change in fields.values()):
            del restored[name]  # the save added this player
    for e in entries:
        if e["id"] == entry["id"]:
            e["undone"] = True
    _write_log(path, entries)
    return restored, entry


def describe(entry):
    """One line per player for the 'Recent edits' list."""
    return [
        f"{name}: added" if not any(c["old_present"] for c in fields.values())
        else f"{name}: " + ", ".join(f"{col} {c['old']} → {c['new']}" for col, c in fields.items())
        for name, fields in entry["changes"].items()
    ]


def changes_for(path, name, limit=10):
    """The newest log entries that touched one player, newest first."""
    return [e for e in reversed(_read_log(path)) if name in e["changes"]][:limit]


# --- tracker links ----------------------------------------------


def tracker_links(players, input_dir):
    """{player name: tracker.gg profile link}: the one stored on the player, else the latest
    sign-up CSV's, matched by Discord name. Players without a valid link are left out."""
    links = {}
    missing = {}
    for name, info in players.items():
        url = valid_tracker_url(info.get("tracker_url"))
        if url:
            links[name] = url
        elif info.get("discord"):
            missing[str(info["discord"]).strip().casefold()] = name
    if not missing:
        return links
    try:
        csv_path = find_latest_csv(input_dir)
    except FileNotFoundError:
        return links
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            name = missing.get(str(row.get("discord") or "").strip().casefold())
            status = str(row.get("status") or "").strip().casefold()
            url = valid_tracker_url(row.get("tracker"))
            if name and url and status not in EXCLUDED_STATUSES:
                links[name] = url
    return links
