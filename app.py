"""VCC Tools front panel: players, the published rating list, teams and the season setup.

Run with start_app.bat. The pipeline steps run the normal commands
(python -m teamMaker.core.<command>) and show their output; scores always come from
teamMaker.core.scoring, never from this file.
"""

import contextlib
import copy
import html
import io
import json
import os
import shutil
import subprocess
import sys
import time
import warnings
from collections import Counter, defaultdict

import pandas as pd
import streamlit as st
from ruamel.yaml import YAML

from teamMaker.core import (
    build_teams,
    cpsat,
    player_edits,
    rating_list,
    render,
    scoring,
)
from teamMaker.core import generate_players as gp
from teamMaker.core import roles as role_mod
from teamMaker.core.config import BASE_DIR, load_team_config
from teamMaker.core.generate_players import backup_file
from teamMaker.core.teams_output import _stats

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE_DIR, "data")
INPUT = os.path.join(DATA, "input")
OUT = os.path.join(BASE_DIR, "output")
HISTORY = os.path.join(OUT, "history")
CONFIG_DIR = os.path.join(BASE_DIR, "config")
CONFIG_YAML = os.path.join(CONFIG_DIR, "config.yaml")
CONFIG_EXAMPLE = os.path.join(CONFIG_DIR, "config.example.yaml")
CHANGE_LOG = os.path.join(DATA, "app_change_history.jsonl")
TEAMS_JSON = os.path.join(OUT, "teams.json")
TEAMS_PNG = os.path.join(OUT, "teams.png")
GROUPS_PNG = os.path.join(OUT, "groups.png")
LOGO = os.path.join(BASE_DIR, "assets", "vcc-logo.png")  # vlrcommunitycup.com's own icon
TABLE_HEIGHT = 560  # the long tables scroll inside this; everything else fits its content

st.set_page_config(page_title="VCC Tools", page_icon=LOGO, layout="wide")
st.logo(LOGO, size="large")
st.html("""<style>
  .block-container { padding-top: 2.2rem; padding-bottom: 2rem; }
  h1 { padding-top: 0; }
  table.team { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
  table.team td { padding: 3px 4px; border-top: 1px solid rgba(128, 128, 128, 0.18); vertical-align: middle; }
  table.team tr:first-child td { border-top: none; }
  table.team td.i { width: 26px; }
  table.team td.n { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 0; width: 55%; }
  table.team td.r { opacity: 0.7; white-space: nowrap; }
  table.team td.s { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
</style>""")


# --- helpers ----------------------------------------------------------------


def run_command(module, *args):
    """Run python -m teamMaker.core.<module> and return (ok, output)."""
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    proc = subprocess.run(
        [sys.executable, "-W", "ignore", "-m", f"teamMaker.core.{module}", *args],
        cwd=ROOT, check=False, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env,
    )
    return proc.returncode == 0, (proc.stdout + proc.stderr).strip()


def show_output(ok, output, label="Output"):
    (st.success if ok else st.error)("Done." if ok else "The command failed, see the output below.")
    with st.expander(label, expanded=not ok):
        st.code(output or "(no output)", language=None)


def read_json(path):
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    """Write JSON the way the scripts do, replacing the file only when fully written."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def load_config():
    with contextlib.redirect_stdout(io.StringIO()):
        return load_team_config()


def score_all(players, cfg):
    """Return (scores, breakdowns, ranked) from the real scoring engine.

    ranked is the score with only rank, peak and tracker (no VCC history, ping or
    new-player step), i.e. the baseline that VCC history adjusts.
    """
    scoring._reset_caches()
    ranked_cfg = {
        **cfg, "use_returning_player_stats": False, "use_ping_adjustment": False,
        "use_new_player_debuff": False, "top_tier_compression": {"enabled": False},
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores, breakdowns = scoring.score_players(players, cfg)
        ranked, _ = scoring.score_players(players, ranked_cfg)
    return scores, breakdowns, ranked


def save_players(path, before, after, note):
    """Back up players.json, write the new version, log the change and refresh the score files."""
    backup = backup_file(path)
    write_json(path, after)
    player_edits.record_changes(CHANGE_LOG, before, after, note)
    run_command("scoring")
    return backup


def team_of(doc):
    """{player name: team name, "sub" or "excluded"} from a teams.json document."""
    out = {}
    if doc:
        for team in doc["teams"]:
            for p in team["players"]:
                out[p["name"]] = team["name"]
        out.update({p["name"]: "sub" for p in doc.get("subs", [])})
        out.update({p["name"]: "excluded" for p in doc.get("excluded", [])})
    return out


def tenths(player):
    return round(player["score"] * 10)


def rank_image(rank):
    """The rank icon's link on the valorant-api.com CDN, or None for an unknown rank."""
    return render.rank_icon_url(rank)


def cap(value):
    """'substitute' -> 'Substitute', ['duelist', 'flex'] -> 'Duelist, Flex'; None stays empty."""
    if isinstance(value, (list, tuple)):
        return ", ".join(cap(v) for v in value)
    return str(value)[:1].upper() + str(value)[1:] if value else ""


def table_height(rows):
    """Height that shows every row of a small table without an inner scrollbar."""
    return 38 + 35 * max(rows, 1)


# --- players ----------------------------------------------------------------

EDIT_COLS = player_edits.EDIT_COLS
SAME_AS_CURRENT = "(same as current)"


def page_players():
    cfg = load_config()
    path = build_teams._players_path(cfg)
    players = read_json(path)
    if not players:
        st.info("No players yet. Import the sign-up CSV under Season setup, or add players one by one.")
        if st.button("Add a player", icon=":material/person_add:"):
            add_player_dialog({}, cfg, path)
        return
    scores, bds, ranked = score_all(players, cfg)
    missing = player_edits.still_missing(
        read_json(os.path.join(DATA, "players_missing.json")) or {}, players, cfg.get("rank_values", {}))
    teams = team_of(read_json(TEAMS_JSON))
    links = player_edits.tracker_links(players, INPUT)

    status = Counter(p.get("status", "?") for p in players.values())
    returning = sum(1 for p in players.values() if p.get("is_returning_player"))
    cols = st.columns([1, 1, 1, 1, 1, 1.2], vertical_alignment="center")
    cols[0].metric("Players", len(players))
    cols[1].metric("Active", status.get("active", 0))
    cols[2].metric("Substitutes", status.get("substitute", 0))
    cols[3].metric("Returning / new", f"{returning} / {len(players) - returning}")
    cols[4].metric("Missing ratings", len(missing))
    adding = cols[5].button("Add a player", icon=":material/person_add:", type="primary", width="stretch",
                            help="A late sign-up, without importing the CSV again")
    if adding:
        add_player_dialog(players, cfg, path)

    rows = []
    for name, p in players.items():
        flags = []
        if bds[name]["tracker_components"].get("tracker_missing"):
            flags.append("No tracker score")
        if name in missing:
            flags.append("Missing rating")
        rank = scoring.canonical_rank(p.get("current_rank"), cfg.get("rank_values", {}))
        peak = scoring.canonical_rank(p.get("peak_rank"), cfg.get("rank_values", {})) or rank
        rows.append({
            "player": name, "rank_icon": rank_image(rank), "peak_icon": rank_image(peak), "score": scores[name],
            "ranked": round(ranked[name], 1),
            "tracker": p.get("tracker_current"), "region": p.get("region"), "status": cap(p.get("status")),
            "team": cap(teams.get(name, "—")), "returning": bool(p.get("is_returning_player")),
            "warning": ", ".join(flags), "profile": links.get(name), "rank": rank,
        })

    list_tab, edits_tab, overview_tab = st.tabs(
        ["Players", f"Recent edits ({len(player_edits.recent_changes(CHANGE_LOG, 20))})", "Overview"])
    with list_tab:
        table = filter_players(pd.DataFrame(rows), list(cfg.get("rank_values", {}))).sort_values(
            "score", ascending=False)
        version = st.session_state.setdefault("players_table", 0)
        picked = st.dataframe(
            table, hide_index=True, width="stretch", height=TABLE_HEIGHT, row_height=38, placeholder="—",
            on_select="rerun", selection_mode="single-row", key=f"players_table_{version}",
            column_order=["rank_icon", "peak_icon", "player", "profile", "score", "tracker", "region",
                          "status", "team", "warning"],
            column_config={
                "rank_icon": st.column_config.ImageColumn("Rank", width=56),
                "peak_icon": st.column_config.ImageColumn("Peak", width=56),
                "player": st.column_config.TextColumn("Player", width=150),
                "score": st.column_config.NumberColumn("Score", format="%.1f", width=70),
                "tracker": st.column_config.NumberColumn("Tracker", width=75),
                "region": st.column_config.TextColumn("Region", width=70),
                "status": st.column_config.TextColumn("Status", width=90),
                "team": st.column_config.TextColumn("Team", width=80),
                "warning": st.column_config.TextColumn("Warning", width="medium"),
                "profile": st.column_config.LinkColumn("Tracker.gg", display_text="Open", width=80),
            },
            alt="Players with rank, score, team and warnings; select a row to open the player",
        )
        st.caption(f"{len(table)} of {len(players)} players. Click a row to open the player.")
        chosen = [i for i in (picked.selection.rows if picked else []) if i < len(table)]
        if chosen and not adding:  # Streamlit allows one dialog per run
            player_dialog(table.iloc[chosen[0]]["player"], players, path, cfg, scores, bds, ranked, links)
    with edits_tab:
        show_recent_edits(path, players)
    with overview_tab:
        if missing:
            st.warning("Missing ratings: " + ", ".join(missing))
        left, right = st.columns(2)
        with left:
            st.markdown("**Players per current rank**")
            order = list(cfg.get("rank_values", {}))
            ranks = Counter(r["rank"] or "Missing" for r in rows)
            st.bar_chart(pd.Series({r: ranks[r] for r in [*order, "Missing"] if r in ranks}), sort=False,
                         x_label="", y_label="Players")
        with right:
            st.markdown("**Players per score**")
            bands = Counter(int(s // 5) * 5 for s in scores.values())
            st.bar_chart(pd.Series({f"{b}–{b + 5}": bands[b] for b in sorted(bands)}), sort=False,
                         x_label="", y_label="Players")


def filter_players(table, rank_order):
    c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1, 1], vertical_alignment="bottom")
    search = c1.text_input("Search", placeholder="Player name", label_visibility="collapsed")
    present = set(table["rank"].dropna())
    picks = {
        "rank": c2.selectbox("Rank", ["All ranks", *[r for r in reversed(rank_order) if r in present]],
                             label_visibility="collapsed"),
        "region": c3.selectbox("Region", ["All regions", *sorted(table["region"].dropna().unique())],
                               label_visibility="collapsed"),
        "status": c4.selectbox("Status", ["All statuses", *sorted(table["status"].dropna().unique())],
                               label_visibility="collapsed"),
        "team": c5.selectbox("Team", ["All teams", *sorted(table["team"].unique(), key=str.casefold)],
                             label_visibility="collapsed"),
    }
    c1, c2 = st.columns([1, 3], vertical_alignment="center")
    kind = c1.segmented_control("Players", ["All", "Returning", "New"], default="All", label_visibility="collapsed")
    only_flagged = c2.toggle("Only players with a warning")
    if search:
        table = table[table["player"].str.contains(search, case=False, regex=False)]
    for column, pick in picks.items():
        if not pick.startswith("All "):
            table = table[table[column] == pick]
    if kind in ("Returning", "New"):
        table = table[table["returning"] == (kind == "Returning")]
    if only_flagged:
        table = table[table["warning"] != ""]
    return table


def _close_player():
    st.session_state["players_table"] = st.session_state.get("players_table", 0) + 1  # clears the selection


@st.dialog("Player", width="large", on_dismiss=_close_player)
def player_dialog(name, players, path, cfg, scores, bds, ranked, links):
    """One player's edit form, score breakdown and edit history."""
    player = players[name]
    rank = scoring.canonical_rank(player.get("current_rank"), cfg.get("rank_values", {}))
    head = st.columns([1, 8, 3], vertical_alignment="center")
    if rank_image(rank):
        head[0].image(rank_image(rank), width=48)
    head[1].markdown(f"### {name}")
    head[1].caption(" · ".join(x for x in (rank, cap(player.get("status")), player.get("region")) if x))
    head[2].metric("Score", scores[name])
    if links.get(name):
        st.link_button("Open Tracker.gg profile", links[name], icon=":material/open_in_new:")
    history = player_edits.changes_for(CHANGE_LOG, name)
    edit_tab, why_tab, history_tab = st.tabs(["Edit", "Why this score", f"History ({len(history)})"])
    with edit_tab:
        edit_player(name, players, path, cfg, scores)
    with why_tab:
        show_breakdown(player, bds[name], ranked[name])
    with history_tab:
        if not history:
            st.caption("No edits saved for this player yet.")
        for entry in history:
            line = next(iter(player_edits.describe({"changes": {name: entry["changes"][name]}})))
            st.caption(f"{entry['time'].replace('T', ' ')}{' · undone' if entry['undone'] else ''}: "
                       f"{line.split(': ', 1)[1]}")


def edit_player(name, players, path, cfg, scores):
    player = players[name]
    rank_values = cfg.get("rank_values", {})
    ranks = list(rank_values)
    regions = sorted(player_edits.known_regions(players, cfg))
    old = {c: player_edits.clean(player.get(c), c) for c in EDIT_COLS}
    for col in ("current_rank", "peak_rank"):  # stored spelling may differ ("platinum 3")
        old[col] = scoring.canonical_rank(old[col], rank_values) or old[col]
    key = f"edit_{name}_{st.session_state.setdefault('editor_version', 0)}"

    def pick(label, options, value, col, fmt=str):
        index = options.index(value) if value in options else None
        return st.selectbox(label, options, index=index, key=f"{key}_{col}", format_func=fmt)

    def number(label, value, col, help=None):
        step = 1.0 if isinstance(value, float) else 1  # the widget needs value and step of one type
        return st.number_input(label, value=value, step=step, key=f"{key}_{col}", help=help)

    c1, c2 = st.columns(2)
    new = {"current_rank": None, "peak_rank": None}
    with c1:
        new["current_rank"] = pick("Current rank", ranks, old["current_rank"], "current_rank")
        new["peak_rank"] = pick("Peak rank", [SAME_AS_CURRENT, *ranks], old["peak_rank"] or SAME_AS_CURRENT, "peak_rank")
        new["peak_rank_act"] = st.text_input("Peak act", old["peak_rank_act"] or "", key=f"{key}_act",
                                             help="When the peak was, e.g. S25A6 or E8A1")
        new["tracker_current"] = number("Tracker Score (current act)", old["tracker_current"], "tracker_current")
        new["tracker_peak"] = number("Tracker Score (peak act)", old["tracker_peak"], "tracker_peak")
    with c2:
        new["current_rr"] = number("Current RR", old["current_rr"], "current_rr", "Only used for Immortal 3 and Radiant")
        new["peak_rr"] = number("Peak RR", old["peak_rr"], "peak_rr", "Only used for Immortal 3 and Radiant")
        new["ping"] = number("Ping (ms)", old["ping"], "ping", "Blank uses the estimate for the region")
        new["region"] = pick("Region", regions, old["region"], "region")
        new["status"] = pick("Status", list(player_edits.STATUSES), old["status"], "status", cap)
    role_options = [*role_mod.MAIN_ROLES, role_mod.FLEX]
    new["role"] = st.multiselect("Roles (main first)", role_options, [r for r in old["role"] if r in role_options],
                                 key=f"{key}_role", format_func=cap)
    new["is_returning_player"] = st.checkbox("Returning player", old["is_returning_player"], key=f"{key}_ret")
    if new["peak_rank"] == SAME_AS_CURRENT:
        new["peak_rank"] = None
    new = {c: player_edits.clean(new[c], c) for c in EDIT_COLS}
    if new == old:
        st.caption("Change a field to see the new score before saving. A backup of players.json is made first.")
        return
    errors = player_edits.validate_edit(name, new, rank_values, set(regions))
    for e in errors:
        st.error(e)
    updated, edit_warnings = player_edits.apply_edits(players, {name: new})
    for w in edit_warnings:
        st.warning(w)
    new_score = score_all(updated, cfg)[0][name]
    st.metric("Score", new_score, round(new_score - scores[name], 1))
    st.caption("After the list is published, a change is an individual correction: the next published list shows it.")
    c1, c2 = st.columns(2)
    if c1.button("Save", type="primary", icon=":material/save:", disabled=bool(errors), key=f"{key}_save"):
        backup = save_players(path, players, updated, f"Players page: {name}")
        st.session_state["editor_version"] += 1
        _close_player()
        st.session_state["flash"] = f"Saved {name}. Backup: {os.path.basename(backup)}"
        st.rerun()
    if c2.button("Discard changes", key=f"{key}_reset"):
        st.session_state["editor_version"] += 1
        st.rerun()


@st.dialog("Add a player", width="large")
def add_player_dialog(players, cfg, path):
    ratings_cfg = gp.load_config()
    rank_values = cfg.get("rank_values", {})
    ranks = list(rank_values)
    with st.form("add_player"):
        c1, c2 = st.columns(2)
        display = c1.text_input("Name", help="As it should appear on the lists")
        discord = c2.text_input("Discord")
        server = c1.selectbox("Server", list(ratings_cfg.get("region_map", {})))
        status = c2.selectbox("Status", ["Approved", "Substitute"])
        roles = st.multiselect("Roles (main first)", [*role_mod.MAIN_ROLES, role_mod.FLEX], format_func=cap)
        c1, c2, c3 = st.columns(3)
        current = c1.selectbox("Current rank", ranks, index=None)
        peak = c2.selectbox("Peak rank", ranks, index=None, help="Blank counts as the current rank")
        act = c3.text_input("Peak act", help="e.g. S25A6")
        tracker_cur = c1.number_input("Tracker Score (current act)", value=None, step=1)
        tracker_peak = c2.number_input("Tracker Score (peak act)", value=None, step=1, help="Blank uses the current one")
        ping = c3.number_input("Ping (ms)", value=None, step=1, help="Blank uses the estimate for the server")
        tracker_url = st.text_input("Tracker.gg profile link")
        returning = st.checkbox("Returning player")
        vcc_profile = st.text_input("VCC profile link (returning players)")
        seasons = st.multiselect("Earlier VCC seasons played", list(ratings_cfg.get("season_map", {})))
        submitted = st.form_submit_button("Add", type="primary")
    if not submitted:
        return
    row = {
        "display_name": display, "discord": discord, "server": server, "status": status, "stack": "Solo",
        "roles": ", ".join(roles), "returning": "TRUE" if returning else "FALSE", "vcc_profile": vcc_profile,
        "previous_seasons": ", ".join(seasons), "tracker": tracker_url, "current_rank": current or "",
        "peak_rank": peak or "", "peak_rank_act": act,
        **{col: "" if value is None else str(value) for col, value in
           (("tracker_current", tracker_cur), ("tracker_peak", tracker_peak), ("ping", ping))},
    }
    try:
        with st.spinner("Loading VCC history..." if returning else "Adding..."), \
                contextlib.redirect_stdout(io.StringIO()):
            name, entry, notes = player_edits.new_player(
                row, players, ratings_cfg, rank_values, player_edits.known_regions(players, cfg))
    except ValueError as exc:
        st.error(str(exc))
        return
    updated = {**players, name: entry}
    save_players(path, players, updated, f"Added {name}")
    st.session_state["flash"] = f"Added {name}." + (" Check: " + "; ".join(notes) if notes else "")
    st.rerun()


def show_recent_edits(path, players):
    """The edit log, with a button that reverts the newest save."""
    entries = player_edits.recent_changes(CHANGE_LOG, 20)
    if not entries:
        st.caption("Edits saved in the app are listed here, so they can be checked and undone.")
        return
    last = player_edits.last_undoable(CHANGE_LOG)
    if last and st.button(f"Undo the last save ({last['time'].replace('T', ' ')})", icon=":material/undo:"):
        try:
            restored, entry = player_edits.undo_last(CHANGE_LOG, players)
        except ValueError as exc:
            st.error(str(exc))
            return
        backup = backup_file(path)
        write_json(path, restored)
        run_command("scoring")
        st.session_state["editor_version"] = st.session_state.get("editor_version", 0) + 1
        st.session_state["flash"] = f"Undid the save of {entry['time']}. Backup: {os.path.basename(backup)}"
        st.rerun()
    rows = [{"time": e["time"].replace("T", " "), "player": line.split(": ", 1)[0],
             "change": line.split(": ", 1)[1], "undone": e["undone"]}
            for e in entries for line in player_edits.describe(e)]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=min(TABLE_HEIGHT, table_height(len(rows))),
                 column_config={"time": "Time", "player": "Player", "change": st.column_config.TextColumn("Change", width="large"),
                                "undone": st.column_config.CheckboxColumn("Undone")},
                 alt="The latest saved edits, newest first")


def show_breakdown(player, bd, ranked):
    adv = bd["advanced_components"]
    prev = adv.get("previous_season", {})
    ping = adv.get("ping_adjustment", {})
    debuff = adv.get("new_player_debuff", {})
    track = bd["tracker_components"]
    steps = [("Rank and peak", f"{bd['base_score']:.2f}")]
    if track.get("enabled"):
        steps.append(("Tracker (added, then × consistency factor)",
                      f"+{track['tracker_total']:.2f}, ×{track['consistency_factor']:.2f}"))
    elif track.get("tracker_missing"):
        steps.append(("Tracker", "⚠ missing: skipped, the score is about 6 points lower"))
    else:
        steps.append(("Tracker", "disabled"))
    steps.append(("= Ranked", f"{ranked:.2f}"))
    if prev.get("pre_blend_score") is not None:
        steps.append(("VCC history",
                      (f"{prev['pre_blend_score']:.2f} × {prev['ranked_weight']:.2f} + "
                       f"{prev['previous_season_score']:.2f} × {prev['previous_weight']:.2f} = "
                       f"{prev['post_blend_score']:.2f}")))
    else:
        steps.append(("VCC history", "none" if not player.get("is_returning_player") else "no usable season"))
    steps.append(("Ping", f"×{ping['multiplier']:.2f} ({ping.get('ping_source')}, {ping.get('ping')} ms)"
                  if ping.get("enabled") else "disabled"))
    steps.append(("New-player step", f"×{debuff['debuff_multiplier']:.2f}" if debuff.get("debuff_multiplier") else "not applied"))
    steps.append(("Final score", f"{bd['final_score']:.2f}"))
    st.table(pd.DataFrame(steps, columns=["Step", "Value"]).set_index("Step"), alt="Step by step score calculation")


# --- rating list ------------------------------------------------------------


def page_rating_list():
    cfg = load_config()
    players = read_json(build_teams._players_path(cfg))
    if not players:
        st.info("No players yet.")
        return
    scores = score_all(players, cfg)[0]
    previous = rating_list.latest_snapshot()
    rows = rating_list.build_rows(scores, players, previous)
    gone = rating_list.removed(scores, previous)

    cols = st.columns([1, 1, 1, 1, 1, 1.4], vertical_alignment="center")
    cols[0].metric("Players", len(rows))
    cols[1].metric("Up", sum(1 for r in rows if (r["change"] or 0) > 0))
    cols[2].metric("Down", sum(1 for r in rows if (r["change"] or 0) < 0))
    cols[3].metric("New", sum(r["new"] for r in rows))
    cols[4].metric("Left the pool", len(gone), help=", ".join(gone) or None)
    if cols[5].button("Publish this list", type="primary", icon=":material/publish:", width="stretch",
                      help="Saves it as the new reference and writes a PNG and CSV to post"):
        paths = rating_list.publish(rows)
        st.session_state["flash"] = f"Published. Files: {os.path.relpath(paths['png'] or paths['csv'], ROOT)}"
        st.rerun()
    st.caption(f"Changes are against the list published {previous['published_at'].replace('T', ' ')}."
               if previous else "Nothing published yet: this will be the first list of the season.")

    published = rating_list.list_snapshots()
    current_tab, published_tab = st.tabs(["Current list", f"Published lists ({len(published)})"])
    with current_tab:
        c1, c2 = st.columns([2, 3], vertical_alignment="center")
        search = c1.text_input("Search", placeholder="Player name", label_visibility="collapsed", key="list_search")
        only_changed = c2.toggle("Only players whose rating changed", disabled=not previous)
        rank_values = cfg.get("rank_values", {})
        table = pd.DataFrame([{"pos": i, "rank": rank_image(scoring.canonical_rank(r["rank"], rank_values)),
                               "player": r["player"], "rating": r["rating"], "change": rating_list.change_label(r)}
                              for i, r in enumerate(rows, start=1)])
        if search:
            table = table[table["player"].str.contains(search, case=False, regex=False)]
        if only_changed:
            table = table[table["change"] != ""]
        st.dataframe(table, hide_index=True, width=640, height=TABLE_HEIGHT, row_height=38,
                     column_config={"pos": st.column_config.NumberColumn("#", width=50),
                                    "rank": st.column_config.ImageColumn("Rank", width=60),
                                    "player": st.column_config.TextColumn("Player", width=300),
                                    "rating": st.column_config.NumberColumn("Rating", format="%.1f", width=90),
                                    "change": st.column_config.TextColumn("Change", width=90)},
                     alt="The rating list, highest first, with the change since the last published list")
    with published_tab:
        if not published:
            st.caption("Published lists appear here, with the image to post.")
        else:
            pick = st.selectbox("List", published, label_visibility="collapsed",
                                format_func=lambda f: "Published " + time.strftime(
                                    "%d %b %Y, %H:%M", time.strptime(f[len(rating_list.PREFIX):-5], "%Y%m%d-%H%M%S")))
            png = os.path.join(rating_list.LIST_DIR, pick[:-5] + ".png")
            st.caption(f"Files: {os.path.relpath(rating_list.LIST_DIR, ROOT)}/{pick[:-5]}.png, .csv and .json")
            if os.path.isfile(png):
                st.image(png, alt="The published rating list as an image")


# --- teams ------------------------------------------------------------------


def recompute(doc):
    """Refresh every derived field of a teams document after a swap."""
    for t in doc["teams"]:
        t["players"].sort(key=lambda p: (-p["score"], p["name"]))
        penalty, assigned = role_mod.best_role_assignment([p["role"] for p in t["players"]])
        for p, role in zip(t["players"], assigned):
            p["assigned_role"] = role
        total = sum(tenths(p) for p in t["players"])
        t["team_score"] = total / 10.0
        t["avg_score"] = round(total / 10.0 / len(t["players"]), 2)
        t["role_penalty"] = penalty
        regions = Counter(p.get("region") for p in t["players"])
        t["regions"] = dict(sorted(regions.items(), key=lambda kv: (-kv[1], str(kv[0]))))
        units = Counter(p["stack_id"] if p.get("stack_id") is not None else ("solo", p["name"]) for p in t["players"])
        t["formation"] = "+".join(str(s) for s in sorted(units.values(), reverse=True))
    opt = [t for t in doc["teams"] if not t["fixed"]]
    summary = doc["summary"]
    summary["optimized"] = _stats([sum(tenths(p) for p in t["players"]) for t in opt])
    summary["all"] = _stats([sum(tenths(p) for p in t["players"]) for t in doc["teams"]])
    summary["role_penalty"] = {
        "optimized": sum(t["role_penalty"] for t in opt),
        "fixed": sum(t["role_penalty"] for t in doc["teams"] if t["fixed"]),
        "total": sum(t["role_penalty"] for t in doc["teams"]),
    }


def swap_warnings(doc, original):
    warns = []
    where = defaultdict(set)
    members = defaultdict(list)
    for t in doc["teams"]:
        for p in t["players"]:
            if p.get("stack_id") is not None:
                where[p["stack_id"]].add(t["id"])
                members[p["stack_id"]].append(p["name"])
    for sid, team_ids in where.items():
        if len(team_ids) > 1:
            warns.append(f"Stack split: {', '.join(members[sid])} are on different teams. Stacks should never be split.")
    old_by_team = {t["id"]: t for t in original["teams"]}
    for t in doc["teams"]:
        before = old_by_team[t["id"]]
        if t["fixed"] and {p["name"] for p in t["players"]} != {p["name"] for p in before["players"]}:
            warns.append(f"{t['name']} is a fixed team (a complete 5-stack) and was changed.")
        if t["role_penalty"] > before["role_penalty"]:
            warns.append(f"{t['name']}: role penalty {before['role_penalty']} → {t['role_penalty']} (roles fit worse).")
    return warns


def list_history():
    if not os.path.isdir(HISTORY):
        return []
    return sorted((f for f in os.listdir(HISTORY) if f.endswith(".json")), reverse=True)


def page_teams():
    cfg = load_config()
    doc = read_json(TEAMS_JSON)
    if not doc:
        st.info("No teams yet.")
        build_teams_panel(cfg)
        return
    stamp = os.path.getmtime(TEAMS_JSON)
    if st.session_state.get("teams_stamp") != stamp:
        st.session_state.update(teams_stamp=stamp, edit_doc=copy.deepcopy(doc), orig_doc=copy.deepcopy(doc), undo=[])
    edit, orig = st.session_state["edit_doc"], st.session_state["orig_doc"]
    changed = edit != orig

    meta = edit["meta"]
    s = edit["summary"]["all"]
    cols = st.columns(4)
    cols[0].metric("Teams", s["n"])
    delta = f"{s['range'] - orig['summary']['all']['range']:+.1f} vs generated" if changed else None
    cols[1].metric("Strongest − weakest", s["range"], delta, delta_color="inverse")
    cols[2].metric("Average total", s["mean"])
    cols[3].metric("Role penalty", edit["summary"]["role_penalty"]["total"])
    state = "manually edited, not saved" if changed else "manually edited" if meta.get("manual_edit") else "generated"
    st.caption(f"Generated {meta['generated_at'].replace('T', ' ')} with seed {meta['seed']} · {cap(state)}"
               + (". Manually edited teams are not optimized." if changed or meta.get("manual_edit") else ""))

    tabs = st.tabs(["Teams", "Swap players", f"Subs and excluded ({len(edit['subs']) + len(edit['excluded'])})",
                    "Groups", "Build teams", "Earlier builds"])
    with tabs[0]:
        show_team_cards(edit, cfg)
        if os.path.isfile(TEAMS_PNG):
            with st.expander("The image to publish (teams.png)"):
                st.image(TEAMS_PNG, alt="The generated teams as one image")
    with tabs[1]:
        swap_panel(edit, orig, changed, cfg)
    with tabs[2]:
        subs_and_excluded(edit)
    with tabs[3]:
        if st.button("Make groups", icon=":material/groups:", type="primary"):
            with st.spinner("Making groups..."):
                show_output(*run_command("make_groups"), label="Group output")
        if os.path.isfile(GROUPS_PNG):
            st.image(GROUPS_PNG, alt="The generated groups as one image")
    with tabs[4]:
        build_teams_panel(cfg)
    with tabs[5]:
        earlier_builds(cfg)


def show_team_cards(doc, cfg):
    players = read_json(build_teams._players_path(cfg)) or {}
    rank_values = cfg.get("rank_values", {})
    per_row = 4
    for i in range(0, len(doc["teams"]), per_row):
        for col, team in zip(st.columns(per_row), doc["teams"][i:i + per_row]):
            with col.container(border=True):
                st.markdown(f"**{team['name']}**{' · fixed' if team['fixed'] else ''}")
                st.caption(f"Total {team['team_score']} · average {team['avg_score']}")
                rows = []
                for p in team["players"]:
                    icon = rank_image(scoring.canonical_rank(players.get(p["name"], {}).get("current_rank"),
                                                             rank_values))
                    img = f'<img src="{icon}" alt="" width="22" height="22">' if icon else ""
                    rows.append(f"<tr><td class='i'>{img}</td><td class='n'>{html.escape(p['name'])}</td>"
                                f"<td class='r'>{cap(p['assigned_role'])}</td><td class='s'>{p['score']:.1f}</td></tr>")
                st.html(f"<table class='team'>{''.join(rows)}</table>")


def subs_and_excluded(edit):
    if edit["subs"]:
        st.markdown("**Substitutes**")
        st.dataframe(pd.DataFrame([{"player": p["name"], "score": p["score"], "region": p["region"],
                                    "why": cap(p["reason"])} for p in edit["subs"]]),
                     hide_index=True, width="stretch", height=table_height(len(edit["subs"])),
                     column_config={"player": "Player", "score": st.column_config.NumberColumn("Score", format="%.1f"),
                                    "region": "Region", "why": st.column_config.TextColumn("Why", width="large")},
                     alt="Substitutes with score, region and reason")
    if edit["excluded"]:
        st.markdown("**Excluded**")
        st.dataframe(pd.DataFrame([{"player": p["name"], "why": cap(p["reason"])} for p in edit["excluded"]]),
                     hide_index=True, width="stretch", height=table_height(len(edit["excluded"])),
                     column_config={"player": "Player", "why": st.column_config.TextColumn("Why", width="large")},
                     alt="Excluded players with the reason")
    if not edit["subs"] and not edit["excluded"]:
        st.caption("Every player is on a team.")


def swap_panel(edit, orig, changed, cfg):
    options = {f"{p['name']} ({t['name']}, {p['score']})": (t["id"], p["name"])
               for t in edit["teams"] for p in t["players"]}
    a, b = st.columns(2)
    first = a.selectbox("Player A", list(options))
    others = [label for label, (tid, _) in options.items() if tid != options[first][0]]
    second = b.selectbox("Player B (another team)", others)
    c1, c2, c3 = st.columns(3)
    if c1.button("Swap", icon=":material/swap_horiz:", width="stretch", disabled=not second):
        (ta, na), (tb, nb) = options[first], options[second]
        if ta != tb:
            st.session_state["undo"].append(copy.deepcopy(edit))
            team_a = next(t for t in edit["teams"] if t["id"] == ta)
            team_b = next(t for t in edit["teams"] if t["id"] == tb)
            pa = next(p for p in team_a["players"] if p["name"] == na)
            pb = next(p for p in team_b["players"] if p["name"] == nb)
            team_a["players"].remove(pa)
            team_b["players"].remove(pb)
            team_a["players"].append(pb)
            team_b["players"].append(pa)
            recompute(edit)
            st.rerun()
    if c2.button("Undo last swap", icon=":material/undo:", disabled=not st.session_state["undo"], width="stretch"):
        st.session_state["edit_doc"] = st.session_state["undo"].pop()
        st.rerun()
    if changed:
        for w in swap_warnings(edit, orig):
            st.warning(w)
        if c3.button("Save edited teams", type="primary", width="stretch"):
            backup = backup_file(TEAMS_JSON)
            edit["meta"]["manual_edit"] = True
            edit["meta"]["edited_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            write_json(TEAMS_JSON, edit)
            build_teams._render_png(edit, TEAMS_PNG, cfg)
            st.session_state["flash"] = f"Saved. Backup: {os.path.basename(backup)}. Make the groups again."
            st.rerun()
    else:
        st.caption("Swapped teams show on the Teams tab before you save them.")


def build_teams_panel(cfg):
    c1, c2, c3 = st.columns(3)
    tighter = c1.checkbox("Tighter team totals (slower)", help="--tighter")
    no_limit = c2.checkbox("No time limit (can take tens of minutes)", help="--no-time-limit")
    solvers = ["annealing", "ortools", "both"] if cpsat.available() else ["annealing"]
    solver = c3.selectbox("Solver", solvers, format_func=lambda s: {"ortools": "OR-Tools"}.get(s, cap(s)),
                          help="OR-Tools: pip install -r requirements-ortools.txt")
    st.caption(f"Seed: {cfg.get('random_seed')} (set in Season setup). A normal run stops after "
               f"about {cfg['optimizer']['time_limit_s']} seconds. Building replaces the current teams "
               f"(they stay under Earlier builds).")
    if st.button("Build teams", type="primary", icon=":material/play_arrow:"):
        args = (["--tighter"] if tighter else []) + (["--no-time-limit"] if no_limit else []) \
            + (["--solver", solver] if solver != "annealing" else [])
        with st.spinner("Building teams, this takes a few minutes..."):
            ok, output = run_command("build_teams", *args)
        if ok and os.path.isfile(TEAMS_JSON):
            os.makedirs(HISTORY, exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            shutil.copy2(TEAMS_JSON, os.path.join(HISTORY, f"teams_{stamp}.json"))
            if os.path.isfile(TEAMS_PNG):
                shutil.copy2(TEAMS_PNG, os.path.join(HISTORY, f"teams_{stamp}.png"))
        show_output(ok, output, "Build output")


def earlier_builds(cfg):
    history = list_history()
    if not history:
        st.caption("Every build made here is kept, so you can come back to it.")
        return
    pick = st.selectbox("Build", history, label_visibility="collapsed")
    old = read_json(os.path.join(HISTORY, pick))
    st.write(f"Seed {old['meta']['seed']}, {old['summary']['teams']} teams, "
             f"strongest − weakest {old['summary']['all']['range']}")
    if st.button("Restore as current teams"):
        backup_file(TEAMS_JSON)
        shutil.copy2(os.path.join(HISTORY, pick), TEAMS_JSON)
        build_teams._render_png(old, TEAMS_PNG, cfg)
        st.session_state["flash"] = "Restored. Make the groups again."
        st.rerun()


# --- season setup -----------------------------------------------------------

FIELDS = [  # (label, path, kind, extra)
    ("Random seed (the VCC season number)", ("random_seed",), "int", (0, 10**6)),
    ("Current Valorant season", ("current_season",), "int", (1, 99)),
    ("Current act", ("current_act",), "int", (1, 9)),
    ("VCC history weight, recent returning players", ("recent_data_previous_weight",), "float", (0.0, 1.0)),
    ("VCC history weight, older returning players", ("older_data_previous_weight",), "float", (0.0, 1.0)),
    ("Optimizer time limit (seconds)", ("optimizer", "time_limit_s"), "int", (1, 7200)),
    ("Optimizer target range (points)", ("optimizer", "target_range"), "float", (0.0, 5.0)),
]
GROUP_FIELDS = [
    ("Groups: count", ("groups", "count"), "int", (1, 12)),
    ("Groups: server", ("groups", "server"), "text", None),
    ("Groups: NA server", ("groups", "na_server"), "text", None),
]
SCORING_KEYS = {"recent_data_previous_weight", "older_data_previous_weight"}


def get_path(cfg, path):
    for key in path:
        cfg = (cfg or {}).get(key)
    return cfg


def save_config(changes):
    """Write the changed keys into config.yaml (comments kept); restore it if it stops loading."""
    existed = os.path.isfile(CONFIG_YAML)
    backup = backup_file(CONFIG_YAML) if existed else None
    if not existed:
        shutil.copy2(CONFIG_EXAMPLE, CONFIG_YAML)
    yaml = YAML()
    yaml.preserve_quotes = True
    with open(CONFIG_YAML, encoding="utf-8") as f:
        data = yaml.load(f)
    for path, value in changes.items():
        node = data
        for key in path[:-1]:
            if node.get(key) is None:
                node[key] = {}
            node = node[key]
        node[path[-1]] = value
    tmp = CONFIG_YAML + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.dump(data, f)
    os.replace(tmp, CONFIG_YAML)
    try:
        load_config()
    except Exception as exc:  # never leave a config the scripts cannot read
        if backup:
            shutil.copy2(backup, CONFIG_YAML)
        else:
            os.remove(CONFIG_YAML)
        raise ValueError(f"Rejected, config.yaml was restored: {exc}") from exc
    return backup


def page_setup():
    st.caption("Things done once a season: import the sign-ups, set the season, archive the last VCC season.")
    signups, settings, archive = st.tabs(["Sign-ups", "Settings", "VCC archive"])
    with signups:
        setup_signups()
    with settings:
        setup_settings()
    with archive:
        setup_archive()


def setup_signups():
    csvs = sorted(
        (f for f in os.listdir(INPUT) if f.lower().endswith(".csv")),
        key=lambda f: os.path.getmtime(os.path.join(INPUT, f)),
    ) if os.path.isdir(INPUT) else []
    st.write("Sign-up CSV in use: **" + (csvs[-1] if csvs else "none") + "** (the newest one in `teamMaker/data/input/`)")
    upload = st.file_uploader("Upload the sign-up CSV", type="csv")
    if upload is not None:
        target = os.path.join(INPUT, os.path.basename(upload.name))
        ok_to_save = not os.path.exists(target) or st.checkbox(f"{upload.name} already exists, replace it")
        if ok_to_save and st.button("Save CSV"):
            os.makedirs(INPUT, exist_ok=True)
            with open(target, "wb") as f:
                f.write(upload.getbuffer())
            st.session_state["flash"] = f"Saved {upload.name}."
            st.rerun()
    st.divider()
    c1, c2, c3 = st.columns(3)
    if c1.button("1. Check the CSV", help="generate_players --dry-run: only reports problems"):
        st.session_state["signup_run"] = (*run_command("generate_players", "--dry-run"), "Check")
    if c2.button("2a. Add new sign-ups only", help="generate_players --keep-existing"):
        st.session_state["signup_run"] = (*run_command("generate_players", "--keep-existing"), "Add new")
    with c3:
        sure = st.checkbox("I know this overwrites every player")
        if st.button("2b. Rebuild all players", disabled=not sure,
                     help="generate_players: ratings edited in the app are replaced by the CSV (a backup is kept)"):
            st.session_state["signup_run"] = (*run_command("generate_players"), "Rebuild")
    if "signup_run" in st.session_state:
        ok, output, label = st.session_state["signup_run"]
        show_output(ok, output, f"{label}: console output")


def setup_settings():
    cfg = load_config()
    fields = FIELDS + (GROUP_FIELDS if cfg.get("groups") else [])
    st.caption("Only the settings that change from season to season. Everything else: teamMaker/config/config.example.yaml.")
    values = {}
    for label, path, kind, extra in fields:
        current = get_path(cfg, path)
        key = "set_" + "_".join(path)
        if kind == "int":
            values[path] = int(st.number_input(label, extra[0], extra[1], int(current), 1, key=key))
        elif kind == "float":
            step = 0.1 if extra[1] > 5 else 0.01
            values[path] = float(st.number_input(label, float(extra[0]), float(extra[1]), float(current), step, key=key))
        elif kind == "choice":
            values[path] = st.selectbox(label, extra, index=extra.index(current), key=key,
                                        format_func=lambda v: cap(v.replace("_", " ")))
        else:
            values[path] = st.text_input(label, current, key=key)
    if not cfg.get("groups"):
        st.caption("The groups: block is not enabled in config.yaml (see the README), so its settings are hidden.")
    changes = {p: v for p, v in values.items() if v != get_path(cfg, p)}
    if not changes:
        st.info("No changes.")
        return
    st.write("Changes to save:")
    st.dataframe(pd.DataFrame([{"Setting": ".".join(p), "Old": get_path(cfg, p), "New": v}
                               for p, v in changes.items()]), hide_index=True, width="stretch",
                 height=table_height(len(changes)), alt="Settings changes waiting to be saved")
    if any(p[0] in SCORING_KEYS for p in changes):
        players = read_json(build_teams._players_path(cfg)) or {}
        new_cfg = {**cfg, **{p[0]: v for p, v in changes.items() if len(p) == 1}}
        old_scores, new_scores = score_all(players, cfg)[0], score_all(players, new_cfg)[0]
        moved = [{"Player": n, "Old": old_scores[n], "New": new_scores[n],
                  "Change": round(new_scores[n] - old_scores[n], 1)} for n in players if old_scores[n] != new_scores[n]]
        st.warning("This changes the player scores, and with them the published rating list. "
                   "Do it before the teams are made.")
        if moved:
            st.dataframe(pd.DataFrame(moved).sort_values("Change"), hide_index=True, width="stretch",
                         height=min(TABLE_HEIGHT, table_height(len(moved))), alt="Players whose score changes with these settings")
    if st.button("Save to config.yaml", type="primary", icon=":material/save:"):
        try:
            backup = save_config(changes)
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.session_state["flash"] = "Saved." + (
                f" Backup: {os.path.basename(backup)}" if backup else " Created config.yaml from the example.")
            for _, path, _, _ in fields:
                st.session_state.pop("set_" + "_".join(path), None)
            st.rerun()


def setup_archive():
    st.subheader("Archive last season's VCC ratings")
    st.caption("Register the season in player_ratings_config.yaml first (README: Once per VCC season).")
    if st.button("Preview (dry run)"):
        st.session_state["archive_preview"] = run_command("scrape_distributions", "--dry-run")
    if "archive_preview" in st.session_state:
        show_output(*st.session_state["archive_preview"], label="Dry run")
        if st.session_state["archive_preview"][0] and st.checkbox("The preview looks right") \
                and st.button("Archive now"):
            show_output(*run_command("scrape_distributions"), label="Archive")


# --- app --------------------------------------------------------------------

pages = [
    st.Page(page_players, title="Players", icon=":material/groups:", url_path="players", default=True),
    st.Page(page_rating_list, title="Rating list", icon=":material/format_list_numbered:", url_path="ratings"),
    st.Page(page_teams, title="Teams", icon=":material/sports_esports:", url_path="teams"),
    st.Page(page_setup, title="Season setup", icon=":material/settings:", url_path="setup"),
]
page = st.navigation(pages)
st.title(page.title)
if "flash" in st.session_state:
    st.success(st.session_state.pop("flash"))
page.run()
