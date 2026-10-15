"""The published rating list: dated snapshots, changes since the last one, CSV and PNG.

The rulebook uses this list to pick substitutes, so every published version is kept in
output/rating_lists/ and the next one shows what moved since. Pure functions apart from
the file writes; no Streamlit.
"""

import csv
import json
import math
import os
import time

from teamMaker.core import render
from teamMaker.core.config import BASE_DIR

LIST_DIR = os.path.join(BASE_DIR, "output", "rating_lists")
PREFIX = "rating_list_"
ROWS_PER_COLUMN = 35

# Image layout (logical pixels, drawn at render.SCALE).
COL_W = 260
ROW_H = 20
HEAD_H = 24
POS_W = 30
RATING_W = 44
CHANGE_W = 46
RANK_W = 34
ICON = 18
UP_COLOR = (130, 220, 130)
DOWN_COLOR = (240, 120, 110)
NEW_COLOR = (255, 210, 80)
HEAD_COLOR = (150, 150, 150)


def build_rows(scores, players, previous=None):
    """Rows of the list, highest rating first.

    Args:
        scores: {name: rating} from scoring.score_players (already rounded to 1 decimal).
        players: players.json dict, for the current rank.
        previous: An earlier snapshot (load_snapshot) to compare against, or None.

    Returns:
        List of {"player", "rating", "rank", "change", "new"}. ``change`` is the difference
        from the previous list (None without one, or for a player who was not on it);
        ``new`` marks a player who was not on the previous list.
    """
    before = (previous or {}).get("ratings", {})
    rows = []
    for name, rating in scores.items():
        old = before.get(name, {}).get("rating")
        rows.append({
            "player": name,
            "rating": rating,
            "rank": players.get(name, {}).get("current_rank"),
            "change": None if old is None else round(rating - old, 1),
            "new": bool(before) and name not in before,
        })
    rows.sort(key=lambda r: (-r["rating"], r["player"].casefold()))
    return rows


def removed(scores, previous):
    """Names on the previous list that are no longer in the player pool."""
    return sorted((n for n in (previous or {}).get("ratings", {}) if n not in scores), key=str.casefold)


def change_label(row):
    """'+0.4', '-1.2', 'New' or '' for one row."""
    if row["new"]:
        return "New"
    if not row["change"]:
        return ""
    return f"{row['change']:+.1f}"


# --- snapshots --------------------------------------------------------------


def list_snapshots(directory=LIST_DIR):
    """Snapshot file names, newest first."""
    if not os.path.isdir(directory):
        return []
    return sorted((f for f in os.listdir(directory) if f.startswith(PREFIX) and f.endswith(".json")), reverse=True)


def load_snapshot(name, directory=LIST_DIR):
    with open(os.path.join(directory, name), encoding="utf-8") as f:
        return json.load(f)


def latest_snapshot(directory=LIST_DIR):
    """The newest published snapshot, or None."""
    names = list_snapshots(directory)
    return load_snapshot(names[0], directory) if names else None


def publish(rows, directory=LIST_DIR, title="VCC Rating List", now=None):
    """Save rows as a new snapshot plus its CSV and PNG.

    Returns:
        {"json": path, "csv": path, "png": path or None (no Pillow)}.
    """
    now = now or time.localtime()
    stem = os.path.join(directory, PREFIX + time.strftime("%Y%m%d-%H%M%S", now))
    os.makedirs(directory, exist_ok=True)
    snapshot = {
        "published_at": time.strftime("%Y-%m-%dT%H:%M:%S", now),
        "ratings": {r["player"]: {"rating": r["rating"], "rank": r["rank"]} for r in rows},
    }
    tmp = stem + ".json.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False)
    os.replace(tmp, stem + ".json")
    write_csv(rows, stem + ".csv")
    png = stem + ".png"
    dated = f"{title} · {time.strftime('%d %b %Y', now)}"
    return {"json": stem + ".json", "csv": stem + ".csv", "png": png if render_png(rows, png, dated) else None}


def write_csv(rows, path):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        out = csv.writer(f)
        out.writerow(["#", "player", "rating", "rank", "change"])
        for pos, r in enumerate(rows, start=1):
            out.writerow([pos, r["player"], f"{r['rating']:.1f}", r["rank"] or "", change_label(r)])


def render_png(rows, path, title=""):
    """Draw the list in balanced columns of at most ROWS_PER_COLUMN. Returns False when Pillow is missing."""
    if not render.pillow_available():
        return False
    from PIL import Image, ImageDraw

    s = render.SCALE
    per_col = max(1, math.ceil(len(rows) / max(1, math.ceil(len(rows) / ROWS_PER_COLUMN))))
    columns = [rows[i:i + per_col] for i in range(0, len(rows), per_col)] or [[]]
    per_col = max(len(c) for c in columns)
    log_w = len(columns) * COL_W + (len(columns) - 1) * render.COL_GAP + 2 * render.MARGIN
    top = render.TITLE_H if title else render.MARGIN
    log_h = top + HEAD_H + per_col * ROW_H + render.MARGIN
    img = Image.new("RGB", (log_w * s, log_h * s), render.BG_COLOR)
    draw = ImageDraw.Draw(img)
    fonts = {"title": render.load_font(24 * s, bold=True), "row": render.load_font(11 * s),
             "bold": render.load_font(11 * s, bold=True), "head": render.load_font(9 * s, bold=True)}
    if title:
        title = render.fit_text(draw, title, fonts["title"], (log_w - 2 * render.MARGIN) * s)
        draw.text((log_w * s // 2, 14 * s), title, font=fonts["title"], fill=render.WHITE, anchor="mt")

    name_w = COL_W - POS_W - RANK_W - RATING_W - CHANGE_W
    pos = 0
    for c, column in enumerate(columns):
        x0 = render.MARGIN + c * (COL_W + render.COL_GAP)
        y = top
        heads = [("#", POS_W, "l"), ("Player", name_w, "l"), ("", RANK_W, "l"),
                 ("Rating", RATING_W, "r"), ("Change", CHANGE_W, "r")]
        _draw_cells(draw, x0, y, HEAD_H, heads, fonts["head"], HEAD_COLOR)
        y += HEAD_H
        for i, r in enumerate(column):
            pos += 1
            bg = render.ROW_BG if i % 2 == 0 else render.ROW_ALT
            draw.rectangle([x0 * s, y * s, (x0 + COL_W) * s - 1, (y + ROW_H) * s - 1], fill=bg)
            label = change_label(r)
            color = NEW_COLOR if r["new"] else UP_COLOR if (r["change"] or 0) > 0 else DOWN_COLOR
            icon = render.rank_icon(r["rank"], ICON * s)
            cells = [(str(pos), POS_W, "l", HEAD_COLOR, fonts["row"]),
                     (r["player"], name_w, "l", render.WHITE, fonts["row"]),
                     ("" if icon else r["rank"] or "", RANK_W, "l", HEAD_COLOR, fonts["row"]),
                     (f"{r['rating']:.1f}", RATING_W, "r", render.SCORE_COLOR, fonts["bold"]),
                     (label, CHANGE_W, "r", color, fonts["row"])]
            x = x0
            for text, width, align, fill, font in cells:
                _cell(draw, x, y, width, ROW_H, text, font, fill, align)
                x += width
            if icon:
                ix = x0 + POS_W + name_w + (RANK_W - ICON) / 2
                img.paste(icon, (round(ix * s), round((y + (ROW_H - ICON) / 2) * s)), icon)
            y += ROW_H
    render._save(img, path)
    return True


def _draw_cells(draw, x, y, height, heads, font, fill):
    for text, width, align in heads:
        _cell(draw, x, y, width, height, text, font, fill, align)
        x += width


def _cell(draw, x, y, width, height, text, font, fill, align):
    s = render.SCALE
    pad = 6
    text = render.fit_text(draw, text, font, (width - 2 * pad) * s)
    if not text:
        return
    mid = math.floor((y + height / 2) * s)
    if align == "r":
        draw.text(((x + width - pad) * s, mid), text, font=font, fill=fill, anchor="rm")
    else:
        draw.text(((x + pad) * s, mid), text, font=font, fill=fill, anchor="lm")
