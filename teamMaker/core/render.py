"""PNG rendering of team and group sheets (Pillow).

Layout follows the original groups renderer: everything is laid out in logical
pixels and drawn at ``SCALE`` times that size. Each team is a 310 px card with
a coloured header (name, optional 5-STACK badge, team score) and one row per
player (name, role icons, discord, score, region tag).

Pillow is optional: :func:`pillow_available` reports whether it can be used and
the ``render_*`` functions return ``False`` without it.
"""

import colorsys
import os
from typing import Any, cast

from teamMaker.core.config import BASE_DIR

try:
    from PIL import Image, ImageDraw, ImageFont

    _PIL_OK = True
except ImportError:  # pragma: no cover - depends on the environment
    Image = ImageDraw = ImageFont = cast(Any, None)
    _PIL_OK = False

ROLE_DIR = os.path.join(BASE_DIR, "assets", "roles")

SCALE = 2
BORDER_W = 2
SCORE_W = 26
SRV_W = 20
TEAM_W = 310
HEADER_H = 28
ROW_H = 20
BOX_PAD = 4
COL_GAP = 8
ROW_GAP = 14
GP = 10
GLABEL_H = 26
INNER_ROW = 8
TITLE_H = 52
MARGIN = 18
NAME_MAX_W = 112
MIN_DISCORD_W = 24

BG_COLOR = (0, 0, 0)
ROW_BG = (20, 20, 20)
ROW_ALT = (32, 32, 32)
WHITE = (255, 255, 255)
GROUP_BG = (22, 22, 22)
GROUP_OUTLINE = (55, 55, 55)
SCORE_COLOR = (140, 215, 140)
TEAM_SCORE_COLOR = (255, 255, 190)
DISCORD_COLOR = (150, 150, 150)
ICON_ALPHA_DIM = 0.5

# Medium-dark colours so white header text stays readable.
TEAM_PALETTE = [
    (196, 48, 48),
    (0, 140, 170),
    (140, 50, 180),
    (40, 140, 60),
    (170, 130, 20),
    (200, 90, 30),
    (50, 90, 200),
    (200, 50, 130),
    (30, 140, 100),
    (150, 100, 30),
    (70, 130, 160),
    (150, 50, 60),
    (100, 130, 30),
    (120, 90, 190),
    (180, 90, 140),
    (40, 110, 130),
    (170, 70, 90),
    (80, 90, 170),
    (60, 130, 130),
    (140, 100, 60),
    (110, 70, 150),
    (170, 110, 20),
    (90, 120, 80),
    (160, 60, 160),
]

STACK_COLORS = [
    (255, 210, 50),
    (100, 210, 255),
    (130, 255, 110),
    (255, 110, 190),
    (190, 130, 255),
    (255, 160, 70),
]

REGION_TAGS = {"EU": "EU", "NA": "NA", "MENA": "ME", "ME": "ME"}
REGION_COLORS = {
    "EU": (130, 180, 255),
    "NA": (100, 220, 140),
    "ME": (255, 180, 80),
}

_FONT_REGULAR = [
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
]
_FONT_BOLD = [
    "C:/Windows/Fonts/segoeuib.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
]

_ICON_CACHE = {}


def pillow_available():
    """Return True if Pillow could be imported."""
    return _PIL_OK


def load_font(size, bold=False):
    """Load a TrueType font at a physical pixel size.

    Search order: Windows (Segoe UI, Arial), DejaVu, macOS Arial, then
    Pillow's built-in font.

    Args:
        size: Font size in physical pixels.
        bold: Prefer a bold face.

    Returns:
        A Pillow font object.
    """
    for path in _FONT_BOLD if bold else _FONT_REGULAR:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default(size=size)


def text_width(draw, text, font):
    """Return the rendered width of text in physical pixels."""
    return int(round(draw.textlength(text, font=font)))


def fit_text(draw, text, font, max_width):
    """Clip text to max_width pixels, ending with an ellipsis when cut.

    Args:
        draw: ImageDraw instance used for measuring.
        text: Text to fit.
        font: Font used to render the text.
        max_width: Available width in pixels.

    Returns:
        The original text if it fits, else a shortened text ending in "...",
        or an empty string if not even the ellipsis fits.
    """
    if not text or max_width <= 0:
        return ""
    if text_width(draw, text, font) <= max_width:
        return text
    ellipsis = "..."
    if text_width(draw, ellipsis, font) > max_width:
        return ""
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if text_width(draw, text[:mid].rstrip() + ellipsis, font) <= max_width:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo].rstrip() + ellipsis


def team_color(index):
    """Return the header colour of the team at a zero-based index.

    The first 24 come from a hand-picked palette; further ones step the hue by
    the golden ratio so colours never repeat.
    """
    if index < len(TEAM_PALETTE):
        return TEAM_PALETTE[index]
    hue = (0.11 + 0.618033988749895 * (index - len(TEAM_PALETTE) + 1)) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.6, 0.68)
    return (int(r * 255), int(g * 255), int(b * 255))


def _role_icon(role, size_px, alpha):
    """Load a role icon from assets/roles, cached, or None if unavailable."""
    key = (role, size_px, alpha)
    if key in _ICON_CACHE:
        return _ICON_CACHE[key]
    icon = None
    path = os.path.join(ROLE_DIR, f"{role}.webp")
    if os.path.exists(path):
        try:
            icon = Image.open(path).convert("RGBA").resize((size_px, size_px), Image.Resampling.LANCZOS)
            r, g, b, a = icon.split()
            a = a.point(lambda v: int(v * alpha))
            icon = Image.merge("RGBA", (r, g, b, a))
        except (OSError, ValueError):
            icon = None
    _ICON_CACHE[key] = icon
    return icon


def _fonts():
    """Return the font set used by every sheet."""
    return {
        "small": load_font(10 * SCALE),
        "tiny": load_font(9 * SCALE),
        "hdr": load_font(12 * SCALE, bold=True),
        "badge": load_font(8 * SCALE, bold=True),
        "glbl": load_font(14 * SCALE, bold=True),
        "title": load_font(28 * SCALE, bold=True),
    }


def team_card_height(team):
    """Return the logical height of a team card."""
    return HEADER_H + len(team.get("players", [])) * ROW_H + BOX_PAD


def render_team_card(img, draw, x, y, team, fonts, color=None):
    """Draw one team card with its top-left corner at logical (x, y).

    The card is always exactly TEAM_W wide: names and discords are clipped
    with an ellipsis instead of growing the card.

    Args:
        img: Target RGB image (used for pasting role icons).
        draw: ImageDraw of img.
        x: Left edge in logical pixels.
        y: Top edge in logical pixels.
        team: A team entry of teams.json.
        fonts: Font dict from _fonts().
        color: Header colour, default derived from the team id.

    Returns:
        The logical height of the card.
    """
    R = SCALE
    if color is None:
        color = team_color(int(team.get("id", 1)) - 1)
    players = team.get("players", [])
    f_hdr, f_small, f_tiny = fonts["hdr"], fonts["small"], fonts["tiny"]

    draw.rectangle([x * R, y * R, (x + TEAM_W) * R - 1, (y + HEADER_H) * R - 1], fill=color)
    mid = y + HEADER_H / 2
    score_str = f"{team.get('team_score', 0):.1f}"
    sw = text_width(draw, score_str, f_hdr)
    score_left = (x + TEAM_W) * R - sw - 6 * R
    draw.text((score_left, mid * R), score_str, font=f_hdr, fill=TEAM_SCORE_COLOR, anchor="lm")

    right_limit = score_left - 6 * R
    is_five = bool(team.get("fixed")) or any(p.get("formation") == "5-stack" for p in players)
    if is_five:
        f_badge = fonts["badge"]
        label = "5-STACK"
        bw = text_width(draw, label, f_badge) + 8 * R
        bx1 = right_limit
        bx0 = bx1 - bw
        draw.rounded_rectangle(
            [bx0, (mid - 7) * R, bx1, (mid + 7) * R],
            radius=3 * R,
            fill=(255, 210, 50),
        )
        draw.text(((bx0 + bx1) / 2, mid * R), label, font=f_badge, fill=(30, 30, 30), anchor="mm")
        right_limit = bx0 - 6 * R
    name = fit_text(draw, team.get("name", ""), f_hdr, right_limit - (x + BORDER_W + 5) * R)
    draw.text(((x + BORDER_W + 5) * R, mid * R), name, font=f_hdr, fill=WHITE, anchor="lm")

    stack_colors = {}
    for p in players:
        sid = p.get("stack_id")
        if p.get("formation", "solo") != "solo" and sid is not None and sid not in stack_colors:
            stack_colors[sid] = STACK_COLORS[len(stack_colors) % len(STACK_COLORS)]

    icon_size = (ROW_H - 8) * R
    for i, p in enumerate(players):
        py = y + HEADER_H + i * ROW_H
        pmid = (py + ROW_H / 2) * R
        draw.rectangle(
            [x * R, py * R, (x + TEAM_W) * R - 1, (py + ROW_H) * R - R - 1],
            fill=ROW_BG if i % 2 == 0 else ROW_ALT,
        )
        sid = p.get("stack_id")
        if sid in stack_colors and p.get("formation", "solo") != "solo":
            draw.rectangle(
                [x * R, py * R, (x + BORDER_W) * R - 1, (py + ROW_H) * R - R - 1],
                fill=stack_colors[sid],
            )

        # Right side: region tag, then score
        tag = REGION_TAGS.get(str(p.get("region") or "").upper(), str(p.get("region") or "")[:2].upper())
        tag_w = text_width(draw, tag, f_tiny)
        draw.text(
            ((x + TEAM_W) * R - tag_w - 3 * R, pmid),
            tag,
            font=f_tiny,
            fill=REGION_COLORS.get(tag, (180, 180, 180)),
            anchor="lm",
        )
        score_right = (x + TEAM_W - SRV_W) * R - 4 * R
        left_limit = score_right
        if p.get("score") is not None:
            ps = f"{p['score']:.1f}"
            psw = text_width(draw, ps, f_tiny)
            draw.text((score_right - psw, pmid), ps, font=f_tiny, fill=SCORE_COLOR, anchor="lm")
            left_limit = score_right - max(psw, (SCORE_W - 4) * R) - 2 * R

        # Left side: name (clipped), role icons
        roles = p.get("role") or []
        if isinstance(roles, str):
            roles = [roles]
        assigned = p.get("assigned_role")
        icon_roles = [r for r in roles if os.path.exists(os.path.join(ROLE_DIR, f"{r}.webp"))]
        if assigned and assigned not in icon_roles and os.path.exists(
            os.path.join(ROLE_DIR, f"{assigned}.webp")
        ):
            icon_roles.append(assigned)
        icons_w = len(icon_roles) * (icon_size + R) if icon_roles else 0

        name_x = (x + BORDER_W + 3) * R
        name_max = min(NAME_MAX_W * R, left_limit - name_x - icons_w - 4 * R)
        pname = fit_text(draw, p.get("name", ""), f_small, name_max)
        draw.text((name_x, pmid), pname, font=f_small, fill=WHITE, anchor="lm")
        block_right = name_x + text_width(draw, pname, f_small)

        icon_x = block_right + 4 * R
        for role in icon_roles:
            alpha = 1.0 if role == assigned else ICON_ALPHA_DIM
            icon = _role_icon(role, icon_size, alpha)
            if icon:
                img.paste(icon, (int(icon_x), int(pmid - icon_size / 2)), icon)
                icon_x += icon_size + R
                block_right = icon_x - R

        # Middle: discord, clipped to the free space
        disc = p.get("discord") or ""
        if disc:
            disc_x = block_right + 5 * R
            avail = left_limit - disc_x
            if avail >= MIN_DISCORD_W * R:
                disc = fit_text(draw, disc, f_tiny, avail)
                dw = text_width(draw, disc, f_tiny)
                draw.text((left_limit - dw, pmid), disc, font=f_tiny, fill=DISCORD_COLOR, anchor="lm")

    return team_card_height(team)


def _new_canvas(log_w, log_h, title, fonts):
    """Create the canvas with the centred title drawn."""
    img = Image.new("RGB", (log_w * SCALE, log_h * SCALE), BG_COLOR)
    draw = ImageDraw.Draw(img)
    if title:
        title = fit_text(draw, title, fonts["title"], (log_w - 2 * MARGIN) * SCALE)
        tw = text_width(draw, title, fonts["title"])
        draw.text(((log_w * SCALE - tw) // 2, 12 * SCALE), title, font=fonts["title"], fill=WHITE)
    return img, draw


def _chunk(items, size):
    """Split a list into rows of at most size items."""
    return [items[i : i + size] for i in range(0, len(items), size)]


def render_teams_sheet(doc, path, title="", per_row=6):
    """Render every team of a teams.json document as a grid of cards.

    Args:
        doc: Parsed teams.json document.
        path: Output PNG path.
        title: Title drawn above the grid; empty for none.
        per_row: Cards per row.

    Returns:
        True if the file was written, False if Pillow is unavailable.
    """
    if not _PIL_OK:
        print("Pillow not installed (pip install Pillow): skipping PNG output.")
        return False
    teams = doc.get("teams", [])
    per_row = max(1, int(per_row))
    rows = _chunk(teams, per_row)
    fonts = _fonts()
    cols = min(per_row, max(len(teams), 1))
    log_w = cols * TEAM_W + (cols - 1) * COL_GAP + 2 * MARGIN
    top = TITLE_H if title else MARGIN
    row_hs = [max(team_card_height(t) for t in r) for r in rows]
    log_h = top + sum(row_hs) + max(len(rows) - 1, 0) * ROW_GAP + MARGIN
    img, draw = _new_canvas(log_w, log_h, title, fonts)
    y = top
    for row, row_h in zip(rows, row_hs):
        x = MARGIN
        for team in row:
            render_team_card(img, draw, x, y, team, fonts)
            x += TEAM_W + COL_GAP
        y += row_h + ROW_GAP
    _save(img, path)
    return True


def render_groups_sheet(groups_doc, teams_doc, path, title="", per_row=6):
    """Render groups (boxes of team cards) from groups.json and teams.json.

    Args:
        groups_doc: Parsed groups.json document; each group has index, name,
            server, team_ids, mean_team_score and na_teams.
        teams_doc: Parsed teams.json document providing the team entries.
        path: Output PNG path.
        title: Title drawn above the groups; empty for none.
        per_row: Maximum cards per row inside a group.

    Returns:
        True if the file was written, False if Pillow is unavailable.

    Raises:
        KeyError: If a group references a team id missing from teams_doc.
    """
    if not _PIL_OK:
        print("Pillow not installed (pip install Pillow): skipping PNG output.")
        return False
    by_id = {t["id"]: t for t in teams_doc.get("teams", [])}
    groups = groups_doc.get("groups", [])
    per_row = max(1, int(per_row))
    fonts = _fonts()

    layouts = []
    for g in groups:
        rows = _chunk([by_id[tid] for tid in g.get("team_ids", [])], per_row)
        row_hs = [max(team_card_height(t) for t in r) for r in rows]
        widths = [len(r) * TEAM_W + (len(r) - 1) * COL_GAP for r in rows]
        gw = (max(widths) if widths else TEAM_W) + 2 * GP
        gh = GLABEL_H + sum(row_hs) + max(len(rows) - 1, 0) * INNER_ROW + 2 * GP
        layouts.append((rows, row_hs, gw, gh))

    max_gw = max([gw for _, _, gw, _ in layouts] or [TEAM_W + 2 * GP])
    layouts = [(rows, row_hs, max_gw, gh) for rows, row_hs, _, gh in layouts]
    log_w = max_gw + 2 * MARGIN
    top = TITLE_H if title else MARGIN
    log_h = top + sum(gh for *_, gh in layouts) + max(len(layouts) - 1, 0) * ROW_GAP + MARGIN
    img, draw = _new_canvas(log_w, log_h, title, fonts)

    y = top
    for g, (rows, row_hs, gw, gh) in zip(groups, layouts):
        gx = MARGIN
        draw.rectangle(
            [gx * SCALE, y * SCALE, (gx + gw) * SCALE - 1, (y + gh) * SCALE - 1],
            fill=GROUP_BG,
            outline=GROUP_OUTLINE,
        )
        label = g.get("name") or f"Group {g.get('index', '')}".strip()
        if g.get("server"):
            label = f"{label} \u00b7 {g['server']}"
        info = []
        if g.get("mean_team_score") is not None:
            info.append(f"avg {g['mean_team_score']:.1f}")
        if g.get("na_teams") is not None:
            info.append(f"NA teams {g['na_teams']}")
        info_str = "   ".join(info)
        iw = text_width(draw, info_str, fonts["tiny"])
        draw.text(
            ((gx + gw - GP) * SCALE - iw, (y + GLABEL_H / 2 + 2) * SCALE),
            info_str,
            font=fonts["tiny"],
            fill=(150, 150, 150),
            anchor="lm",
        )
        label = fit_text(draw, label, fonts["glbl"], (gw - 2 * GP) * SCALE - iw - 12 * SCALE)
        draw.text(
            ((gx + GP) * SCALE, (y + GLABEL_H / 2 + 2) * SCALE),
            label,
            font=fonts["glbl"],
            fill=(200, 200, 200),
            anchor="lm",
        )
        ty = y + GLABEL_H + GP
        for row, row_h in zip(rows, row_hs):
            tx = gx + GP
            for team in row:
                render_team_card(img, draw, tx, ty, team, fonts)
                tx += TEAM_W + COL_GAP
            ty += row_h + INNER_ROW
        y += gh + ROW_GAP
    _save(img, path)
    return True


def _save(img, path):
    """Save the image as PNG, creating the parent directory."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    img.save(path, dpi=(144, 144))
