"""Tests for the PNG renderer (synthetic data only)."""

import copy

import pytest

pytest.importorskip("PIL")

from PIL import Image, ImageDraw

from teamMaker.core import render


def _player(i, discord="", stack=None, formation="solo", region="EU"):
    return {
        "name": f"Player{i}",
        "discord": discord,
        "score": 20.0 + i,
        "role": ["duelist", "flex"],
        "assigned_role": "duelist",
        "region": region,
        "server": "",
        "formation": formation,
        "stack_id": stack,
        "is_returning": False,
        "ranks": {"current": None, "peak": None},
    }


def _team(tid, discord="", fixed=False):
    players = [_player(i, discord if i == 0 else "") for i in range(5)]
    if fixed:
        for p in players:
            p.update(formation="5-stack", stack_id=99)
    return {
        "id": tid,
        "name": f"Team {tid}",
        "fixed": fixed,
        "formation": "1+1+1+1+1",
        "team_score": 110.0,
        "avg_score": 22.0,
        "role_penalty": 0.5,
        "regions": {"EU": 5},
        "players": players,
    }


def _doc(n=3, discord=""):
    return {"schema_version": 1, "teams": [_team(i + 1, discord) for i in range(n)]}


def test_fit_text_clips_with_ellipsis():
    draw = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    font = render.load_font(20)
    long = "x" * 60
    out = render.fit_text(draw, long, font, 100)
    assert out.endswith("...")
    assert render.text_width(draw, out, font) <= 100
    assert render.fit_text(draw, "ab", font, 100) == "ab"
    assert render.fit_text(draw, "ab", font, 0) == ""


def test_team_color_unique_beyond_palette():
    colors = [render.team_color(i) for i in range(60)]
    assert len(set(colors)) == 60


def test_long_discord_does_not_widen_card(tmp_path):
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    assert render.render_teams_sheet(_doc(2, ""), str(a), "T", per_row=2)
    assert render.render_teams_sheet(_doc(2, "d" * 60), str(b), "T", per_row=2)
    with Image.open(a) as ia, Image.open(b) as ib:
        assert ia.size == ib.size
        expected_w = (2 * render.TEAM_W + render.COL_GAP + 2 * render.MARGIN) * render.SCALE
        assert ia.size[0] == expected_w


def test_long_discord_stays_inside_card(tmp_path):
    path = tmp_path / "t.png"
    render.render_teams_sheet(_doc(1, "d" * 60), str(path), "", per_row=1)
    with Image.open(path) as im:
        # Right of the card must be untouched background (the margin).
        right = im.crop((im.size[0] - render.MARGIN * render.SCALE + 2, 0, im.size[0], im.size[1]))
        assert right.getextrema() == ((0, 0), (0, 0), (0, 0))


def test_five_stack_badge_and_rows(tmp_path):
    doc = {"schema_version": 1, "teams": [_team(1), _team(2, fixed=True)]}
    path = tmp_path / "t.png"
    assert render.render_teams_sheet(doc, str(path), "", per_row=6)
    with Image.open(path) as im:
        assert im.size[1] > 0


def test_groups_sheet(tmp_path):
    teams_doc = _doc(4)
    groups_doc = {
        "groups": [
            {"index": 1, "name": "Group 1", "server": "London", "team_ids": [1, 2],
             "mean_team_score": 110.0, "regions": {"EU": 10}, "na_teams": 0},
            {"index": 2, "name": "Group 2", "server": "Frankfurt", "team_ids": [3, 4],
             "mean_team_score": 110.0, "regions": {"EU": 10}, "na_teams": 0},
        ]
    }
    path = tmp_path / "g.png"
    assert render.render_groups_sheet(groups_doc, teams_doc, str(path), "T", per_row=6)
    assert path.exists()
    bad = copy.deepcopy(groups_doc)
    bad["groups"][0]["team_ids"] = [42]
    with pytest.raises(KeyError):
        render.render_groups_sheet(bad, teams_doc, str(tmp_path / "x.png"), "T")


def test_pillow_available():
    assert render.pillow_available() is True
