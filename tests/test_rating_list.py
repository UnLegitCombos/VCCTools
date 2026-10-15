"""Tests for the published rating list (synthetic data only)."""

import csv
import time

from teamMaker.core import rating_list as rl
from teamMaker.core import render

PLAYERS = {"Ann": {"current_rank": "Gold 1"}, "bob": {"current_rank": "Diamond 2"}, "Cid": {}}
NOW = time.strptime("2026-10-15 12:00:00", "%Y-%m-%d %H:%M:%S")


def test_rows_sorted_by_rating_then_name_without_a_previous_list():
    rows = rl.build_rows({"Ann": 20.0, "bob": 24.5, "Cid": 20.0}, PLAYERS)
    assert [r["player"] for r in rows] == ["bob", "Ann", "Cid"]
    assert all(r["change"] is None and not r["new"] for r in rows)
    assert rows[0]["rank"] == "Diamond 2"


def test_changes_new_and_removed_against_the_previous_list():
    previous = {"ratings": {"Ann": {"rating": 20.0}, "bob": {"rating": 24.5}, "Gone": {"rating": 30.0}}}
    rows = {r["player"]: r for r in rl.build_rows({"Ann": 21.2, "bob": 24.5, "Cid": 18.0}, PLAYERS, previous)}
    assert rows["Ann"]["change"] == 1.2
    assert rl.change_label(rows["Ann"]) == "+1.2"
    assert rl.change_label(rows["bob"]) == ""
    assert rows["Cid"]["new"] and rl.change_label(rows["Cid"]) == "New"
    assert rl.removed({"Ann": 1, "bob": 1, "Cid": 1}, previous) == ["Gone"]


def test_rank_tiers_and_icon_links():
    assert [render.rank_tier(r) for r in ("Iron 1", "gold 2", "Immortal 3", "Radiant", "Gold", None)] == \
        [3, 13, 26, 27, None, None]
    assert render.rank_icon_url("Diamond 2").endswith("/19/smallicon.png")


def test_publish_writes_snapshot_and_csv_and_becomes_the_latest(tmp_path, monkeypatch):
    monkeypatch.setattr(render, "rank_icon", lambda rank, size: None)  # no downloads in tests
    rows = rl.build_rows({"Ann": 20.0, "bob": 24.5}, PLAYERS)
    paths = rl.publish(rows, directory=str(tmp_path), now=NOW)
    assert rl.list_snapshots(str(tmp_path)) == ["rating_list_20261015-120000.json"]
    latest = rl.latest_snapshot(str(tmp_path))
    assert latest is not None
    assert latest["published_at"] == "2026-10-15T12:00:00"
    assert latest["ratings"]["bob"] == {"rating": 24.5, "rank": "Diamond 2"}
    with open(paths["csv"], encoding="utf-8-sig", newline="") as f:
        assert list(csv.reader(f))[1] == ["1", "bob", "24.5", "Diamond 2", ""]


def test_no_snapshots_yet(tmp_path):
    assert rl.latest_snapshot(str(tmp_path / "missing")) is None
