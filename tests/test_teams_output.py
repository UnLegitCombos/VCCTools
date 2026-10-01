import json

import pytest

from teamMaker.core import optimizer as opt
from teamMaker.core import report, teams_output
from teamMaker.core.config import DEFAULTS, deep_merge
from tests.synthetic_players import make_players


def _run(seed=5, team_order="shuffled"):
    players, scores = make_players([5, 4, 3, 2, 2] + [1] * 12, subs=2, seed=3)
    for i, name in enumerate(players):
        players[name]["discord"] = f"user{i}"
    players["P000"]["discord"] = "nämeé"
    config = deep_merge(DEFAULTS, {})
    config["optimizer"] = dict(config["optimizer"], iterations=3000, restarts=1)
    config["output"] = {"teams_png": False, "team_order": team_order}
    config["random_seed"] = seed
    build = opt.build_units(players, scores, config)
    kept, dropped = opt.select_units(build.units)
    result = opt.optimize_teams(kept, config, seed)
    doc = teams_output.build_teams_doc(players, scores, build, dropped, result, config, seed)
    return players, scores, build, dropped, result, doc


def test_document_structure():
    players, scores, build, dropped, result, doc = _run()
    assert doc["schema_version"] == 1
    assert set(doc) == {"schema_version", "meta", "summary", "teams", "subs", "excluded"}
    teams = doc["teams"]
    assert [t["id"] for t in teams] == list(range(1, len(teams) + 1))
    assert [t["fixed"] for t in teams] == [False] * (len(teams) - 1) + [True]
    for team in teams:
        assert len(team["players"]) == 5
        assert team["team_score"] == pytest.approx(sum(p["score"] for p in team["players"]), abs=1e-6)
        assert team["avg_score"] == pytest.approx(team["team_score"] / 5, abs=0.01)
        for p in team["players"]:
            assert "stack_id" in p and "group_id" not in p
            assert p["assigned_role"]
            assert set(p["ranks"]) == {"current", "peak"}
    placed = (
        [p["name"] for t in teams for p in t["players"]]
        + [s["name"] for s in doc["subs"]]
        + [e["name"] for e in doc["excluded"]]
    )
    assert sorted(placed) == sorted(players)
    assert len(doc["excluded"]) == 4
    assert all("not allowed" in e["reason"] for e in doc["excluded"])
    assert all(s["reason"] for s in doc["subs"])
    assert doc["summary"]["all"]["n"] == len(teams)
    assert doc["summary"]["optimized"]["n"] == len(teams) - 1


def test_stack_ids_only_for_real_stacks():
    *_, doc = _run()
    for team in doc["teams"]:
        for p in team["players"]:
            if p["formation"] == "solo":
                assert p["stack_id"] is None
            else:
                assert p["stack_id"] is not None


def test_same_seed_same_document_and_team_shuffle():
    *_, doc1 = _run(seed=5)
    *_, doc2 = _run(seed=5)
    for d in (doc1, doc2):
        d["meta"].pop("generated_at")
    assert doc1 == doc2
    *_, sorted_doc = _run(seed=5, team_order="sorted")
    opt_scores = [t["team_score"] for t in sorted_doc["teams"] if not t["fixed"]]
    assert opt_scores == sorted(opt_scores, reverse=True)


def test_write_and_load_roundtrip_utf8(tmp_path):
    *_, doc = _run()
    path = tmp_path / "teams.json"
    teams_output.write_teams_json(doc, str(path))
    raw = path.read_text(encoding="utf-8")
    assert "nämeé" in raw and "\\u00e4" not in raw
    assert teams_output.load_teams_json(str(path)) == json.loads(raw)


def test_load_rejects_unknown_schema(tmp_path):
    path = tmp_path / "teams.json"
    path.write_text(json.dumps({"schema_version": 2}), encoding="utf-8")
    with pytest.raises(ValueError):
        teams_output.load_teams_json(str(path))
    path.write_text(json.dumps({"teams": []}), encoding="utf-8")
    with pytest.raises(ValueError):
        teams_output.load_teams_json(str(path))


def test_report_is_ascii_and_lists_sections():
    players, scores, build, dropped, result, doc = _run()
    text = report.format_report(doc, {"players": len(players), "elapsed_s": 1.0}, {"teams": "x.json"}, ["a warning"])
    for heading in ("INPUT AND SELECTION", "OPTIMIZATION", "BALANCE", "TEAMS", "SUBSTITUTES", "WARNINGS", "OUTPUT FILES"):
        assert heading in text
    assert "a warning" in text
    # The only non-ASCII content allowed is player-supplied text (the synthetic name
    # here uses ASCII), so the report itself must be ASCII.
    doc_ascii = json.loads(json.dumps(doc).replace("\\u00e4", "a").replace("\\u00e9", "e"))
    assert report.format_report(doc_ascii).isascii()
