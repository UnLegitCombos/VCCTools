import math
import os

import pytest

from teamMaker.core import scrape_distributions as sd
from teamMaker.core.utils import vcc

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "vcc_stats_small.html")


@pytest.fixture(scope="module")
def html():
    with open(FIXTURE, encoding="utf-8") as f:
        return f.read()


@pytest.fixture(autouse=True)
def clear_cache():
    vcc._SEASON_CACHE.clear()
    yield
    vcc._SEASON_CACHE.clear()


def test_extract_flight_payload_joins_chunks(html):
    payload = vcc.extract_flight_payload(html)
    # the rows list is split across two push chunks, so joining is required
    assert payload.count('"player_id"') == 6
    assert '"totalCount":6' in payload


def test_extract_stats_rows_returns_all_rows(html):
    rows = vcc.extract_stats_rows(vcc.extract_flight_payload(html))
    assert len(rows) == 6
    assert rows[0]["player_id"] == "synthetic-p1"
    assert vcc.extract_total_count(vcc.extract_flight_payload(html)) == 6


def test_extract_stats_rows_ignores_unrelated_rows_lists():
    payload = '{"rows":[{"a":1}],"x":1} {"rows":[{"player_id":"z","rating":1.0,"total_rounds":10}],"totalCount":1}'
    rows = vcc.extract_stats_rows(payload)
    assert [r["player_id"] for r in rows] == ["z"]


def test_extract_stats_rows_warns_on_count_mismatch(capsys):
    payload = '"rows":[{"player_id":"z","rating":1.0,"total_rounds":10}],"totalCount":5'
    assert len(vcc.extract_stats_rows(payload)) == 1
    assert "totalCount is 5" in capsys.readouterr().out


def test_extract_flight_payload_without_scripts():
    assert vcc.extract_flight_payload("<html></html>") == ""
    assert vcc.extract_stats_rows("") == []


def test_compute_adjusted_ratings_matches_site_formula(html):
    rows = vcc.extract_stats_rows(vcc.extract_flight_payload(html))
    ar = vcc.compute_adjusted_ratings(rows)
    avg_rating = sum(r["rating"] for r in rows) / 6
    avg_rounds = sum(r["total_rounds"] for r in rows) / 6
    # p1: rating 1.45 (> average), 400 rounds
    f = avg_rounds * 1.5 / math.sqrt(400 / 10)
    expected = (1.45 * 400 + avg_rating * f) / (400 + f)
    assert ar["synthetic-p1"] == pytest.approx(expected)
    # p5: rating 0.80 (< average), 80 rounds, E = 0.1
    f = avg_rounds * 0.1 / math.sqrt(80 / 10)
    expected = (0.80 * 80 + avg_rating * f) / (80 + f)
    assert ar["synthetic-p5"] == pytest.approx(expected)


def test_adjusted_rating_regresses_towards_mean_more_for_few_rounds():
    rows = [
        {"player_id": "a", "rating": 1.6, "total_rounds": 400},
        {"player_id": "b", "rating": 1.6, "total_rounds": 40},
        {"player_id": "c", "rating": 0.4, "total_rounds": 200},
    ]
    ar = vcc.compute_adjusted_ratings(rows)
    assert ar["a"] > ar["b"]
    assert 1.0 < ar["b"] < 1.6


def test_compute_adjusted_ratings_empty():
    assert vcc.compute_adjusted_ratings([]) == {}


def test_parse_html_table(html):
    rows = vcc.parse_html_table(html)
    assert len(rows) == 4
    assert rows[0]["player_id"] == "synthetic-p1"
    assert rows[0]["rounds"] == 400
    assert rows[0]["ar"] == pytest.approx(1.415, abs=0.001)


def test_payload_beats_truncated_table(monkeypatch, html):
    monkeypatch.setattr(vcc, "fetch_html", lambda url, ua, timeout=20: html)
    stats = vcc.fetch_season_stats("S99", "https://x.test/e?tab=stats", "ua")
    assert stats is not None
    assert stats.source == "payload"
    assert not stats.truncated
    assert len(stats.ar_values) == 6
    assert stats.ar_values == sorted(stats.ar_values, reverse=True)
    assert stats.total_count == 6


def test_html_fallback_when_payload_missing(monkeypatch, html, capsys):
    no_payload = html.replace("self.__next_f.push", "self.__other.push")
    monkeypatch.setattr(vcc, "fetch_html", lambda url, ua, timeout=20: no_payload)
    stats = vcc.fetch_season_stats("S99", "https://x.test/e?tab=stats", "ua")
    assert stats is not None
    assert stats.source == "html"
    assert stats.truncated
    assert len(stats.ar_values) == 4
    assert "falling back" in capsys.readouterr().out


def test_self_check_warns_when_table_disagrees(monkeypatch, html, capsys):
    tampered = html.replace(">1.415<", ">1.900<")
    monkeypatch.setattr(vcc, "fetch_html", lambda url, ua, timeout=20: tampered)
    vcc.fetch_season_stats("S99", "https://x.test/e?tab=stats", "ua")
    assert "computed aR differs" in capsys.readouterr().out


def test_self_check_silent_when_consistent(monkeypatch, html, capsys):
    monkeypatch.setattr(vcc, "fetch_html", lambda url, ua, timeout=20: html)
    vcc.fetch_season_stats("S99", "https://x.test/e?tab=stats", "ua")
    assert "warning" not in capsys.readouterr().out


def test_fetch_failure_returns_none(monkeypatch):
    monkeypatch.setattr(vcc, "fetch_html", lambda url, ua, timeout=20: None)
    assert vcc.fetch_season_stats("S99", "https://x.test/e", "ua") is None
    assert vcc.lookup_player(None, "p") == (None, None)


def test_load_season_stats_fetches_once(monkeypatch, html):
    calls = []

    def fake(url, ua, timeout=20):
        calls.append(url)
        return html

    monkeypatch.setattr(vcc, "fetch_html", fake)
    a = vcc.load_season_stats("S99", "https://x.test/e?tab=stats", "ua")
    b = vcc.load_season_stats("S99", "https://x.test/e?tab=stats", "ua")
    assert a is b
    assert len(calls) == 1


def test_lookup_player(monkeypatch, html):
    monkeypatch.setattr(vcc, "fetch_html", lambda url, ua, timeout=20: html)
    stats = vcc.load_season_stats("S99", "https://x.test/e?tab=stats", "ua")
    ar, rounds = vcc.lookup_player(stats, "synthetic-p6")
    assert rounds == 40
    assert ar is not None
    assert vcc.lookup_player(stats, "nobody") == (None, None)


def test_build_stats_url():
    link = "https://x.test/e/s?tab=stats&stage=group-stage"
    assert vcc.build_stats_url(link, "all") == "https://x.test/e/s?tab=stats"
    assert vcc.build_stats_url(link, "playoffs").endswith("tab=stats&stage=playoffs")


# --- archive shrink guard / frozen policy ---


def _stats(values, source="payload"):
    return vcc.SeasonStats(
        code="S99", url="u", stage="all", source=source, ar_values=values
    )


def test_new_season_is_written():
    assert sd.decide_write(None, _stats([1.0, 0.9]), refresh=False)[0]


def test_existing_season_is_frozen():
    write, reason = sd.decide_write([1.0, 0.9], _stats([1.1, 1.0]), refresh=False)
    assert not write and "frozen" in reason


def test_refresh_replaces_when_not_smaller():
    assert sd.decide_write([1.0, 0.9], _stats([1.1, 1.0, 0.8]), refresh=True)[0]
    assert sd.decide_write([1.0, 0.9], _stats([1.1, 1.0]), refresh=True)[0]


def test_refresh_refuses_shrink():
    write, reason = sd.decide_write([1.0, 0.9, 0.8], _stats([1.1, 1.0]), refresh=True)
    assert not write and "fewer" in reason


def test_html_source_is_never_written():
    assert not sd.decide_write(None, _stats([1.0], source="html"), refresh=True)[0]
    assert not sd.decide_write([1.0], _stats([1.0], source="html"), refresh=True)[0]


def test_summarize_and_zscore():
    n, mean, std = sd.summarize([1.0, 2.0, 3.0])
    assert (n, mean) == (3, 2.0)
    assert std == pytest.approx(math.sqrt(2 / 3))
    assert sd.zscore(2.0, [1.0, 1.0]) == 0.0
