"""VLR Community Cup (VCC) season stats scraping.

The stats page is a Next.js app. Its server-rendered HTML table only holds the
top ~60 rows, but the complete row list (every player of the season) is
embedded in the flight payload (``self.__next_f.push([1, "..."])`` scripts).
This module reads that payload, recomputes the site's adjusted rating (aR) from
the raw ratings and falls back to the (truncated) HTML table if the payload
format ever changes.
"""

import json
import math
import re
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from bs4 import BeautifulSoup

FLIGHT_MARKER = "self.__next_f.push("
REQUIRED_ROW_KEYS = ("player_id", "rating", "total_rounds")
AR_TOLERANCE = 0.005

# aR formula: F = avgRounds * E / sqrt(rounds / 10), E = 1.5 if rating is above
# the season average else 0.1, aR = (rating*rounds + avgRating*F) / (rounds + F)
E_ABOVE_AVERAGE = 1.5
E_BELOW_AVERAGE = 0.1

_SEASON_CACHE = {}


@dataclass
class SeasonStats:
    """Stats of one VCC season.

    Attributes:
        code: Season code, e.g. "S13".
        url: The URL the data was fetched from.
        stage: Stage filter used ("all" means no stage filter).
        source: "payload" (complete) or "html" (truncated fallback).
        players: Mapping of player_id to (adjusted rating, rounds).
        ar_values: All adjusted ratings, rounded and sorted descending.
        total_count: Row count the site reports, when known.
        truncated: True if the data is known or suspected to be incomplete.
        fetched_at: UTC ISO timestamp of the fetch.
    """

    code: str
    url: str
    stage: str
    source: str
    players: dict = field(default_factory=dict)
    ar_values: list = field(default_factory=list)
    total_count: int | None = None
    truncated: bool = False
    fetched_at: str = ""


def build_stats_url(link: str, stage: str = "all") -> str:
    """Return the stats URL for a stage.

    Args:
        link: Season link from the config (may carry a stage query param).
        stage: "all" removes the stage filter, anything else sets it.

    Returns:
        The rewritten URL.
    """
    parts = urlparse(link)
    query = [(k, v) for k, v in parse_qsl(parts.query) if k != "stage"]
    if stage and stage != "all":
        query.append(("stage", stage))
    return urlunparse(parts._replace(query=urlencode(query)))


def fetch_html(url, user_agent, timeout=20):
    """Download a page.

    Args:
        url: Page URL.
        user_agent: User-Agent header value.
        timeout: Request timeout in seconds.

    Returns:
        The page HTML, or None if the request failed.
    """
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"  [vcc] fetch error for {url}: {e}")
        return None


def extract_flight_payload(html):
    """Concatenate the string chunks of every ``self.__next_f.push([1, ...])``.

    Each push argument is decoded with ``json.JSONDecoder.raw_decode``, which
    handles the escaping that a regex would get wrong.

    Args:
        html: Page HTML.

    Returns:
        The concatenated flight payload (empty string if none was found).
    """
    decoder = json.JSONDecoder()
    chunks = []
    pos = 0
    while True:
        start = html.find(FLIGHT_MARKER, pos)
        if start < 0:
            break
        arg_start = start + len(FLIGHT_MARKER)
        try:
            arg, end = decoder.raw_decode(html, arg_start)
        except ValueError:
            pos = arg_start
            continue
        pos = end
        if (
            isinstance(arg, list)
            and len(arg) > 1
            and arg[0] == 1
            and isinstance(arg[1], str)
        ):
            chunks.append(arg[1])
    return "".join(chunks)


def _is_stats_rows(value):
    return (
        isinstance(value, list)
        and len(value) > 0
        and all(
            isinstance(r, dict) and all(k in r for k in REQUIRED_ROW_KEYS)
            for r in value
        )
    )


def extract_stats_rows(payload):
    """Find the stats rows inside a flight payload.

    Takes the first ``"rows":[`` list whose dicts carry player_id, rating and
    total_rounds. Warns if the number of rows differs from ``totalCount``.

    Args:
        payload: Output of extract_flight_payload.

    Returns:
        A list of row dicts (empty if none was found).
    """
    decoder = json.JSONDecoder()
    marker = '"rows":'
    pos = 0
    while True:
        idx = payload.find(marker, pos)
        if idx < 0:
            return []
        list_start = idx + len(marker)
        pos = list_start
        try:
            rows, end = decoder.raw_decode(payload, list_start)
        except ValueError:
            continue
        if not _is_stats_rows(rows):
            continue
        match = re.match(r'[^{}\[\]]{0,200}?"totalCount":(\d+)', payload[end:])
        if match and int(match.group(1)) != len(rows):
            print(
                f"  [vcc] warning: payload has {len(rows)} rows but "
                f"totalCount is {match.group(1)}"
            )
        return rows


def extract_total_count(payload):
    """Return the ``totalCount`` reported next to the stats rows, or None."""
    match = re.search(r'"totalCount":(\d+)', payload)
    return int(match.group(1)) if match else None


def compute_adjusted_ratings(rows):
    """Compute the site's adjusted rating for every row.

    Uses unweighted averages of rating and rounds over all rows.

    Args:
        rows: Row dicts with player_id, rating and total_rounds.

    Returns:
        Dict of player_id to unrounded aR.
    """
    if not rows:
        return {}
    avg_rating = sum(r["rating"] for r in rows) / len(rows)
    avg_rounds = sum(r["total_rounds"] for r in rows) / len(rows)
    result = {}
    for r in rows:
        rounds = r["total_rounds"]
        rating = r["rating"]
        if rounds <= 0:
            result[r["player_id"]] = rating
            continue
        e = E_ABOVE_AVERAGE if rating > avg_rating else E_BELOW_AVERAGE
        f = avg_rounds * e / math.sqrt(rounds / 10)
        result[r["player_id"]] = (rating * rounds + avg_rating * f) / (rounds + f)
    return result


def parse_html_table(html):
    """Parse the server-rendered stats table (fallback, truncated to ~60 rows).

    Args:
        html: Page HTML.

    Returns:
        A list of dicts with player_id, ar and rounds (None when unparsable).
    """
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if not table:
        return []
    thead = table.find("thead")
    thead_row = thead.find("tr") if thead else None
    ths = thead_row.find_all("th") if thead_row else []
    ar_idx = rounds_idx = None
    for i, th in enumerate(ths):
        title = th.get("title")
        text = th.get_text(strip=True)
        if (title and "Adjusted rating" in title) or text == "aR":
            ar_idx = i
        if text == "RND":
            rounds_idx = i
    tbody = table.find("tbody")
    if ar_idx is None or not tbody:
        return []

    out = []
    for row in tbody.find_all("tr"):
        link = row.find("a", href=re.compile(r"/player/([^/?#]+)"))
        if not link:
            continue
        match = re.search(r"/player/([^/?#]+)", str(link.get("href", "")))
        if not match:
            continue
        player_id = match.group(1)
        tds = row.find_all("td")
        entry = {"player_id": player_id, "ar": None, "rounds": None}
        if ar_idx < len(tds):
            try:
                entry["ar"] = float(tds[ar_idx].get_text(strip=True))
            except ValueError:
                pass
        if rounds_idx is not None and rounds_idx < len(tds):
            try:
                entry["rounds"] = int(tds[rounds_idx].get_text(strip=True))
            except ValueError:
                pass
        out.append(entry)
    return out


def _self_check(computed, table_rows, code):
    """Warn if computed aR values differ from the aR shown in the HTML table."""
    bad = [
        (t["player_id"], computed[t["player_id"]], t["ar"])
        for t in table_rows
        if t["ar"] is not None
        and t["player_id"] in computed
        and abs(computed[t["player_id"]] - t["ar"]) > AR_TOLERANCE
    ]
    if bad:
        pid, mine, shown = bad[0]
        print(
            f"  [vcc] warning: {code}: computed aR differs from the page for "
            f"{len(bad)} row(s) (e.g. {pid}: {mine:.3f} vs {shown:.3f}); "
            f"the site formula may have changed"
        )
    return not bad


def _build_stats(html, code, url, stage, decimals):
    """Build SeasonStats from page HTML (payload first, HTML table fallback)."""
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    payload = extract_flight_payload(html)
    rows = extract_stats_rows(payload) if payload else []
    table_rows = parse_html_table(html)

    if rows:
        computed = compute_adjusted_ratings(rows)
        _self_check(computed, table_rows, code)
        rounds = {r["player_id"]: r["total_rounds"] for r in rows}
        players = {
            pid: (round(ar, 3), rounds[pid]) for pid, ar in computed.items()
        }
        values = sorted((round(ar, decimals) for ar in computed.values()), reverse=True)
        return SeasonStats(
            code=code,
            url=url,
            stage=stage,
            source="payload",
            players=players,
            ar_values=values,
            total_count=extract_total_count(payload) or len(rows),
            truncated=False,
            fetched_at=fetched_at,
        )

    if table_rows:
        print(
            f"  [vcc] warning: {code}: no flight payload found, falling back to "
            f"the HTML table ({len(table_rows)} rows, truncated)"
        )
        players = {
            t["player_id"]: (t["ar"], t["rounds"])
            for t in table_rows
            if t["ar"] is not None
        }
        values = sorted(
            (round(t["ar"], decimals) for t in table_rows if t["ar"] is not None),
            reverse=True,
        )
        return SeasonStats(
            code=code,
            url=url,
            stage=stage,
            source="html",
            players=players,
            ar_values=values,
            total_count=None,
            truncated=True,
            fetched_at=fetched_at,
        )
    return None


def fetch_season_stats(code, link, user_agent, stage="all", decimals=2):
    """Fetch and parse one season (no caching).

    Args:
        code: Season code, e.g. "S13".
        link: Season stats link from the config.
        user_agent: User-Agent header value.
        stage: Stage filter; "all" removes it.
        decimals: Decimals for ar_values.

    Returns:
        SeasonStats, or None if the page could not be fetched or parsed.
    """
    url = build_stats_url(link, stage)
    html = fetch_html(url, user_agent)
    if html is None:
        return None
    stats = _build_stats(html, code, url, stage, decimals)
    if stats is None:
        print(f"  [vcc] no stats found at {url}")
    return stats


def load_season_stats(code, link, user_agent, stage="all", decimals=2):
    """Fetch a season once per process and reuse the result.

    Args:
        code: Season code, e.g. "S13".
        link: Season stats link from the config.
        user_agent: User-Agent header value.
        stage: Stage filter; "all" removes it.
        decimals: Decimals for ar_values.

    Returns:
        SeasonStats, or None on failure (failures are not cached).
    """
    key = (code, link, stage, decimals)
    if key not in _SEASON_CACHE:
        stats = fetch_season_stats(code, link, user_agent, stage, decimals)
        if stats is None:
            return None
        _SEASON_CACHE[key] = stats
    return _SEASON_CACHE[key]


def lookup_player(stats, player_id):
    """Return (aR, rounds) for a player, or (None, None) if not found."""
    if stats is None or player_id not in stats.players:
        return None, None
    return stats.players[player_id]
