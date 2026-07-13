import json
import re
import subprocess
import time
from urllib.parse import quote


# Cloudflare blocks Python's TLS fingerprint; curl passes through fine.
def _curl_get(url, user_agent, params=None):
    if params:
        qs = "&".join(f"{k}={quote(str(v))}" for k, v in params.items())
        url = f"{url}?{qs}"

    cmd = [
        "curl.exe", "-s",
        "-H", f"User-Agent: {user_agent}",
        "-H", "Accept: application/json",
        "-H", "Origin: https://tracker.gg",
        "-H", "Referer: https://tracker.gg/",
        url,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        if result.returncode != 0:
            print(f"  [curl] error ({result.returncode}) for {url}")
            return None
        data = json.loads(result.stdout)
        if data.get("errors"):
            print(f"  [tracker] API error: {data['errors']}")
            return None
        return data
    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception) as e:
        print(f"  [curl] exception for {url}: {e}")
        return None


# "Season 26 Act 3" -> "S26A3", "Episode 7 Act 3" -> "E7A3"
def _act_name_to_code(name):
    m = re.match(r"Season\s+(\d+)\s+Act\s+(\d+)", name, re.IGNORECASE)
    if m:
        return f"S{m.group(1)}A{m.group(2)}"
    m = re.match(r"Episode\s+(\d+)\s+Act\s+(\d+)", name, re.IGNORECASE)
    if m:
        return f"E{m.group(1)}A{m.group(2)}"
    return name


# Returns tracker_current, tracker_peak, current_rank, peak_rank, peak_rank_act
def fetch_tracker_data(name, tag, config):
    ua = config["tracker_api"]["user_agent"]
    delay = config["tracker_api"].get("request_delay", 0.25)

    profile_url = (
        config["tracker_api"]["profile_url"]
        .replace("{name}", quote(name))
        .replace("{tag}", quote(tag))
    )
    segment_url = (
        config["tracker_api"]["segment_url"]
        .replace("{name}", quote(name))
        .replace("{tag}", quote(tag))
    )

    profile = _curl_get(profile_url, ua)
    if not profile:
        return {}

    seasons_by_id = {
        s["id"]: s["name"]
        for s in profile["data"]["metadata"].get("seasons", [])
    }
    default_season_id = profile["data"]["metadata"].get("defaultSeason")

    # If a specific act is configured, find its UUID so we pull from that act
    # instead of the defaultSeason (which may be a fresh act with reset ranks).
    configured_act = config.get("current_act")
    current_season_id = None
    if configured_act:
        for sid, sname in seasons_by_id.items():
            if _act_name_to_code(sname) == configured_act:
                current_season_id = sid
                break
        if not current_season_id:
            print(f"  [tracker] WARNING: current_act '{configured_act}' not found in seasons list; falling back to defaultSeason")
    if not current_season_id:
        current_season_id = default_season_id

    # Fetch the target act's segment data explicitly so we always get the right act
    time.sleep(delay)
    seg_data = _curl_get(
        segment_url, ua,
        params={"playlist": "competitive", "seasonId": current_season_id},
    )
    if not seg_data or not seg_data.get("data"):
        print(f"  [tracker] no segment data for {name}#{tag} act={configured_act or current_season_id}")
        return {}

    stats = seg_data["data"][0].get("stats", {})

    tracker_current = (stats.get("trnPerformanceScore") or {}).get("value")

    # rank.metadata.tierName is the current rank; rank.displayValue is always empty
    current_rank = (stats.get("rank") or {}).get("metadata", {}).get("tierName")

    peak_meta = (stats.get("peakRank") or {}).get("metadata", {})
    peak_rank = peak_meta.get("tierName")
    peak_act_id = peak_meta.get("actId")
    peak_act_name = seasons_by_id.get(peak_act_id, "")
    peak_rank_act = _act_name_to_code(peak_act_name) if peak_act_name else None

    # If peak is in the current act we already have the score
    if peak_act_id and peak_act_id == current_season_id:
        tracker_peak = tracker_current
    elif peak_act_id:
        # One extra call for the past peak act's tracker score
        time.sleep(delay)
        seg_data = _curl_get(
            segment_url, ua,
            params={"playlist": "competitive", "seasonId": peak_act_id},
        )
        if seg_data and seg_data.get("data"):
            seg_stats = seg_data["data"][0].get("stats", {})
            tracker_peak = (seg_stats.get("trnPerformanceScore") or {}).get("value")
        else:
            tracker_peak = None
    else:
        tracker_peak = None

    return {
        "tracker_current": tracker_current,
        "tracker_peak": tracker_peak,
        "current_rank": current_rank,
        "peak_rank": peak_rank,
        "peak_rank_act": peak_rank_act,
    }
