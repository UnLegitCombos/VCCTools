import csv
import glob
import io
import json
import os
import sys
import time

import yaml

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8")

from teamMaker.core.utils.helpers import (
    parse_roles,
    parse_tracker_url,
    parse_vcc_player_id,
    parse_vcc_player_name,
)
from teamMaker.core.utils.tracker import fetch_tracker_data
from teamMaker.core.utils.vcc import scrape_adjusted_rating

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_config():
    with open(os.path.join(BASE_DIR, "config", "player_ratings_config.yaml")) as f:
        return yaml.safe_load(f)

def find_latest_csv(data_dir):
    files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    return max(files, key=os.path.getmtime)

# Pull all fields from the CSV row
def process_row(row, config):
    tracker_url = row.get("tracker", "").strip()
    vcc_profile = row.get("vcc_profile", "").strip()
    discord = row.get("discord", "").strip()
    display_name = row.get("display_name", "").strip()
    rank_csv = row.get("rank", "").strip()
    roles_str = row.get("roles", "").strip()
    server = row.get("server", "").strip()
    returning = row.get("returning", "").strip().upper() == "TRUE"
    prev_seasons_str = row.get("previous_seasons", "").strip()
    submission_id = row.get("tally_submission_id", "").strip()

    # Player key: prefer VCC profile slug -> display_name -> discord handle
    player_name = parse_vcc_player_name(vcc_profile) or display_name or discord
    if not player_name:
        return None, None, None
    player_name = player_name[0].upper() + player_name[1:]

    print(f"Processing: {player_name}")

    # --- Tracker.gg data ---
    name, tag = parse_tracker_url(tracker_url)
    tracker_data = {}
    tracker_issue = None

    if not tracker_url:
        tracker_issue = "no tracker URL"
    elif not name or not tag:
        tracker_issue = f"could not parse name#tag from: {tracker_url}"
    else:
        tracker_data = fetch_tracker_data(name, tag, config)
        if not tracker_data:
            tracker_issue = f"API returned no data for {name}#{tag}"
        else:
            print(
                f"  current={tracker_data.get('tracker_current')}  "
                f"peak={tracker_data.get('tracker_peak')}  "
                f"rank={tracker_data.get('current_rank')}  "
                f"peak_rank={tracker_data.get('peak_rank')} ({tracker_data.get('peak_rank_act')})"
            )
        time.sleep(config["tracker_api"].get("request_delay", 0.25))

    # --- VCC previous-season adjusted ratings ---
    prev_stats = []
    if returning and prev_seasons_str and vcc_profile:
        player_id = parse_vcc_player_id(vcc_profile)
        season_map = config.get("season_map", {})
        seasons = config.get("seasons", {})
        ua = config["tracker_api"]["user_agent"]
        delay = config["tracker_api"].get("request_delay", 0.25)

        min_rounds = config.get("min_rounds", 0)
        if player_id:
            for season_display in prev_seasons_str.split(","):
                season_key = season_map.get(season_display.strip())
                if not season_key or season_key not in seasons:
                    continue
                season_info = seasons[season_key]
                code = season_info.get("code", season_key.replace("_", ""))
                link = season_info.get("link")
                if link:
                    print(f"  Scraping {code} aR for {player_name}...")
                    ar, rounds = scrape_adjusted_rating(player_id, link, ua)
                    if ar is not None:
                        if rounds is None or rounds >= min_rounds:
                            prev_stats.append({"season": code, "adjusted_rating": ar})
                        else:
                            print(f"  Skipping {code} for {player_name}: {rounds} rounds < min_rounds ({min_rounds})")
                    time.sleep(delay)

    entry = {
        "current_rank": tracker_data.get("current_rank") or rank_csv,
        "peak_rank": tracker_data.get("peak_rank"),
        "peak_rank_act": tracker_data.get("peak_rank_act"),
        "group_id": int(submission_id) if submission_id.isdigit() else None,
        "tracker_peak": tracker_data.get("tracker_peak"),
        "tracker_current": tracker_data.get("tracker_current"),
        "role": parse_roles(roles_str),
        "region": config.get("region_map", {}).get(server, "EU"),
        "is_returning_player": returning,
    }
    if prev_stats:
        entry["previous_season_stats"] = prev_stats

    return player_name, entry, tracker_issue

def main():
    config = load_config()
    data_dir = os.path.join(BASE_DIR, "data", config.get("data_dir", "input"))
    output_path = os.path.join(BASE_DIR, "data", config.get("output", "players.json"))

    csv_path = find_latest_csv(data_dir)
    print(f"CSV: {csv_path}\n")

    players = {}
    missing = {}
    missing_path = output_path.replace(".json", "_missing.json")

    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        if row.get("status", "").strip().lower() != "approved":
            continue
        player_name, entry, tracker_issue = process_row(row, config)
        if player_name is None:
            print(f"Skipping row (no name): {row}")
            continue
        players[player_name] = entry
        if tracker_issue:
            missing[player_name] = {
                "reason": tracker_issue,
                "discord": row.get("discord", "").strip(),
                "tracker_url": row.get("tracker", "").strip(),
            }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(players, f, indent=2)

    print(f"\nDone. Output: {output_path}")
    if missing:
        with open(missing_path, "w", encoding="utf-8") as f:
            json.dump(missing, f, indent=2)
        print(f"Missing tracker data for {len(missing)} player(s): {missing_path}")

if __name__ == "__main__":
    main()