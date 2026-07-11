#!/usr/bin/env python3
import json
import os
import urllib.request

import yaml
from bs4 import BeautifulSoup

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_config():
    with open(os.path.join(SCRIPT_DIR, "player_ratings_config.yaml")) as f:
        return yaml.safe_load(f)


def scrape_all_adjusted_ratings(url, user_agent):
    """Fetch all adjusted rating values from a VCC season stats page."""
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"  fetch error for {url}: {e}")
        return []

    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if not table:
        print(f"  no table found at {url}")
        return []

    # Find the aR column index by its title attribute
    thead = table.find("thead")
    thead_row = thead.find("tr") if thead else None
    ths = thead_row.find_all("th") if thead_row else []
    ar_idx = None
    for i, th in enumerate(ths):
        title = th.get("title")
        if (title and "Adjusted rating" in title) or th.get_text(strip=True) == "aR":
            ar_idx = i
            break

    if ar_idx is None:
        print(f"  aR column not found at {url}")
        return []

    tbody = table.find("tbody")
    if not tbody:
        return []

    ratings = []
    for row in tbody.find_all("tr"):
        tds = row.find_all("td")
        if ar_idx < len(tds):
            try:
                ratings.append(float(tds[ar_idx].get_text(strip=True)))
            except ValueError:
                pass

    return ratings


def main():
    config = load_config()
    seasons = config.get("SEASONS", {})
    ua = config["TRACKER_API"]["USER_AGENT"]
    output_path = os.path.join(SCRIPT_DIR, "season_distributions.json")

    # Load existing file so we can update without clobbering other seasons
    if os.path.isfile(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            distributions = json.load(f)
    else:
        distributions = {}

    for season_key, season_info in seasons.items():
        code = season_info.get("CODE")
        link = season_info.get("LINK")
        if not code or not link:
            print(f"Skipping {season_key}: missing CODE or LINK")
            continue

        print(f"Scraping {code} distribution from {link}...")
        ratings = scrape_all_adjusted_ratings(link, ua)

        if not ratings:
            print(f"  No ratings found for {code}, skipping.")
            continue

        distributions[code] = ratings
        print(f"  Collected {len(ratings)} ratings for {code}.")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(distributions, f, indent=2)

    print(f"\nDistributions written to {output_path}")


if __name__ == "__main__":
    main()
