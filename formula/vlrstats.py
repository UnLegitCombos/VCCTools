import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm


def scrape_vlr_stats_for_url(url, year, group, show_progress=False):
    """
    Scrapes the given VLR stats page using your column indexing
    and returns a list of lists. Each row also has a 'year' field.
    """
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to retrieve page: {url}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    stats_table = soup.find("table", class_="wf-table")
    if not stats_table:
        print(f"No table found on page: {url}")
        return []

    table_body = stats_table.find("tbody")
    rows = table_body.find_all("tr")

    all_data = []
    iterator = tqdm(
        rows, desc=f"Scraping {group} {year}", unit=" player", disable=not show_progress
    )
    for row in iterator:
        cols = row.find_all("td")
        if len(cols) < 18:
            continue
        try:
            player_name = cols[0].get_text(strip=True)
            rounds_played = int(cols[2].get_text(strip=True))
            rating_val = float(cols[3].get_text(strip=True))
            acs_val = float(cols[4].get_text(strip=True))
            kd_val = float(cols[5].get_text(strip=True))
            kast_val = float(cols[6].get_text(strip=True).replace("%", "")) / 100.0
            adr_val = float(cols[7].get_text(strip=True))
            kpr_val = float(cols[8].get_text(strip=True))
            apr_val = float(cols[9].get_text(strip=True))
            # Avoid division by zero for dpr_val
            dpr_val = (
                round(float(cols[17].get_text(strip=True)) / rounds_played, 2)
                if rounds_played != 0
                else 0.0
            )
            fkpr_val = float(cols[10].get_text(strip=True))
            fdpr_val = float(cols[11].get_text(strip=True))
            hs_val = float(cols[12].get_text(strip=True).replace("%", "")) / 100.0
            cl_text = cols[13].get_text(strip=True).replace("%", "")
            cl_val = float(cl_text) / 100.0 if cl_text else 0.0

            row_data = [
                player_name,
                rounds_played,
                rating_val,
                acs_val,
                kd_val,
                kast_val,
                adr_val,
                kpr_val,
                apr_val,
                dpr_val,
                fkpr_val,
                fdpr_val,
                hs_val,
                cl_val,
                group,
                year,
            ]
            all_data.append(row_data)
        except (ValueError, IndexError):
            continue

    return all_data


def scrape_all_years_combined(output_csv="vlr_data.csv"):
    """
    Scrapes 3 pages (2023, 2024, 2025) using the same indexing, merges
    them into one CSV with the same columns + a 'year' column, and sorts by rating (descending).
    """

    # The 3 VLR URLs for each year
    url_2025 = "https://www.vlr.gg/stats/?event_group_id=74&event_id=all&region=all&min_rounds=150&min_rating=0&agent=all&map_id=all&timespan=all"
    url_2024 = "https://www.vlr.gg/stats/?event_group_id=61&event_id=all&region=all&min_rounds=200&min_rating=0&agent=all&map_id=all&timespan=all"
    url_2023 = "https://www.vlr.gg/stats/?event_group_id=45&event_id=all&region=all&min_rounds=200&min_rating=0&agent=all&map_id=all&timespan=all"
    url_ch25 = "https://www.vlr.gg/stats/?event_group_id=75&event_id=all&region=all&min_rounds=200&min_rating=0&agent=all&map_id=all&timespan=all"
    url_ch24 = "https://www.vlr.gg/stats/?event_group_id=59&region=all&min_rounds=200&min_rating=0&agent=all&map_id=all&timespan=all"
    url_ch23 = "https://www.vlr.gg/stats/?event_group_id=31&region=all&min_rounds=200&min_rating=0&agent=all&map_id=all&timespan=all"
    url_gc25 = "https://www.vlr.gg/stats/?event_group_id=76&event_id=all&region=all&min_rounds=200&min_rating=0&agent=all&map_id=all&timespan=all"
    url_gc24 = "https://www.vlr.gg/stats/?event_group_id=62&region=all&min_rounds=200&min_rating=0&agent=all&map_id=all&timespan=all"
    url_gc23 = "https://www.vlr.gg/stats/?event_group_id=38&region=all&min_rounds=200&min_rating=0&agent=all&map_id=all&timespan=all"

    # List of all scraping tasks
    scraping_tasks = [
        (url_2023, "2023", "vct"),
        (url_ch23, "2023", "chall"),
        (url_gc23, "2023", "gc"),
        (url_2024, "2024", "vct"),
        (url_ch24, "2024", "chall"),
        (url_gc24, "2024", "gc"),
        (url_2025, "2025", "vct"),
        (url_ch25, "2025", "chall"),
        (url_gc25, "2025", "gc"),
    ]

    all_data = []

    # Single progress bar for all scraping tasks
    with tqdm(
        total=len(scraping_tasks), desc="Scraping VLR data", unit=" source"
    ) as pbar:
        for url, year, group in scraping_tasks:
            pbar.set_postfix_str(f"{group} {year}")
            data = scrape_vlr_stats_for_url(url, year, group, show_progress=False)
            all_data.extend(data)
            pbar.update(1)

    # Convert the list of lists into a DataFrame
    headers = [
        "player+team",
        "rounds_played",
        "vlr_rating",
        "acs",
        "kd",
        "kast",
        "adr",
        "kpr",
        "apr",
        "dpr",
        "fkpr",
        "fdpr",
        "hs_percent",
        "cl_percent",
        "group",
        "year",
    ]
    df = pd.DataFrame(all_data, columns=headers)

    # Sort by rating in descending order, then by acs as secondary
    df.sort_values(by=["vlr_rating", "acs"], ascending=[False, False], inplace=True)

    # Write to CSV in the correct path (formula/data/vlr_data.csv)
    output_path = "formula/data/vlr_data.csv"
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Saved {len(df)} total rows (sorted by rating, acs) to {output_path}")


if __name__ == "__main__":
    scrape_all_years_combined()
