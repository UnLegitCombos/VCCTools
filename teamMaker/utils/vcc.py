import re
import urllib.request

from bs4 import BeautifulSoup

# Fetch the VCC season stats page and return (aR, rounds) for the player.
# Returns (None, None) if the player or columns are not found.
def scrape_adjusted_rating(player_id, season_url, user_agent):
    req = urllib.request.Request(season_url, headers={"User-Agent": user_agent})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"  [vcc] fetch error for {season_url}: {e}")
        return None, None

    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if not table:
        print(f"  [vcc] no table found at {season_url}")
        return None, None

    thead = table.find("thead")
    ths = thead.find("tr").find_all("th") if thead else []
    ar_idx = None
    rounds_idx = None
    for i, th in enumerate(ths):
        title = th.get("title", "")
        text = th.get_text(strip=True)
        if "Adjusted rating" in title or text == "aR":
            ar_idx = i
        if text == "RND":
            rounds_idx = i

    if ar_idx is None:
        print(f"  [vcc] aR column not found at {season_url}")
        return None, None

    tbody = table.find("tbody")
    if not tbody:
        return None, None

    for row in tbody.find_all("tr"):
        link = row.find("a", href=re.compile(f"/player/{re.escape(player_id)}/"))
        if link:
            tds = row.find_all("td")
            ar = None
            rounds = None
            if ar_idx < len(tds):
                try:
                    ar = float(tds[ar_idx].get_text(strip=True))
                except ValueError:
                    pass
            if rounds_idx is not None and rounds_idx < len(tds):
                try:
                    rounds = int(tds[rounds_idx].get_text(strip=True))
                except ValueError:
                    pass
            return ar, rounds

    print(f"  [vcc] player {player_id} not found in table")
    return None, None
