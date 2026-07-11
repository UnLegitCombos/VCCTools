import re
from urllib.parse import unquote

# Returns (name, tag) from a tracker.gg URL, e.g. /riot/yukky%23zzz -> ("yukky", "zzz")
def parse_tracker_url(url):
    if not url:
        return None, None
    match = re.search(r"/riot/([^/?#]+)", url)
    if match:
        decoded = unquote(match.group(1))
        if "#" in decoded:
            name, tag = decoded.split("#", 1)
            return name.strip(), tag.strip()
    return None, None

# Extracts the slug name from vlrcommunitycup.com/player/{uuid}/{name}
def parse_vcc_player_name(url):
    if not url:
        return None
    match = re.search(r"/player/[^/]+/([^/?#]+)", url)
    return match.group(1) if match else None


# Extracts the UUID from vlrcommunitycup.com/player/{uuid}/{name}
def parse_vcc_player_id(url):
    if not url:
        return None
    match = re.search(r"/player/([a-f0-9-]{36})/", url)
    return match.group(1) if match else None


def parse_roles(roles_str):
    return [r.strip().lower() for r in roles_str.split(",") if r.strip()]
