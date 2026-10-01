import re

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
    match = re.search(r"/player/([a-f0-9-]{36})(?:[/?#]|$)", url)
    return match.group(1) if match else None


def parse_roles(roles_str):
    return [r.strip().lower() for r in roles_str.split(",") if r.strip()]
