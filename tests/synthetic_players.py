"""Synthetic player data for optimizer tests (no real players)."""

import random

ROLE_SETS = [
    ["duelist"],
    ["initiator"],
    ["controller", "sentinel"],
    ["sentinel"],
    ["duelist", "initiator"],
    ["flex"],
]
REGIONS = ["EU"] * 7 + ["NA"] * 2 + ["MENA"]


def make_players(sizes, seed=0, subs=0):
    """Build a players dict from a list of stack sizes.

    Args:
        sizes: Stack size of each group, in signup order.
        seed: Seed for scores, roles and regions.
        subs: Number of extra players with a null group_id.

    Returns:
        Tuple (players, scores) with synthetic names ``P000``...
    """
    rng = random.Random(seed)
    players = {}
    scores = {}
    n = 0
    for gid, size in enumerate(sizes, start=1):
        for _ in range(size):
            name = f"P{n:03d}"
            players[name] = {
                "current_rank": "Gold 1",
                "peak_rank": "Gold 2",
                "group_id": gid,
                "role": rng.choice(ROLE_SETS),
                "region": rng.choice(REGIONS),
                "is_returning_player": rng.random() < 0.5,
                "signup_index": n,
            }
            scores[name] = round(rng.uniform(5, 35), 1)
            n += 1
    for _ in range(subs):
        name = f"P{n:03d}"
        players[name] = {
            "group_id": None,
            "role": rng.choice(ROLE_SETS),
            "region": "EU",
            "is_returning_player": False,
            "signup_index": n,
        }
        scores[name] = round(rng.uniform(5, 35), 1)
        n += 1
    return players, scores
