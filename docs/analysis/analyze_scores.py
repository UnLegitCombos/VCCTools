"""Read-only analysis of the current scoring on the real players.json (run from repo root)."""

import json
import math
import statistics as st
import sys
import warnings

sys.path.insert(0, ".")
warnings.simplefilter("ignore")

from teamMaker.core import build_teams  # noqa: E402
from teamMaker.core.config import load_team_config  # noqa: E402
from teamMaker.core.scoring import (  # noqa: E402
    VCC_SCORE_BASE,
    VCC_SCORE_PER_SD,
    compute_player_score_detailed,
    score_players,
)

cfg = load_team_config()
with open(sys.argv[1] if len(sys.argv) > 1 else build_teams._players_path(cfg), encoding="utf-8") as f:
    players = json.load(f)
statuses = {}
for p in players.values():
    statuses[p.get("status")] = statuses.get(p.get("status"), 0) + 1
print("players", len(players), "statuses", statuses)
print("vcc_mode:", cfg["vcc_mode"], "| compression:", cfg["top_tier_compression"].get("enabled"))
print("rank_values:", {k: v for k, v in cfg["rank_values"].items() if k.split()[0] in ("Gold", "Platinum", "Diamond", "Ascendant", "Immortal", "Radiant")})


def corr(a, b):
    ma, mb = st.mean(a), st.mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    return num / math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))


rows = []
for name, p in players.items():
    bd = compute_player_score_detailed(p, cfg)
    rc, tc, adv = bd["rank_components"], bd["tracker_components"], bd["advanced_components"]
    base = bd["base_score"]
    tracker_add = tc.get("tracker_total", 0.0)
    cons = tc.get("consistency_factor", 1.0)
    pre_vcc = (base + tracker_add) * cons
    prev = adv.get("previous_season", {})
    post_vcc = prev.get("post_blend_score", pre_vcc) if prev.get("enabled") and prev.get("previous_season_score") else pre_vcc
    ping = adv.get("ping_adjustment", {})
    rows.append({
        "name": name, "rank": p.get("current_rank"), "peak": p.get("peak_rank"),
        "cur": rc["current_rank_value"], "peakv": rc["peak_rank_value"], "base": base,
        "tracker": tracker_add, "cons_gain": (base + tracker_add) * (cons - 1), "pre_vcc": pre_vcc,
        "vcc_delta": post_vcc - pre_vcc, "vcc_score": prev.get("previous_season_score") or None,
        "ping_mult": ping.get("multiplier", 1.0), "region": p.get("region"),
        "returning": p.get("is_returning_player", False), "final": bd["final_score"],
    })

final = score_players(players, cfg)
if isinstance(final, tuple):
    final = final[0]
for r in rows:
    r["published"] = final[r["name"]] if isinstance(final, dict) else r["final"]

f = [r["final"] for r in rows]
print(f"\nfinal score: min {min(f):.1f} median {st.median(f):.1f} max {max(f):.1f} sd {st.pstdev(f):.2f}")
print(f"rank part (0.7 cur + 0.3 peak): mean {st.mean(r['base'] for r in rows):.1f}")
print(f"tracker add-on: mean {st.mean(r['tracker'] for r in rows):.2f}, range {min(r['tracker'] for r in rows):.2f}-{max(r['tracker'] for r in rows):.2f}")
print(f"consistency gain: mean {st.mean(r['cons_gain'] for r in rows):.2f}, range {min(r['cons_gain'] for r in rows):.2f}-{max(r['cons_gain'] for r in rows):.2f}")
v = [r["vcc_delta"] for r in rows if r["returning"]]
print(f"VCC history change (returning, n={len(v)}): mean {st.mean(v):+.2f}, range {min(v):+.2f} to {max(v):+.2f}, |.|>2: {sum(abs(x) > 2 for x in v)}")
pm = [r for r in rows if r["ping_mult"] < 1]
print(f"ping penalty applied to {len(pm)} players (regions {sorted({r['region'] for r in pm})}), multipliers {sorted({round(r['ping_mult'], 3) for r in pm})}")
print(f"new players (x0.95): {sum(not r['returning'] for r in rows)}")

ret = [r for r in rows if r["vcc_score"]]
pre = [r["pre_vcc"] for r in ret]
vz = [(r["vcc_score"] - VCC_SCORE_BASE) / VCC_SCORE_PER_SD for r in ret]
print(f"\ncorr(ranked score, VCC history z) over {len(ret)} returning: {corr(pre, vz):.2f}")
print(f"corr(current rank value, VCC z): {corr([r['cur'] for r in ret], vz):.2f}")
print(f"corr(tracker add-on, VCC z): {corr([r['tracker'] for r in ret], vz):.2f}")

# How far apart do 'similar' players end up: rank ties with different final scores.
by_rank = {}
for r in rows:
    by_rank.setdefault(r["rank"], []).append(r["final"])
print("\nspread of final score within the same current rank (n>=3):")
for k, vals in sorted(by_rank.items(), key=lambda kv: -st.mean(kv[1])):
    if len(vals) >= 3:
        print(f"  {k:14} n={len(vals):2} {min(vals):5.1f} - {max(vals):5.1f}")

# Missing tracker effect on a median player.
mid = sorted(rows, key=lambda r: r["final"])[len(rows) // 2]
p = dict(players[mid["name"]])
p["tracker_current"] = p["tracker_peak"] = None
print(f"\nmedian player {mid['final']:.1f} -> without Tracker {compute_player_score_detailed(p, cfg)['final_score']:.1f}")

# blend vs ranked_baseline.
alt = {**cfg, "vcc_mode": "ranked_baseline"}
fb = {n: compute_player_score_detailed(players[n], alt)["final_score"] for n in players}
below = [r for r in ret if r["final"] < r["pre_vcc"] * (1 if r["returning"] else 0.95) * r["ping_mult"] - 1e-9]
print(f"\nreturning with history scored below their ranked-only score: {len(below)} of {len(ret)}")
d = [r["final"] - fb[r["name"]] for r in rows]
print(f"blend minus ranked_baseline: mean {st.mean(d):+.2f}, min {min(d):+.2f}, max {max(d):+.2f}")
print("players crossing the 24 line between modes:", sum((r["final"] >= 24) != (fb[r["name"]] >= 24) for r in rows))
print("missing tracker_current:", sum(1 for p in players.values() if p.get("tracker_current") in (None, "")))

# Peak-act history length per returning player.
hist = [len(p.get("previous_season_stats") or []) for p in players.values() if p.get("is_returning_player")]
print("VCC seasons per returning player:", {n: hist.count(n) for n in sorted(set(hist))})
