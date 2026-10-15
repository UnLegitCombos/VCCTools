"""Does Tracker (relative to rank), being new, or region/ping explain VCC results beyond rank? Read-only."""

import json
import math
import statistics as st
import sys
import warnings

sys.path.insert(0, ".")
warnings.simplefilter("ignore")

from teamMaker.core.config import load_team_config  # noqa: E402
from teamMaker.core.scoring import (  # noqa: E402
    _get_season_distributions,
    normalize_season,
    rank_to_numeric,
)

cfg = load_team_config()
with open(sys.argv[1], encoding="utf-8") as f:
    players = json.load(f)
dist = _get_season_distributions(cfg)
rv = cfg["rank_values"]


def fit(xs, ys):
    mx, my = st.mean(xs), st.mean(ys)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return b, my - b * mx


def corr(a, b):
    ma, mb = st.mean(a), st.mean(b)
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / math.sqrt(
        sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))


def season_z(entry):
    s = normalize_season(entry.get("season", ""))
    r = float(entry.get("adjusted_rating", 0) or 0)
    d = dist.get(s)
    if not d or r <= 0 or len(d) < 10:
        return None
    m, sd = st.mean(d), st.pstdev(d)
    return (r - m) / sd if sd else None


# 1. Tracker is lobby-relative: how does it relate to rank?
allp = [p for p in players.values() if p.get("tracker_current") is not None and rank_to_numeric(p.get("current_rank", ""), rv)]
rank = [rank_to_numeric(p["current_rank"], rv) for p in allp]
trk = [float(p["tracker_current"]) for p in allp]
print(f"all players n={len(allp)}: corr(rank, tracker_current) = {corr(rank, trk):.2f}")
bt, at = fit(rank, trk)
print(f"  expected tracker = {at:.0f} + {bt:.1f} x rank value (Gold 1=10, Asc 1=19, Imm 1=23)")

# 2. VCC history (recency-weighted z) vs rank, Tracker residual, newness, region.
rows = []
for p in players.values():
    hist = p.get("previous_season_stats") or []
    if isinstance(hist, dict):
        hist = [hist]
    zs = [(normalize_season(e.get("season", "")), season_z(e)) for e in hist]
    zs = [(s, z) for s, z in zs if z is not None]
    r = rank_to_numeric(p.get("current_rank", ""), rv)
    if not zs or not r or p.get("tracker_current") is None:
        continue
    zs.sort(key=lambda t: int(t[0][1:]))
    rows.append({"rank": r, "tracker": float(p["tracker_current"]), "latest": zs[-1][1], "first": zs[0][1],
                     "n": len(zs), "region": (p.get("region") or "EU").upper()})
print(f"\nreturning players with usable history: {len(rows)}")
b, a = fit([r["rank"] for r in rows], [r["latest"] for r in rows])
for r in rows:
    r["resid"] = r["latest"] - (a + b * r["rank"])
    r["tres"] = r["tracker"] - (at + bt * r["rank"])
print(f"latest VCC z ~ rank: slope {b:.3f} z per rank point, corr {corr([r['rank'] for r in rows], [r['latest'] for r in rows]):.2f}")
print(f"raw tracker vs VCC z: corr {corr([r['tracker'] for r in rows], [r['latest'] for r in rows]):.2f}")
print(f"tracker ABOVE/BELOW what's normal for the rank vs VCC result beyond rank: corr {corr([r['tres'] for r in rows], [r['resid'] for r in rows]):.2f}")
bb, _ = fit([r["tres"] for r in rows], [r["resid"] for r in rows])
print(f"  => +100 tracker above normal for rank ~ {bb * 100:+.2f} VCC z ~ {bb * 100 / b:+.2f} rank points")

# 3. New players: first VCC season vs later seasons, relative to (current) rank.
multi = [r for r in rows if r["n"] >= 2]
fr = [r["first"] - (a + b * r["rank"]) for r in multi]
print(f"\nplayers with 2+ seasons (n={len(multi)}): first-season result vs rank-expected: mean {st.mean(fr):+.2f} z "
      f"(latest season: {st.mean(r['resid'] for r in multi):+.2f} z) -> first season minus latest = "
      f"{st.mean(r['first'] - r['latest'] for r in multi):+.2f} z = {st.mean(r['first'] - r['latest'] for r in multi) / b:+.2f} rank points")
one = [r for r in rows if r["n"] == 1]
print(f"players whose only season so far was their first (n={len(one)}): result vs rank-expected {st.mean(r['resid'] for r in one):+.2f} z "
      f"= {st.mean(r['resid'] for r in one) / b:+.2f} rank points")

# 4. Region (ping proxy).
for reg in sorted({r["region"] for r in rows}):
    g = [r["resid"] for r in rows if r["region"] == reg]
    print(f"region {reg:5} n={len(g):2}: VCC result vs rank-expected {st.mean(g):+.2f} z = {st.mean(g) / b:+.2f} rank points")
print("players with an actual ping value:", sum(1 for p in players.values() if p.get("ping") not in (None, "")))
