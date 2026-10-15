# VCC Tools redesign (S16)

> **Parked until after the UI work (2026-10-15).** The rating math and engine changes go into Season 16 on the `dev16` branch. The UI rethink and finishing S15 happen on `dev`. This file keeps the questions, the analysis and the partial answers so the work can pick up where it stopped. Every answer below can still be argued; none of them is final.

Everything gets rethought one layer at a time, foundations first. Each layer covers what happens today, what the real data says, the options, and the decisions to make.

| Layer | Status |
|---|---|
| 1. What the tool must achieve | mostly decided |
| 2. Player rating math | partly decided; see "Where it stopped" |
| 3. Team balancing | not started |
| 4. Groups | not started (note: the friend's S15 config turns `groups:` on and lowers `na_server_min_share` from 0.5 to 0.2) |
| 5. Data and season workflow | not started |
| 6. Settings | not started |
| 7. App and how the organiser uses it | moved: being done now for S15 on `dev` |
| 8. Code and migration | not started |

## Where it stopped

Open questions, in the order to pick them up:

1. **D3, VCC history.** Proposal: instead of averaging two scales (which today lowers all 68 returning players by 2.7 on average), move a rating up or down by how much the player beat or missed what their rank predicts. The user asked why; the answer is under "Decisions so far". No decision yet.
2. **D4, Tracker.** The user's position: refit it, and a missing value should warn or error. Counter-proposal from the analysis: Tracker above or below normal for the rank shows no signal (corr 0.02), so take it out of the rating and use it only as a flag for underranked players or smurfs (a Tracker far above normal for the rank), which feeds section 7 adjustments. A missing value then just means no flag. Undecided.
3. **D5, new players.** The user's position: keep a factor, because new players are an unknown. Counter-argument: uncertain doesn't mean worse; the data suggests 0 to ~0.5 points, not today's ~1.2; uncertainty could instead be handled in team balancing (don't put several unknowns on one team). Undecided on the size.
4. **D7, manual adjustments.** Raw-field edits (the user: they show how big a change really is) vs a separate adjustment field (survives rating updates, keeps the real ranks). Middle ground: an adjustment field shown as "≈ N divisions". Under discussion.
5. **D2 scope.** Fit only 3 or 4 weights; 68 players with history aren't enough for a full model.
6. Then layers 3 to 8.

Scripts used for the numbers (read-only, run from the repo root with the players.json path as argument): `analyze_scores.py` and `analyze_residuals.py`. They are saved in `docs/analysis/`.

Data used: the S15 data from the friend's copy (`teamMaker/data/players.json`, 99 players: 82 active, 14 substitutes, 3 with no status; 73 returning, 68 with usable VCC history; and the 99-row S15 sign-up CSV), scored with the current `dev` code and `config.yaml` (`vcc_mode: blend`). Nothing was changed. Rulebook: "VLR Community Cup EMEA Official Tournament Guidelines, Autumn 2026".

---

## Layer 1: What the tool must achieve

The tool has two jobs.

**A. Make the teams.** Once per season: about 80 to 100 sign-ups go into teams of five, with duos and trios kept together. The result is published, so it has to look fair and be hard to argue with.

**B. Publish the rating list.** Computed once before the teams, then usually twice more during the season. Under the rulebook, this list is how substitutes are chosen, and the limits are written in absolute rating points:

| Rule (rulebook sections 2 and 8) | Points |
|---|---|
| Player rated below 24: sub may be up to | +2.0 |
| ...but never a sub rated above | 25.5 |
| Player rated 24 or more: sub may be up to | +1.5 |
| Several subs: total upgrade at most | +3.0 |
| A sub this far above loses the map | +2.01 |

Section 7 also allows adjusting a rating "if actual performance differs excessively from registered rank".

**The rating scale is part of the rulebook.** Today scores run from 12.2 to 35.4 with a median of 23.9. One rank division is worth about 1 point (2 points from Immortal up). So "+2" means about two divisions below Immortal and one division above it. If the math changes the scale, either the rulebook numbers change too, or the new math is pinned to the same scale.

What the rating list needs:

- **Explainable:** a player can see why they got their number.
- **Stable:** between updates, a rating moves only when the player's inputs change, never because of side effects in the formula.
- **Means strength:** two players 2 points apart are one or two divisions apart in skill. Anything that isn't skill (ping, being new) is a visible separate line, not hidden inside the number.
- **Manual adjustments** (section 7) are a normal, recorded step, with a reason.

For whoever runs it: the organiser may not be technical, so nothing should require editing YAML or reading code.

---

## Layer 2: Player rating math

### Today, step by step

From `teamMaker/core/scoring.py`, `compute_player_score_detailed`, in this order. The average effect is measured on the 99 S15 players.

| Step | Formula | Average effect |
|---|---|---|
| Rank | 0.7 × current + 0.3 × peak; an old peak fades 10% per act toward the current rank | 19.0 |
| + Tracker | 0.2·√(tracker current) + 0.1·log(1 + tracker peak) | +4.9 (2.6 to 6.8) |
| × Consistency | the **whole** score × (1 + 0.1 × current/peak) | +2.2 |
| VCC history, `blend` (returning players) | 0.65 × score + 0.35 × history score | −2.7 (down to −6.1) |
| × Ping (16 NA/MENA players) | ×0.84 to ×0.95 | up to −16% |
| × New player (26 players) | ×0.95 | about −1.2 |

Top-tier compression is in the code but switched off.

### What the data shows

**1. VCC history works as a penalty, not a correction.** `blend` mixes 65% of the ranked score with 35% of a VCC-history score that sits on a different scale: it is centred on 18, while ranked scores for returning players average about 27. As a result, **all 68** returning players with history end up *below* their ranked-only score, by 2.7 points on average and up to 6.1. New players only lose about 1.2 points from their ×0.95. So a veteran is rated lower than a newcomer with the same rank, which is the opposite of the intent. Switching to the existing `ranked_baseline` mode moves **17 players** across the 24 line, the rulebook's sub threshold.

**2. Tracker Score adds ~5 points but barely predicts VCC results.** Across the 68 returning players, the ranked score correlates 0.88 with their VCC history, and current rank alone 0.87. The Tracker part alone correlates only **0.12**. So the Tracker term moves ratings by up to ~4 points between players of the same rank, based on something that hardly tracks how they actually play in VCC.

**3. A missing Tracker Score costs about 6 points.** Without Tracker Scores, the median player drops from 23.9 to 17.8, which is three sub windows. Everyone has one this season, but a late sign-up without one would get a badly wrong rating. Missing data should count as "unknown", not as "bad".

**4. Nobody can say what one point is.** Added terms (Tracker) and multipliers (consistency, ping, new player) are mixed, so the order of the steps changes the result, and each step has its own hand-picked constant. The consistency factor also multiplies the rank part, not just the Tracker part.

**5. Hidden multipliers distort substitutions.** The ×0.95 new-player factor and the ping factor (as low as ×0.84) make those players look weaker on the list than they are. Under the sub rules, a team can pick a new or high-ping player as a "cheap" sub who is actually stronger. A team replacing a new player also gets a tighter limit than it should.

**6. The same rank gives very different ratings.** For example Ascendant 1 runs from 17.4 to 26.0, and Immortal 2 from 27.0 to 35.4. Part of that is real information (peak, history); a lot of it is the effects above.

**What works:** rank predicts VCC performance well. So the core idea is sound: rank first, with VCC history as a correction. What's broken is everything added on top of it.

### Options

- **A. Patch.** Keep the formula; switch to `ranked_baseline`, treat a missing Tracker Score as neutral, drop the consistency factor. Smallest change, but it keeps the many hand-picked constants.
- **B. Rebuild on one scale (recommended).** Rating = rank points + a small Tracker adjustment + a capped VCC-history correction, all added together and on today's point scale, so the rulebook numbers keep their meaning. Every part shows on the player's breakdown. Ping and the new-player factor come out of the published number (D5, D6). The weights are checked against the returning players' VCC history rather than guessed, which given finding 2 probably means a much smaller Tracker weight.
- **C. Statistical model.** Fit every weight from past seasons, so a rating is the predicted VCC strength with an uncertainty range. The most rigorous option, but it rests on only about 68 players with history, and it's harder to explain to players.

### Decisions for layers 1 and 2

Answer with something like "D1 yes, D2 B, rest as recommended", or change any of them.

| # | Decision | Recommendation |
|---|---|---|
| D1 | Keep today's point scale (median ~24, ~1 point per division, 2 from Immortal) so the rulebook sub limits stay valid? | Yes |
| D2 | Which approach to the math: A, B or C? | B, with weights checked against past data |
| D3 | VCC history becomes a capped correction that moves a rating up *or* down by how much the player beat or missed what their rank predicts (the `ranked_baseline` idea). `blend` is removed. | Yes; the cap is still to be set (today ±3) |
| D4 | Tracker: shrink it to what the data supports, and treat a missing Tracker Score as neutral plus a "provisional" flag. | Yes |
| D5 | New-player ×0.95: remove it from the published rating and mark new players as "provisional" on the list. Section 7 adjustments fix real misratings mid-season. | Remove. Handle newness in team balancing (layer 3) if needed |
| D6 | Ping: take it out of the published rating and use it only in team balancing and server choice? | Yes. Rate skill; ping is a team and server matter, and the rulebook already has a server rule |
| D7 | Manual adjustments (section 7) become their own field: "± points, reason, date", shown on the list, instead of editing raw rank fields. | Yes |
| D8 | Team balancing uses the same number as the published list. | Yes; ping and newness become separate balancing rules in layer 3 |

### Follow-up analysis (after first review)

Tracker Score is measured against the player's own lobby (a Gold-skill player scores ~700 in Silver lobbies and ~450 in Gold), so the useful signal should be "Tracker above or below what's normal for the rank". Tested on the S15 data:

| Check | Result |
|---|---|
| Tracker vs rank, all 99 players | corr 0.20; expected Tracker ≈ 305 + 9.4 × rank value (~400 at Gold 1, ~520 at Immortal 1) |
| Tracker above/below normal for rank vs VCC result beyond rank (68 returning) | corr **0.02**: no measurable signal (caveat: Tracker is from this act, VCC history from past seasons) |
| First VCC season vs latest, players with 2+ seasons (51) | first season ~0.8 rank points worse (biased: compared against today's rank) |
| Players whose only season was their first (17) | +0.4 rank points better than rank predicts |
| VCC result vs rank-expected, by region | EU +0.4 (57), NA **−1.6** (9), MENA −3.8 (2) rank points |
| Players with a real ping value | 11 of 99 (others use region estimates) |

### Decisions so far

| # | Decision |
|---|---|
| D1 | Scale can be revisited; any change comes with matching rulebook numbers |
| D2 | B + C: weights fitted from data (C), published as a simple additive formula (B) |
| D3 | **Open.** Proposal: VCC history moves a rating up or down by how much the player beat or missed what their rank predicts, instead of averaging two scales (which today lowers all 68 returning players) |
| D4 | Tracker refit (expect a weight near zero); missing Tracker blocks the run with a clear warning until fixed or confirmed |
| D5 | Keep a new-player factor, sized by data (evidence suggests 0 to ~0.8, today ~1.2); uncertainty can also be handled in team balancing |
| D6 | Keep ping, sized by data (NA ~−1.6 rank points observed; today ×0.9 ≈ −2.5) |
| D7 | **Open.** Raw-field edits (shows how big a change is) vs a separate adjustment field (survives rating updates, keeps true ranks). Middle ground: adjustment field shown as "≈ N divisions" |
| D8 | Team balancing uses the same published number, ping and newness included |
