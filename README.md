# VCC TOOLS

A set of tools designed to simplify and enhance the Valorant Community Cup (VCC) tournament experience:

- **Team Maker**: Quickly create balanced teams, respecting player groupings like duos or trios, and split them into groups.
- **Playoffs Scenarios Generator** _(coming soon)_: Simulate and explore playoff outcomes.

## Table of Contents

- [Team Maker](#team-maker)
  - [Pipeline](#pipeline)
  - [Setup](#setup)
  - [CSV rating columns](#csv-rating-columns)
  - [Configuration](#configuration)
  - [Player overrides](#player-overrides)
  - [VCC scraping policy](#vcc-scraping-policy)
  - [Stack rules](#stack-rules)
  - [Output: teams.json](#output-teamsjson)
  - [Output: groups.json](#output-groupsjson)
  - [Troubleshooting](#troubleshooting)
  - [Tests](#tests)

---

## Team Maker

Builds balanced teams of 5 from the sign-up sheet, keeps stacks together, and optionally splits the teams into groups (pools) by server and strength.

### Pipeline

Run every command from the repo root.

```bash
# 1. Fill in the rank/tracker columns of the sign-up CSV and drop it into teamMaker/data/input/
# 2. CSV -> players.json (adds VCC history from vlrcommunitycup.com)
python -m teamMaker.core.generate_players            # --dry-run only checks the CSV

# 3. players.json -> teams.json + teams.png
python -m teamMaker.core.build_teams

# 4. teams.json -> groups.json + groups.png (optional)
python -m teamMaker.core.make_groups
```

Other commands:

```bash
python -m teamMaker.core.scoring                     # write player_scores*.json only
python -m teamMaker.core.scrape_distributions        # update archive/season_distributions.json
python -m teamMaker.core.scrape_distributions --refresh --dry-run
```

Outputs land in `teamMaker/output/`: `teams.json`, `teams.png`, `groups.json`, `groups.png`, `player_scores.json`, `player_scores_minimal.json`.

### Setup

```bash
pip install -r requirements.txt          # Pillow is needed for the PNGs
pip install -r requirements-dev.txt      # adds pytest
```

1. Optional: create `teamMaker/config/config.yaml` (gitignored, see [Configuration](#configuration)). Without it `config.example.yaml` is used as-is.
2. Set the season links in `teamMaker/config/player_ratings_config.yaml` (`season_map`, `seasons`, `vcc`).
3. Fill in the [rating columns](#csv-rating-columns) and put the sign-up CSV in `teamMaker/data/input/` (the most recently modified `*.csv` is used). Approved rows become active players, `substitute` rows become subs, other statuses are skipped and counted.
4. Run the pipeline above. A hand-written `teamMaker/data/players.json` also works (see `teamMaker/data/playersexample.json`).

Stacks are read from the Tally submission id (or an optional `group_override` column) and written as `group_id` in `players.json`. Names that collide become `Name (discord)` with a warning. `players_missing.json` lists players with missing ratings or VCC history, with the reason.

### CSV rating columns

Ranks and tracker.gg scores are typed into the sign-up CSV by hand (tracker.gg blocks automated lookups). Rank names are matched case-insensitively against `rank_values`.

| Column | Example | Notes |
| --- | --- | --- |
| `current_rank` | `Diamond 2` | Required. Falls back to the sign-up `rank` column. A tier without division (`Diamond`) is read as `Diamond 2` with a warning. |
| `current_rr` | `250` | Optional; only used for the ranks in `rr_granularity_ranks` (Immortal 3, Radiant). |
| `peak_rank` | `Ascendant 1` | Missing peaks count as the current rank. Warns if below the current rank. |
| `peak_rank_act` | `S25A6`, `E8A1` | Optional; feeds the Episode-era peak-act bonus. tracker.gg's `E26: A5` is read as `S26A5`. |
| `tracker_current` | `812` | tracker.gg Tracker Score (0-1000) for the current act. Without it the tracker part of the score is skipped. |
| `tracker_peak` | `950` | Tracker Score of the peak act; defaults to `tracker_current`. |

Missing required values go to `players_missing.json`; malformed values, out-of-range scores and peak-below-current are warnings. After loading VCC history, players whose rank and VCC aR disagree by `vcc_check_threshold` (default 1.5) standard deviations or more are flagged, which usually means a typo in `current_rank`. VCC history still feeds the score as before.

### Configuration

`config.yaml` may start from a profile and override only what differs:

```yaml
extends: config_tighter_teams.yaml   # path relative to this file; child wins, dicts merge
random_seed: 7
optimizer:
  restarts: 12
```

`config_tighter_teams.yaml` (which itself extends `config.example.yaml`) uses a fixed seed and a stricter optimizer (more restarts, `target_range: 0.05`, `role_weight: 0.2`). Cycles in `extends` are detected. Unknown and deprecated keys print warnings; `role_balance_weight`, `max_time`, `early_termination_threshold` and `max_restarts` are mapped to their optimizer equivalents, other old annealing keys are ignored.

`config.example.yaml` documents every key. Main groups:

- **Scoring**: `mode` (basic/advanced), `weight_current`/`weight_peak`, `use_tracker`, `weight_current_tracker`/`weight_peak_tracker`, `use_peak_act` with `peak_act_*`, `current_season`, `current_act`, `acts_per_season`, `rank_values`.
- **Returning players**: `use_returning_player_stats`, `recent_data_*_weight`, `older_data_*_weight`, `latest_season_available` (`auto` = newest season in `archive/season_distributions.json`), `use_new_player_debuff`/`new_player_debuff`.
- **Ping / region**: `use_ping_adjustment`, `ping_breakpoints`, `use_region_ping_estimates`, `region_ping_estimates`, or the flat `use_region_debuff`/`non_eu_debuff`.
- **RR granularity**: `use_immortal_rr_granularity`, `radiant_rr_threshold`, `rr_granularity_ranks`, `rr_bonus_cap`.
- **Compression**: `top_tier_compression: {enabled, knee_percentile, slope}` (off by default, never raises a score).
- **`random_seed`**: `null` draws a seed, prints it and records it in the output.
- **`optimizer:`** `iterations`, `restarts`, `time_limit_s`, `t0`, `t_end`, `target_range`, `range_weight`, `std_weight`, `role_weight`, `cluster_regions`, `cluster_max_teams`, `cluster_weight`. The objective is `range_weight*range + std_weight*std + role_weight*role_penalty + cluster_weight*extra_cluster_teams`, computed on tenths of a point. The cluster term only applies when `groups:` is configured.
- **`output:`** `teams_png` (true/false), `team_order` (`shuffled` with the seed so numbering does not reveal strength, or `by_score`).
- **`groups:`** `count`, `servers` (one per group, e.g. `[London, Frankfurt, Frankfurt]`), `sizes` (`auto` or a list), `region_server_cost` (per-player cost of a region on a server, e.g. NA on Frankfurt), `balance_weight`, `std_weight`, `mode` (`auto`, `manual`, `reuse`), `manual` (list of team id lists), `pinned` (`{team_id: group_number}`), `iterations`, `restarts`, `title`.

`reuse` mode keeps the groups in `output/groups.json` and fails loudly if the teams changed since (for example after rebuilding with another seed).

### Player overrides

Copy `teamMaker/data/overrides.example.yaml` to `teamMaker/data/overrides.yaml` (gitignored). `generate_players` applies it after fetching data, keyed by player name or discord handle (case-insensitive). Allowed fields: `ping`, `group_id`, `status` (`active`, `substitute`, `skip`), `role`, `region`. This replaces editing `players.json` by hand.

### VCC scraping policy

- The vlrcommunitycup.com page only renders the top 60 rows in HTML; the full table is read from the embedded page payload. The HTML table is a truncated fallback only.
- `vcc.stats_stage: all` uses every stage of a season. Each season page is fetched once per run.
- Seasons already in `archive/season_distributions.json` are **frozen**. `scrape_distributions` only adds new seasons unless `vcc.refresh_existing: true` or `--refresh` is given, and even then a write with fewer values or one from the HTML fallback is refused. A per-season diff is printed; `--dry-run` writes nothing.
- To add a season, add it to `season_map` and `seasons` in `player_ratings_config.yaml`, then run `scrape_distributions`.

### Stack rules

Allowed stack sizes are **1, 2, 3 and 5**. A 5-stack becomes a fixed team and is not optimized. **A 4-stack is invalid**: those players are excluded with a reason (stacks above 5 too), and `generate_players` warns about it and about stacks whose declared size does not match the members found. Fix it with a `group_id` override or by changing the CSV. A missing `group_id` is treated as a solo with a warning. Extra players (total mod 5) become subs, keeping returning players and dropping the latest sign-ups first.

### Output: teams.json

`schema_version: 1`. The same seed and config give an identical file except `meta.generated_at`.

- `meta`: `generated_at`, `seed`, `seed_generated`, `config_sources`, `players_file`, `optimizer`, `stopped_by`, `restarts`.
- `summary`: `teams`, `players_in_teams`, `optimized` and `all` (min, max, range, mean, std), `role_penalty` (optimized, fixed, total), `cluster`.
- `teams[]`: `id`, `name`, `fixed`, `formation`, `team_score`, `avg_score`, `role_penalty`, `regions`, `players[]` (`name`, `discord`, `score`, `role`, `assigned_role`, `region`, `server`, `formation`, `stack_id`, `is_returning`, `ranks`).
- `subs[]` and `excluded[]`, each with a `reason`.

### Output: groups.json

Written by `make_groups`: the team ids per group, server and mean score per group, total region cost, and the seed used. The console output lists the number of NA teams outside London.

### Troubleshooting

- **Players end up in `players_missing.json`**: a rating column is empty or holds an unknown rank (fix the CSV), or a returning player has no valid VCC profile link or was not found in a season's stats. Run `generate_players --dry-run` to check the CSV without VCC requests.
- **Different result on each run**: `random_seed: null` draws a new seed every time. Set an integer seed (printed at the start of the run and stored in `teams.json`) to reproduce a run. Runs bound by `iterations` are reproducible; runs stopped by `time_limit_s` can differ slightly.
- **No PNG**: install Pillow (`pip install -r requirements.txt`); the JSON and the console report are always written.
- **`make_groups` reuse error**: the teams changed since `groups.json` was written; use `mode: auto` or rerun in the right order.

### Tests

```bash
python -m pytest
```

Tests use synthetic data only and never call vlrcommunitycup.com.

---
