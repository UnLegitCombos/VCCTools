# VCC Tools

Tools for running the Valorant Community Cup (VCC). The **Team Maker** turns the sign-up sheet into balanced teams of 5, keeps duos and trios together, and splits the teams into groups.

## Contents

- [Step by step](#step-by-step)
- [Commands](#commands)
- [The sign-up CSV](#the-sign-up-csv)
- [How players are scored](#how-players-are-scored)
- [How teams and groups are made](#how-teams-and-groups-are-made)
- [Configuration](#configuration)
- [Output files](#output-files)
- [Troubleshooting](#troubleshooting)
- [Tests](#tests)

## Step by step

Run every command from the repo root.

### First time on a machine

```bash
pip install -r requirements.txt        # requirements-dev.txt as well for the tests
```

Settings live in `teamMaker/config/config.yaml`. Without that file `config.example.yaml` is used, and it already matches the defaults.

### Once per VCC season

1. Register the season that just ended, so returning players' VCC history counts. In `teamMaker/config/player_ratings_config.yaml`:

   ```yaml
   season_map:
     "EMEA Season 15": "S_15"     # exactly as the sign-up form lists it in previous_seasons

   seasons:
     S_15:
       id: "<event id>"
       code: "S15"
       link: "https://vlrcommunitycup.com/events/<event id>/...?tab=stats"
   ```

2. Archive its stats (preview first with `--dry-run`):

   ```bash
   python -m teamMaker.core.scrape_distributions
   ```

3. In `config.yaml`, set `random_seed` to the new VCC season number, and `current_season` / `current_act` to the current Valorant season and act (used to age peak ranks).
4. Remove last season's CSV from `teamMaker/data/input/`.
5. If NA players signed up, uncomment the `groups:` block in `config.yaml` (see [groups](#how-teams-and-groups-are-made)).

### Every time you make teams

1. Export the sign-up sheet as CSV and set each row's `status` (see [statuses](#statuses)).
2. Fill in the [rating columns](#rating-columns) for every player who plays. The `tracker` column has each player's profile link.
3. Put the CSV in `teamMaker/data/input/`.
4. Check the CSV, fix what it reports in the sheet, and repeat until it is clean:

   ```bash
   python -m teamMaker.core.generate_players --dry-run
   ```

5. Build the player list. This loads VCC history and writes `teamMaker/data/players.json`. A warning that a rank disagrees with the player's VCC history usually means a typo.

   ```bash
   python -m teamMaker.core.generate_players
   ```

6. Build the teams, then check `teamMaker/output/teams.png` and the subs listed in the console:

   ```bash
   python -m teamMaker.core.build_teams              # add --tighter for closer team totals (slower)
   ```

7. Optional: split the teams into groups (`groups.png`):

   ```bash
   python -m teamMaker.core.make_groups
   ```

8. Publish `teams.png` (and `groups.png`) and keep `teams.json`. A rerun gives very similar, but not always identical, teams.

### Late sign-ups

1. Add them to the sheet and replace the CSV in `teamMaker/data/input/`.
2. Add only the new players, leaving everyone already in `players.json` untouched, then fill in their ratings:

   ```bash
   python -m teamMaker.core.generate_players --keep-existing
   ```

3. Rebuilding the teams reshuffles everyone, so only do it while the teams can still change.

## Commands

| Command | Does | Options |
| --- | --- | --- |
| `generate_players` | Sign-up CSV → `players.json` (with VCC history) | `--dry-run` checks the CSV only; `--keep-existing` only adds new sign-ups |
| `build_teams` | `players.json` → `teams.json`, `teams.png` | `--tighter` for closer team totals |
| `make_groups` | `teams.json` → `groups.json`, `groups.png` | |
| `scrape_distributions` | Archives a VCC season's ratings | `--dry-run`, `--refresh` |
| `scoring` | Writes the player scores only | |

Run each as `python -m teamMaker.core.<command>`. `players.json` can also be written by hand; see `teamMaker/data/playersexample.json`.

## The sign-up CSV

The newest `*.csv` in `teamMaker/data/input/` is used.

### Statuses

| `status` | Result |
| --- | --- |
| `Denied`, `Investigate` | Left out |
| `Substitute` | Substitute |
| Anything else (`Approved`, `Pending`, `Pending Rating`, ...) | Plays |

If the same Discord signed up more than once, only their latest row counts.

### Stacks

Players from the same Tally submission are a stack. To merge rows by hand, give them the same value in an optional `group_override` column. Stacks of **1, 2, 3 or 5** are allowed: a 5-stack is a fixed team, and a **4-stack is invalid** (its players are left out, with a warning).

### Rating columns

Typed in by hand, because tracker.gg blocks automated lookups. Rank names ignore case and spacing.

| Column | Example | Notes |
| --- | --- | --- |
| `current_rank` | `Diamond 2` | Required. Falls back to the sign-up `rank` column. A tier alone (`Diamond`) counts as division 2, with a warning. |
| `current_rr` | `250` | Only used for Immortal 3 and Radiant. |
| `peak_rank` | `Ascendant 1` | Blank counts as the current rank. |
| `peak_rank_act` | `S25A6`, `E8A1` | When the peak was. tracker.gg's `E26: A5` is read as `S26A5`. |
| `tracker_current` | `812` | Tracker Score (0-1000) for the current act. |
| `tracker_peak` | `950` | Tracker Score of the peak act. Blank uses `tracker_current`. |
| `ping` | `45` | Optional, in ms. Blank uses an estimate for the player's server. |

Missing ratings are listed in `teamMaker/data/players_missing.json`; typos and odd values are printed as warnings.

## How players are scored

1. **Rank**: 70% current rank + 30% peak rank. An older peak counts less: a peak above the current rank slides 10% of the way back toward it for every act since `peak_rank_act`.
2. **Tracker**: the Tracker Scores add a smaller part on top.
3. **VCC history** (returning players): blended in at 35% when they played the latest archived season, 25% when their last season is older. Each season's adjusted rating is compared against that whole season.
4. **New players** (no VCC history): × 0.95.
5. **Ping**: high ping lowers the score, from nothing up to 70 ms to 20% at 200 ms.

## How teams and groups are made

**Teams**: `build_teams` keeps stacks together and searches for the split with the closest team totals and a sensible role spread. It stops after 3 minutes, or earlier once all teams are within 0.1 points with a perfect role spread. If the players do not divide into teams of 5, the extras become subs: new players and the latest sign-ups first. `--tighter` waits for 0.05 points (up to 5 minutes) and favours strength over roles.

**Groups**: `make_groups` splits the teams into 3 groups with the closest average strength. Every group plays on **Frankfurt**. If NA teams can be gathered into one group while the group averages stay within 1 point of the most even split, and that group ends up at least half NA players, it plays on **London**.

The `groups:` block in `config.yaml` is commented out for now. That is fine for groups, but uncommenting it also makes `build_teams` keep NA players on as few teams as one group can hold, which makes a London group far more likely. Uncomment it in seasons with NA players.

## Configuration

| File | Holds |
| --- | --- |
| `teamMaker/config/config.yaml` | Scoring, optimizer, output and group settings. Gitignored; the built-in defaults and `config.example.yaml` match it. |
| `teamMaker/config/player_ratings_config.yaml` | VCC seasons and links, server names, sign-up checks. |
| `teamMaker/config/config_tighter_teams.yaml` | The optimizer settings `--tighter` applies. |

The main settings in `config.yaml`:

| Setting | Default | What it does |
| --- | --- | --- |
| `weight_current`, `weight_peak` | 0.7, 0.3 | Share of current and peak rank |
| `peak_act_decay_rate` | 0.9 | How much of an old peak's lead over the current rank is kept per act |
| `current_season`, `current_act` | 26, 5 | The current Valorant act, used to age peaks |
| `recent_data_previous_weight` | 0.35 | VCC history share for recent returning players (`older_data_previous_weight`: 0.25) |
| `new_player_debuff` | 0.95 | Multiplier for players without VCC history |
| `ping_breakpoints` | 70 ms → 0% ... 200 ms → 20% | Ping penalty curve |
| `random_seed` | 15 | The VCC season number; `null` draws a new seed each run |
| `optimizer.time_limit_s`, `optimizer.target_range` | 180, 0.1 | When team building stops |
| `groups.na_server`, `groups.na_server_min_share`, `groups.na_balance_tolerance` | London, 0.5, 1.0 | The London group rule; `na_server: null` keeps the servers as listed |

`config.example.yaml` documents every key. The `optimizer:`, `output:` and `groups:` blocks are commented out there because the defaults apply; uncomment a key to change it. Unknown or removed keys print a warning.

### VCC stats

- The full season table is read from the page's embedded data (the visible table only shows the top 60).
- Seasons already in `teamMaker/archive/season_distributions.json` are frozen. `scrape_distributions` only adds new ones unless you pass `--refresh`, and it never replaces a season with fewer values.

## Output files

| File | Contents |
| --- | --- |
| `teamMaker/data/players.json` | Every player with ratings, stack and VCC history |
| `teamMaker/data/players_missing.json` | Players with missing ratings or VCC history, and why |
| `teamMaker/data/backups/` | The previous `players.json` from each `generate_players` run (newest 10), in case hand edits were overwritten |
| `teamMaker/output/teams.json`, `teams.png` | The teams, subs and excluded players (with reasons), the seed and settings used |
| `teamMaker/output/groups.json`, `groups.png` | The groups, their servers and average strength |
| `teamMaker/output/player_scores.json` | Each player's score breakdown |

## Troubleshooting

- **Players in `players_missing.json`**: a rating is blank or the rank is misspelled, or a returning player has no valid VCC profile link. Check with `generate_players --dry-run`.
- **Teams differ between reruns**: runs usually stop at the time limit, so results can differ slightly even with the same seed. Keep the `teams.json` you published.
- **No PNG**: install Pillow (`pip install -r requirements.txt`). The JSON and console output are always written.
- **`make_groups` reuse error**: the teams changed since `groups.json` was written; use `mode: auto`.

## Tests

```bash
python -m pytest
```

The tests use made-up data and never contact vlrcommunitycup.com. GitHub runs them, plus the Pyright type check, on every push (`.github/workflows/checks.yml`); a red cross next to a commit means something broke.
