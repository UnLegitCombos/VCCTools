# VCC TOOLS

A set of tools designed to simplify and enhance the Valorant Community Cup (VCC) tournament experience:

- **Team Maker**: Quickly create balanced teams, respecting player groupings like duos or trios.
- **Playoffs Scenarios Generator** _(coming soon)_: Simulate and explore playoff outcomes.

## Table of Contents

- [Team Maker](#team-maker)
  - [Features](#features-team-maker)
  - [Usage](#usage-team-maker)

---

## Team Maker

Efficiently create fair and balanced teams while keeping groups intact. Supports both basic and advanced optimization modes with sophisticated balancing algorithms.

### Features (Team Maker)

#### Core Features

- Forms balanced teams of 5 from any player count
- Keeps duos/trios together automatically
- Supports both **Basic** and **Advanced** optimization modes
- Optimizes teams based on rank, tracker scores, and custom criteria
- Configurable via `config.yaml` with extensive customization options

#### Advanced Features (Advanced Mode Only)

- **Peak Rank Act Decay**: Considers historical peak ranks with configurable decay over Valorant episodes/acts
- **Role Balancing**: Intelligently balances team compositions across Valorant agent roles (Duelist, Initiator, Controller, Sentinel, Flex)
- **Region Debuff**: Applies ping penalties for non-EU players to account for latency differences
- **Previous Season Stats**: Incorporates percentile-based rankings from previous competitive seasons (S8/S9) using adjusted rating data
- **Tracker Score Integration**: Combines current and peak tracker.gg performance metrics
- **Simulated Annealing Optimization**: Uses advanced optimization algorithms for superior team balance

### Usage (Team Maker)

#### Quick Start

1. Copy configuration template: `teamMaker/config/config.example.yaml` → `teamMaker/config/config.yaml`
2. Build `teamMaker/data/players.json` from the season sign-up form export:
   - Drop the sign-up CSV export into `teamMaker/data/input/`
   - Configure `teamMaker/config/player_ratings_config.yaml` (current act, season links, etc.)
   - Run `python -m teamMaker.core.generate_players` (from the repo root) — scrapes tracker.gg and vlrcommunitycup.com and writes `teamMaker/data/players.json` directly
   - **Note:** the generator doesn't know about manually-assigned `group_id` or `ping` values — merge those in by hand after generating, or maintain `teamMaker/data/players.json` by hand instead (copy `teamMaker/data/playersexample.json` as a starting template)
3. Update player details in `teamMaker/data/players.json` as needed
4. Customize settings in `teamMaker/config/config.yaml` (optional)
5. Execute (from the repo root):

```bash
python -m teamMaker.core.build_teams
```

#### Configuration Options

**Basic Settings** (apply to both modes):

- `mode`: Choose `"basic"` or `"advanced"`
- `use_tracker`: Enable tracker.gg score integration
- `weight_current`/`weight_peak`: Balance between current and peak ranks
- `weight_current_tracker`/`weight_peak_tracker`: Tracker score weighting

**Advanced Settings** (advanced mode only):

- `use_peak_act`: Enable peak rank act decay
- `peak_act_decay_rate`: Decay rate per act (0.9-0.99)
- `use_role_balancing`: Enable role-based team composition balancing
- `role_balance_weight`: Strength of role balancing effect
- `use_region_debuff`: Apply ping penalty for non-EU players
- `use_returning_player_stats`: Include previous season performance data
- `weight_previous_season`: Weight for previous season stats in scoring

#### Player Data Format

Players can include:

- Basic info: `name`, `current_rank`, `peak_rank`, `group` (for duos/trios)
- Tracker scores: `current_tracker`, `peak_tracker`
- Advanced: `role`, `region`, `peak_act` (episode.act format)
- Previous seasons: `previous_season` object with S8/S9 data

---
