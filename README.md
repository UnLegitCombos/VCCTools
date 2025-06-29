# VCC TOOLS

A set of tools designed to simplify and enhance the Valorant Community Cup (VCC) tournament experience:

- **Team Maker**: Quickly create balanced teams, respecting player groupings like duos or trios.
- **Stats Tracker**: Automatically track detailed match statistics with Google Sheets integration.
- **VLR Formula Approximator**: Analyze player stats, scrape data from VLR.gg, and generate accurate formulas for predicting VLR ratings.
- **Playoffs Scenarios Generator** _(coming soon)_: Simulate and explore playoff outcomes.

## Table of Contents

- [Team Maker](#team-maker)

  - [Features](#features-team-maker)
  - [Usage](#usage-team-maker)

- [Stats Tracker](#stats-tracker)

  - [Features](#features-stats-tracker)
  - [Usage](#usage-stats-tracker)

- [VLR Formula Approximator](#vlr-formula-approximator)
  - [Features](#features-vlr-formula-approximator)
  - [Usage](#usage-vlr-formula-approximator)

---

## Team Maker

Efficiently create fair and balanced teams while keeping groups intact. Supports both basic and advanced optimization modes with sophisticated balancing algorithms.

### Features (Team Maker)

#### Core Features

- Forms balanced teams of 5 from any player count
- Keeps duos/trios together automatically
- Supports both **Basic** and **Advanced** optimization modes
- Optimizes teams based on rank, tracker scores, and custom criteria
- Configurable via `config.json` with extensive customization options

#### Advanced Features (Advanced Mode Only)

- **Peak Rank Act Decay**: Considers historical peak ranks with configurable decay over Valorant episodes/acts
- **Role Balancing**: Intelligently balances team compositions across Valorant agent roles (Duelist, Initiator, Controller, Sentinel, Flex)
- **Region Debuff**: Applies ping penalties for non-EU players to account for latency differences
- **Previous Season Stats**: Incorporates percentile-based rankings from previous competitive seasons (S8/S9) using adjusted rating data
- **Tracker Score Integration**: Combines current and peak tracker.gg performance metrics
- **Simulated Annealing Optimization**: Uses advanced optimization algorithms for superior team balance

### Usage (Team Maker)

#### Quick Start

1. Copy configuration template: `teamMaker/config.example.json` → `teamMaker/config.json`
2. Copy player template: `teamMaker/playersexample.json` → `teamMaker/players.json`
3. Update player details in `teamMaker/players.json`
4. Customize settings in `teamMaker/config.json` (optional)
5. Execute:

```bash
python teamMaker/teams.py
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

## Stats Tracker

Automatically gathers and uploads detailed match statistics to Google Sheets.

### Features (Stats Tracker)

- Fetches matches automatically via Henrik's VALORANT API.
- Filters out non-tournament matches intelligently.
- Provides extensive statistics (ACS, K/D, ADR, KAST, clutch rates, etc.).
- Automatic Google Sheets integration.

### Usage (Stats Tracker)

1. Setup configurations:

   - Copy and configure: `stats/config.example.json` → `stats/config.json`
   - Copy and configure: `stats/credentials.example.json` → `stats/credentials.json`

2. Run the tracker:

```bash
python stats/main.py
```

---

## VLR Formula Approximator

Analyze Valorant player statistics and generate reliable VLR rating predictions.

### Features (VLR Formula Approximator)

- Scrapes and compiles comprehensive data from VLR.gg (2023–2025).
- Generates accurate predictive formulas for VLR ratings.
- Visualizes correlations and model accuracy.

### Usage (VLR Formula Approximator)

- Data scraping:

```bash
python formula/vlrstats.py
```

- Formula generation:

```bash
python formula/formula.py
```

---
