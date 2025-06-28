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

Efficiently create fair and balanced teams while keeping groups intact.

### Features (Team Maker)

- Forms balanced teams of 5 from any player count.
- Keeps duos/trios together.
- Optimizes teams based on rank, scores, or custom criteria.
- Quick and easy configuration via `config.json`.

### Usage (Team Maker)

1. Update player details in `teamMaker/players.json`.
2. Customize settings in `teamMaker/config.json` (optional).
3. Execute:

```bash
python teamMaker/teams.py
```

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
