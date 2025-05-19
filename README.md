# VCC TOOLS

A set of tools for VCC (Valorant Community Cup):

- **Team Maker**: Create balanced teams from groups of players, respecting group constraints and optimizing for fair matches.
- **VLR Formula Approximator**: Analyze Valorant player statistics and generate formulas to approximate VLR rating using various modeling approaches.
- **Automatic VLR Stats Scraper**: Download and aggregate player stats from VLR.gg for multiple years and event types.
- **Playoffs Scenarios Generator**: Simulate and analyze playoff scenarios (details coming soon).

## Table of Contents

- [Team Maker](#team-maker)
  - [Features](#features)
  - [Usage](#usage)
- [VLR Formula Approximator](#vlr-formula-approximator)

## Team Maker

This tool creates balanced teams of 5 players, respecting group constraints (duos/trios) and optimizing for fair matches using simulated annealing.

### Features

- Creates balanced teams of 5 from any number of players
- Supports groups of 1, 2, or 3 (duos/trios stay together)
- Optimizes for minimal team score difference (rank, tracker score, or custom)
- Configurable weights and options in `config.json`
- Fast optimization (usually <2.0 score diff)
- Supports both Windows and Unix paths

### Usage

1. Edit `teamMaker/players.json` with your player and group data
2. (Optional) Adjust `teamMaker/config.json` for custom weights or settings
3. Run:

```bash
python teamMaker/teams.py
```

## VLR Formula Approximator

This set of tools analyzes Valorant player statistics, generates formulas to approximate VLR rating, and provides advanced modeling and visualization.

### Features

- Scrapes and aggregates VLR.gg stats for VCT, Challengers, and Game Changers (2023–2025)
- Outputs a combined CSV (`formula/data/vlr_data.csv`) with all relevant stats, year, and event type
- Generates formulas to calculate VLR rating from in-game stats
- Visualizes stat correlations and model predictions
- Compares datasets for improvement

### Usage

#### 1. Scrape and Aggregate VLR Stats

Run the stats scraper to download and combine all VLR.gg stats:

```bash
python formula/vlrstats.py
```

This will create or update `formula/data/vlr_data.csv` with all stats for 2023–2025, including VCT, Challengers, and Game Changers.

#### 2. Run the Formula Approximator

Standard version:

```bash
python formula/formula.py
```
