# VCC TOOLS

A set of tools for VCC (Valorant Community Cup):

- **Team Maker**: Create balanced teams from groups of players, respecting group constraints and optimizing for fair matches.
- **Stats Tracker**: Comprehensive tournament statistics tracking with Google Sheets integration using Henrik's VALORANT API.
- **VLR Formula Approximator**: Analyze Valorant player statistics and generate formulas to approximate VLR rating using various modeling approaches.
- **Automatic VLR Stats Scraper**: Download and aggregate player stats from VLR.gg for multiple years and event types.
- **Playoffs Scenarios Generator**: Simulate and analyze playoff scenarios (details coming soon).

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
    - [1. Scrape and Aggregate VLR Stats](#1-scrape-and-aggregate-vlr-stats)
    - [2. Run the Formula Approximator](#2-run-the-formula-approximator)

## Team Maker

This tool creates balanced teams of 5 players, respecting group constraints (duos/trios) and optimizing for fair matches using simulated annealing.

### Features (Team Maker)

- Creates balanced teams of 5 from any number of players
- Supports groups of 1, 2, or 3 (duos/trios stay together)
- Optimizes for minimal team score difference (rank, tracker score, or custom)
- Configurable weights and options in `config.json`
- Fast optimization (usually <2.0 score diff)
- Supports both Windows and Unix paths

### Usage (Team Maker)

1. Edit `teamMaker/players.json` with your player and group data
2. (Optional) Adjust `teamMaker/config.json` for custom weights or settings
3. Run:

```bash
python teamMaker/teams.py
```

## Stats Tracker

A comprehensive VALORANT tournament statistics tracking system that fetches match data from Henrik's API and automatically uploads detailed statistics to Google Sheets.

### Features (Stats Tracker)

- **Automated Match Detection**: Fetches custom matches for tracked players
- **Intelligent Filtering**: Automatic and manual filtering of tournament vs non-tournament matches
- **Comprehensive Statistics**: Calculates 25+ metrics including:
  - ACS (Average Combat Score), K/D, ADR (Average Damage per Round)
  - KAST (Kill/Assist/Survive/Trade percentage)
  - First Kills/Deaths per round, Clutch success rate
  - Custom rating system based on multiple performance factors
- **Google Sheets Integration**: Automatic upload and organization of match data
- **Player Management**: Tracks multiple players across tournaments

### Usage (Stats Tracker)

1. **Setup Configuration**:

   - Copy `stats/config.example.json` to `stats/config.json`
   - Add your Henrik API key and Google Sheet name
   - Copy `stats/credentials.example.json` to `stats/credentials.json`
   - Add your Google Service Account credentials

2. **Run the tracker**:

```bash
python stats/main.py
```

For detailed setup instructions including API keys and Google Sheets configuration, see `stats/README.md`.

## VLR Formula Approximator

This set of tools analyzes Valorant player statistics, generates formulas to approximate VLR rating, and provides advanced modeling and visualization.

### Features (VLR Formula Approximator)

- Scrapes and aggregates VLR.gg stats for VCT, Challengers, and Game Changers (2023–2025)
- Outputs a combined CSV (`formula/data/vlr_data.csv`) with all relevant stats, year, and event type
- Generates formulas to calculate VLR rating from in-game stats
- Visualizes stat correlations and model predictions
- Compares datasets for improvement

### Usage (VLR Formula Approximator)

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
