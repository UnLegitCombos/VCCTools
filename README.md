# VCC TOOLS

A set of tools for VCC (Valorant Community Cup):

- Team Maker
- VLR Formula Aprrox.
- Automatic stats
- Playoffs scenarios generator

## Team Maker

This tool optimizes the formation of balanced teams from groups of players using a simulated annealing algorithm that efficiently creates well-balanced teams.

### Features

- Creates balanced teams of 5 players from pre-defined groups
- Groups can have 1, 2, or 3 players and must stay together
- Optimizes team balance based on player ranks and optional tracker scores
- Configurable scoring weights through config.json
- Fast and efficient optimization using simulated annealing
- Typically produces teams with score differences under 2.0

### Usage

1. Configure player data in `players.json`
2. Adjust parameters in `config.json` if needed
3. Run the script:

```bash
python teamMaker/teams.py
# Or if using Windows Command Prompt:
# python teamMaker\teams.py
```

##
