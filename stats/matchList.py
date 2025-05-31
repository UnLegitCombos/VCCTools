import json
import os
import logging

matchListFile = "matchList.json"


def loadMatchList():
    """Load the match list from disk, ensuring proper structure."""
    if os.path.exists(matchListFile):
        try:
            with open(matchListFile, "r") as f:
                data = json.load(f)

                # Handle old format (simple list)
                if isinstance(data, list):
                    return {"tournament_matches": data, "filtered_matches": []}
                # Handle new dictionary format
                elif isinstance(data, dict):
                    if "tournament_matches" not in data:
                        data["tournament_matches"] = []
                    if "filtered_matches" not in data:
                        data["filtered_matches"] = []
                    return data
                else:
                    logging.error("Invalid matchList format")
                    return {"tournament_matches": [], "filtered_matches": []}
        except Exception as e:
            logging.error(f"Error loading match list: {e}")
            return {"tournament_matches": [], "filtered_matches": []}
    else:
        return {"tournament_matches": [], "filtered_matches": []}


def saveMatchList(matchList):
    """Save the match list to disk."""
    try:
        with open(matchListFile, "w") as f:
            json.dump(matchList, f, indent=2)
    except Exception as e:
        logging.error(f"Error saving match list: {e}")


def addTournamentMatch(matchList, matchId):
    """Add a match ID to the tournament matches list."""
    if matchId not in matchList["tournament_matches"]:
        matchList["tournament_matches"].append(matchId)
        return True
    return False


def addFilteredMatch(matchList, matchId):
    """Add a match ID to the filtered matches list."""
    if matchId not in matchList["filtered_matches"]:
        matchList["filtered_matches"].append(matchId)
        return True
    return False


def isMatchProcessed(matchList, matchId):
    """Check if a match is in either the tournament or filtered list."""
    return (
        matchId in matchList["tournament_matches"]
        or matchId in matchList["filtered_matches"]
    )
