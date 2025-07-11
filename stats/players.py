import json
import os
import logging

masterListFile = "stats/masterList.json"


def loadMasterList():
    if os.path.exists(masterListFile):
        try:
            with open(masterListFile, "r") as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error loading master list: {e}")
            return []
    else:
        return []


def saveMasterList(masterList):
    try:
        with open(masterListFile, "w") as f:
            json.dump(masterList, f, indent=2)
    except Exception as e:
        logging.error(f"Error saving master list: {e}")


def updateMasterList(masterList, player):
    """
    Add the given player if not already in masterList (matching by puuid).
    """
    if not any(p["puuid"] == player["puuid"] for p in masterList):
        masterList.append(player)
        saveMasterList(masterList)
