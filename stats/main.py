import logging
import json
import os
from players import loadMasterList, saveMasterList, updateMasterList
from matchList import (
    loadMatchList,
    saveMatchList,
    isMatchProcessed,
    addTournamentMatch,
    addFilteredMatch,
)
from apiHandler import fetchCustomMatchHistory
from statsCalculations import (
    calculateAcs,
    calculateKast,
    calculateFirstKillAndDeaths,
    calculateKd,
    calculateAdr,
    calculatePerRoundStat,
    calculateFirstStatPerRound,
    calculateClPercent,
    calculateHsPercentage,
    computeRating,
)
from sheetsHandler import initSheet, updateSheet

logging.basicConfig(
    filename="tournamentStats.log",
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s: %(message)s",
)


# Load API key from config file
def load_api_key():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
            return config.get("apiKey")
    except Exception as e:
        logging.error(f"Failed to load API key from config: {e}")
        return None


apiKey = load_api_key()


def displayMatchesForSelection(matches, matchList, filteringMode="automatic"):
    """
    Given a list of match dicts from the API, deduplicate by matchId.
    Prompt the user to select which matches are for the tournament.
    Return only those the user picks.
    """
    logging.debug(
        "Entering displayMatchesForSelection with %d match entries.", len(matches)
    )

    unprocessedMatches = [
        match
        for match in matches
        if not isMatchProcessed(matchList, match["metadata"]["match_id"])
    ]

    if not unprocessedMatches:
        logging.info("No unprocessed matches found.")
        return []

    uniqueById = {}
    for match in unprocessedMatches:
        matchId = match["metadata"]["match_id"]
        uniqueById[matchId] = match

    uniqueMatches = list(uniqueById.values())
    if not uniqueMatches:
        logging.info("No unique matches found.")
        return []

    if filteringMode == "manual":
        print("\nSelect matches to FILTER OUT (skip reviewing in the future):")

        for idx, match in enumerate(uniqueMatches):
            matchId = match["metadata"]["match_id"]
            mapName = match["metadata"]["map"]["name"]
            startTime = match["metadata"]["started_at"]
            trackerUrl = f"https://tracker.gg/valorant/match/{matchId}"
            print(f"[{idx}] {mapName} at {startTime} -> {trackerUrl}")

        filterSelection = input(
            "Enter comma-separated indices of tournament matches (blank = select none): "
        ).strip()

        if filterSelection:
            try:
                filterIndices = [int(x.strip()) for x in filterSelection.split(",")]
                for idx in filterIndices:
                    if 0 <= idx < len(uniqueMatches):
                        matchId = uniqueMatches[idx]["metadata"]["match_id"]
                        addFilteredMatch(matchList, matchId)
                        logging.info(f"Match {matchId} marked as filtered.")
                saveMatchList(matchList)

                uniqueMatches = [
                    uniqueMatches[i]
                    for i in range(len(uniqueMatches))
                    if i not in filterIndices
                ]

                if not uniqueMatches:
                    logging.info("No matches left after filtering.")
                    return []
            except Exception as e:
                logging.error(f"Error parsing filter selection: {e}")

    print("\nSelect which custom matches are tournament matches:")
    for idx, match in enumerate(uniqueMatches):
        matchId = match["metadata"]["match_id"]
        mapName = match["metadata"]["map"]["name"]
        startTime = match["metadata"]["started_at"]
        trackerUrl = f"https://tracker.gg/valorant/match/{matchId}"
        print(f"[{idx}] {mapName} at {startTime} -> {trackerUrl}")

    selection = input(
        "Enter comma-separated indices of tournament matches (blank = select none): "
    ).strip()

    if not selection:
        logging.debug("No selection made, returning empty list.")
        # If automatic filtering, add all unselected matches to filter list
        if filteringMode == "automatic":
            for match in uniqueMatches:
                matchId = match["metadata"]["match_id"]
                addFilteredMatch(matchList, matchId)
                logging.debug(f"Auto-added match {matchId} to filter list.")
            saveMatchList(matchList)
        return []

    try:
        indices = [int(x.strip()) for x in selection.split(",")]
        chosen = [uniqueMatches[i] for i in indices if 0 <= i < len(uniqueMatches)]
        logging.debug("User selected %d matches.", len(chosen))

        if filteringMode == "automatic":
            for i, match in enumerate(uniqueMatches):
                if i not in indices:
                    matchId = match["metadata"]["match_id"]
                    addFilteredMatch(matchList, matchId)
                    logging.debug(f"Auto-added match {matchId} to filter list.")
            saveMatchList(matchList)

        return chosen
    except Exception as e:
        logging.error(f"Error parsing selection: {e}")
        return []


def processMatch(matchDict, masterList):
    """
    Convert a single match-dict into rows for the Google Sheet.
    Update the master list with newly encountered players.
    """
    logging.debug("Entered processMatch.")

    rows = []
    kastCounts, kastFractions = calculateKast(matchDict)
    firstKills, firstDeaths = calculateFirstKillAndDeaths(matchDict)

    matchId = matchDict["metadata"]["match_id"]
    mapName = matchDict["metadata"]["map"]["name"]
    startedAt = matchDict["metadata"]["started_at"]
    logging.info(f"Processing match {matchId} on {mapName} at {startedAt}.")

    allPlayers = matchDict["players"]

    for p in allPlayers:
        kd = calculateKd(p)
        adr, totalDamage = calculateAdr(matchDict, p)
        kpr = calculatePerRoundStat(matchDict, p, "kills")
        apr = calculatePerRoundStat(matchDict, p, "assists")
        dpr = calculatePerRoundStat(matchDict, p, "deaths")
        fkpr = calculateFirstStatPerRound(matchDict, p, firstKills)
        fdpr = calculateFirstStatPerRound(matchDict, p, firstDeaths)
        cl = calculateClPercent(p, matchDict)
        clFrac = cl[0]
        clWins = cl[1]
        clAttempts = cl[2]
        kastFrac = kastFractions.get(p["puuid"], 0)
        kastCount = kastCounts.get(p["puuid"], 0)
        rating = computeRating(kd, adr, kpr, apr, dpr, fkpr, fdpr, clFrac, kastFrac)

        newPlayer = {
            "puuid": p["puuid"],
            "name": p["name"],
            "tag": p["tag"],
        }
        updateMasterList(masterList, newPlayer)

        stats = p["stats"]
        row = [
            matchId,
            mapName,
            len(matchDict["rounds"]),
            startedAt,
            f"{p['name']}#{p['tag']}",
            p["agent"]["name"],
            stats["kills"],
            stats["deaths"],
            stats["assists"],
            firstKills.get(p["puuid"], 0),
            firstDeaths.get(p["puuid"], 0),
            calculateAcs(matchDict, p),
            kd,
            totalDamage,
            adr,
            kpr,
            apr,
            dpr,
            fkpr,
            fdpr,
            calculateHsPercentage(p),
            clFrac,
            clWins,
            clAttempts,
            kastFrac,
            kastCount,
            rating,
        ]
        rows.append(row)

    logging.info(f"Finished processing match {matchId} => generated {len(rows)} rows.")
    return rows


def promptForNewPlayer():
    """
    Ask the user for the initial player's (name#tag) if none exist.
    Returns (playerName, playerTag).
    """
    print("No players in master list. Please enter a player to start tracking:")
    name = input("Player name (no tag): ").strip()
    tag = input("Player tag (numbers or text): ").strip()
    return (name, tag)


def choosePlayerMode(masterList):
    """
    If the master list is not empty, ask the user whether
    to process just one “default” player or all in the master list.
    Return either:
       [ (playerName, playerTag) ]  # single
    or
       [ (nameTag for each in masterList) ] # entire list
    """
    if not masterList:
        return []

    print("\nPlayers in master list:")
    for idx, p in enumerate(masterList):
        print(f"   [{idx}] {p['name']}#{p['tag']}")

    choice = input(
        "Use [0] to pick a single player by index, or [1] to process ALL players in the master list? (0/1): "
    ).strip()

    if choice == "0":
        idx = input("Enter the index of the player: ").strip()
        try:
            idx = int(idx)
            if 0 <= idx < len(masterList):
                pl = masterList[idx]
                return [(pl["name"], pl["tag"])]
            else:
                print("Invalid index, defaulting to index 0.")
                pl = masterList[0]
                return [(pl["name"], pl["tag"])]
        except:
            print("Invalid input, defaulting to index 0.")
            pl = masterList[0]
            return [(pl["name"], pl["tag"])]
    else:
        return [(p["name"], p["tag"]) for p in masterList]


def main():
    region = "eu"
    logging.info("Initializing Google Sheets.")
    sheet = initSheet()

    masterList = loadMasterList()
    matchList = loadMatchList()

    filteringMode = "automatic"  # or "manual"

    if not masterList:
        pName, pTag = promptForNewPlayer()
        masterList.append({"puuid": None, "name": pName, "tag": pTag})
        saveMasterList(masterList)
        selectedPlayers = [(pName, pTag)]
    else:
        selectedPlayers = choosePlayerMode(masterList)

    allNewMatches = []

    for playerName, playerTag in selectedPlayers:
        logging.info(f"Fetching matches for {playerName}#{playerTag}.")
        matches, error = fetchCustomMatchHistory(region, playerName, playerTag, apiKey)
        if error:
            logging.error(f"{playerName}#{playerTag}: {error}")
            continue

        newlyFound = []
        for m in matches:
            mid = m["metadata"]["match_id"]
            if mid not in matchList:
                newlyFound.append(m)

        logging.debug(
            "Player %s#%s => found %d new matches",
            playerName,
            playerTag,
            len(newlyFound),
        )
        allNewMatches.extend(newlyFound)

    if not allNewMatches:
        print("\nNo new matches found across all selected players. Exiting.")
        logging.info("No new matches across all players, done.")
        return  # Sort matches by timestamp (oldest first) for chronological processing and sheet order
    allNewMatches.sort(key=lambda match: match["metadata"]["started_at"], reverse=False)
    logging.info(f"Sorted {len(allNewMatches)} matches by timestamp (oldest first).")

    # Now, show them as a single consolidated list
    selectedMatches = displayMatchesForSelection(
        allNewMatches, matchList, filteringMode
    )
    if not selectedMatches:
        print("\nNo matches were selected. Exiting.")
        logging.info("User did not select any matches to process, done.")
        return

    allRows = []  # Process matches in chronological order (oldest first)
    for matchDict in selectedMatches:
        matchId = matchDict["metadata"]["match_id"]
        if matchId in matchList:
            logging.info(f"Match {matchId} already processed, skipping.")
            continue

        logging.debug("User selected match %s for processing.", matchId)
        addTournamentMatch(matchList, matchId)
        rows = processMatch(matchDict, masterList)
        # Add these rows to maintain chronological order in the sheet
        allRows.extend(rows)

    saveMatchList(matchList)

    if allRows:
        logging.info(f"Updating Google Sheets with {len(allRows)} new rows.")
        updateSheet(sheet, allRows)
        print("\nSpreadsheet successfully updated with new matches!")
    else:
        print("\nNo new matches to update. Exiting.")


if __name__ == "__main__":
    logging.info("Script started.")
    main()
    logging.info("Script finished.")
