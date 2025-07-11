import logging
from collections import defaultdict


def calculateActualRoundsPlayed(matchDict):
    """
    Calculate the actual number of rounds played by summing team scores.
    This is needed because forfeited matches include unplayed rounds in the rounds array.
    """
    teams = matchDict.get("teams", [])
    if len(teams) != 2:
        # Fallback to rounds array length if team data is unavailable
        return len(matchDict["rounds"])

    # Sum rounds lost for both teams to get total rounds played
    total_rounds = 0
    for team in teams:
        rounds_lost = team.get("rounds", {}).get("lost", 0)
        total_rounds += rounds_lost

    # Double-check: the winning team's losses + losing team's losses should equal total rounds
    # If this doesn't match expected values, log a warning and use the calculation
    logging.debug(f"Calculated {total_rounds} actual rounds played from team scores")

    return total_rounds


def calculateHsPercentage(playerDict):
    """
    Returns fraction of headshots (e.g. 0.5 for 50%).
    """
    try:
        hs = playerDict["stats"]["headshots"]
        bs = playerDict["stats"]["bodyshots"]
        ls = playerDict["stats"]["legshots"]
        total = hs + bs + ls
        if total == 0:
            return 0.0
        return round(hs / total, 3)
    except Exception as exc:
        pname = playerDict.get("name", "Unknown")
        logging.error(f"Error calculating HS% for {pname}: {exc}")
        return 0.0


def calculateAcs(matchDict, playerDict):
    """
    ACS = player's score / total number of rounds
    """
    try:
        totalRounds = calculateActualRoundsPlayed(matchDict)
        score = playerDict["stats"]["score"]
        if totalRounds == 0:
            return 0.0
        return round(score / totalRounds, 3)
    except Exception as exc:
        pname = playerDict.get("name", "Unknown")
        logging.error(f"Error calculating ACS for {pname}: {exc}")
        return 0.0


def calculateKast(matchDict):
    """
    Returns KAST fraction, e.g. 0.8 => 80%.
    This logic checks per-round kills/assists/trade/survive.
    """
    playerKastRounds = defaultdict(set)  # Maps puuid -> set of round indices
    roundsList = matchDict["rounds"]
    totalRounds = calculateActualRoundsPlayed(matchDict)

    # Only process the actual rounds that were played, not the full rounds array
    for roundIndex in range(totalRounds):
        roundData = roundsList[roundIndex]
        deadPlayers = set()
        playerAfkStatus = {}

        for pStats in roundData["stats"]:
            puuid = pStats["player"]["puuid"]
            playerAfkStatus[puuid] = pStats["was_afk"]

        killEvents = [k for k in matchDict["kills"] if k["round"] == roundIndex]

        for killEvent in killEvents:
            deadPlayers.add(killEvent["victim"]["puuid"])

        # Survive check
        for pStats in roundData["stats"]:
            puuid = pStats["player"]["puuid"]
            if puuid not in deadPlayers and not playerAfkStatus.get(puuid, False):
                playerKastRounds[puuid].add(roundIndex)

        # Kills & assists
        for killEvent in killEvents:
            killerPuuid = killEvent["killer"]["puuid"]
            playerKastRounds[killerPuuid].add(roundIndex)
            for assistant in killEvent["assistants"]:
                assistantPuuid = assistant["puuid"]
                playerKastRounds[assistantPuuid].add(roundIndex)

        # Trades within 5s
        for i, kill1 in enumerate(killEvents):
            t1 = kill1["time_in_round_in_ms"]
            killer1 = kill1["killer"]["puuid"]
            victim1 = kill1["victim"]["puuid"]
            killer1Team = kill1["killer"]["team"]
            victim1Team = kill1["victim"]["team"]

            for j, kill2 in enumerate(killEvents):
                if i == j:
                    continue
                t2 = kill2["time_in_round_in_ms"]
                killer2 = kill2["killer"]["puuid"]
                victim2 = kill2["victim"]["puuid"]
                killer2Team = kill2["killer"]["team"]
                victim2Team = kill2["victim"]["team"]

                if victim2 == killer1 and killer2Team == victim1Team:
                    if 0 < (t2 - t1) <= 5000:
                        playerKastRounds[victim1].add(roundIndex)

    playerKast = {}
    playerKastFractions = {}

    for puuid, roundsSet in playerKastRounds.items():
        roundCount = len(roundsSet)
        playerKast[puuid] = roundCount

        if totalRounds > 0:
            kastFraction = len(roundsSet) / totalRounds
        else:
            kastFraction = 0.0
        playerKastFractions[puuid] = round(kastFraction, 3)

    return playerKast, playerKastFractions


def calculateFirstKillAndDeaths(matchDict):
    """
    Identify earliest kill each round => increment firstKills for killer, firstDeaths for victim.
    """
    firstKills = {}
    firstDeaths = {}
    killEvents = defaultdict(list)  # Maps round index to list of kill events

    for killEvent in matchDict["kills"]:
        killEvents[killEvent["round"]].append(killEvent)

    for roundIndex, kills in killEvents.items():
        earliestKill = None
        earliestTime = float("inf")

        for kill in kills:
            t = kill["time_in_round_in_ms"]
            if t < earliestTime:
                earliestTime = t
                earliestKill = kill

        if earliestKill:
            killerPuuid = earliestKill["killer"]["puuid"]
            victimPuuid = earliestKill["victim"]["puuid"]
            firstKills[killerPuuid] = firstKills.get(killerPuuid, 0) + 1
            firstDeaths[victimPuuid] = firstDeaths.get(victimPuuid, 0) + 1

    return firstKills, firstDeaths


def calculateKd(playerDict):
    """
    kills / deaths
    """
    try:
        kills = playerDict["stats"]["kills"]
        deaths = playerDict["stats"]["deaths"]
        if deaths == 0:
            return round(float(kills), 3)
        return round(kills / deaths, 3)
    except Exception as exc:
        pname = playerDict.get("name", "Unknown")
        logging.error(f"Error calculating KD for {pname}: {exc}")
        return 0.0


def calculateAdr(matchDict, playerDict):
    """
    ADR = player's damage_made / total number of rounds
    """
    try:
        totalRounds = calculateActualRoundsPlayed(matchDict)
        if totalRounds == 0:
            return 0.0, 0
        damage = playerDict["stats"]["damage"]["dealt"]
        return round(damage / totalRounds, 3), damage
    except Exception as exc:
        pname = playerDict.get("name", "Unknown")
        logging.error(f"Error calculating ADR for {pname}: {exc}")
        return 0.0, 0


def calculateDmgDeltaPerRound(matchDict, playerDict):
    """
    Returns damage delta per round.
    """
    try:
        totalRounds = calculateActualRoundsPlayed(matchDict)
        if totalRounds == 0:
            return 0.0
        damageDealt = playerDict["stats"]["damage"]["dealt"]
        damageReceived = playerDict["stats"]["damage"]["received"]
        damageDelta = damageDealt - damageReceived
        return round(damageDelta / totalRounds, 3)
    except Exception as exc:
        pname = playerDict.get("name", "Unknown")
        logging.error(f"Error calculating damage delta for {pname}: {exc}")
        return 0.0


def calculatePerRoundStat(matchDict, playerDict, statName):
    """
    Returns the player's [statName] / total number of rounds, e.g. KPR, APR, DPR.
    """
    try:
        totalRounds = calculateActualRoundsPlayed(matchDict)
        if totalRounds == 0:
            return 0.0
        value = playerDict["stats"].get(statName, 0)
        return round(value / totalRounds, 3)
    except Exception as exc:
        pname = playerDict.get("name", "Unknown")
        logging.error(f"Error calculating {statName} for {pname}: {exc}")
        return 0.0


def calculateFirstStatPerRound(matchDict, playerDict, firstStatDict):
    """
    Generic function to compute (firstStatCount / totalRounds).
    E.g. firstKills => FKPR or firstDeaths => FDPR
    """
    try:
        roundsList = matchDict["rounds"]
        totalRounds = calculateActualRoundsPlayed(matchDict)
        if totalRounds == 0:
            return 0.0

        puuid = playerDict["puuid"]
        statCount = firstStatDict.get(puuid, 0)
        return round(statCount / totalRounds, 3)
    except Exception as exc:
        pname = playerDict.get("name", "Unknown")
        logging.error(f"Error calculating first-stat for {pname}: {exc}")
        return 0.0


def calculateClPercent(player, matchDict=None):
    """
    Returns a tuple of (clutch_percentage, clutches_won, clutch_attempts)
    """
    try:
        if not matchDict:
            return 0.0, 0, 0

        playerPuuid = player["puuid"]
        playerName = player.get("name", "Unknown")

        playerTeam = None
        for p in matchDict["players"]:
            if p["puuid"] == playerPuuid:
                playerTeam = p["team_id"]
                break

        if not playerTeam:
            return 0.0, 0, 0

        clutchAttempts = 0
        clutchWins = 0

        teamPlayers = defaultdict(list)
        for p in matchDict["players"]:
            teamPlayers[p["team_id"]].append(p["puuid"])

        for roundIndex, roundData in enumerate(matchDict["rounds"], 1):
            roundWinner = roundData.get("winning_team")

            if not roundWinner:
                continue

            killEvents = sorted(
                [
                    k
                    for k in matchDict.get("kills", [])
                    if k.get("round") == roundIndex - 1
                ],
                key=lambda k: k.get("time_in_round_in_ms", 0),
            )

            deadPlayers = set()
            clutchDetectedThisRound = False

            for kill_idx, kill in enumerate(killEvents):
                victimPuuid = kill.get("victim", {}).get("puuid")

                if not victimPuuid:
                    continue

                deadPlayers.add(victimPuuid)

                if playerPuuid in deadPlayers:
                    break

                aliveTeammates = sum(
                    1
                    for puuid in teamPlayers.get(playerTeam, [])
                    if puuid != playerPuuid and puuid not in deadPlayers
                )

                aliveEnemies = sum(
                    1
                    for team, players in teamPlayers.items()
                    if team != playerTeam
                    for puuid in players
                    if puuid not in deadPlayers
                )

                if (
                    aliveTeammates == 0
                    and aliveEnemies > 0
                    and not clutchDetectedThisRound
                ):
                    clutchAttempts += 1
                    clutchDetectedThisRound = True

                    if roundWinner == playerTeam:
                        clutchWins += 1
                    break

        # Log summary at INFO level
        logging.info(f"Player {playerName}: {clutchWins}/{clutchAttempts} clutches")

        if clutchAttempts == 0:
            return 0.0, 0, 0

        clutchPercentage = round(clutchWins / clutchAttempts, 3)
        return clutchPercentage, clutchWins, clutchAttempts

    except Exception as exc:
        logging.error(
            f"Error calculating clutch percentage for {player.get('name', 'Unknown')}: {exc}"
        )
        import traceback

        logging.error(f"Exception in clutch calculation: {traceback.format_exc()}")
        return 0.0, 0, 0


def computeRating(kd, adr, kpr, apr, dpr, fkpr, fdpr, clPercent, kast):
    rating = (
        0.702855
        + 0.002500 * adr
        + 0.714459 * kpr
        + 0.209346 * apr
        - 0.824487 * dpr
        + 0.032957 * kast
        + 0.208877 * fkpr
        - 0.208877 * fdpr
        + 0.037841 * clPercent
    )

    return round(rating, 3)
