import requests
import logging


def fetchCustomMatchHistory(region, playerName, playerTag, apiKey=None):
    """
    Fetch only custom matches for the given player using Henrik's API v4.
    Returns (matches, errorMessage).
    If an error occurs, matches will be an empty list, and errorMessage will be non-empty.
    """
    try:
        url = f"https://api.henrikdev.xyz/valorant/v4/matches/{region}/pc/{playerName}/{playerTag}"
        headers = {"Authorization": apiKey} if apiKey else {}
        params = {"mode": "custom", "size": 10}

        response = requests.get(url, headers=headers, params=params)

        if response.status_code != 200:
            errText = f"Error fetching match history: {response.status_code} - {response.text}"
            logging.error(errText)
            return ([], errText)

        data = response.json()
        allMatches = data.get("data", [])
        logging.info(f"Fetched {len(allMatches)} matches from the API.")

        if not allMatches:
            return ([], "No custom matches found")

        filtered = []
        for match in allMatches:
            queueInfo = match.get("metadata", {}).get("queue", {})
            modeType = queueInfo.get("mode_type", {})
            players = len(match.get("players", {}))

            if modeType == "Standard" and players == 10:
                filtered.append(match)

            if not filtered:
                return [], "No matches found that meet 'Standard mode' + 10 players"

        return filtered, None  # No error

    except Exception as e:
        logging.error(f"Unexpected error fetching match history: {e}")
        return [], str(e)
