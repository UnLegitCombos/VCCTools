import requests
import logging
import time


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


def fetchCustomMatchHistoryBatched(region, playerList, apiKey=None, batchSize=3, delay=5):
    """
    Fetch custom matches for multiple players with batching and rate limiting.

    Args:
        region: The region to fetch from
        playerList: List of tuples (playerName, playerTag)
        apiKey: API key for Henrik's API
        batchSize: Number of concurrent requests per batch
        delay: Delay in seconds between batches

    Returns:
        List of tuples: [(playerName, playerTag, matches, error), ...]
    """
    results = []

    # Split players into batches
    batches = [playerList[i : i + batchSize] for i in range(0, len(playerList), batchSize)]
    totalBatches = len(batches)

    print(f"Fetching match history for {len(playerList)} players in {totalBatches} batches...")
    logging.info(f"Starting batched API requests: {totalBatches} batches of {batchSize} players each")

    for batchIndex, batch in enumerate(batches):
        print(f"Processing batch {batchIndex + 1}/{totalBatches} ({len(batch)} players)")
        logging.info(f"Processing API batch {batchIndex + 1}/{totalBatches}")

        batchResults = []

        # Process each player in the current batch
        for playerName, playerTag in batch:
            logging.info(f"Fetching matches for {playerName}#{playerTag}")
            print(f"  Fetching: {playerName}#{playerTag}")

            matches, error = fetchCustomMatchHistory(region, playerName, playerTag, apiKey)
            batchResults.append((playerName, playerTag, matches, error))

            if error:
                logging.error(f"{playerName}#{playerTag}: {error}")
                print(f"    Error: {error}")
            else:
                print(f"    Found {len(matches)} matches")

        results.extend(batchResults)

        # Add delay between batches (except for the last batch)
        if batchIndex < totalBatches - 1:
            print(f"  Waiting {delay} seconds before next batch...")
            logging.info(f"Waiting {delay} seconds before next API batch")
            time.sleep(delay)

    logging.info(f"Completed all API batches. Total results: {len(results)}")
    return results
