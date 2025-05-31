import gspread
import logging
import time
import json
import os


def load_config():
    """
    Load configuration from config.json file.
    Returns:
        dict: Configuration settings
    """
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
            return config
    except Exception as e:
        logging.error(f"Failed to load configuration: {e}")
        return {"sheetId": "VCC S9 SB"}  # Default value


def initSheet():
    """
    Initializes the 'Database' sheet in the spreadsheet specified in config.json.
    If no headers exist, append them.
    """
    config = load_config()
    sheet_id = config.get("sheetId", "VCC S9 SB")  # Default value if not found

    try:
        creds = gspread.service_account(filename="credentials.json")
        sheet = creds.open(sheet_id).worksheet("Database")
        # Check if headers exist; if not, add them.
        values = sheet.get_all_values()
        if not values or not values[0]:
            headers = [
                "Match ID",
                "Map",
                "Rounds",
                "Game Start",
                "Player",
                "Agent",
                "Kills",
                "Deaths",
                "Assists",
                "First Kills",
                "First Deaths",
                "ACS",
                "KD",
                "Total Damage",
                "ADR",
                "KPR",
                "APR",
                "DPR",
                "FKPR",
                "FDPR",
                "HS (frac)",
                "Clutch (frac)",
                "Clutch wins",
                "Clutch attempts",
                "KAST (frac)",
                "KAST (counts)",
                "Rating",
            ]
            sheet.append_row(headers)
        return sheet
    except Exception as e:
        logging.error(f"Error initializing Google Sheets: {e}")
        raise


def updateSheet(sheet, rows):
    try:
        if not rows:
            logging.warning("No rows to update in Google Sheet.")
            return

        BATCH_SIZE = 10
        DELAY = 10

        batches = [rows[i : i + BATCH_SIZE] for i in range(0, len(rows), BATCH_SIZE)]
        totalBatches = len(batches)
        eta = totalBatches * DELAY / 60  # ETA in minutes
        print(f"Total batches to update: {totalBatches}, ETA: {eta:.2f} minutes")

        for i, batch in enumerate(batches):
            try:
                print(
                    f"Updating sheet - batch {i+1}/{totalBatches} ({len(batch)} rows)"
                )
                # Use append_rows instead of append_row for better efficiency
                sheet.append_rows(batch)

                # Add delay if there are more batches to process
                if i < totalBatches - 1:
                    logging.info(f"Waiting {DELAY} seconds before next batch...")
                    time.sleep(DELAY)

            except gspread.exceptions.APIError as apiError:
                if "Quota exceeded" in str(apiError):

                    retryDelay = 60  # Retry after 1 minute
                    logging.warning(
                        f"Quota exceeded. Retrying in {retryDelay} seconds..."
                    )
                    print(f"Rate limit reached. Retrying in {retryDelay} seconds...")
                    time.sleep(retryDelay)

                    sheet.append_rows(batch)
                else:
                    logging.error(f"API error while updating batch {i+1}: {apiError}")
                    raise

    except Exception as e:
        logging.error(f"Error updating Google Sheet: {e}")
        print(f"Error updating sheet: {e}")
        raise
