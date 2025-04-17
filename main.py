from config import RAW_DATA_DIR, FORECASTS_DIR, SIGNALS_DIR, TRADES_DIR, MODELS_DIR,PREDICTIONS_DIR, TICKERS_DIR
import os
import pandas as pd
import yfinance as yf
import pprint

from data.loader import fetch_and_update_data  # Import the function from loader.py

# Choose Ticker file name in tickers directory
TICKER_FILE = "dev_tickers1.csv"


# Load tickers from the TICKERS_DIR
def load_tickers(ticker_filename=TICKER_FILE):
    """
    Load tickers from a file in the TICKERS_DIR.
    Assumes the file is named 'tickers.csv' and contains a column 'ticker'.
    """
    tickers_file = os.path.join(TICKERS_DIR, ticker_filename)
    if os.path.exists(tickers_file):
        tickers_df = pd.read_csv(tickers_file)
        return tickers_df['ticker'].tolist()
    else:
        raise FileNotFoundError(f"Tickers file not found in {TICKERS_DIR}. Please ensure {ticker_filename} exists.")
    
    

if __name__ == "__main__":
    # Step 1: Load tickers
    try:
        tickers = load_tickers()
        print(f"Loaded tickers: {tickers}")
    except FileNotFoundError as e:
        print(e)
        exit(1)

    # Step 2: Fetch and update data for the tickers
    data = fetch_and_update_data(tickers)
    # Step 3: Confirm data is updated and ready
    print("Data has been updated and is ready for use.")
    pprint.pprint(data)
