from config import RAW_DATA_DIR, FORECASTS_DIR, SIGNALS_DIR, TRADES_DIR, MODELS_DIR,PREDICTIONS_DIR, TICKERS_DIR, TICKER_FILE
import os
import pandas as pd
import yfinance as yf
import pprint
import h5py
from data.loader import fetch_and_update_data_hdf5  # Import the function from loader.py

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
    
def print_h5_structure(name, obj):
    print(name)

if __name__ == "__main__":
    # Step 1: Load tickers
    try:
        tickers = load_tickers()
        print(f"Loaded tickers: {tickers}")
    except FileNotFoundError as e:
        print(e)
        exit(1)

    # Step 2: Fetch and update data for the tickers
    data = fetch_and_update_data_hdf5(tickers, start_date="2020-01-01")
    
    # Step 3: Confirm data is updated and ready
    print("Data has been updated and is ready for use.")
    print(data)
    # Step 4: Display the structure of the .h5 file
    
