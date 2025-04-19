from config import RAW_DATA_DIR, FORECASTS_DIR, SIGNALS_DIR, TRADES_DIR, MODELS_DIR,PREDICTIONS_DIR, TICKERS_DIR, TICKER_FILE
import os
import pandas as pd
import yfinance as yf
import pprint
import h5py
from data.loader import fetch_and_update_data_hdf5  # Import the function from loader.py
from data.preprocessing import  run_preprocessing  # Import the function from preprocessing.py
from models.bayesian_filter_cointegration import bayesian_interference_check  # Import the function from bayesian_filter_cointegration.py       
from datetime import datetime,timedelta  # Import datetime to define 'today'
from models.cointegration import cointegration_checker  # Import the function from cointegration.py
today = (datetime.today() - timedelta(100)).strftime('%Y-%m-%d')  # Define 'today' as the current date


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
   
    # Step 3: Preprocessing, remove NaN and take the lgiast #max_windowsize days
    # Starting from Starting date, taking only the needed data from the database
    pre_processed = run_preprocessing(data,end_date = today, max_cointegration_window_size=500)

    #print(pre_processed['XOM']["Close_XOM"])
    # Step 4: Use Bayesian interference model to check and update possible cointegration pairs
    possible_cointegration_pairs = bayesian_interference_check(pre_processed, end_date=today)

    for i,el in enumerate([1,23,3]):
        print(f"Pair {i}: {el}")  
    # Step 5: Check for Cointegration pairs
    cointegration_checker(pre_processed, possible_cointegration_pairs, end_date=today, window_sizes= [ 350,250, 180], sig_lvl= 0.15,plot=True)

