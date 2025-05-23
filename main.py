from config import RAW_DATA_DIR, FORECASTS_DIR, SIGNALS_DIR, TRADES_DIR, MODELS_DIR,PREDICTIONS_DIR, TICKERS_DIR, TICKER_FILE
import os
import pandas as pd
import yfinance as yf
import pprint
import h5py
from data.loader import fetch_and_update_data_hdf5, fetch_data_hdf5_between_dates # Import the function from loader.py
from data.preprocessing import  run_preprocessing, run_preprocessing_between_dates  # Import the function from preprocessing.py
from models.bayesian_filter_cointegration import bayesian_interference_check  # Import the function from bayesian_filter_cointegration.py       
from datetime import datetime,timedelta  # Import datetime to define 'today'
from models.cointegration import cointegration_checker  # Import the function from cointegration.py
from models.ecm import ECM_model  # Import the function from ecm.py
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
    data = fetch_data_hdf5_between_dates(tickers, start_date='2020-04-20',end_date='2021-04-15')
    print(data[list(data.keys())[-1]].head())  # Print the first few rows of the last dataset
    # Step 3: Preprocessing, remove NaN and take the lgiast #max_windowsize days
    # Starting from Starting date, taking only the needed data from the database


    pre_processed = run_preprocessing_between_dates(data,end_date = '2021-04-15', start_date='2020-04-20')
    print(pre_processed[list(data.keys())[-1]].head())
    # Step 4: Use Bayesian interference model to check and update possible cointegration pairs
    possible_cointegration_pairs = bayesian_interference_check(pre_processed, end_date='2021-04-15')
    
    # Step 5: Check for Cointegration pairs
    cointegration_res = cointegration_checker(pre_processed, possible_cointegration_pairs, window_sizes=[250, 70, 40], sig_lvl= 0.05)
    print("Cointegration results:", cointegration_res)
    print(cointegration_res)

    # Step 6: Setup spread simullation for backtesting
    #ECM_model(cointegration_res, pre_processed)
