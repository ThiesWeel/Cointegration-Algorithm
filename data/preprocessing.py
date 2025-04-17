import pandas as pd
import os
import sys

# Add the parent directory to the system path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_DATA_DIR, FORECASTS_DIR, SIGNALS_DIR, TRADES_DIR, MODELS_DIR, PREDICTIONS_DIR, TICKERS_DIR, TICKER_FILE

import numpy as np
from scipy.stats import pearsonr
from loader import fetch_and_update_data_hdf5
import pprint
"""
Data Preprocessing

Cleans and standardizes raw stock and option data.
Handles:
- Industry filtering
- Handling NaNs and missing values
- Rolling window slicing for time-series operations
- Bayesian inference for identifying stock pairs for cointegration analysis
"""

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
    
# Step 1: Load tickers
try:
    tickers = load_tickers()
    print(f"Loaded tickers: {tickers}")
except FileNotFoundError as e:
    print(e)
    exit(1)

# Step 2: Fetch and update data for the tickers
data = fetch_and_update_data_hdf5(tickers, start_date="2023-01-01")
# Step 3: Confirm data is updated and ready
print("Data has been updated and is ready for use.")

def clean_data(data):
    """
    Cleans the raw data by handling NaNs and filtering invalid entries.
    
    Args:
        data (dict): Dictionary containing raw stock and option data.
    
        pd.DataFrame: Cleaned data.
    """
    df = pd.DataFrame(data)
    df = df.dropna()  # Drop rows with NaN values
    return df
