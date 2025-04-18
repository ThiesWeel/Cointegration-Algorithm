"""
preprocessing.py

This script preprocesses financial time series data stored in an HDF5 file.
"""

import pandas as pd
import os
import sys
import h5py
import logging
import numpy as np
from scipy.stats import pearsonr
from datetime import datetime, timedelta

today = datetime.today().strftime('%Y-%m-%d')  # Define 'today' as the current date

# Add the parent directory to the system path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DATA_DIR, TICKER_FILE, TICKERS_DIR, BASE_DATABASE

# Create the log file path dynamically with a timestamp
LOG_FILE = os.path.join(
    "logs/preprocessing",
    f"preprocessing_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
)

# Ensure the logs directory exists
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# Data file name
DATA_FILE = f"{TICKER_FILE}_raw.h5"

# Configure logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.getLogger().addHandler(console_handler)


def log_message(message):
    """
    Log a message to the log file.
    """
    with open(LOG_FILE, "a") as log_file:
        log_file.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")

# Ensure the folder exists
PROCESSED_DATA_DIR = os.path.join(BASE_DATABASE, "preprocessed")
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)



def _save_data_hdf5(data, file_name):
    """
    Save data locally in HDF5 format within the given directory and with the specified file name.
    """
    log_message(f"Saving data to {file_name} in HDF5 format...")
    file_path = os.path.join(PROCESSED_DATA_DIR, file_name)

    try:
        with pd.HDFStore(file_path, mode="w") as store:
            for ticker, df in data.items():
                store.put(ticker, df, format="table", data_columns=True)
        log_message(f"Data successfully saved to {file_path}.")
    except Exception as e:
        log_message(f"Error saving data to {file_path}: {e}")

def _handle_nan_values(data):
    """
    Removes all rows (dates) with NaN values across all DataFrames in the data dictionary.
    If a date has a NaN value in any DataFrame, it is removed from all DataFrames.

    Args:
        data (dict): Dictionary of DataFrames for each ticker.

    Returns:
        dict: Cleaned data with synchronized NaN removal.
    """
    logging.info("Starting NaN handling...")

    # Step 1: Identify all dates with NaN values across all DataFrames
    all_nan_dates = set()
    for ticker, df in data.items():
        nan_dates = df[df.isna().any(axis=1)].index  # Dates with NaN values in this DataFrame
        all_nan_dates.update(nan_dates)  # Add these dates to the global set
        logging.info(f"Ticker: {ticker} - Found {len(nan_dates)} NaN dates.")

    logging.info(f"Total NaN dates to remove: {len(all_nan_dates)}")

    # Step 2: Remove these dates from all DataFrames
    cleaned_data = {}
    for ticker, df in data.items():
        cleaned_df = df.drop(index=all_nan_dates, errors="ignore")  # Drop NaN dates
        cleaned_data[ticker] = cleaned_df
        logging.info(f"Ticker: {ticker} - Rows after NaN removal: {len(cleaned_df)}")

    logging.info("NaN handling completed.")
    return cleaned_data


#LOAD FUNCTIONALITY MISSING?!


def run_preprocessing(data, end_date, max_cointegration_window_size, output_file=True):
    """
    Cleans and slices the data for cointegration analysis by:

    1. Removing rows with NaN values using _handle_nan_values.
    2. Keeping only the last max_cointegration_window_size days (e.g., 256 days).
    3. Adjusting the slicing to start from the specified start_date.

    Optionally saves the preprocessed data to a file.

    Args:
        data (dict): Dictionary of DataFrames for each ticker.
        max_cointegration_window_size (int): Maximum window size for slicing.
        end_date (str): Ending date for slicing.
        output_file (bool): Whether to save the preprocessed data to a file.

    Returns:
        dict: Preprocessed data with normalized keys (no leading `/`).
    """
    # Data file name
    start_date = (pd.to_datetime(end_date) - timedelta(days=max_cointegration_window_size)).strftime('%Y-%m-%d')
    DATA_FILE = f"{start_date}_to_{end_date}_processed.h5"

    # Step 0: Check if the data has been preprocessed before
    if os.path.exists(os.path.join(PROCESSED_DATA_DIR, DATA_FILE)):
        logging.info(f"Preprocessed data file {DATA_FILE} already exists. Loading from file.")
        with pd.HDFStore(os.path.join(PROCESSED_DATA_DIR, DATA_FILE), mode="r") as store:
            # Normalize keys by stripping leading `/`
            return {key.lstrip("/"): store[key] for key in store.keys()}

    else:
        # Step 1: Handle NaN values
        logging.info("Starting NaN handling...")
        cleaned_data = _handle_nan_values(data)

        # Step 2: Slice data based on max_cointegration_window_size and start_date
        logging.info("Slicing data based on max_cointegration_window_size and start_date...")
        sliced_data = {}
        for ticker, df in cleaned_data.items():
            if start_date in df.index:
                df = df.loc[start_date:]  # Slice from the start_date
            sliced_data[ticker] = df.tail(max_cointegration_window_size)  # Keep only the last N days
            logging.info(f"Ticker: {ticker} - Final row count after slicing: {len(df)}")

        logging.info("Data preprocessing completed.")

        # Step 3: Save preprocessed data if output_file is provided
        if output_file:
            _save_data_hdf5(sliced_data, DATA_FILE)
            logging.info(f"Preprocessed data saved to {DATA_FILE}.")

        # Normalize keys by stripping leading `/`
        return {key.lstrip("/"): df for key, df in sliced_data.items()}
