"""
Data Loader

This module handles historical stock data acquisition via the yfinance API.
Includes functions to:
- Download daily resolution data
- Perform rolling updates of the last 7 days
- Cache data locally in HDF5 format
"""

import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import sys
import logging

# Add the parent directory to the system path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TICKER_FILE
from logger_factory import get_logger

# Define the base directory for the database
BASE_DATABASE = "dev_database"  # Change from "data" to "dev_database"

# Data file name
DATA_FILE = f"{TICKER_FILE}_raw.h5"

# Ensure the folder exists
RAW_DIR = os.path.join(BASE_DATABASE, "raw")
os.makedirs(RAW_DIR, exist_ok=True)

# Create the log file path
LOG_FILE = os.path.join(
    "logs/loader",
    f"loader_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
)

# Ensure the logs directory exists
if not os.path.exists(os.path.dirname(LOG_FILE)):
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# Configure loggers
file_logger = get_logger(f"loader_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}", f"logs/loader/loader_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
summary_logger = get_logger("summary_logger", to_terminal=True)

def log_message(msg):
    file_logger.info(msg)

def log_summary(msg):
    summary_logger.info(msg)

def check_existing_data_hdf5(ticker):
    """
    Check if data for the given ticker exists locally in HDF5 format within the 'raw' subdirectory.
    """
    log_message(f"Checking if data for {ticker} exists locally in HDF5 format within 'raw' subdirectory...")
    file_path = os.path.join(RAW_DIR, DATA_FILE)
    if os.path.exists(file_path):
        try:
            with pd.HDFStore(file_path, mode="r") as store:
                if ticker in store:
                    log_message(f"Data for {ticker} found locally in 'raw' subdirectory.")
                    data = store[ticker]
                    if isinstance(data, pd.DataFrame):
                        return data
                    else:
                        log_message(f"Data for {ticker} is not a valid DataFrame.")
                        return None
        except Exception as e:
            log_message(f"Error reading data for {ticker} from HDF5: {e}")
            return None
    log_message(f"No local data found for {ticker} in 'raw' subdirectory.")
    return None


def download_data(ticker, start_date, end_date):
    """
    Download historical stock data using yfinance.
    """
    log_message(f"Downloading data for {ticker} from {start_date} to {end_date}...")
    try:
        data = yf.download(ticker, start=start_date, end=end_date,progress=False,show_errors=False)
        if not data.empty:
            log_message(f"Data for {ticker} downloaded successfully.")
            return data
        else:
            log_message(f"No data downloaded for {ticker}.")
            return None
    except Exception as e:
        log_message(f"Failed to download data for {ticker}: {e}")
        return None


def save_data_hdf5(ticker, data):
    """
    Save data locally in HDF5 format within the 'raw' subdirectory.
    """
    log_message(f"Saving data for {ticker} locally in HDF5 format within 'raw' subdirectory...")
    file_path = os.path.join(RAW_DIR, DATA_FILE)

    # Ensure the data is a valid DataFrame
    if not isinstance(data, pd.DataFrame):
        log_message(f"Data for {ticker} is not a valid DataFrame. Skipping save.")
        return

    # Flatten multi-index columns if present
    if isinstance(data.columns, pd.MultiIndex):
        log_message(f"Flattening multi-index columns for {ticker}.")
        data.columns = ['_'.join(map(str, col)).strip() for col in data.columns]

    try:
        with pd.HDFStore(file_path, mode="a") as store:
            store.put(ticker, data, format="table", data_columns=True)
        log_message(f"Data for {ticker} saved to {file_path}.")
    except Exception as e:
        log_message(f"Error saving data for {ticker} to HDF5: {e}")


def update_data_hdf5(ticker):
    """
    Update the last 7 days of data for the given ticker in HDF5 format.
    If the data has been updated in the last 7 days, overwrite the old data for those days and append new data.
    """
    log_message(f"Updating data for {ticker}...")
    existing_data = check_existing_data_hdf5(ticker)
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")  # Always fetch the last 7 days

    # Remove the last 7 days from the existing data if it exists
    if existing_data is not None:
        log_message(f"Existing data found for {ticker}. Removing the last 7 days to prepare for update.")
        existing_data = existing_data[existing_data.index < start_date]

    # Download new data for the last 7 days
    log_message('debuggy')
    new_data = download_data(ticker, start_date, end_date)
    if new_data is not None and not new_data.empty:
        log_message(f"New data for {ticker} downloaded successfully. Combining with existing data.")
        
        if isinstance(new_data.columns, pd.MultiIndex):
            log_message(f"Flattening multi-index columns for new data of {ticker}.")
            new_data.columns = ['_'.join(map(str, col)).strip() for col in new_data.columns]
        
        # Combine the existing data (excluding the last 7 days) with the new data
        updated_data = pd.concat([existing_data, new_data]).drop_duplicates() if existing_data is not None else new_data
        save_data_hdf5(ticker, updated_data)
        log_message(f"Data for {ticker} updated successfully.")
        return updated_data
    else:
        log_message(f"No new data available for {ticker}.")
        return existing_data


def fetch_and_update_data_hdf5(tickers, start_date):
    """
    Fetch and update data for a list of tickers in one step using HDF5.
    Returns a dictionary of DataFrames for the tickers.
    """
    log_message(f"Starting data fetch and update for tickers: {tickers}")
    log_message(f"Date range: {start_date} to {datetime.now().strftime('%Y-%m-%d')}")

    data_dict = {}
    last_7_days_loaded = []
    full_dataset_loaded = []
    unavailable_tickers = []

    for ticker in tickers:
        log_message(f"Processing ticker: {ticker}")
        try:
            existing_data = check_existing_data_hdf5(ticker)
            if existing_data is not None:
                # Call the update_data_hdf5 function to handle the update logic
                updated_data = update_data_hdf5(ticker)
                if updated_data is not None:
                    data_dict[ticker] = updated_data
                    last_7_days_loaded.append(ticker)
                else:
                    log_message(f"No new data available for {ticker}.")
                    data_dict[ticker] = existing_data
            else:
                # Load the full dataset starting from START_DATE
                log_message(f"No existing data found for {ticker}. Downloading full dataset from {start_date}.")
                new_data = download_data(ticker, start_date, datetime.now().strftime("%Y-%m-%d"))
                if new_data is not None and not new_data.empty:
                    
                    if isinstance(new_data.columns, pd.MultiIndex):
                        log_message(f"Flattening multi-index columns for new data of {ticker}.")
                        new_data.columns = ['_'.join(map(str, col)).strip() for col in new_data.columns]
                    
                    save_data_hdf5(ticker, new_data)
                    log_message(f"Full dataset for {ticker} downloaded and saved.")
                    data_dict[ticker] = new_data
                    full_dataset_loaded.append(ticker)
                else:
                    log_message(f"No data downloaded for {ticker}.")
                    unavailable_tickers.append(ticker)
        except Exception as e:
            log_message(f"Error processing ticker {ticker}: {e}")
            unavailable_tickers.append(ticker)

    # Write a concise summary to the log file
    summary = (
        f"\nSummary:\n"
        f"Date range: {start_date} to {datetime.now().strftime('%Y-%m-%d')}\n"
        f"Tickers with last 7 days loaded: {last_7_days_loaded}\n"
        f"Tickers with full dataset loaded: {full_dataset_loaded}\n"
        f"Unavailable tickers: {unavailable_tickers}\n"
    )
    log_message(summary)  # Write the summary to the log file
    log_summary(summary)  # Print the summary to the terminal

    log_message("Data fetch and update completed.")
    return data_dict