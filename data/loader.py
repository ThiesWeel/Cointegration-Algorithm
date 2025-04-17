"""
Data Loader

This module handles historical stock data acquisition via the yfinance API.
Includes functions to:
- Download daily resolution data
- Perform rolling updates of the last 7 days
- Cache data locally in Parquet or HDF5 format
"""

import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from config import BASE_DATABASE  # Ensure this is defined in your config.py

# Ensure the folder exists
LOG_DIR = "logs/loader"
os.makedirs(LOG_DIR, exist_ok=True)

# Create the log file path
LOG_FILE = os.path.join(
    LOG_DIR,
    f"loader_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
)


def log_message(message):
    """
    Log a message to the log file.
    """
    with open(LOG_FILE, "a") as log_file:
        log_file.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")

def check_existing_data(ticker):
    """
    Check if data for the given ticker exists locally.
    """
    log_message(f"Checking if data for {ticker} exists locally...")
    file_path = os.path.join(BASE_DATABASE, f"{ticker}.parquet")
    if os.path.exists(file_path):
        log_message(f"Data for {ticker} found locally.")
        return pd.read_parquet(file_path)
    log_message(f"No local data found for {ticker}.")
    return None

def check_existing_data_hdf5(ticker):
    """
    Check if data for the given ticker exists locally in HDF5 format.
    """
    log_message(f"Checking if data for {ticker} exists locally in HDF5 format...")
    file_path = os.path.join(BASE_DATABASE, "data.h5")
    if os.path.exists(file_path):
        with pd.HDFStore(file_path, mode="r") as store:
            if ticker in store:
                log_message(f"Data for {ticker} found locally.")
                return store[ticker]
    log_message(f"No local data found for {ticker}.")
    return None


def download_data(ticker, start_date, end_date):
    """
    Download historical stock data using yfinance.
    """
    log_message(f"Downloading data for {ticker} from {start_date} to {end_date}...")
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if not data.empty:
            log_message(f"Data for {ticker} downloaded successfully.")
            return data
        else:
            log_message(f"No data downloaded for {ticker}.")
            return None
    except json.JSONDecodeError as e:
        log_message(f"Failed to download data for {ticker}: {e}")
        return None

def save_data(ticker, data):
    """
    Save data locally in Parquet format.
    """
    log_message(f"Saving data for {ticker} locally...")
    if not os.path.exists(BASE_DATABASE):
        os.makedirs(BASE_DATABASE)
        log_message(f"Created directory: {BASE_DATABASE}")
    file_path = os.path.join(BASE_DATABASE, f"{ticker}.parquet")
    data.to_parquet(file_path)
    log_message(f"Data for {ticker} saved to {file_path}.")

def save_data_hdf5(ticker, data):
    """
    Save data locally in HDF5 format.
    """
    log_message(f"Saving data for {ticker} locally in HDF5 format...")
    if not os.path.exists(BASE_DATABASE):
        os.makedirs(BASE_DATABASE)
        log_message(f"Created directory: {BASE_DATABASE}")
    file_path = os.path.join(BASE_DATABASE, "data.h5")
    with pd.HDFStore(file_path, mode="a") as store:
        store.put(ticker, data, format="table", data_columns=True)
    log_message(f"Data for {ticker} saved to {file_path}.")

def update_data(ticker):
    """
    Update the last 7 days of data for the given ticker.
    """
    log_message(f"Updating data for {ticker}...")
    existing_data = check_existing_data(ticker)
    if existing_data is not None:
        last_date = existing_data.index[-1]
        start_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
        log_message(f"Existing data found. Updating from {start_date}.")
    else:
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")  # Default to 1 year back
        log_message(f"No existing data found. Downloading data from {start_date}.")
    end_date = datetime.now().strftime("%Y-%m-%d")
    new_data = download_data(ticker, start_date, end_date)
    if existing_data is not None and not new_data.empty:
        log_message(f"Combining existing data with newly downloaded data for {ticker}.")
        updated_data = pd.concat([existing_data, new_data]).drop_duplicates()
        save_data(ticker, updated_data)
        return updated_data
    elif not new_data.empty:
        log_message(f"New data for {ticker} downloaded and saved.")
    else:
        log_message(f"No new data available for {ticker}.")
    return new_data if not new_data.empty else existing_data

def update_data_hdf5(ticker):
    """
    Update the last 7 days of data for the given ticker in HDF5 format.
    """
    log_message(f"Updating data for {ticker}...")
    existing_data = check_existing_data_hdf5(ticker)
    if existing_data is not None:
        last_date = existing_data.index[-1]
        start_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
        log_message(f"Existing data found. Updating from {start_date}.")
    else:
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")  # Default to 1 year back
        log_message(f"No existing data found. Downloading data from {start_date}.")
    end_date = datetime.now().strftime("%Y-%m-%d")
    new_data = download_data(ticker, start_date, end_date)

    if new_data is not None and not new_data.empty:
        if existing_data is not None:
            log_message(f"Combining existing data with newly downloaded data for {ticker}.")
            updated_data = pd.concat([existing_data, new_data]).drop_duplicates()
            save_data_hdf5(ticker, updated_data)
            return updated_data
        else:
            log_message(f"New data for {ticker} downloaded and saved.")
            save_data_hdf5(ticker, new_data)
            return new_data
    else:
        log_message(f"No new data available for {ticker}.")
        return existing_data

def fetch_and_update_data(tickers):
    """
    Fetch and update data for a list of tickers in one step.
    Returns a dictionary of DataFrames for the tickers.
    """
    log_message(f"Starting data fetch and update for tickers: {tickers}")
    data_dict = {}
    unavailable_tickers = []

    for ticker in tickers:
        log_message(f"Processing ticker: {ticker}")
        try:
            data = update_data(ticker)
            if data is not None:
                data_dict[ticker] = data
                log_message(f"Data for {ticker} is ready.")
            else:
                log_message(f"No data available for {ticker}. Adding to unavailable tickers.")
                unavailable_tickers.append(ticker)
        except Exception as e:
            log_message(f"Error processing ticker {ticker}: {e}")
            unavailable_tickers.append(ticker)

    # Log unavailable tickers at the top of the log file
    if unavailable_tickers:
        with open(LOG_FILE, "r+") as log_file:
            content = log_file.read()
            log_file.seek(0, 0)
            log_file.write(f"Unavailable tickers: {unavailable_tickers}\n\n" + content)

    log_message("Data fetch and update completed.")
    return data_dict

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
                # Update only the last 7 days
                last_date = existing_data.index[-1]
                update_start_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
                log_message(f"Existing data found for {ticker}. Updating from {update_start_date}.")
                new_data = download_data(ticker, update_start_date, datetime.now().strftime("%Y-%m-%d"))
                if new_data is not None and not new_data.empty:
                    updated_data = pd.concat([existing_data, new_data]).drop_duplicates()
                    save_data_hdf5(ticker, updated_data)
                    data_dict[ticker] = updated_data
                    last_7_days_loaded.append(ticker)
                else:
                    log_message(f"No new data available for {ticker}.")
            else:
                # Load the full dataset starting from START_DATE
                log_message(f"No existing data found for {ticker}. Downloading full dataset from {start_date}.")
                new_data = download_data(ticker, start_date, datetime.now().strftime("%Y-%m-%d"))
                if new_data is not None and not new_data.empty:
                    save_data_hdf5(ticker, new_data)
                    data_dict[ticker] = new_data
                    full_dataset_loaded.append(ticker)
                else:
                    log_message(f"No data downloaded for {ticker}.")
                    unavailable_tickers.append(ticker)
        except Exception as e:
            log_message(f"Error processing ticker {ticker}: {e}")
            unavailable_tickers.append(ticker)

    # Write a concise summary to the log file
    with open(LOG_FILE, "a") as log_file:
        log_file.write("\nSummary:\n")
        log_file.write(f"Date range: {start_date} to {datetime.now().strftime('%Y-%m-%d')}\n")
        log_file.write(f"Tickers with last 7 days loaded: {last_7_days_loaded}\n")
        log_file.write(f"Tickers with full dataset loaded: {full_dataset_loaded}\n")
        log_file.write(f"Unavailable tickers: {unavailable_tickers}\n")

    log_message("Data fetch and update completed.")
    return data_dict