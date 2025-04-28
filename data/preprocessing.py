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
from logger_factory import get_logger

today = datetime.today().strftime('%Y-%m-%d')  # Define 'today' as the current date

# Add the parent directory to the system path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DATA_DIR, TICKER_FILE, TICKERS_DIR, BASE_DATABASE

# Data file name
DATA_FILE = f"{TICKER_FILE}_raw.h5"

# Configure loggers
file_logger = get_logger(f"preprocessing_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}", f"logs/preprocessing/preprocessing_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", to_terminal=False)
summary_logger = get_logger("summary_logger", to_terminal=True)

def log_message(msg):
    file_logger.info(msg)

def log_summary(msg):
    summary_logger.info(msg)

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
    log_message("Starting NaN handling...")  # Use file logger

    # Step 1: Identify all dates with NaN values across all DataFrames
    all_nan_dates = set()
    for ticker, df in data.items():
        nan_dates = df[df.isna().any(axis=1)].index  # Dates with NaN values in this DataFrame
        all_nan_dates.update(nan_dates)  # Add these dates to the global set
        log_message(f"Ticker: {ticker} - Found {len(nan_dates)} NaN dates.")  # Use file logger

    log_message(f"Total NaN dates to remove: {len(all_nan_dates)}")  # Use file logger

    # Step 2: Remove these dates from all DataFrames
    cleaned_data = {}
    for ticker, df in data.items():
        cleaned_df = df.drop(index=all_nan_dates, errors="ignore")  # Drop NaN dates
        cleaned_data[ticker] = cleaned_df
        log_message(f"Ticker: {ticker} - Rows after NaN removal: {len(cleaned_df)}")  # Use file logger

    log_message("NaN handling completed.")  # Use file logger
    return cleaned_data

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
    log_message("Starting preprocessing...")
    # Data file name
    start_date = (pd.to_datetime(end_date) - timedelta(days=max_cointegration_window_size)).strftime('%Y-%m-%d')
    DATA_FILE = f"{start_date}_to_{end_date}_processed.h5"

    # Initialize summary variables
    tickers_processed = []
    rows_removed = {}
    rows_remaining = {}

    # Step 0: Check if the data has been preprocessed before
    if os.path.exists(os.path.join(PROCESSED_DATA_DIR, DATA_FILE)):
        log_message(f"Preprocessed data file {DATA_FILE} already exists. Loading from file.")
        with pd.HDFStore(os.path.join(PROCESSED_DATA_DIR, DATA_FILE), mode="r") as store:
            # Normalize keys by stripping leading `/`
            preprocessed_data = {key.lstrip("/"): store[key] for key in store.keys()}
        log_summary(f"Preprocessed data loaded from {DATA_FILE}.")
        return preprocessed_data

    else:
        # Step 1: Handle NaN values
        log_message("Starting NaN handling...")
        cleaned_data = _handle_nan_values(data)

        # Collect NaN removal stats
        for ticker, df in data.items():
            rows_removed[ticker] = len(df) - len(cleaned_data[ticker])

        # Step 2: Slice data based on max_cointegration_window_size and start_date
        log_message("Slicing data based on max_cointegration_window_size and start_date...")
        sliced_data = {}
        for ticker, df in cleaned_data.items():
            if start_date in df.index:
                df = df.loc[start_date:]  # Slice from the start_date
            df = df.tail(max_cointegration_window_size)  # Keep only the last N days

            # Apply log transform to all columns except those containing 'Volume'
            for col in df.columns:
                if 'Volume' not in col:
                    df[col] = np.log(df[col])

            sliced_data[ticker] = df
            rows_remaining[ticker] = len(df)
            tickers_processed.append(ticker)
            log_message(f"Ticker: {ticker} - Final row count after slicing: {len(df)}")

        log_message("Data preprocessing completed.")

        # Step 3: Save preprocessed data if output_file is provided
        if output_file:
            _save_data_hdf5(sliced_data, DATA_FILE)
            log_message(f"Preprocessed data saved to {DATA_FILE}.")

        # Generate a summary
        summary = (
            f"\nSummary:\n"
            f"Date range: {start_date} to {end_date}\n"
            f"Tickers processed: {tickers_processed}\n"
            f"Rows removed due to NaN values: {rows_removed}\n"
            f"Rows remaining after slicing: {rows_remaining}\n"
            f"Preprocessed data saved to: {DATA_FILE if output_file else 'Not saved'}\n"
        )

        # Log the summary to the file and display it in the terminal
        log_message(summary)
        log_summary(summary)

        # Normalize keys by stripping leading `/`
        return {key.lstrip("/"): df for key, df in sliced_data.items()}

def run_preprocessing_between_dates(data, start_date, end_date, output_file=True):
    """
    Cleans and slices the data for cointegration analysis by:

    1. Removing rows with NaN values using _handle_nan_values.
    2. Slicing the data between start_date and end_date (inclusive).
    3. Applies log transform to all columns except those containing 'Volume'.

    Optionally saves the preprocessed data to a file.

    Args:
        data (dict): Dictionary of DataFrames for each ticker.
        start_date (str): Starting date for slicing (inclusive).
        end_date (str): Ending date for slicing (inclusive).
        output_file (bool): Whether to save the preprocessed data to a file.

    Returns:
        dict: Preprocessed data with normalized keys (no leading `/`).
    """
    log_message("Starting preprocessing (between dates)...")
    DATA_FILE = f"{start_date}_to_{end_date}_processed.h5"

    tickers_processed = []
    rows_removed = {}
    rows_remaining = {}

    # Step 0: Check if the data has been preprocessed before
    if os.path.exists(os.path.join(PROCESSED_DATA_DIR, DATA_FILE)):
        log_message(f"Preprocessed data file {DATA_FILE} already exists. Loading from file.")
        with pd.HDFStore(os.path.join(PROCESSED_DATA_DIR, DATA_FILE), mode="r") as store:
            preprocessed_data = {key.lstrip("/"): store[key] for key in store.keys()}
        log_summary(f"Preprocessed data loaded from {DATA_FILE}.")
        return preprocessed_data

    else:
        # Step 1: Handle NaN values
        log_message("Starting NaN handling...")
        cleaned_data = _handle_nan_values(data)

        # Collect NaN removal stats
        for ticker, df in data.items():
            rows_removed[ticker] = len(df) - len(cleaned_data[ticker])

        # Step 2: Slice data between start_date and end_date
        log_message("Slicing data between start_date and end_date...")
        sliced_data = {}
        for ticker, df in cleaned_data.items():
            df = df.loc[(df.index >= start_date) & (df.index <= end_date)]
            # Apply log transform to all columns except those containing 'Volume'
            for col in df.columns:
                if 'Volume' not in col:
                    df[col] = np.log(df[col])
            sliced_data[ticker] = df
            rows_remaining[ticker] = len(df)
            tickers_processed.append(ticker)
            log_message(f"Ticker: {ticker} - Final row count after slicing: {len(df)}")

        log_message("Data preprocessing (between dates) completed.")

        # Step 3: Save preprocessed data if output_file is provided
        if output_file:
            _save_data_hdf5(sliced_data, DATA_FILE)
            log_message(f"Preprocessed data saved to {DATA_FILE}.")

        # Generate a summary
        summary = (
            f"\nSummary:\n"
            f"Date range: {start_date} to {end_date}\n"
            f"Tickers processed: {tickers_processed}\n"
            f"Rows removed due to NaN values: {rows_removed}\n"
            f"Rows remaining after slicing: {rows_remaining}\n"
            f"Preprocessed data saved to: {DATA_FILE if output_file else 'Not saved'}\n"
        )

        log_message(summary)
        log_summary(summary)

        return {key.lstrip("/"): df for key, df in sliced_data.items()}
