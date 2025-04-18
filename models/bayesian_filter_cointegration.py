"""
Bayesian Confidence Framework

Maintains posterior probabilities of cointegration:
- Uses priors based on prev. calculated correlation and sector similarity
- Updates posterior probabilities based on new data
- if posterior probability > threshold, consider the pair as cointegrated
- if there has been found a cointegrated pair, update the prior for that pair, set to 1 and always pass to cointegration test

- Stores and retrieves confidence scores for candidate pairs
"""
import pandas as pd
import os
import sys
import h5py
import logging
import numpy as np
from scipy.stats import pearsonr
from datetime import datetime
import warnings

today = datetime.today().strftime('%Y-%m-%d')  # Define 'today' as the current date

# Add the parent directory to the system path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BAYESIAN1_DIR, TICKER_FILE, TICKERS_DIR, BASE_DATABASE

# Create the log file path dynamically with a timestamp
LOG_FILE = os.path.abspath(os.path.join(
    "logs/bayesian1",
    f"bayesian1_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
))

# Ensure the logs directory exists
if not os.path.exists(os.path.dirname(LOG_FILE)):
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

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


# Ensure the folder existsF
BAYESIAN1_DIR = os.path.join(BASE_DATABASE, "bayesian1")
os.makedirs(BAYESIAN1_DIR, exist_ok=True)

def correlation_test(pre_processed_data, ticker1, ticker2):
    """
    Calculate the correlation between two time series and log any warnings.
    """
    log_message(f"INFO: Calculating correlation between {ticker1} and {ticker2}")
    
    # Ensure the ticker names match the keys in pre_processed_data
    ticker1_key = ticker1.lstrip("/")  # Remove leading '/' if present
    ticker2_key = ticker2.lstrip("/")
    
    # Ensure the data is in the correct format
    if ticker1_key not in pre_processed_data or ticker2_key not in pre_processed_data:
        log_message(f"ERROR: Tickers {ticker1} or {ticker2} not found in pre-processed data.")
        return None

    try:
        # Capture warnings from pearsonr
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")  # Capture all warnings
            
            # Access the 'Close' columns for the tickers
            close_ticker1 = pre_processed_data[ticker1_key][f'Close_{ticker1_key}']
            close_ticker2 = pre_processed_data[ticker2_key][f'Close_{ticker2_key}']
            
            # Calculate the correlation
            corr, _ = pearsonr(close_ticker1, close_ticker2)

            # Log any warnings raised by pearsonr
            for warning in w:
                log_message(f"WARNING: {warning.message}")

        return corr

    except ValueError as e:
        log_message(f"ERROR: ValueError while calculating correlation for {ticker1} and {ticker2}: {e}")
        return None


def bayesian_interference_check(pre_processed, end_date):
    """
    1. Check if DATA_FILE exists in table_path.
    2. If it exists, load the table.
    3. If it does not exist, create a new CSV file using tickers from TICKER_FILE.
    """
    log_message(f"DEBUG: Keys in pre_processed: {list(pre_processed.keys())}")
    DATA_FILE = f'{TICKER_FILE}_Bayesian1_intereference_table_{end_date}.csv'
    table_path = os.path.join(BAYESIAN1_DIR, DATA_FILE)

    # Scenario 1: table for this data already exists, with most recent data.
    if False: #os.path.exists(table_path):
        log_message(f"INFO: Loading existing table from {table_path}")
        # Step 0: load the table
        prior_table = pd.read_csv(table_path, index_col=0)

    # Scenario 2: table for this data does not exist, create a new one.
    else:
        log_message(f"INFO: Table {DATA_FILE} not found. Creating a new one.")
        
        # Step 0.1: Check if the ticker file exists
        ticker_file_path = os.path.join(TICKERS_DIR, TICKER_FILE)

        if not os.path.exists(ticker_file_path):
            log_message(f"ERROR: Ticker file {ticker_file_path} does not exist.")
            return None

        # Step 0.2: Load tickers and their metadata
        tickers_data = pd.read_csv(ticker_file_path, header=None)
        tickers = tickers_data.iloc[1:, 0].tolist()
        tickers2 = [key.lstrip("/") for key in pre_processed.keys()]
        industries = tickers_data.iloc[1:, 1].tolist()
        regions = tickers_data.iloc[1:, 2].tolist()

        # Filter out elements not in tickers2
        filtered_data = [(ticker, industry, region) for ticker, industry, region in zip(tickers, industries, regions) if ticker in tickers2]
        tickers, industries, regions = zip(*filtered_data) if filtered_data else ([], [], [])
        log_message(f'{tickers}, {industries}, {regions}')
        log_message(f"INFO: Loaded {len(tickers)} tickers from {ticker_file_path}.")

        # Step 0.3: Create a DataFrame with tickers as both rows and columns
        prior_table = pd.DataFrame(
            0.18,  # Default value for mismatched industry and region
            index=tickers,
            columns=tickers
        )

        # Step 0.4: Set diagonal to 1
        np.fill_diagonal(prior_table.values, 1)

        # Step 0.5: Update values based on industry and region match
        for i in range(len(tickers)):
            for j in range(i + 1, len(tickers)):  # Only iterate over elements above the diagonal
                if industries[i] == industries[j] and regions[i] == regions[j]:
                    prior_table.iloc[i, j] = 0.7
                    prior_table.iloc[j, i] = 0.7  # Ensure symmetry
                    log_message(f"INFO: Matched industry and region for {tickers[i]} and {tickers[j]}")
                else:
                    prior_table.iloc[i, j] = 0.18
                    prior_table.iloc[j, i] = 0.18
                    log_message(f"INFO: Mismatched industry and region for {tickers[i]}: {industries[i]}, {regions[i]} and {tickers[j]}: {industries[j]}, {regions[j]}.")

        log_message("INFO: Prior table created based on industry and region matches.")

    # Step 1: Calculate correlations for priors above threshold
    log_message("INFO: Starting correlation calculations...")
    correlation_results = []
    for i in range(len(prior_table.columns)):
        for j in range(i + 1, len(prior_table.columns)):
            ticker1 = prior_table.columns[i]
            ticker2 = prior_table.columns[j]

            # Check if the correlation is above the threshold
            if prior_table.iloc[i, j] > 0.5:
                corr = correlation_test(pre_processed, ticker1, ticker2)
                if corr is not None:
                    prior_table.iloc[i, j] = corr
                    prior_table.iloc[j, i] = corr
                    correlation_results.append((ticker1, ticker2, corr))
                    log_message(f"INFO: Correlation between {ticker1} and {ticker2}: {corr:.4f}")
                else:
                    log_message(f"ERROR: Correlation calculation failed for {ticker1} and {ticker2}.")
                    prior_table.iloc[i, j] = 0.18
                    prior_table.iloc[j, i] = 0.18

    log_message(f"INFO: Correlation calculations completed. Total correlations calculated: {len(correlation_results)}")

    post_table = prior_table

    # Step 2: Save the updated post_table with the name DATA_FILE in the BAYESIAN1_DIR
    post_table.to_csv(os.path.join(BAYESIAN1_DIR, DATA_FILE), index=True)
    log_message(f"INFO: Updated table saved to {os.path.join(BAYESIAN1_DIR, DATA_FILE)}")
   
    # Step 3: Check for correlated pairs in post_table and group them for the cointegration test
    correlated_pairs = []
    for i in range(len(post_table.columns)):
        for j in range(i + 1, len(post_table.columns)):
            if 0.8 < abs(post_table.iloc[i, j]) < 1:
                ticker1 = post_table.columns[i]
                ticker2 = post_table.columns[j]
                correlated_pairs.append(tuple(sorted((ticker1, ticker2))))

    log_message(f"INFO: Correlated pairs identified: {correlated_pairs}")
    print(post_table)
    return correlated_pairs



