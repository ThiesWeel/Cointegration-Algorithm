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
import numpy as np
from scipy.stats import pearsonr
from datetime import datetime
import warnings
from config import TICKER_FILE, BASE_DATABASE, BAYESIAN1_DIR, TICKERS_DIR
# Import logger_factory
from logger_factory import get_logger

# Configure loggers
file_logger = get_logger(f"bayesian_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}", f"logs/bayesian1/bayesian_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
summary_logger = get_logger("summary_logger", to_terminal=True)

def log_message(msg):
    file_logger.info(msg)

def log_summary(msg):
    summary_logger.info(msg)

# Ensure the folder exists
BAYESIAN1_DIR = os.path.join("dev_database", "bayesian1")
os.makedirs(BAYESIAN1_DIR, exist_ok=True)

def correlation_test(pre_processed_data, ticker1, ticker2):
    """
    Calculate the correlation between two time series and log any warnings.
    """
    log_message(f"Calculating correlation between {ticker1} and {ticker2}")
    
    # Ensure the ticker names match the keys in pre_processed_data
    ticker1_key = ticker1.lstrip("/")  # Remove leading '/' if present
    ticker2_key = ticker2.lstrip("/")
    
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
    # Sanitize end_date for filename (remove spaces, colons, keep only date part)
    safe_end_date = str(end_date).replace(":", "-").replace(" ", "_").split(" ")[0]
    DATA_FILE = f'Bayesian1_interference_table_{safe_end_date}.csv'
    table_path = os.path.join(BAYESIAN1_DIR, DATA_FILE)

    # Scenario 1: table for this data already exists, with most recent data.
    if os.path.exists(table_path):
        log_message(f"Loading existing table from {table_path}")
        prior_table = pd.read_csv(table_path, index_col=0)

    # Scenario 2: table for this data does not exist, create a new one.
    else:
        log_message(f"Table {DATA_FILE} not found. Creating a new one.")
        
        # Step 0.1: Check if the ticker file exists
        ticker_file_path = os.path.join( TICKERS_DIR, TICKER_FILE)

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
        log_message(f"Loaded {len(tickers)} tickers from {ticker_file_path}.")

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
                    prior_table.iloc[i, j] = 0.8
                    prior_table.iloc[j, i] = 0.8  # Ensure symmetry

    # Step 1: Calculate correlations for priors above threshold
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

    # Save the updated table
    post_table = prior_table
    os.makedirs(BAYESIAN1_DIR, exist_ok=True)
    post_table.to_csv(os.path.join(BAYESIAN1_DIR, DATA_FILE), index=True)

    # Identify correlated pairs
    correlated_pairs = []
    for i in range(len(post_table.columns)):
        for j in range(i + 1, len(post_table.columns)):
            if 0.7 < abs(post_table.iloc[i, j]) < 1:
                ticker1 = post_table.columns[i]
                ticker2 = post_table.columns[j]
                correlated_pairs.append(tuple(sorted((ticker1, ticker2))))

    # Print the final table to the terminal
    print("\nFinal correlation table:")
    print(post_table)

    # Also log the table to the log file
    log_message("Final correlation table:\n" + post_table.to_string())

    # Generate and log the summary
    summary = (
        f"\nSummary:\n"
        f"Date range: {end_date}\n"
        f"Total tickers processed: {len(post_table.columns)}\n"
        f"Correlated pairs identified: {correlated_pairs}\n"
        f"Updated table saved to: {os.path.join(BAYESIAN1_DIR, DATA_FILE)}\n"
    )
    log_message(summary)
    log_summary(summary)

    return correlated_pairs



