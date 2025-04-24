"""
Cointegration Analysis Module

General Purpose:
----------------
This module provides functions to analyze cointegration relationships between pairs of financial time series.
It supports:
- Running Engle-Granger cointegration tests on multiple window sizes for each pair.
- Extracting per-window cointegration parameters (alpha, beta, p-value, etc.).
- Aggregating these parameters using p-value-based weighting to produce a robust, combined set of parameters for each pair.
- Logging all results for traceability and debugging.

Typical Usage:
--------------
- Use `run_cointegration_analysis` to process all pairs and get both per-window and weighted parameters.
- Use the results for further modeling, e.g., ECM or Monte Carlo forecasting.

Details:
--------
- Engle-Granger two-step test for cointegration.
- Z-score computation for spread.
- Pair selection logic based on cointegration test results.
"""

import pandas as pd
import os
import sys
import logging
import numpy as np
from scipy.stats import pearsonr
from datetime import datetime, timedelta
# Add the project root to sys.path so logger_factory can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logger_factory import get_logger
from statsmodels.tsa.stattools import coint, adfuller

today = datetime.today().strftime('%Y-%m-%d')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import COINTEGRATION_DIR, TICKER_FILE, TICKERS_DIR, BASE_DATABASE

# Configure loggers for file and terminal output
file_logger = get_logger(
    f"cointegration_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    f"logs/cointegration/cointegration_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
    to_terminal=False
)
summary_logger = get_logger("summary_logger", to_terminal=True)

def log_message(msg):
    file_logger.info(msg)

def log_summary(msg):
    summary_logger.info(msg)


def cointegration_test(closing_price_1, closing_price_2, window_sizes=None, significance_level=0.05):
    """
    Run Engle-Granger cointegration tests at specified fixed window sizes.

    Args:
        closing_price_1 (pd.Series): Price series for ticker 1.
        closing_price_2 (pd.Series): Price series for ticker 2.
        window_sizes (list): List of window sizes (int) to test.
        significance_level (float): Not used here, but can be used for filtering.

    Returns:
        list of dict: Each dict contains results for one window (alpha, beta, p-value, etc.).
    """
    if window_sizes is None:
        log_message("No window sizes provided. !!")
        return ValueError("window_sizes must be provided")
    # Align the two price series on their dates and drop missing values
    aligned = pd.concat([closing_price_1, closing_price_2], axis=1, join='inner').dropna()
    aligned.columns = ['A', 'B']

    results = []
    for win in window_sizes:
        if len(aligned) < win:
            log_message(f"Not enough data for window size {win}. Skipping.")
            continue
        # Take the last 'win' rows for the rolling window
        window_data = aligned.iloc[-win:]
        # Engle-Granger cointegration test
        coint_res = coint(window_data['A'], window_data['B'])
        pval = coint_res[1]
        # OLS regression to get cointegration vector (beta, alpha)
        X = np.vstack([window_data['B'], np.ones(len(window_data['B']))]).T
        beta, alpha = np.linalg.lstsq(X, window_data['A'], rcond=None)[0]
        # Calculate residuals and ADF test on residuals
        residuals = window_data['A'] - (beta * window_data['B'] + alpha)
        adf_res = adfuller(residuals)
        adf_stat = adf_res[0]
        adf_pval = adf_res[1]
        residual_std = np.std(residuals)
        # Store all results in a dict for this window
        results.append({
            'window_size': win,
            'window_start': window_data.index[0],
            'window_end': window_data.index[-1],
            'alpha': alpha,
            'beta': beta,
            'p_value': pval,
            'adf_stat': adf_stat,
            'adf_pval': adf_pval,
            'residual_std': residual_std
        })
        log_message(
            f"Win {win}: α={alpha:.4f}, β={beta:.4f}, p={pval:.4g}, "
            f"ADF={adf_stat:.4g}, ADF_p={adf_pval:.4g}, {window_data.index[0].date()} to {window_data.index[-1].date()}"
        )
    return results
def cointegration_checker(pre_processed, possible_pairs, window_sizes=[30, 60, 250], sig_lvl=0.05):
    """
    For each pair, run cointegration tests for each window size and return raw results.
    Only include pairs where ALL p-values are below the significance level.

    Args:
        pre_processed (dict): Dict of DataFrames for each ticker, with aligned log prices.
        possible_pairs (list): List of (ticker1, ticker2) tuples.
        window_sizes (list): List of window sizes to use.
        sig_lvl (float): Significance threshold for p-values.

    Returns:
        dict: 
          {
            (ticker1, ticker2): {
                'windows': [...],
                'per_window_results': [ ...results for each window... ],
                'log_prices': {
                    'A': pd.Series,  # log(P_ticker1)
                    'B': pd.Series   # log(P_ticker2)
                }
            }, ...
          }
    """
    results = {}
    for pair in possible_pairs:
        ticker1, ticker2 = pair
        if ticker1 not in pre_processed or ticker2 not in pre_processed:
            log_message(f"Ticker {ticker1} or {ticker2} not found in pre-processed data.")
            continue

        # Use already log-transformed and aligned series
        log_price_1 = pre_processed[ticker1][f"Close_{ticker1}"]
        log_price_2 = pre_processed[ticker2][f"Close_{ticker2}"]

        aligned_log_prices = pd.DataFrame({
            "A": log_price_1,
            "B": log_price_2
        })

        # Run cointegration tests
        per_window = cointegration_test(aligned_log_prices['A'], aligned_log_prices['B'], window_sizes, sig_lvl)

        # Skip if any window fails the p-value test
        if not per_window or any(r['p_value'] >= sig_lvl for r in per_window):
            log_message(f"Pair {pair}: Skipped because at least one p-value >= significance level ({sig_lvl}).")
            continue

        # Store raw results only — beta selection handled later
        results[pair] = {
            'windows': window_sizes,
            'per_window_results': per_window,
            'log_prices': {
                'A': aligned_log_prices['A'],
                'B': aligned_log_prices['B']
            }
        }

        log_message(f"Pair {pair}: Passed all p-value checks.")

    log_summary(
        f"Cointegration analysis complete for {len(results)} pairs. "
        f"Each pair includes per-window results and log prices."
    )

    return results

