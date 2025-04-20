"""
Error Correction Model (ECM)

Fits ECM to cointegrated pairs:
- Estimates alpha (adjustment speed) and beta (spread relation)
- Extracts residuals
- Generates model structure for forecasting
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
    f"ecm_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    f"logs/ecm/ecm_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
    to_terminal=False
)
summary_logger = get_logger("summary_logger", to_terminal=True)

def log_message(msg):
    file_logger.info(msg)

def log_summary(msg):
    summary_logger.info(msg)


def ECM_model(cointegration_res, pre_processed):
    """
    Extracts all needed features for ECM modeling from cointegration results and preprocessed price data.
    Returns a dictionary with all relevant features for each cointegrated pair.
    """
    ecm_features = {}

    for pair, res in cointegration_res.items():
        ticker1, ticker2 = pair
        # Extract weighted alpha and beta (long-term equilibrium parameters)
        alpha = res['weighted']['alpha']
        beta = res['weighted']['beta']
        weights = res['weighted']['weights']
        used_pvals = res['weighted']['used_pvals']
        per_window = res['per_window']

        # Get aligned price series for the pair
        price1 = pre_processed[ticker1][f"Close_{ticker1}"]
        price2 = pre_processed[ticker2][f"Close_{ticker2}"]
        aligned = pd.concat([price1, price2], axis=1, join='inner').dropna()
        aligned.columns = [ticker1, ticker2]

        # Compute spread (residuals) using weighted alpha and beta
        spread = aligned[ticker1] - (beta * aligned[ticker2] + alpha)

        # Store all features needed for ECM
        ecm_features[pair] = {
            'alpha': alpha,
            'beta': beta,
            'weights': weights,
            'used_pvals': used_pvals,
            'per_window': per_window,
            'price1': aligned[ticker1],
            'price2': aligned[ticker2],
            'spread': spread
        }

        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [1, 1]})

        # Top subplot: Spread
        axs[0].plot(spread.index, spread.values, label=f"Spread: {ticker1} - β*{ticker2} - α", color='tab:blue')
        axs[0].set_title(f"Spread for pair {ticker1} & {ticker2}")
        axs[0].set_ylabel("Spread")
        axs[0].legend()

        # Bottom subplot: Price action
        axs[1].plot(aligned.index, aligned[ticker1], label=ticker1, color='tab:orange')
        axs[1].plot(aligned.index, aligned[ticker2], label=ticker2, color='tab:green')
        axs[1].set_title(f"Price Action: {ticker1} & {ticker2}")
        axs[1].set_xlabel("Date")
        axs[1].set_ylabel("Price")
        axs[1].legend()

        plt.tight_layout()
        plt.show()
    return ecm_features


