"""
Cointegration Analysis

Performs correlation filtering and cointegration testing:
- Pearson correlation within industry
- Engle-Granger two-step test
- Z-score computation for spread
- Pair selection logic
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
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.stattools import adfuller
from logger_factory import get_logger

today = datetime.today().strftime('%Y-%m-%d')
# Add the parent directory to the system path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import COINTEGRATION_DIR, TICKER_FILE, TICKERS_DIR, BASE_DATABASE

# Configure loggers
file_logger = get_logger(f"cointegration_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}", f"logs/cointegration/cointegration_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", to_terminal=False)
summary_logger = get_logger("summary_logger", to_terminal=True)

def log_message(msg):
    file_logger.info(msg)

def log_summary(msg):
    summary_logger.info(msg)

def cointegration_test(closing_price_1, closing_price_2, window_sizes=None, significance_level=0.05):
    """
    Runs Engle-Granger cointegration tests at specified fixed window sizes.

    Parameters:
    - closing_price_1, closing_price_2: Series or DataFrames with closing prices and datetime index.
    - window_sizes: list of window sizes (e.g., [30, 60, 250]). Default: [30, 60, 120]
    - significance_level: threshold for p-value to declare cointegration.

    Returns:
    - results_dict: {
        'windowsizes': [...],
        'p_values': [...],
        'cointegration_vectors': [...],
        'date_ranges': [...],
        'adf_stats': [...],
        'adf_pvalues': [...],
        'residual_std': [...],
    }
    """
   
    # Align the two series
    aligned = pd.concat([closing_price_1, closing_price_2], axis=1, join='inner').dropna()
    aligned.columns = ['A', 'B']

    results_dict = {
        'windowsizes': [],
        'p_values': [],
        'cointegration_vectors': [],
        'date_ranges': [],
        'adf_stats': [],
        'adf_pvalues': [],
        'residual_std': [],
    }

    for win in window_sizes:
        if len(aligned) < win:
            log_message(f"Not enough data for window size {win}. Skipping.")    
            continue  # skip if not enough data

        window_data = aligned.iloc[-win:]
        # Engle-Granger cointegration test
        coint_res = coint(window_data['A'], window_data['B'])
        pval = coint_res[1]
        # OLS regression to get cointegration vector (beta)
        X = np.vstack([window_data['B'], np.ones(len(window_data['B']))]).T
        beta, alpha = np.linalg.lstsq(X, window_data['A'], rcond=None)[0]
        # Calculate residuals and ADF test on residuals
        residuals = window_data['A'] - (beta * window_data['B'] + alpha)
        adf_res = adfuller(residuals)
        adf_stat = adf_res[0]
        adf_pval = adf_res[1]
        residual_std = np.std(residuals)

        results_dict['windowsizes'].append(win)
        results_dict['p_values'].append(pval)
        results_dict['cointegration_vectors'].append((beta, alpha))
        results_dict['date_ranges'].append(f"{window_data.index[0].date()} - {window_data.index[-1].date()}")
        results_dict['adf_stats'].append(adf_stat)
        results_dict['adf_pvalues'].append(adf_pval)
        results_dict['residual_std'].append(residual_std)

    return results_dict

def cointegration_checker(pre_processed, possible_pairs, end_date, window_sizes, sig_lvl=0.05, plot=False):
    import matplotlib.pyplot as plt

    log_message(f"Running cointegration_checker with window_sizes: {window_sizes}")
    results = {}

    for pair in possible_pairs:
        ticker1, ticker2 = pair
        if ticker1 not in pre_processed or ticker2 not in pre_processed:
            log_message(f"Ticker {ticker1} or {ticker2} not found in pre-processed data.")
            continue

        closing_price_1 = pre_processed[ticker1][f"Close_{ticker1}"]
        closing_price_2 = pre_processed[ticker2][f"Close_{ticker2}"]

        # Align and drop NA
        aligned = pd.concat([closing_price_1, closing_price_2], axis=1, join='inner').dropna()
        aligned.columns = ['A', 'B']

        if len(aligned) < max(window_sizes):
            log_message(f"Not enough data for pair {pair}. Skipping.")
            continue

        # Use initial_check_cycle_detector to get windows based on crossings
        cycle_results = initial_check_cycle_detector(
            pre_processed, [pair], end_date, window_sizes, sig_lvl=sig_lvl, plot=False
        )
        pair_windows = cycle_results.get(pair, [])
        results[pair] = pair_windows

        if len(pair_windows) == 0:
            continue

        # Plot all windows in the same figure
        fig, ax = plt.subplots(figsize=(16, 7))
        for idx, win in enumerate(pair_windows):
            mask = (aligned.index >= win['window_start']) & (aligned.index <= win['window_end'])
            sub_aligned = aligned.loc[mask]
            if len(sub_aligned) == 0:
                continue
            spread = sub_aligned['A'] - (win['beta'] * sub_aligned['B'] + win['alpha'])
            spread_norm = (spread - spread.mean()) / spread.std()
            ax.plot(
                sub_aligned.index, spread_norm, 
                label=f"Win {idx+1}: {win['window_start'].date()} to {win['window_end'].date()} | "
                      f"α={win['alpha']:.2f}, β={win['beta']:.2f}, p={win['p_value']:.4f}, days={win['window_size']}"
            )

        ax.axhline(0, color='gray', linestyle='--', linewidth=1, label='Zero')
        ax.axhline(0.8, color='gray', linestyle='--', linewidth=1, label='+0.8')
        ax.axhline(-0.8, color='gray', linestyle='--', linewidth=1, label='-0.8')
        ax.set_title(f"{ticker1}-{ticker2} | Cointegration windows")
        ax.legend(fontsize=9)
        plt.tight_layout()
        plt.show()

    log_summary(f"Cointegration check complete for {len(results)} pairs.")
    return results


    
def detect_hysteresis_crossings(spread_norm):
    """
    Detects zero crossings after being above +0.8 or below -0.8 (hysteresis logic).
    Returns the indices where crossings occur.
    """
    above = spread_norm > 0.8
    below = spread_norm < -0.8
    state = 0  # 0: neutral, 1: above, -1: below
    crossings = []
    last_cross = -100  # initialize to a large negative value
    for i in range(1, len(spread_norm)):
        if state == 0:
            if above.iloc[i-1]:
                state = 1
            elif below.iloc[i-1]:
                state = -1
        elif state == 1:
            if spread_norm.iloc[i-1] > 0 and spread_norm.iloc[i] <= 0:
                if i - last_cross > 40:
                    crossings.append(i)
                    last_cross = i
                state = 0
        elif state == -1:
            if spread_norm.iloc[i-1] < 0 and spread_norm.iloc[i] >= 0:
                if i - last_cross > 40:
                    crossings.append(i)
                    last_cross = i
                state = 0
        # Update state if we go back above/below threshold
        if spread_norm.iloc[i] > 0.8:
            state = 1
        elif spread_norm.iloc[i] < -0.8:
            state = -1
    return crossings

def initial_check_cycle_detector(pre_processed, possible_pairs, end_date, window_sizes, sig_lvl=0.05, plot=False):
    """
    For each pair, detect crossings, and for each crossing (from the right), 
    use the 2nd, 3rd, ... crossing as the window start and the most recent date as the window end.
    For each such window, run the cointegration test and store the results.
    Returns a dict: {(ticker1, ticker2): [ { 'window_start': ..., 'window_end': ..., 'window_size': ..., 'p_value': ..., ... }, ... ]}
    """
    import matplotlib.pyplot as plt
    log_message(window_sizes)
    results_dict = {}

    for pair in possible_pairs:
        ticker1, ticker2 = pair
        if ticker1 not in pre_processed or ticker2 not in pre_processed:
            log_message(f"Ticker {ticker1} or {ticker2} not found in pre-processed data.")
            continue
        closing_price_1 = pre_processed[ticker1][f"Close_{ticker1}"]
        closing_price_2 = pre_processed[ticker2][f"Close_{ticker2}"]

        # Use the largest window size for crossing detection
        max_win = max(window_sizes)
        aligned = pd.concat([closing_price_1, closing_price_2], axis=1, join='inner').dropna()
        aligned.columns = ['A', 'B']
        if len(aligned) < max_win:
            log_message(f"Not enough data for window size {max_win}. Skipping pair {pair}.")
            continue
        window_data = aligned.iloc[-max_win:]
        spread = window_data['A'] - (window_data['B'] * 1 + 0)  # Use beta=1, alpha=0 for crossing detection
        spread_norm = (spread - spread.mean()) / spread.std()
        crossings = detect_hysteresis_crossings(spread_norm)
        crossing_indices = [window_data.index[i] for i in crossings]

        # Reverse to start from the most recent crossing
        crossing_indices = crossing_indices[::-1]

        pair_results = []
        for i in range(1, len(crossing_indices)):
            window_start = crossing_indices[i]
            window_end = window_data.index[-1]
            # Get the window data
            mask = (aligned.index >= window_start) & (aligned.index <= window_end)
            sub_aligned = aligned.loc[mask]
            if len(sub_aligned) < min(window_sizes):
                continue
            # Run cointegration test on this window
            coint_res = coint(sub_aligned['A'], sub_aligned['B'])
            pval = coint_res[1]
            X = np.vstack([sub_aligned['B'], np.ones(len(sub_aligned['B']))]).T
            beta, alpha = np.linalg.lstsq(X, sub_aligned['A'], rcond=None)[0]
            residuals = sub_aligned['A'] - (beta * sub_aligned['B'] + alpha)
            adf_res = adfuller(residuals)
            adf_stat = adf_res[0]
            adf_pval = adf_res[1]
            residual_std = np.std(residuals)
            pair_results.append({
                'window_start': window_start,
                'window_end': window_end,
                'window_size': len(sub_aligned),
                'p_value': pval,
                'beta': beta,
                'alpha': alpha,
                'adf_stat': adf_stat,
                'adf_pval': adf_pval,
                'residual_std': residual_std
            })
            if plot:
                fig, ax = plt.subplots(figsize=(14, 5))
                spread = sub_aligned['A'] - (beta * sub_aligned['B'] + alpha)
                spread_norm = (spread - spread.mean()) / spread.std()
                ax.plot(sub_aligned.index, spread_norm, label="Normalized Spread")
                ax.axhline(0, color='gray', linestyle='--', linewidth=1, label='Zero')
                ax.axhline(0.8, color='gray', linestyle='--', linewidth=1, label='+0.8')
                ax.axhline(-0.8, color='gray', linestyle='--', linewidth=1, label='-0.8')
                ax.set_title(f"{ticker1}-{ticker2} | Window: {window_start} to {window_end} | p-value: {pval:.4f}")
                ax.legend()
                plt.tight_layout()
                plt.show()
        results_dict[(ticker1, ticker2)] = pair_results

    return results_dict
