from statsmodels.tsa.stattools import adfuller
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from itertools import combinations
import seaborn as sns
from tqdm import tqdm  # Progress bar
import cupy as cp  # GPU acceleration
from itertools import combinations

tickers_all = [
    "XOM",  "CVX",  "COP",  "OXY",  "PSX",  "MPC",  "VLO",  "EOG",
    "DVN",  "HES",  "APA",  "MRO",  "MUR",  "CHK",  "FANG", "EQT",
    "RRC",  "SWN",  "GPOR", "MTDR", "SM",   "AR",   "WTI",  "CRC", "CRK"
]
#tickers to consider
tickers = tickers_all[: ]
vol_tick = ["^OVX"] # Volatility measure

# FUCNTIONS TO DECIDE WHICH COINTEGRATIONS WE CHECK
def rank_by_correlation(data):
    """
    Ranks columns in a DataFrame by their average pairwise correlation
    based on the returns and returns the reordered DataFrame.
    
    Parameters:
    data (pd.DataFrame): Input DataFrame with numeric data (e.g., stock prices).
    
    Returns:
    pd.DataFrame: DataFrame with columns reordered by their average pairwise correlation of returns.
    """
    # Calculate daily returns
    returns = data.pct_change().dropna()
    
    # Calculate correlation matrix of returns
    correlation_matrix = returns.corr()
    
    # Calculate average correlation for each column
    avg_correlation = correlation_matrix.mean().sort_values(ascending=False)
    
    # Order columns based on average correlation
    ordered_columns = avg_correlation.index.tolist()
    
    # Return the reordered DataFrame based on original data
    return data[ordered_columns]

def ticker_cor_filter(data):
    """
    Filters the dataset to exclude dates where OVX is in the top 5%.
    Returns the filtered dataset for further analysis.
    
    Parameters:
    data (pd.DataFrame): The input DataFrame containing OVX and stock data.

    Returns:
    pd.DataFrame: The filtered dataset.
    """
    # Step 1: Calculate the 95th percentile threshold for OVX
    ovx_threshold = vol_data['^OVX'].quantile(0.95)

    # Step 2: Identify dates where OVX is in the top 5%
    high_ovx_dates = data[vol_data['^OVX'] > ovx_threshold].index

    # Step 3: Filter the dataset to exclude high-OVX dates
    filtered_data = data.drop(index=high_ovx_dates)

    return filtered_data


def get_top_groups_gpu(data, sizes, top_n_per_size=5):
    """
    Optimized GPU-accelerated version to generate groups of tickers
    and select top groups based on average correlation.
    """
    # Calculate daily returns and correlation matrix
    returns = cp.array(data.pct_change().dropna().values)  # Move data to GPU
    correlation_matrix = cp.corrcoef(returns, rowvar=False)  # GPU-based correlation matrix
    columns = data.columns.tolist()

    # Store results for each group size
    top_groups = {}

    for size in tqdm(sizes, desc="Processing Group Sizes"):
        ticker_combinations = list(combinations(range(len(columns)), size))
        
        group_corrs_gpu = []
        for group in tqdm(ticker_combinations, desc=f"Calculating Correlations for Size {size}", leave=False):
            indices = cp.array(group)
            sub_corr_matrix = correlation_matrix[indices][:, indices]
            avg_corr = (sub_corr_matrix.sum() - cp.trace(sub_corr_matrix)) / (size * (size - 1))
            group_corrs_gpu.append((group, avg_corr))

        # Move all GPU results back to CPU in one batch
        group_corrs = [(tuple(columns[i] for i in group), avg_corr.get()) for group, avg_corr in group_corrs_gpu]

        # Sort groups by average correlation and select top N
        group_corrs = sorted(group_corrs, key=lambda x: x[1], reverse=True)
        top_groups[size] = group_corrs[:top_n_per_size]

    return top_groups

def get_top_groups(data, sizes, top_n_per_size=5):
    """
    Generates groups of tickers of specified sizes and selects the top groups
    based on average group correlation.

    Parameters:
    data (pd.DataFrame): The input DataFrame with stock price or return data.
    sizes (list): List of group sizes to evaluate (e.g., [3, 4, 5]).
    top_n_per_size (int): Number of top groups to return per group size.

    Returns:
    dict: A dictionary where keys are group sizes and values are the top groups
          with their average correlations.
    """
    # Calculate daily returns and correlation matrix
    returns = data.pct_change().dropna()
    correlation_matrix = returns.corr()

    # Store results for each group size
    top_groups = {}

    # Loop through each group size with a progress bar
    for size in tqdm(sizes, desc="Processing Group Sizes"):
        # Generate all combinations of tickers for the given group size
        ticker_combinations = list(combinations(correlation_matrix.columns, size))
        
        # Calculate average correlation for each group with an inner progress bar
        group_corrs = []
        for group in tqdm(ticker_combinations, desc=f"Calculating Correlations for Size {size}", leave=False):
            # Extract the submatrix for this group
            sub_corr_matrix = correlation_matrix.loc[group, group]
            
            # Compute the average correlation (excluding diagonal values)
            avg_corr = (sub_corr_matrix.values.sum() - sub_corr_matrix.values.trace()) / (size * (size - 1))
            group_corrs.append((group, avg_corr))
        
        # Sort groups by average correlation (descending)
        group_corrs = sorted(group_corrs, key=lambda x: x[1], reverse=True)
        
        # Select the top N groups for this size
        top_groups[size] = group_corrs[:top_n_per_size]

    return top_groups


def filter_high_correlation_groups(cor_group_dict, considered_group_sizes, threshold=0.82):
    """
    Filters ticker groups based on a minimum average correlation threshold.

    Parameters:
    cor_group_dict (dict): Dictionary where keys are group sizes, and values
                           are lists of tuples (tickers, average correlation).
    considered_group_sizes (list): List of group sizes to consider for filtering.
    threshold (float): Minimum correlation value to include a group.

    Returns:
    list: A list of ticker groups (tuples) that meet the correlation threshold.
    """
    high_correlation_groups = []

    # Iterate through the specified group sizes
    for size in considered_group_sizes:
        if size in cor_group_dict:
            for group, avg_corr in cor_group_dict[size]:
                # Check if the group's average correlation exceeds the threshold
                if avg_corr > threshold:
                    high_correlation_groups.append(group)

    return high_correlation_groups

# FUNCTIONS TO calc COINTEGRATION eigenv. AND CALCULATE IT FOR 

def spread_calc_check_all_tf(tickers, days_to_check=60,plot = False):
    """
    Calculates the best eigenvector from the Johansen cointegration test for 
    multiple timeframes, evaluates stationarity using the ADF test, and checks 
    for significant spread deviations (|Z| > 2) in the last specified calendar days. 
    Optionally generates subplots for visualization.

    Parameters:
    tickers (list): List of tickers to analyze.
    days_to_check (int): Number of calendar days to check for |Z| > 2.
    plot (bool): If True and |Z| > 2 is true for any of the last days_to_check;
                 generates subplots showing normalized spreads and 
                 the volatility index (^OVX_Z). Default is False.

    Returns:
    tuple:
        metadata_list (list): Contains metadata for the group (e.g., tickers, ADF p-values, time frames).
        details_df (pd.DataFrame): DataFrame with raw spreads, normalized spreads, 
                                   ^OVX_Z values, and signal information.
                                   Returns None, None if conditions are not met.
    """
    def calculate_best_eigenvector(group_data_windowed):
        """
        Calculates the best eigenvector from the Johansen cointegration test.

        Parameters:
        group_data_windowed (pd.DataFrame or np.ndarray): The time series data for the tickers.

        Returns:
        np.ndarray: The best eigenvector (corresponding to the largest eigenvalue).
        """        
        data_array = group_data_windowed.values if hasattr(group_data_windowed, 'values') else group_data_windowed
        johansen_result = coint_johansen(data_array, det_order=0, k_ar_diff=1)
        eigenvectors = johansen_result.evec
        eigenvalues = johansen_result.eig
        best_index = eigenvalues.argmax()
        
        return eigenvectors[:, best_index]

    # Window sizes, i.e., last n days on which the spread is calculated and checked
    tf_days_list = [504, 378, 252]  # Longest to shortest window

    group_data = data[list(tickers)]

    # Prepare to check ADF test results and |Z| conditions
    all_adf_p_values = []
    spread_data = {}  # Store normalized spreads and ADF results
    z_signal_all = True  # Track if all timeframes show |Z| > 2 in the last calendar days

    # Define the cutoff date based on calendar days
    cutoff_date = group_data.index.max() - pd.Timedelta(days=days_to_check)

    # Normalize vol_data once for the largest window
    max_window = max(tf_days_list)
    vol_data_windowed = vol_data.tail(max_window)
    vol_data['^OVX_Z'] = (vol_data_windowed['^OVX'] - vol_data_windowed['^OVX'].mean()) / vol_data_windowed['^OVX'].std()

    for window in tf_days_list:
        # Filter data for the current window size
        group_data_windowed = group_data.tail(window)
        eigen_vect_windowed = calculate_best_eigenvector(group_data_windowed)
        spread = group_data_windowed.dot(eigen_vect_windowed)

        # Perform ADF test for stationarity
        adf_result = adfuller(spread)
        all_adf_p_values.append(adf_result[1])  # Collect p-values
        # Determine if all ADF p-values are below 0.05
        adf_all_significant = all(p < 0.05 for p in all_adf_p_values)
   
        # Normalize the spread
        spread_norm = (spread - spread.mean()) / spread.std()
        spread_data[window] = (spread_norm, adf_result)  # Store spread and ADF result

        # Filter by the last calendar days
        recent_spread = spread_norm[spread_norm.index > cutoff_date]

        # Check if |Z| > 2 in the recent calendar days for this timeframe
        if not (recent_spread.abs() > 2).any():
            z_signal_all = False  # If any timeframe fails, set to False

    # Plot only if all ADF tests are significant and all timeframes show |Z| > 2 in the last calendar days
    if adf_all_significant and z_signal_all and plot:
        fig, axes = plt.subplots(len(tf_days_list), 1, figsize=(16, 4 * len(tf_days_list)), sharex=True)
        axes = axes.flatten() if len(tf_days_list) > 1 else [axes]

        for idx, (window, ax) in enumerate(zip(tf_days_list, axes)):
            spread_norm, adf_result = spread_data[window]

            # Plot the spread
            ax.plot(spread_norm, label=f"Spread (Window: {window} days)", color="blue")

            # Plot normalized vol_data
            ax.plot(vol_data['^OVX_Z'], label="^OVX Z-Score", color="orange", alpha=0.8)

            # Add horizontal lines for thresholds
            ax.axhline(2, color="red", linestyle="--", alpha=0.5, label="Thresholds")
            ax.axhline(-2, color="blue", linestyle="--", alpha=0.5)

            # Add titles, labels, and legends
            ax.set_title(f"Spread for {window}-day Window, of group {tickers}\nADF Statistic: {adf_result[0]:.2f}, p-value: {adf_result[1]:.4f}")
            ax.set_ylabel("Spread / Z-Score")
            ax.legend()
            ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.7)

        # Set common x-axis label and x-axis limits to the largest window frame
        axes[-1].set_xlabel("Date")
        axes[0].set_xlim(vol_data.tail(max_window).index[0], vol_data.tail(max_window).index[-1])

        # Adjust layout and show the final figure
        plt.tight_layout()
        plt.show()
        
   

    if adf_all_significant and z_signal_all:
        metadata_list = [
            tickers,
            all_adf_p_values,  # List of ADF p-values for each time frame
            tf_days_list       # List of time frames considered
        ]
        
        # Initialize the details DataFrame
        spread_df = pd.DataFrame()

        # Add raw and normalized spreads for each time frame
        for window in tf_days_list:
            raw_spread = spread_data[window][0] * spread_data[window][0].std() + spread_data[window][0].mean()  # Reverse normalization
            spread_df[f"raw_spread_{window}"] = raw_spread
            spread_df[f"norm_spread_{window}"] = spread_data[window][0]

        # Add normalized vol_data
        spread_df["^OVX_Z"] = vol_data['^OVX_Z'].reindex(spread_df.index)

        # Add the signal column
        spread_df["Signal"] = spread_df[[f"norm_spread_{window}" for window in tf_days_list]].apply(
            lambda row: all(abs(val) > 2 for val in row), axis=1
        )
        details_df = spread_df

        print(f"Z/ADF met for {tickers}")
        return metadata_list, details_df

    else:
        print(f"Z/ADF not met for {tickers}")
        return None, None


def plot_norm_ts(norm_ret):
    # Plot normalized returns
    plt.figure(figsize=(14, 8))
    for column in norm_ret.columns:
        plt.plot(norm_ret.index, norm_ret[column], label=column, alpha=0.7, linewidth=1)

    # Add a title and labels
    plt.title("Normalized Returns of Selected Energy Stocks (2020-2023)", fontsize=16)
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Normalized Returns", fontsize=14)

    # Add a legend with a better layout
    plt.legend(loc='upper left', fontsize=8, ncol=2, frameon=True)

    # Improve the grid and overall appearance
    plt.grid(visible=True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    # Show the plot
    plt.show()

def plot_ts(data_):
    # Plot the raw time series data
    plt.figure(figsize=(14, 8))
    for column in data_.columns:
        plt.plot(data_.index, data_[column], label=column, alpha=0.7, linewidth=1)

    # Add title and labels
    plt.title("Time Series of Selected Energy Stocks (2020-2023)", fontsize=16)
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Stock Price", fontsize=14)

    # Add a legend with a better layout
    plt.legend(loc='upper left', fontsize=8, ncol=2, frameon=True)

    # Improve the grid and overall appearance
    plt.grid(visible=True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    # Show the plot
    plt.show()
 
def plot_norm_ts(norm_ret):
    # Plot normalized returns
    plt.figure(figsize=(14, 8))
    for column in norm_ret.columns:
        plt.plot(norm_ret.index, norm_ret[column], label=column, alpha=0.7, linewidth=1)

    # Add a title and labels
    plt.title("Normalized Returns of Selected Energy Stocks (2020-2023)", fontsize=16)
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Normalized Returns", fontsize=14)

    # Add a legend with a better layout
    plt.legend(loc='upper left', fontsize=8, ncol=2, frameon=True)

    # Improve the grid and overall appearance
    plt.grid(visible=True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    # Show the plot
    plt.show()

def ADF(spread_df):
    result = adfuller(spread_df['Spread'])
    print(f"ADF Statistic: {result[0]}")
    print(f"p-value: {result[1]}")
    print("Critical Values:", result[4])
    if result[1] < 0.05:
      print("The spread is stationary.")
    else:
      print("The spread is not stationary.")

########## Import data van yfinance
# Fetch historical daily close prices for each stock
start_date = "2020-01-01"
end_date = "2023-12-31"

data_unf = yf.download(tickers, start=start_date, end=end_date)['Close']
vol_data = yf.download(vol_tick, start=start_date, end=end_date)['Close']
data = ticker_cor_filter(data_unf)
vol_data = vol_data.reindex(data.index)

# Calculate daily returns
returns = data.pct_change().dropna()

########## Get highly correlated stock groups and filter on corr threshold
considered_group_sizes = [3 ]
cor_group_dict = get_top_groups_gpu(data,considered_group_sizes)
        
pot_coin_groups_list = filter_high_correlation_groups(cor_group_dict,considered_group_sizes)

########## Calc. spread through Johnson and Test Coint. 
# Initialize storage
all_metadata = []  # To store metadata_list from each iteration
all_details = []  # To store all details_df

for i in tqdm(range(len(pot_coin_groups_list)), desc="Testing Groups"):
    ticker_group = pot_coin_groups_list[i]
    metadata, details_df = spread_calc_check_all_tf(ticker_group,60, plot = True)

    if metadata is not None and details_df is not None:
        # Save metadata
        metadata_dict = {
            "tickers": metadata[0],
            "adf_p_values": metadata[1],
            "time_frames": metadata[2],
        }
        all_metadata.append(metadata_dict)

        # Save df with spread
        all_details.append(details_df)


def potential_profit(meta, df, entry_z=2, exit_z=0, z_momentum_threshold=1, plot=False, plot_extra_days=30):
    """
    Simulates trade entries and exits based on Z-score signals and momentum, 
    with entry prices taken from the next day's open price. Optionally generates
    plots of spreads and stock prices.

    Parameters:
    meta (list): List of tickers in the group to analyze.
    df (pd.DataFrame): DataFrame containing spread and Z-score values.
    entry_z (float): Z-score threshold to open a position.
    exit_z (float): Z-score threshold to close a position.
    z_momentum_threshold (float): Minimum Z-score momentum (change) required to open a trade.
    plot (bool): Whether to generate plots for each trade.
    plot_extra_days (int): Number of days before and after a trade to include in the plot.

    Returns:
    list: A list of trades, where each trade is a dictionary with entry/exit details.
    """
    # Add momentum column (change in Z-score)
    df['Z_moment'] = df['norm_spread_504'].diff()  # Using the longest window's normalized spread

    # Fetch next-day open prices for the tickers in meta
    tickers = meta['tickers']
    historical_data = yf.download(tickers, start=df.index.min(), end=df.index.max() + pd.Timedelta(days=1))['Open']

    trades = []  # Store completed trades
    open_trade = None  # Track the currently open trade

    for i, row in df.iterrows():
        z_score = row['norm_spread_504']  # Using the longest window's normalized spread
        z_momentum = row['Z_moment']

        # Check for entry signal: |Z| > entry_z and momentum threshold met
        if open_trade is None and abs(z_score) > entry_z and abs(z_momentum) > z_momentum_threshold:
            # Get next day's open price
            next_day_idx = df.index.get_loc(i) + 1 if df.index.get_loc(i) + 1 < len(df) else None
            if next_day_idx:
                next_day = df.index[next_day_idx]
                if next_day in historical_data.index:
                    open_trade = {
                        "entry_date": next_day,
                        "entry_price": historical_data.loc[next_day].mean(),  # Average price across tickers
                        "entry_z": z_score,
                        "meta": meta['tickers']  # Include metadata like tickers
                    }

        # Check for exit signal: |Z| <= exit_z or momentum reverses direction
        elif open_trade is not None and (abs(z_score) <= exit_z or  z_score < 0):
            # Get the current price for the tickers
            current_date = i
            if current_date in historical_data.index:
                # Save the exit details
                open_trade["exit_date"] = current_date
                open_trade["exit_price"] = historical_data.loc[current_date].mean()  # Average price across tickers
                open_trade["exit_z"] = z_score
                open_trade["profit"] = open_trade["exit_price"] - open_trade["entry_price"]  # Example calculation
                trades.append(open_trade)

                # Plot if enabled
                if plot:
                    start_date = max(df.index.min(), open_trade["entry_date"] - pd.Timedelta(days=plot_extra_days))
                    end_date = min(df.index.max(), open_trade["exit_date"] + pd.Timedelta(days=plot_extra_days))

                    # Plot spread and volatility
                    plt.figure(figsize=(16, 12))
                    plt.subplot(2, 1, 1)
                    df_slice = df.loc[start_date:end_date]
                    plt.plot(df_slice.index, df_slice['norm_spread_504'], label='Spread (Norm)', color='blue')
                    plt.plot(df_slice.index, df_slice['^OVX_Z'], label='Volatility (^OVX Z-Score)', color='orange', alpha=0.8)
                    plt.axhline(entry_z, color='green', linestyle='--', label=f'Entry Z={entry_z}')
                    plt.axhline(-entry_z, color='green', linestyle='--')
                    plt.axhline(exit_z, color='red', linestyle='--', label=f'Exit Z={exit_z}')
                    plt.axhline(-exit_z, color='red', linestyle='--')
                    plt.title(f"Spread and Volatility: {meta['tickers']}")
                    plt.xlabel("Date")
                    plt.ylabel("Z-Score")
                    plt.legend()
                    plt.grid(alpha=0.5)

                    # Plot stock prices with entry/exit points
                    plt.subplot(2, 1, 2)
                    historical_slice = historical_data.loc[start_date:end_date]
                    for ticker in tickers:
                        plt.plot(historical_slice.index, historical_slice[ticker], label=ticker)

                        # Mark entry and exit points
                        if open_trade["entry_date"] in historical_slice.index:
                            entry_price = historical_slice.loc[open_trade["entry_date"], ticker]
                            plt.scatter(open_trade["entry_date"], entry_price, color='black', label=f'{ticker} Entry')
                        if open_trade["exit_date"] in historical_slice.index:
                            exit_price = historical_slice.loc[open_trade["exit_date"], ticker]
                            profit = exit_price - entry_price
                            plt.scatter(open_trade["exit_date"], exit_price, color='red', label=f'{ticker} Exit ({profit:.2f})')

                    plt.title(f"Stock Prices: {meta}")
                    plt.xlabel("Date")
                    plt.ylabel("Price")
                    plt.legend()
                    plt.grid(alpha=0.5)
                    plt.tight_layout()
                    plt.show()

                open_trade = None  # Reset trade

    # Return the list of trades
    return trades






for i in range(len(all_details)):
    metadata = all_metadata[i]
    details_df = all_details[i]
    trades = potential_profit(metadata, details_df, entry_z=2, exit_z = 0.25, z_momentum_threshold=0.1, plot = True)

print(trades)



########## ADF test on spreads and filter on gruops that are cointegrated on every time frame


########## Check for Z > 2 values

