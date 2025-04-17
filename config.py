# config.py

# Choose the active environment
ENV = "dev"  # or "prod"



# Base path to the database
BASE_DATABASE = f"{ENV}_database"

# Ticker dataset name
TICKER_FILE = "dev_tickers1.csv"

# Directory structure within the database
RAW_DATA_DIR = f"{BASE_DATABASE}/raw"
PROCESSED_DATA_DIR = f"{BASE_DATABASE}/processed"
COINTEGRATION_DIR = f"{BASE_DATABASE}/cointegration"
BAYESIAN1_DIR = f"{BASE_DATABASE}/bayesian1"
BAYESIAN2_DIR = f"{BASE_DATABASE}/bayesian2"
FORECASTS_DIR = f"{BASE_DATABASE}/forecasts"
SIGNALS_DIR = f"{BASE_DATABASE}/signals"
TRADES_DIR = f"{BASE_DATABASE}/trades"
MODELS_DIR = f"{BASE_DATABASE}/models"
PREDICTIONS_DIR = f"{BASE_DATABASE}/predictions"
TICKERS_DIR = f"{BASE_DATABASE}/tickers"