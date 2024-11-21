from statsmodels.tsa.stattools import adfuller
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from itertools import combinations
import seaborn as sns
import itertools

tickers_all = [
    "^OVX"]
#tickers to consider
tickers = tickers_all

data = yf.download(tickers, start="2020-01-01", end="2023-12-31")['Close']
print(data)