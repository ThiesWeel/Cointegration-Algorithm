Cointegration Pipeline – Development Database

This folder contains all data used for development, testing, and debugging of the cointegration trading pipeline.

It mirrors the structure of the production database, but contains lightweight or mock data. Files here are not used in live or scheduled runs.

Folder Descriptions:

- raw/: Raw stock and option data downloaded from yfinance. One file per ticker.
- processed/: Cleaned and normalized time series, ready for modeling.
- cointegration/: Z-score histories, ADF results, and selected pairs.
- bayesian1/: "Stores posterior probabilities for pair selection prior to testing"
- bayesian2/: "Stores posterior probabilities of cointegration robustness across multiple window lengths"
- forecasts/: Monte Carlo forecast results and convergence probabilities.
- signals/: Outputs from neural network models (entry scores, monitoring flags).
- trades/: Logs of simulated or executed trades, including timestamps and outcomes.

Usage:
- This database is used by default in development mode.
- To switch to production mode, update the BASE_DATABASE path in `config.py`.

Note:
This folder is ignored in version control and is safe to delete and regenerate.

