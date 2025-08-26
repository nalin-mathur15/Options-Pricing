# European Option Pricer

Simple [web app](https://nalin-mathur15-options-pricing-app-8lgsob.streamlit.app) for European option pricing using **Black–Scholes** (closed-form) and **Monte Carlo** simulations.
Spot prices are fetched from the **Yahoo Finance API** via `pandas-datareader`.
External HTTP calls are cached with 5 minute TTL

## Features
- Fetches latest stock price data from Yahoo Finance API using **pandas-datareader** .
- Caches data using **requests-cache** to avoid duplicate API calls.
- Interactive inputs:
  - Ticker symbol
  - Strike price
  - Risk-free rate (%) (annual, continuously compounded)
  - Volatility σ (%) (annual)
  - Exercise date
  - Monte Carlo simulation controls (number of paths, chunking, antithetic variates)
- Computes **call/put** prices with:
  - Black–Scholes closed form
  - Monte Carlo (GBM, exact solution)
- Visuals:
  - Histogram bar chart of simulated terminal price distribution
  - Recent OHLC data table

## Quickstart

> **Python 3.12+**

```bash
# Install dependencies
python -m pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## Notes

- I set up an on-disk HTTP cache using `requests-cache`:
  ```python
  import requests_cache
  requests_cache.install_cache("yahoo_cache", expire_after=300)
  ```
  This reduces duplicate calls during experimentation.
- Time-to-maturity uses a simple ACT/365 day count. If expiry is today/past, the app uses **1/252** years to avoid zero-maturity issues.
- Prices assume **no dividends**. You can extend `BSInputs` and the pricing functions to include a dividend yield \($q$\) by shifting drift to \($r-q$\).

