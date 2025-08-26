# European Option Pricer 

Simple [web app](https://nalin-mathur15-options-pricing-app-8lgsob.streamlit.app/) for European option pricing using **Black–Scholes** (closed-form), and **Monte Carlo** simulation (GBM, exact solution). Spot prices fetched from **Yahoo Finance** via `yfinance`. Also has capability of **Streamlit caching** for fetched data

> This version is compatible with Python 3.12+ and works on Streamlit Community Cloud (which may use Python 3.13).

## Features
- Fetches recent OHLC + latest price from Yahoo using **yfinance** directly.
- Caches HTTP requests for 5 minutes via **requests-cache** (on-disk SQLite).
- Interactive inputs:
  - Ticker symbol
  - Strike ($K$)
  - Risk-free rate $r$ (annual, continuously-compounded)
  - Volatility $\sigma$ (annual)
  - Expiry date
  - Monte Carlo settings: #paths, chunk size, antithetic variates
- Computes Call/Put prices with BS and Monte Carlo.
- Visuals:
  - Histogram (bar chart) sample of simulated terminal prices \($S_T$\)
  - Recent OHLC data table

## Quickstart (Local)

> Requires **Python 3.12+** (works on 3.13 as well).

```bash
pip install -r requirements.txt
streamlit run app.py
