import math
import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st

from scipy.stats import norm

import requests_cache
import yfinance as yf
from pandas_datareader import data as pdr

# Global HTTP cache for all requests
requests_cache.install_cache("yahoo_cache", backend="sqlite", expire_after=300)

# Utilities
@dataclass
class BSInputs:
    S: float      # spot
    K: float      # strike
    r: float      # risk-free rate
    sigma: float  # volatility
    T: float      # time to maturity

def _clamp_positive(x: float, eps: float = 1e-12) -> float:
    return max(float(x), eps)

def black_scholes_prices(inp: BSInputs):
    S, K, r, sigma, T = inp.S, inp.K, inp.r, inp.sigma, inp.T
    sigma = _clamp_positive(sigma)
    T = _clamp_positive(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)
    Nmd1 = norm.cdf(-d1)
    Nmd2 = norm.cdf(-d2)
    call = S * Nd1 - K * math.exp(-r * T) * Nd2
    put = K * math.exp(-r * T) * Nmd2 - S * Nmd1
    return float(call), float(put), d1, d2

def monte_carlo_prices(inp: BSInputs, n_sims: int = 100_000, chunk_size: int = 250_000, antithetic: bool = True, seed: int = 42):
    S, K, r, sigma, T = inp.S, inp.K, inp.r, inp.sigma, inp.T
    sigma = _clamp_positive(sigma)
    T = _clamp_positive(T)
    rng = np.random.default_rng(seed)

    disc = math.exp(-r * T)
    mu = (r - 0.5 * sigma * sigma) * T
    sig = sigma * math.sqrt(T)

    total = n_sims
    call_sum = 0.0
    put_sum = 0.0
    sims_done = 0

    eff_chunk = max(1, min(chunk_size, total))

    while sims_done < total:
        m = min(eff_chunk, total - sims_done)
        Z = rng.standard_normal(m)
        if antithetic:
            Z = np.concatenate([Z, -Z])
            m = Z.shape[0]

        ST = S * np.exp(mu + sig * Z)
        call_payoff = np.maximum(ST - K, 0.0)
        put_payoff = np.maximum(K - ST, 0.0)

        call_sum += call_payoff.sum()
        put_sum += put_payoff.sum()
        sims_done += m

    call_mc = disc * (call_sum / sims_done)
    put_mc = disc * (put_sum / sims_done)
    return float(call_mc), float(put_mc)

# Cached data fetch
@st.cache_data(show_spinner=False, ttl=300)
def fetch_latest_price(ticker: str):
    end = dt.date.today() + dt.timedelta(days=1)
    start = end - dt.timedelta(days=15)
    df = pdr.get_data_yahoo(ticker, start=start, end=end)
    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'.")
    df = df.sort_index()
    last_row = df.iloc[-1]
    price = float(last_row["Adj Close"])
    last_date = pd.Timestamp(str(df.index[-1])).date()
    return price, last_date, df

# Streamlit
st.set_page_config(page_title="European Option Pricer", page_icon="💹", layout="wide")

st.title("European Option Pricer — Black Scholes & Monte Carlo")
st.caption("Spot data via Yahoo Finance. HTTP calls cached 5 minutes.")

with st.sidebar:
    st.header("Inputs")
    ticker = st.text_input("Ticker", value="AAPL", help="Enter a Yahoo Finance symbol, e.g., AAPL, MSFT, TSLA.")
    default_expiry = dt.date.today() + dt.timedelta(days=30)
    expiry = st.date_input("Exercise date", value=default_expiry)
    K = st.number_input("Strike price (K)", min_value=0.0, value=150.0, step=1.0, format="%.2f")
    r_pct = st.number_input("Risk-free rate r (%)", value=2.00, step=0.05, format="%.3f",
                            help="Annualised compounded risk-free rate.")
    sigma_pct = st.number_input("Volatility σ (%)", min_value=0.0, value=25.00, step=0.10, format="%.3f",
                                help="Annualized volatility in percent.")
    st.divider()
    st.subheader("Monte Carlo settings")
    n_sims = st.number_input("Simulations", min_value=1_000, max_value=5_000_000, value=200_000, step=50_000)
    chunk = st.number_input("Chunk size", min_value=10_000, max_value=1_000_000, value=250_000, step=10_000,
                            help="Bigger = faster but more memory. Used to batch random draws.")
    antithetic = st.toggle("Antithetic variates", value=True)

col_price, col_info = st.columns([1, 2])
try:
    with st.spinner("Fetching latest price..."):
        S, last_date, hist_df = fetch_latest_price(ticker.strip())
    col_price.metric("Spot (Adj Close)", f"${S:,.2f}", help=f"Last available adjusted close on {last_date}")
    col_info.write(f"**Data source**: Yahoo Finance via pandas-datareader (yfinance backend). **Cached** for 5 minutes.")
except Exception as e:
    st.error(str(e))
    st.stop()

today = dt.date.today()
days_to_expiry = (expiry - today).days
if days_to_expiry <= 0:
    st.warning("Expiry is today or in the past. Using one trading day (1/252 years) to avoid zero maturity.")
    T = 1.0 / 252.0
else:
    T = days_to_expiry / 365.0  

inp = BSInputs(
    S=S,
    K=float(K),
    r=float(r_pct) / 100.0,
    sigma=float(sigma_pct) / 100.0,
    T=float(T),
)

call_bs, put_bs, d1, d2 = black_scholes_prices(inp)
call_mc, put_mc = monte_carlo_prices(inp, n_sims=int(n_sims), chunk_size=int(chunk), antithetic=bool(antithetic))

st.subheader("Prices")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Call (Black–Scholes)", f"${call_bs:,.4f}")
c2.metric("Put (Black–Scholes)", f"${put_bs:,.4f}")
c3.metric("Call (Monte Carlo)", f"${call_mc:,.4f}")
c4.metric("Put (Monte Carlo)", f"${put_mc:,.4f}")

with st.expander("Model & inputs details"):
    st.write(
        f"""
- **Ticker**: `{ticker}`  
- **Spot**: ${S:,.4f} (as of **{last_date}**)  
- **Strike**: {inp.K:.4f}  
- **Risk-free r**: {inp.r*100:.4f}%  
- **Volatility σ**: {inp.sigma*100:.4f}%  
- **Time to maturity T**: {inp.T:.6f} years ({days_to_expiry} days)  
- **MC simulations**: {int(n_sims):,} | **Antithetic**: {antithetic} | **Chunk**: {int(chunk):,}
"""
    )
    st.code(
        "Call_BS = S·N(d1) − K·e^{-rT}·N(d2),   Put_BS = K·e^{-rT}·N(−d2) − S·N(−d1)\n"
        "d1 = [ln(S/K) + (r + ½σ²)T] / (σ√T),   d2 = d1 − σ√T",
        language="text"
    )

tab_chart, tab_data = st.tabs(["Distribution (MC)", "Recent OHLC Data"])

with tab_chart:
    plot_sims = min(int(n_sims), 20_000)
    plot_call, plot_put = monte_carlo_prices(inp, n_sims=plot_sims, chunk_size=plot_sims, antithetic=False, seed=7)
    rng = np.random.default_rng(7)
    Z = rng.standard_normal(plot_sims)
    ST = inp.S * np.exp((inp.r - 0.5 * inp.sigma**2) * inp.T + inp.sigma * np.sqrt(inp.T) * Z)
    st.caption("Histogram of simulated terminal prices $S_T$ (sample for visualization).")
    st.bar_chart(pd.Series(ST).rename("S_T"))

with tab_data:
    st.dataframe(hist_df.tail(20), use_container_width=True)
    st.caption("Last 20 rows from Yahoo Finance.")

st.success("Ready. Try different parameters in the sidebar to explore scenarios.")