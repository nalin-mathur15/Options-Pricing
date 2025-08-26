import math
import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm

import requests_cache
import yfinance as yf

# Global HTTP cache (caches underlying HTTP calls yfinance makes)
requests_cache.install_cache(
    cache_name="yahoo_cache",
    backend="sqlite",
    expire_after=300,
)

# Data structures and pricing functions
@dataclass
class BSInputs:
    S: float      # Spot
    K: float      # Strike
    r: float      # Risk-free rate (annual, cont. comp)
    sigma: float  # Volatility (annual)
    T: float      # Time to maturity in years

def _clamp_positive(x: float, eps: float = 1e-12) -> float:
    return max(float(x), eps)

def black_scholes_prices(inp: BSInputs):
    S, K, r, sigma, T = float(inp.S), float(inp.K), float(inp.r), float(inp.sigma), float(inp.T)
    sigma = _clamp_positive(sigma)
    T = _clamp_positive(T)

    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)
    Nmd1 = norm.cdf(-d1)
    Nmd2 = norm.cdf(-d2)

    disc = math.exp(-r * T)
    call = S * Nd1 - K * disc * Nd2
    put = K * disc * Nmd2 - S * Nmd1
    return float(call), float(put), d1, d2

def monte_carlo_prices(inp: BSInputs, n_sims: int = 100000, chunk_size: int = 250000, antithetic: bool = True, seed: int = 42):
    S, K, r, sigma, T = float(inp.S), float(inp.K), float(inp.r), float(inp.sigma), float(inp.T)
    sigma = _clamp_positive(sigma)
    T = _clamp_positive(T)

    rng = np.random.default_rng(int(seed))
    disc = math.exp(-r * T)
    mu = (r - 0.5 * sigma * sigma) * T
    sig = sigma * math.sqrt(T)

    sims_remaining = int(n_sims)
    effective_chunk = max(1, min(int(chunk_size), sims_remaining))
    call_sum = 0.0
    put_sum = 0.0
    total_paths = 0

    while sims_remaining > 0:
        m = min(effective_chunk, sims_remaining)
        Z = rng.standard_normal(m)
        if antithetic:
            Z = np.concatenate((Z, -Z))
        ST = S * np.exp(mu + sig * Z)
        call_sum += np.maximum(ST - K, 0.0).sum()
        put_sum += np.maximum(K - ST, 0.0).sum()
        total_paths += Z.shape[0]
        sims_remaining -= m

    call = disc * (call_sum / total_paths)
    put = disc * (put_sum / total_paths)
    return float(call), float(put)

# Data fetching (yfinance direct) with Streamlit cache
@st.cache_data(show_spinner=False, ttl=300)
def fetch_latest_price(ticker: str):
    end = dt.date.today() + dt.timedelta(days=1)  # pad for TZ
    start = end - dt.timedelta(days=15)
    # auto_adjust=False keeps 'Adj Close' separate
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if df is None or df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'.")
    df = df.sort_index()
    price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    last_price = float(df.iloc[-1][price_col])
    last_date = pd.to_datetime(df.index[-1]).date()
    return last_price, last_date, df

# UI
st.set_page_config(page_title="European Option Pricer", page_icon="💹", layout="wide")
st.title("European Option Pricer — Black–Scholes & Monte Carlo")
st.caption("Prices via `yfinance`.")

with st.sidebar:
    st.header("Inputs")
    ticker = st.text_input("Ticker", value="AAPL", help="Yahoo Finance symbol, e.g., AAPL, MSFT, TSLA")

    default_expiry = dt.date.today() + dt.timedelta(days=30)
    expiry = st.date_input("Exercise date", value=default_expiry)

    K = st.number_input("Strike price (K)", min_value=0.0, value=150.00, step=1.00, format="%.2f")
    r_pct = st.number_input("Risk-free rate r (%)", value=2.00, step=0.05, format="%.3f",
                            help="Annualized continuously-compounded risk-free rate in percent.")
    sigma_pct = st.number_input("Volatility σ (%)", min_value=0.0, value=25.00, step=0.10, format="%.3f",
                                help="Annualized volatility in percent.")

    st.divider()
    st.subheader("Monte Carlo settings")
    n_sims = st.number_input("Simulations", min_value=1000, max_value=5000000, value=200000, step=50000)
    chunk_size = st.number_input("Chunk size", min_value=10000, max_value=1000000, value=250000, step=10000,
                                 help="Bigger = faster but uses more memory (batches random draws).")
    antithetic = st.toggle("Antithetic variates", value=True)

# Fetch data
col_price, col_info = st.columns([1, 2])
try:
    with st.spinner("Fetching latest price..."):
        S, last_date, hist_df = fetch_latest_price(ticker.strip())
    col_price.metric("Spot", f"${S:,.2f}", help=f"Last available on {last_date}")
    col_info.write("**Data source**: Yahoo Finance via `yfinance`. **Cache TTL**: 5 minutes.")
except Exception as e:
    st.error(str(e))
    st.stop()

today = dt.date.today()
days_to_expiry = (expiry - today).days
T = days_to_expiry / 365.0 if days_to_expiry > 0 else (1.0 / 252.0)
if days_to_expiry <= 0:
    st.warning("Expiry is today/past. Using one trading day (1/252 years) to avoid zero maturity.")

# Inputs struct
inp = BSInputs(
    S=float(S),
    K=float(K),
    r=float(r_pct) / 100.0,
    sigma=float(sigma_pct) / 100.0,
    T=float(T),
)

call_bs, put_bs, d1, d2 = black_scholes_prices(inp)
call_mc, put_mc = monte_carlo_prices(inp, n_sims=int(n_sims), chunk_size=int(chunk_size), antithetic=bool(antithetic))

# Results display
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
- **MC simulations**: {int(n_sims):,} | **Antithetic**: {antithetic} | **Chunk**: {int(chunk_size):,}
"""
    )
    st.code(
        "Call = S·N(d1) − K·e^{−rT}·N(d2)\n"
        "Put  = K·e^{−rT}·N(−d2) − S·N(−d1)\n"
        "d1 = [ln(S/K) + (r + ½σ²)T] / (σ√T),   d2 = d1 − σ√T",
        language="text",
    )

# Tabs: distribution and data
tab_chart, tab_data = st.tabs(["Distribution (MC sample)", "Recent OHLC data"])

with tab_chart:
    plot_sims = min(int(n_sims), 20000)
    rng = np.random.default_rng(7)
    Z = rng.standard_normal(plot_sims)
    ST = inp.S * np.exp((inp.r - 0.5 * inp.sigma**2) * inp.T + inp.sigma * math.sqrt(inp.T) * Z)
    st.caption("Histogram of simulated terminal prices $S_T$ (sample).")
    st.bar_chart(pd.Series(ST, name="S_T"))

with tab_data:
    st.dataframe(hist_df.tail(20), use_container_width=True)
    st.caption("Last 20 rows fetched from Yahoo Finance.")

st.success("Ready. Adjust the sidebar and experiment.")
