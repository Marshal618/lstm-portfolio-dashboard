import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objs as go
import quantstats as qs
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from src.core import make_supervised, compute_weights_from_preds, predict_next_log_return
from src.risk import (
    simulate_paths_parametric,
    simulate_paths_bootstrap,
    summarize_terminal,
)

qs.extend_pandas()
np.random.seed(42)

# -----------------------------
# Page + Styling
# -----------------------------
st.set_page_config(page_title="LSTM Portfolio Lab", layout="wide")

st.markdown(
    """
<style>
/* Layout */
.block-container { padding-top: 1.8rem; padding-bottom: 2rem; max-width: 1150px; }
section.main > div { padding-top: 0.75rem; }
body { background-color: #f8fafc; }

/* Typography */
h1 { letter-spacing: -0.02em; font-weight: 850; margin-bottom: 0.25rem; }
h2, h3 { letter-spacing: -0.01em; }

/* Sidebar */
section[data-testid="stSidebar"] { background: #0b1220; }
section[data-testid="stSidebar"] * { color: #e5e7eb !important; }
section[data-testid="stSidebar"] label { color: #e5e7eb !important; }
section[data-testid="stSidebar"] .stCaption { color: #cbd5e1 !important; }

/* Metric cards */
div[data-testid="metric-container"]{
  background: #0f172a;
  border: 1px solid #1f2937;
  padding: 14px 14px;
  border-radius: 16px;
}
div[data-testid="metric-container"] * { color: #e5e7eb !important; }

/* Buttons */
.stButton>button {
  background: #0ea5e9 !important;
  color: white !important;
  border-radius: 12px !important;
  border: none !important;
  padding: 0.65rem 1.1rem !important;
  font-weight: 650 !important;
}
.stButton>button:hover { background: #0284c7 !important; }

/* Plot containers */
[data-testid="stPlotlyChart"]{
  background: white;
  border: 1px solid #e5e7eb;
  border-radius: 16px;
  padding: 8px;
}
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# Robust yfinance price getter
# -----------------------------
@st.cache_data(show_spinner=False)
def get_prices(tickers, start, end):
    df = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=False,
        group_by="column",
        progress=False,
        threads=True,
    )
    if df is None or df.empty:
        raise RuntimeError("yfinance returned empty data (API limit / internet / ticker issue).")

    if isinstance(df.columns, pd.MultiIndex):
        level0 = df.columns.get_level_values(0)
        if "Adj Close" in level0:
            out = df["Adj Close"].copy()
        elif "Close" in level0:
            out = df["Close"].copy()
        else:
            raise KeyError(f"No 'Adj Close' or 'Close'. Columns: {df.columns}")
    else:
        if "Adj Close" in df.columns:
            out = df["Adj Close"].copy()
        elif "Close" in df.columns:
            out = df["Close"].copy()
        else:
            raise KeyError(f"No 'Adj Close' or 'Close'. Columns: {df.columns}")

    if isinstance(out, pd.Series):
        out = out.to_frame(name=tickers if isinstance(tickers, str) else "Close")
    if isinstance(tickers, str):
        out.columns = [tickers]
    return out



# -----------------------------
# CACHED MODEL TRAINING (big speed-up)
# -----------------------------
@st.cache_resource(show_spinner=False)
def train_models_cached(log_returns_train: pd.DataFrame, symbols: tuple, lookback_period: int, epochs: int, fast_mode: bool):
    """
    Cache trained models so repeated runs don't retrain.
    Using tuple(symbols) makes cache keys stable/hashiable.
    """
    models, scalers = {}, {}

    # Smaller model in fast mode
    if fast_mode:
        lstm1, lstm2, batch = 24, 12, 128
    else:
        lstm1, lstm2, batch = 32, 16, 64

    for sym in symbols:
        if sym not in log_returns_train.columns:
            continue

        r = log_returns_train[sym].dropna().values.reshape(-1, 1)
        if len(r) < lookback_period + 80:
            continue

        scaler = MinMaxScaler(feature_range=(0, 1))
        r_scaled = scaler.fit_transform(r).flatten()
        X, y = make_supervised(r_scaled, lookback_period)

        model = Sequential(
            [
                LSTM(lstm1, return_sequences=True, input_shape=(lookback_period, 1)),
                Dropout(0.15),
                LSTM(lstm2),
                Dropout(0.10),
                Dense(1),
            ]
        )
        model.compile(optimizer="adam", loss="mean_squared_error")
        model.fit(X, y, epochs=epochs, batch_size=batch, verbose=0)

        models[sym] = model
        scalers[sym] = scaler

    return models, scalers



# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
<div style="display:flex; align-items:flex-end; gap:12px;">
  <h1 style="margin:0;">LSTM Portfolio Lab</h1>
  <span style="
    background:#0ea5e9;
    color:white;
    padding:4px 10px;
    border-radius:999px;
    font-size:12px;
    font-weight:800;">
    Research Dashboard
  </span>
</div>
<p style="margin-top:6px; color:#64748b; font-size:15px;">
ML-driven signal generation → constrained portfolio construction → systematic backtesting & risk diagnostics.
</p>
""",
    unsafe_allow_html=True,
)

with st.expander("Strategy Overview", expanded=True):
    st.write(
        "This strategy trains an LSTM to forecast next-day log returns for selected equities. "
        "Forecasted signals are converted into portfolio weights via a constrained softmax allocation "
        "with minimum/maximum exposure limits. The portfolio rebalances monthly and is evaluated "
        "with risk-adjusted metrics and benchmark comparison (SPY)."
    )

# -----------------------------
# Sidebar (Control panel)
# -----------------------------
with st.sidebar:
    st.markdown("## Controls")
    st.caption("Configure universe, training settings, and portfolio constraints.")

    st.markdown("### Universe")
    symbols_default = ["TSM", "GOOGL", "EQIX", "ASML", "ALGN", "KO", "DIS", "XOM", "AWK", "BHP", "V"]
    symbols = st.multiselect("Tickers", options=symbols_default, default=symbols_default)

    start_date = st.date_input("Start date", value=pd.to_datetime("2016-01-01"))
    end_date = st.date_input("End date", value=pd.to_datetime("2022-12-31"))
    trade_start = st.date_input("Trade start", value=pd.to_datetime("2017-01-01"))

    st.markdown("### Model")
    fast_mode = st.toggle("Fast mode (recommended)", value=True, help="Trains faster using a smaller model and shorter training window.")

    lookback_period = st.slider("Lookback (days)", 60, 300 if fast_mode else 400, 180 if fast_mode else 252, step=5)
    epochs = st.slider("LSTM epochs", 1, 12 if fast_mode else 30, 4 if fast_mode else 10, step=1)

    st.markdown("### Portfolio Constraints")
    min_weight = st.slider("Min weight", 0.0, 0.10, 0.01, step=0.005)
    max_weight = st.slider("Max weight", 0.10, 1.00, 0.40, step=0.05)

    st.markdown("### Risk-free rate")
    annual_rf = st.number_input("Risk-free rate (annual)", min_value=0.0, max_value=0.20, value=0.03, step=0.005)

    st.markdown("### Training window")
    if fast_mode:
        train_days = st.slider("Days used for training", 300, 900, 700, step=50)
    else:
        train_days = st.slider("Days used for training", 600, 2000, 1400, step=100)

    run = st.button("Run backtest", use_container_width=True)

if not run:
    st.info("Set your parameters on the left, then click **Run backtest**.")
    st.stop()

if len(symbols) < 2:
    st.error("Pick at least 2 tickers.")
    st.stop()

# -----------------------------
# Backtest
# -----------------------------
with st.spinner("Downloading data..."):
    prices = get_prices(symbols, str(start_date), str(end_date))
    prices = prices.dropna(how="all").ffill().dropna()

log_returns = np.log(prices / prices.shift(1)).dropna()

bt_dates = log_returns.index[log_returns.index >= pd.Timestamp(trade_start)]
if bt_dates.empty:
    st.error("No backtest dates found. Try an earlier trade start or wider date range.")
    st.stop()

rebalance_dates = bt_dates.to_series().groupby(pd.Grouper(freq="MS")).head(1)
rebalance_dates = pd.to_datetime(rebalance_dates.values)

# Training data: rolling recent window (speed + avoids overfitting to old regimes)
log_returns_train = log_returns.tail(int(train_days)).copy()

# Train models (CACHED)
with st.spinner("Training LSTMs (cached)..."):
    models, scalers = train_models_cached(
        log_returns_train,
        tuple(symbols),
        lookback_period,
        epochs,
        fast_mode,
    )

if len(models) < 2:
    st.error("Not enough models trained (try fewer tickers, smaller lookback, or longer training window).")
    st.stop()

# Initialize portfolio
initial_investment = 100000.0
weights_history = pd.DataFrame(index=bt_dates, columns=symbols, dtype=float)
portfolio_value = pd.Series(index=bt_dates, dtype=float)
portfolio_value.iloc[0] = initial_investment

current_weights = pd.Series(1 / len(symbols), index=symbols)

# Run backtest
for i, date in enumerate(bt_dates):
    if date in rebalance_dates:
        asof = bt_dates[i - 1] if i > 0 else date
        preds = pd.Series(
            {s: predict_next_log_return(s, asof, log_returns, models, scalers, lookback_period) for s in symbols}
        )
        new_w = compute_weights_from_preds(preds, symbols, min_weight=min_weight, max_weight=max_weight)
        if (not new_w.empty) and np.isfinite(new_w.values).all():
            current_weights = new_w

    weights_history.loc[date] = current_weights.values

    day_log_ret = log_returns.loc[date, symbols].fillna(0.0)
    simple_day_ret = np.exp(day_log_ret) - 1
    port_ret = float((current_weights * simple_day_ret).sum())

    if i > 0:
        portfolio_value.iloc[i] = portfolio_value.iloc[i - 1] * (1 + port_ret)

portfolio_returns = portfolio_value.pct_change().dropna()

# Benchmark (SPY)
spy_prices = get_prices("SPY", str(start_date), str(end_date))["SPY"].ffill().dropna()
spy_returns = spy_prices.pct_change().dropna()

# Align
spy_returns = spy_returns.reindex(portfolio_returns.index).dropna()
portfolio_returns = portfolio_returns.reindex(spy_returns.index).dropna()

if portfolio_returns.empty or spy_returns.empty:
    st.error("Return series alignment failed. Try different dates.")
    st.stop()

# -----------------------------
# KPIs
# -----------------------------
cagr = qs.stats.cagr(portfolio_returns)
vol = qs.stats.volatility(portfolio_returns)
mdd = qs.stats.max_drawdown(portfolio_returns)
sh = qs.stats.sharpe(portfolio_returns, rf=annual_rf)

k1, k2, k3, k4 = st.columns(4)
k1.metric("CAGR", f"{cagr:.2%}")
k2.metric("Volatility", f"{vol:.2%}")
k3.metric("Sharpe", f"{sh:.2f}")
k4.metric("Max Drawdown", f"{mdd:.2%}")

# -----------------------------
# Charts + Report (Tabs)
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Performance", "Portfolio Weights", "Report", "Risk (Monte Carlo)"])


with tab1:
    st.subheader("Equity Curve")
    eq = (1 + portfolio_returns).cumprod()
    eq_b = (1 + spy_returns).cumprod()

    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(x=eq.index, y=eq.values, mode="lines", name="Strategy"))
    fig_eq.add_trace(go.Scatter(x=eq_b.index, y=eq_b.values, mode="lines", name="SPY"))
    fig_eq.update_layout(height=460, xaxis_title="Date", yaxis_title="Growth of $1")
    st.plotly_chart(fig_eq, use_container_width=True)

    st.subheader("Drawdown")
    dd = eq / eq.cummax() - 1
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=dd.index, y=dd.values, mode="lines", name="Drawdown"))
    fig_dd.update_layout(height=300, xaxis_title="Date", yaxis_title="Drawdown")
    st.plotly_chart(fig_dd, use_container_width=True)

with tab2:
    st.subheader("Weights (Monthly Avg)")
    w_month = weights_history.dropna().resample("M").mean()

    fig_w = go.Figure()
    for s in symbols:
        fig_w.add_trace(go.Scatter(x=w_month.index, y=w_month[s], mode="lines", name=s))
    fig_w.update_layout(height=500, xaxis_title="Date", yaxis_title="Weight")
    st.plotly_chart(fig_w, use_container_width=True)

with tab3:
    st.subheader("QuantStats Report")
    st.caption("Full performance report (can be heavy to render).")

    # Optional: let user choose to render full report (avoids slow UI by default)
    render_report = st.toggle("Render full QuantStats report", value=False)
    if render_report:
        qs.reports.full(portfolio_returns, benchmark=spy_returns, rf=annual_rf, compounded=True)
    else:
        st.info("Toggle on to render the full report. (This can take a bit.)")

st.caption("Tip: In **Fast mode**, repeated runs with the same settings reuse cached models and should be much faster.")

with tab4:
    st.subheader("Monte Carlo Risk Simulation")
    st.caption("Simulate many plausible future equity paths using either parametric (Normal) returns or bootstrap resampling.")

    st.markdown(
        """
**What this is (quant explanation):**
- **Backtest** answers: *“What happened historically under my rules?”*
- **Monte Carlo** answers: *“Given the return characteristics I observed, what range of outcomes could happen?”*

**How to interpret:**
- The fan of lines = many possible future portfolio paths.
- The terminal distribution tells you downside risk (loss probability, bad-percentile outcomes) and upside potential.
        """
    )

    mc_col1, mc_col2, mc_col3 = st.columns([1.2, 1.0, 1.0])

    with mc_col1:
        mc_method = st.selectbox("Simulation method", ["Bootstrap (recommended)", "Parametric (Normal)"])
        mc_years = st.slider("Horizon (years)", 1, 5, 1)
        mc_days = int(252 * mc_years)
        mc_sims = st.slider("Number of simulations", 200, 10000, 2000, step=200)
        mc_seed = st.number_input("Random seed", min_value=0, max_value=10_000_000, value=42, step=1)

    # Use your already-computed daily portfolio returns
    # portfolio_returns exists earlier in your script
    if portfolio_returns.empty:
        st.warning("No returns available for Monte Carlo.")
        st.stop()

    try:
        if mc_method.startswith("Bootstrap"):
            eq_paths, meta = simulate_paths_bootstrap(portfolio_returns, n_sims=mc_sims, n_days=mc_days, seed=int(mc_seed))
            method_note = "Bootstrap resamples realized daily returns (keeps fat tails better)."
        else:
            eq_paths, meta = simulate_paths_parametric(portfolio_returns, n_sims=mc_sims, n_days=mc_days, seed=int(mc_seed))
            method_note = "Parametric assumes i.i.d. Normal returns (often underestimates tail risk)."

        stats, terminal = summarize_terminal(eq_paths)

        with mc_col2:
            st.markdown("### Terminal outcomes (× initial capital)")
            st.write(f"Method: {mc_method}")
            st.caption(method_note)
            st.metric("Median (P50)", f"{stats['p50']:.2f}×")
            st.metric("5th percentile (P05)", f"{stats['p05']:.2f}×")
            st.metric("95th percentile (P95)", f"{stats['p95']:.2f}×")

        with mc_col3:
            st.markdown("### Downside risk")
            st.metric("Prob. of loss", f"{stats['prob_loss']:.1%}")
            st.metric("1st percentile (P01)", f"{stats['p01']:.2f}×")
            st.metric("Mean terminal", f"{stats['mean']:.2f}×")

        # Plot: a subset of paths so the chart stays responsive
        import plotly.graph_objs as go
        n_plot = min(200, eq_paths.shape[0])
        plot_idx = np.linspace(0, eq_paths.shape[0] - 1, n_plot).astype(int)
        eq_plot = eq_paths[plot_idx]

        x = list(range(eq_plot.shape[1]))
        fig_mc = go.Figure()

        for i in range(eq_plot.shape[0]):
            fig_mc.add_trace(go.Scatter(x=x, y=eq_plot[i], mode="lines", line=dict(width=1), name=None, showlegend=False))

        # Median path
        median_path = np.median(eq_paths, axis=0)
        fig_mc.add_trace(go.Scatter(x=x, y=median_path, mode="lines", name="Median path", line=dict(width=3)))

        fig_mc.update_layout(
            height=520,
            xaxis_title=f"Trading days (≈ {mc_years} year(s))",
            yaxis_title="Equity (× initial capital)",
        )
        st.plotly_chart(fig_mc, use_container_width=True)

        # Terminal distribution
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=terminal, nbinsx=60, name="Terminal equity"))
        fig_hist.update_layout(height=360, xaxis_title="Terminal equity (× initial)", yaxis_title="Count")
        st.plotly_chart(fig_hist, use_container_width=True)

    except Exception as e:
        st.error(f"Monte Carlo failed: {e}")
with st.expander("Strategy Overview", expanded=True):
    st.markdown(
        """
**Signal model (LSTM):**  
- The LSTM is used as a **forecasting model for next-period log returns** per asset.
- In quant terms, it’s a **nonlinear time-series feature extractor** that maps a lookback window of returns → a one-step-ahead expected return estimate.

**Portfolio construction:**  
- Forecasts are converted to weights using a constrained softmax-style allocation.
- Constraints enforce practical risk controls (min weight / max concentration).

**Why Monte Carlo is added:**  
- A backtest shows historical performance under your rule set.
- Monte Carlo summarizes the **distribution of plausible future outcomes**, helping quantify tail risk and downside probabilities.
        """
    )

