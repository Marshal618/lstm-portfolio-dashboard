import os
import tempfile

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objs as go
import streamlit.components.v1 as components
import quantstats as qs
import joblib  

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

from src.core import make_supervised, compute_weights_from_preds, predict_next_log_return
from src.risk import (
    simulate_paths_parametric,
    simulate_paths_bootstrap,
    summarize_terminal,
)

# Supabase model packs (pretrained)
# Make sure requirements.txt includes:
# supabase, joblib, python-dotenv
from src.model_registry import (
    PackMeta,
    make_pack_id,
    list_packs,
    download_pack,
    upload_pack,
    build_zip_from_models,
    load_models_from_zip,
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
# CACHED MODEL TRAINING (big speed-up within the same Streamlit runtime)
# -----------------------------
@st.cache_resource(show_spinner=False)
def train_models_cached(
    log_returns_train: pd.DataFrame,
    symbols: tuple,
    lookback_period: int,
    epochs: int,
    fast_mode: bool,
):
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


def save_model_pack(models: dict, scalers: dict, pack_dir: str):
    """Local save helper (optional). Supabase pack is the real persistence."""
    os.makedirs(pack_dir, exist_ok=True)

    for sym, model in models.items():
        model_path = os.path.join(pack_dir, f"{sym}.keras")
        model.save(model_path)

    scaler_path = os.path.join(pack_dir, "scalers.joblib")
    joblib.dump(scalers, scaler_path)

    meta_path = os.path.join(pack_dir, "meta.txt")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write("Model pack for LSTM Portfolio Lab\n")
        f.write(f"Symbols: {list(models.keys())}\n")


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
Build a signal, turn it into a portfolio, then stress-test the risk.
</p>
""",
    unsafe_allow_html=True,
)

with st.expander("How this dashboard works", expanded=True):
    st.markdown(
        """
**LSTM (signal):** Looks at recent returns and outputs a *next-step signal* for each stock.  
**Portfolio (rules):** Turns those signals into weights with min/max limits.  
**Backtest (history):** Rebalances monthly and shows what would have happened.  
**Monte Carlo (what-if):** Simulates many possible future paths so you see a range of outcomes.  
**Signals tab:** Lets you inspect what the model output was, and whether it lined up with what happened next.
        """
    )

# -----------------------------
# Sidebar (Control panel)
# -----------------------------
with st.sidebar:
    st.markdown("## Controls")
    st.caption("Pick tickers, choose a model mode, then run the backtest.")

    st.markdown("### Pretrained models (Supabase)")
    model_mode = st.radio("Mode", ["Use pretrained (fast)", "Train & upload (slow)"], index=0)
    force_retrain = st.toggle("Force retrain", value=False, help="Ignore pretrained packs/caches and retrain now.")

    try:
        available_packs = list_packs()
    except Exception:
        available_packs = []

    selected_pack = None
    if model_mode.startswith("Use"):
        if available_packs:
            selected_pack = st.selectbox("Model pack", available_packs, index=0)
        else:
            st.warning("No packs found (or Supabase secrets not set).")

    st.divider()

    st.markdown("### Monte Carlo")
    mc_enabled = st.toggle("Enable Monte Carlo", value=True)
    mc_method = st.selectbox("Method", ["Bootstrap (recommended)", "Parametric (Normal)"])
    mc_years = st.slider("Horizon (years)", 1, 5, 1)
    mc_sims = st.slider("Simulations", 200, 10000, 2000, step=200)
    mc_seed = st.number_input("Seed", min_value=0, max_value=10_000_000, value=42, step=1)

    st.divider()

    st.markdown("### Universe")
    symbols_default = ["TSM", "GOOGL", "EQIX", "ASML", "ALGN", "KO", "DIS", "XOM", "AWK", "BHP", "V"]
    symbols = st.multiselect("Tickers", options=symbols_default, default=symbols_default)

    start_date = st.date_input("Start date", value=pd.to_datetime("2016-01-01"))
    end_date = st.date_input("End date", value=pd.Timestamp.today().normalize().date())  
    trade_start = st.date_input("Trade start", value=pd.to_datetime("2017-01-01"))

    st.markdown("### LSTM training")
    fast_mode = st.toggle("Fast mode", value=True, help="Smaller network + shorter training window.")
    lookback_period = st.slider("Lookback (days)", 60, 300 if fast_mode else 400, 180 if fast_mode else 252, step=5)
    epochs = st.slider("Epochs", 1, 12 if fast_mode else 30, 4 if fast_mode else 10, step=1)

    st.markdown("### Portfolio constraints")
    min_weight = st.slider("Min weight", 0.0, 0.10, 0.01, step=0.005)
    max_weight = st.slider("Max weight", 0.10, 1.00, 0.40, step=0.05)

    st.markdown("### Risk-free rate")
    annual_rf = st.number_input("Risk-free (annual)", min_value=0.0, max_value=0.20, value=0.03, step=0.005)

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
# Backtest Data
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

# Training data: rolling recent window
log_returns_train = log_returns.tail(int(train_days)).copy()

# -----------------------------
# Load pretrained OR train + upload
# -----------------------------
pack_id = make_pack_id(symbols, lookback_period, epochs, int(train_days), fast_mode)

models, scalers = None, None
loaded_from_supabase = False
trained_now = False

# 1) Try loading pack if user chose it and not forcing retrain
if (not force_retrain) and model_mode.startswith("Use") and selected_pack:
    try:
        with st.spinner(f"Loading pretrained pack: {selected_pack}..."):
            zip_bytes = download_pack(selected_pack)
            meta, models, scalers = load_models_from_zip(zip_bytes)
        loaded_from_supabase = True
        st.success(f"Loaded pretrained pack: {selected_pack}")
    except Exception as e:
        st.warning(f"Could not load pretrained pack. Falling back to training. ({e})")
        models, scalers = None, None

# 2) Train ONLY if we still need models
if models is None or scalers is None:
    with st.spinner("Training LSTMs (cached)..."):
        models, scalers = train_models_cached(
            log_returns_train,
            tuple(symbols),
            lookback_period,
            epochs,
            fast_mode,
        )
    trained_now = True

# 3) Safety check
if not isinstance(models, dict) or len(models) < 2:
    st.error("Not enough models trained/loaded (try fewer tickers, smaller lookback, or longer training window).")
    st.stop()

# 4) Upload ONLY in Train & upload mode
if model_mode.startswith("Train"):
    try:
        with st.spinner(f"Uploading model pack to Supabase: {pack_id}..."):
            meta = PackMeta(
                pack_id=pack_id,
                symbols=tuple(symbols),
                lookback=int(lookback_period),
                epochs=int(epochs),
                train_days=int(train_days),
                fast_mode=bool(fast_mode),
            )
            zip_bytes = build_zip_from_models(meta, models, scalers)
            upload_pack(pack_id, zip_bytes)
        st.success(f"Uploaded model pack: {pack_id}")
    except Exception as e:
        st.warning(f"Models are ready, but upload failed: {e}")
else:
    # optional local save only when training occurred and we didn't load from Supabase
    if trained_now and (not loaded_from_supabase):
        pack_dir = "model_pack"
        if not os.path.exists(pack_dir):
            save_model_pack(models, scalers, pack_dir)
            st.info("Saved a local ./model_pack snapshot (optional). Supabase packs are the main persistence.")

# -----------------------------
# Backtest Engine + Signal Diagnostics
# -----------------------------
initial_investment = 100000.0
weights_history = pd.DataFrame(index=bt_dates, columns=symbols, dtype=float)
portfolio_value = pd.Series(index=bt_dates, dtype=float)
portfolio_value.iloc[0] = initial_investment

# Store monthly signal snapshots + realized next-day return for evaluation
preds_history = pd.DataFrame(index=rebalance_dates, columns=symbols, dtype=float)
realized_history = pd.DataFrame(index=rebalance_dates, columns=symbols, dtype=float)

current_weights = pd.Series(1 / len(symbols), index=symbols)

for i, date in enumerate(bt_dates):
    if date in rebalance_dates:
        asof = bt_dates[i - 1] if i > 0 else date

        preds = pd.Series(
            {s: predict_next_log_return(s, asof, log_returns, models, scalers, lookback_period) for s in symbols}
        )

        # record model outputs at rebalance date
        preds_history.loc[date, preds.index] = preds.values

        # record realized next-day return (for signal quality metrics)
        if i < len(bt_dates) - 1:
            next_day = bt_dates[i + 1]
            realized_next = log_returns.loc[next_day, symbols].astype(float)
            realized_history.loc[date, realized_next.index] = realized_next.values

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
# Signal quality metrics (IC + hit-rate)
# -----------------------------
pairs = (
    preds_history.stack().rename("pred").to_frame()
    .join(realized_history.stack().rename("realized"), how="inner")
    .dropna()
)
ic = float(pairs["pred"].corr(pairs["realized"])) if len(pairs) > 2 else np.nan
hit_rate = float((np.sign(pairs["pred"]) == np.sign(pairs["realized"])).mean()) if len(pairs) > 0 else np.nan

# -----------------------------
# QuantStats report as cached HTML (always-on)
# -----------------------------
@st.cache_data(show_spinner=False)
def quantstats_html(returns: pd.Series, benchmark: pd.Series, rf: float) -> str:
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "report.html")
        qs.reports.html(returns, benchmark=benchmark, rf=rf, compounded=True, output=path)
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Performance", "Portfolio Weights", "Report", "Risk (Monte Carlo)", "Signals (Model Output)"]
)

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
    st.caption("Auto-generated diagnostics (cached to keep the app responsive).")
    html = quantstats_html(portfolio_returns, spy_returns, annual_rf)
    components.html(html, height=900, scrolling=True)

with tab4:
    st.subheader("Monte Carlo Risk Simulation")
    st.caption("A quick stress test: what might the next year (or few years) look like if returns behave similarly to the past?")

    if not mc_enabled:
        st.info("Turn on Monte Carlo in the sidebar to run simulations.")
        st.stop()

    st.markdown(
        """
**How to read this:**
- The backtest shows one historical path.
- Monte Carlo generates many plausible future paths so you can see a range: good outcomes, bad outcomes, and everything in between.
- If you care about downside, focus on **probability of loss** and the **5th percentile** outcome.
        """
    )

    mc_days = int(252 * mc_years)

    try:
        if mc_method.startswith("Bootstrap"):
            eq_paths, _ = simulate_paths_bootstrap(
                portfolio_returns, n_sims=int(mc_sims), n_days=int(mc_days), seed=int(mc_seed)
            )
            method_note = "Bootstrap reuses real daily returns (usually better for capturing big swings)."
        else:
            eq_paths, _ = simulate_paths_parametric(
                portfolio_returns, n_sims=int(mc_sims), n_days=int(mc_days), seed=int(mc_seed)
            )
            method_note = "Parametric uses a Normal approximation (fast, but can understate extreme moves)."

        stats, terminal = summarize_terminal(eq_paths)

        c1, c2, c3 = st.columns(3)
        c1.metric("Median outcome", f"{stats['p50']:.2f}×")
        c2.metric("5th percentile", f"{stats['p05']:.2f}×")
        c3.metric("Prob. of loss", f"{stats['prob_loss']:.1%}")
        st.caption(method_note)

        n_plot = min(200, eq_paths.shape[0])
        plot_idx = np.linspace(0, eq_paths.shape[0] - 1, n_plot).astype(int)
        eq_plot = eq_paths[plot_idx]

        x = list(range(eq_plot.shape[1]))
        fig_mc = go.Figure()
        for i in range(eq_plot.shape[0]):
            fig_mc.add_trace(go.Scatter(x=x, y=eq_plot[i], mode="lines", showlegend=False, line=dict(width=1)))

        median_path = np.median(eq_paths, axis=0)
        fig_mc.add_trace(go.Scatter(x=x, y=median_path, mode="lines", name="Median", line=dict(width=3)))
        fig_mc.update_layout(
            height=520,
            xaxis_title=f"Trading days (≈ {mc_years} year(s))",
            yaxis_title="Equity (× initial capital)",
        )
        st.plotly_chart(fig_mc, use_container_width=True)

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=terminal, nbinsx=60, name="Terminal equity"))
        fig_hist.update_layout(height=360, xaxis_title="Terminal equity (× initial)", yaxis_title="Count")
        st.plotly_chart(fig_hist, use_container_width=True)

    except Exception as e:
        st.error(f"Monte Carlo failed: {e}")

with tab5:
    st.subheader("Signals (Model Output)")
    st.caption("These are model outputs used by the strategy to rebalance. They’re useful for transparency and diagnostics.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Information Coefficient (IC)", "—" if np.isnan(ic) else f"{ic:.3f}")
    c2.metric("Directional hit-rate", "—" if np.isnan(hit_rate) else f"{hit_rate:.1%}")
    c3.metric("Signal samples", f"{len(pairs):,}")

    st.markdown(
        """
**How to use this tab:**
- Think of the LSTM output as a *signal score* for “next-step return”.
- A higher score doesn’t guarantee profit — it just means the model is more optimistic **relative to the other stocks**.
- IC and hit-rate are quick checks: did the signal line up with what happened next?
        """
    )

    last_dt = preds_history.dropna(how="all").index.max()
    if pd.isna(last_dt):
        st.info("No signal snapshots were recorded.")
    else:
        pred_row = preds_history.loc[last_dt].dropna()
        real_row = realized_history.loc[last_dt].dropna()

        df_sig = pd.DataFrame(
            {
                "pred_log_return": pred_row,
                "realized_next_log_return": real_row.reindex(pred_row.index),
            }
        )
        df_sig["pred_rank"] = df_sig["pred_log_return"].rank(ascending=False, method="average")
        df_sig["direction_match"] = np.sign(df_sig["pred_log_return"]) == np.sign(df_sig["realized_next_log_return"])
        df_sig = df_sig.sort_values("pred_rank")

        st.write(f"Most recent rebalance snapshot: **{last_dt.date()}**")
        st.dataframe(df_sig, use_container_width=True)

        pts = df_sig.dropna(subset=["pred_log_return", "realized_next_log_return"])
        if len(pts) > 2:
            fig_sc = go.Figure()
            fig_sc.add_trace(
                go.Scatter(
                    x=pts["pred_log_return"],
                    y=pts["realized_next_log_return"],
                    mode="markers",
                    text=pts.index,
                    name="Points",
                )
            )
            fig_sc.update_layout(
                height=420,
                xaxis_title="Predicted next-day log return (signal)",
                yaxis_title="Realized next-day log return",
            )
            st.plotly_chart(fig_sc, use_container_width=True)

st.caption(
    "Tip: **Use pretrained** loads saved models instantly. "
    "**Train & upload** is for creating a new Supabase pack when you change settings."
)



