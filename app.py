import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objs as go
import quantstats as qs
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

qs.extend_pandas()
np.random.seed(42)

st.set_page_config(page_title="LSTM Portfolio Lab", layout="wide")

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
        threads=True
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

def make_supervised(series_1d: np.ndarray, lookback: int):
    X, y = [], []
    for i in range(lookback, len(series_1d)):
        X.append(series_1d[i - lookback:i])
        y.append(series_1d[i])
    X = np.array(X)
    y = np.array(y)
    return X.reshape((X.shape[0], X.shape[1], 1)), y

def compute_weights_from_preds(preds: pd.Series, symbols, min_weight=0.01, max_weight=0.40):
    preds = preds.dropna()
    if preds.empty:
        return pd.Series(dtype=float)

    x = preds.values
    x = (x - x.mean()) / (x.std() + 1e-8)
    expx = np.exp(np.clip(x, -5, 5))
    w = expx / expx.sum()

    weights = pd.Series(w, index=preds.index).reindex(symbols).fillna(0.0)
    weights = weights + min_weight
    weights = weights / weights.sum()

    for _ in range(10):
        over = weights > max_weight
        if not over.any():
            break
        excess = (weights[over] - max_weight).sum()
        weights[over] = max_weight
        under = ~over
        if under.sum() == 0:
            break
        weights[under] = weights[under] + excess * (weights[under] / weights[under].sum())

    return weights / weights.sum()

# -----------------------------
# UI
# -----------------------------
st.title("LSTM Portfolio Lab (Free Quant Dashboard)")

with st.sidebar:
    st.header("Strategy Settings")

    symbols_default = ["TSM","GOOGL","EQIX","ASML","ALGN","KO","DIS","XOM","AWK","BHP","V"]
    symbols = st.multiselect("Tickers", options=symbols_default, default=symbols_default)

    start_date = st.date_input("Start date", value=pd.to_datetime("2016-01-01"))
    end_date = st.date_input("End date", value=pd.to_datetime("2022-12-31"))
    trade_start = st.date_input("Trade start", value=pd.to_datetime("2017-01-01"))

    lookback_period = st.slider("Lookback (days)", 60, 400, 252, step=5)
    epochs = st.slider("LSTM epochs", 3, 30, 10, step=1)

    min_weight = st.slider("Min weight", 0.0, 0.10, 0.01, step=0.005)
    max_weight = st.slider("Max weight", 0.10, 1.00, 0.40, step=0.05)

    annual_rf = st.number_input("Risk-free rate (annual)", min_value=0.0, max_value=0.20, value=0.03, step=0.005)

    run = st.button("Run backtest")

if not run:
    st.info("Set your parameters on the left, then click **Run backtest**.")
    st.stop()

if len(symbols) < 2:
    st.error("Pick at least 2 tickers.")
    st.stop()

# -----------------------------
# Backtest runner
# -----------------------------
with st.spinner("Downloading data..."):
    prices = get_prices(symbols, str(start_date), str(end_date))
    prices = prices.dropna(how="all").ffill().dropna()

log_returns = np.log(prices / prices.shift(1)).dropna()
bt_dates = log_returns.index[log_returns.index >= pd.Timestamp(trade_start)]

# monthly rebalance (first trading day each month)
rebalance_dates = bt_dates.to_series().groupby(pd.Grouper(freq="MS")).head(1)
rebalance_dates = pd.to_datetime(rebalance_dates.values)

# Train models
models, scalers = {}, {}
with st.spinner("Training LSTMs (one per asset)..."):
    for sym in symbols:
        r = log_returns[sym].dropna().values.reshape(-1, 1)
        if len(r) < lookback_period + 20:
            continue

        scaler = MinMaxScaler(feature_range=(0, 1))
        r_scaled = scaler.fit_transform(r).flatten()
        X, y = make_supervised(r_scaled, lookback_period)

        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(lookback_period, 1)),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer="adam", loss="mean_squared_error")
        model.fit(X, y, epochs=epochs, batch_size=32, verbose=0)

        models[sym] = model
        scalers[sym] = scaler

def predict_next_log_return(sym, asof_date):
    if sym not in models:
        return np.nan
    hist = log_returns.loc[:asof_date, sym].dropna()
    if len(hist) < lookback_period:
        return np.nan
    last_window = hist.values[-lookback_period:].reshape(-1, 1)
    scaled = scalers[sym].transform(last_window).flatten()
    X = scaled.reshape((1, lookback_period, 1))
    pred_scaled = models[sym].predict(X, verbose=0)[0, 0]
    pred = scalers[sym].inverse_transform(np.array(pred_scaled).reshape(1, 1))[0, 0]
    return float(pred)

# Initialize portfolio
initial_investment = 100000.0
weights_history = pd.DataFrame(index=bt_dates, columns=symbols, dtype=float)
portfolio_value = pd.Series(index=bt_dates, dtype=float)
portfolio_value.iloc[0] = initial_investment

# simple equal start (safe for any symbol list)
current_weights = pd.Series(1/len(symbols), index=symbols)

for i, date in enumerate(bt_dates):
    if date in rebalance_dates:
        asof = bt_dates[i - 1] if i > 0 else date
        preds = pd.Series({s: predict_next_log_return(s, asof) for s in symbols})
        new_w = compute_weights_from_preds(preds, symbols, min_weight=min_weight, max_weight=max_weight)
        if not new_w.empty and np.isfinite(new_w.values).all():
            current_weights = new_w

    weights_history.loc[date] = current_weights.values

    day_log_ret = log_returns.loc[date, symbols].fillna(0.0)
    simple_day_ret = np.exp(day_log_ret) - 1
    port_ret = float((current_weights * simple_day_ret).sum())

    if i > 0:
        portfolio_value.iloc[i] = portfolio_value.iloc[i - 1] * (1 + port_ret)

portfolio_returns = portfolio_value.pct_change().dropna()

# Benchmark
spy_prices = get_prices("SPY", str(start_date), str(end_date))["SPY"].ffill().dropna()
spy_returns = spy_prices.pct_change().dropna()
spy_returns = spy_returns.reindex(portfolio_returns.index).dropna()
portfolio_returns = portfolio_returns.reindex(spy_returns.index).dropna()

# -----------------------------
# Display
# -----------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Total Return", f"{(portfolio_value.iloc[-1]/portfolio_value.iloc[0]-1)*100:.2f}%")
col2.metric("Sharpe (qs)", f"{qs.stats.sharpe(portfolio_returns, rf=annual_rf):.2f}")
col3.metric("Max Drawdown", f"{qs.stats.max_drawdown(portfolio_returns)*100:.2f}%")

st.subheader("Equity Curve")
eq = (1 + portfolio_returns).cumprod()
eq_b = (1 + spy_returns).cumprod()

fig_eq = go.Figure()
fig_eq.add_trace(go.Scatter(x=eq.index, y=eq.values, mode="lines", name="Strategy"))
fig_eq.add_trace(go.Scatter(x=eq_b.index, y=eq_b.values, mode="lines", name="SPY"))
fig_eq.update_layout(height=450, xaxis_title="Date", yaxis_title="Growth of $1")
st.plotly_chart(fig_eq, use_container_width=True)

st.subheader("Weights (Monthly Avg)")
w_month = weights_history.dropna().resample("M").mean()
fig_w = go.Figure()
for s in symbols:
    fig_w.add_trace(go.Scatter(x=w_month.index, y=w_month[s], mode="lines", name=s))
fig_w.update_layout(height=450, xaxis_title="Date", yaxis_title="Weight")
st.plotly_chart(fig_w, use_container_width=True)

st.subheader("QuantStats Report (in-app)")
qs.reports.full(portfolio_returns, benchmark=spy_returns, rf=annual_rf, compounded=True)
