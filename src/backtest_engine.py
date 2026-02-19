import numpy as np
import pandas as pd


def compute_signal(
    signal_type: str,
    date,
    symbols,
    log_returns,
    models=None,
    scalers=None,   # FIX: was missing from signature; used inside LSTM branch
    lookback=60,
):
    """Returns pd.Series of signal values indexed by symbol."""
    signals = {}

    for sym in symbols:
        hist = log_returns.loc[:date, sym].dropna()

        if len(hist) < lookback:
            signals[sym] = np.nan
            continue

        if signal_type == "LSTM":
            if models is None or scalers is None or sym not in models:
                signals[sym] = np.nan
                continue
            last_window = hist.values[-lookback:].reshape(-1, 1)
            scaled = scalers[sym].transform(last_window).flatten()
            X = scaled.reshape((1, lookback, 1))
            pred_scaled = models[sym].predict(X, verbose=0)[0, 0]
            pred = scalers[sym].inverse_transform(
                np.array(pred_scaled).reshape(1, 1)
            )[0, 0]
            signals[sym] = float(pred)

        elif signal_type == "Momentum":
            signals[sym] = hist[-12:].mean()

        elif signal_type == "LowVol":
            signals[sym] = -hist[-60:].std()

        elif signal_type == "MeanReversion":
            signals[sym] = -hist[-5:].mean()

        else:
            signals[sym] = 0.0

    return pd.Series(signals)


def compute_weights(
    signal: pd.Series,
    method="Softmax",
    min_weight=0.01,
    max_weight=0.40,
):
    signal = signal.dropna()
    if signal.empty:
        return pd.Series(dtype=float)

    if method == "Equal":
        w = np.ones(len(signal)) / len(signal)

    elif method == "VolParity":
        vol = signal.abs() + 1e-6
        inv = 1 / vol
        w = inv / inv.sum()

    elif method == "TopK":
        k = min(5, len(signal))
        top = signal.sort_values(ascending=False).iloc[:k]
        w = np.zeros(len(signal))
        idx = signal.index.get_indexer(top.index)
        w[idx] = 1 / k

    else:  # Softmax default
        x = (signal - signal.mean()) / (signal.std() + 1e-8)
        expx = np.exp(np.clip(x, -5, 5))
        w = expx / expx.sum()

    weights = pd.Series(w, index=signal.index)
    weights = weights + min_weight
    weights = weights / weights.sum()
    weights = weights.clip(upper=max_weight)
    return weights / weights.sum()


# FIX: removed the duplicate simulate_portfolio definition that appeared twice in the file
def simulate_portfolio(
    returns,
    signal_type,
    weight_method,
    symbols,
    models=None,
    scalers=None,
    lookback=60,
    rebalance="M",
    transaction_cost_bps=5,
):
    equity = []
    weights_prev = None
    prev_date = None

    for date in returns.index:
        if weights_prev is None or date.to_period(rebalance) != prev_date.to_period(rebalance):
            signal = compute_signal(
                signal_type, date, symbols, returns, models, scalers, lookback
            )
            weights = compute_weights(signal, method=weight_method)

            if weights_prev is not None and not weights.empty:
                turnover = np.abs(weights - weights_prev.reindex(weights.index).fillna(0)).sum()
                cost = turnover * transaction_cost_bps / 10000
            else:
                cost = 0.0

            if not weights.empty:
                weights_prev = weights

        if weights_prev is None:
            equity.append(0.0)
            prev_date = date
            continue

        daily_ret = (returns.loc[date, symbols] * weights_prev.reindex(symbols).fillna(0)).sum() - cost
        equity.append(daily_ret)
        prev_date = date

    return pd.Series(equity, index=returns.index).cumsum().apply(np.exp)


def run_walk_forward(
    returns,
    signal_type,
    weight_method,
    symbols,
    train_years=3,
    test_months=6,
):
    results = []
    dates = returns.index
    start = dates.min()

    while True:
        train_end = start + pd.DateOffset(years=train_years)
        test_end = train_end + pd.DateOffset(months=test_months)

        test = returns.loc[train_end:test_end]
        if len(test) == 0:
            break

        eq = simulate_portfolio(test, signal_type, weight_method, symbols)
        results.append(eq)
        start = train_end

    if not results:
        return pd.Series(dtype=float)
    return pd.concat(results)
