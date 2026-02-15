import numpy as np
import pandas as pd


def make_supervised(series_1d: np.ndarray, lookback: int):
    X, y = [], []
    for i in range(lookback, len(series_1d)):
        X.append(series_1d[i - lookback : i])
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
        weights[under] = weights[under] + excess * (
            weights[under] / (weights[under].sum() + 1e-12)
        )

    return weights / weights.sum()


def predict_next_log_return(sym, asof_date, log_returns_all, models, scalers, lookback_period):
    if sym not in models:
        return np.nan
    hist = log_returns_all.loc[:asof_date, sym].dropna()
    if len(hist) < lookback_period:
        return np.nan

    last_window = hist.values[-lookback_period:].reshape(-1, 1)
    scaled = scalers[sym].transform(last_window).flatten()
    X = scaled.reshape((1, lookback_period, 1))

    pred_scaled = models[sym].predict(X, verbose=0)[0, 0]
    pred = scalers[sym].inverse_transform(
        np.array(pred_scaled).reshape(1, 1)
    )[0, 0]
    return float(pred)

