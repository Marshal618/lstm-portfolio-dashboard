import numpy as np
import pandas as pd

def simulate_paths_parametric(
    returns: pd.Series,
    n_sims: int = 2000,
    n_days: int = 252,
    seed: int = 42,
):
    """
    Monte Carlo (Parametric):
    Assumes i.i.d. daily returns drawn from N(mu, sigma^2).
    Good for a first-order risk view, but may understate fat tails.
    """
    r = returns.dropna().values
    if len(r) < 30:
        raise ValueError("Not enough returns to run Monte Carlo. Need at least ~30 data points.")

    mu = float(np.mean(r))
    sigma = float(np.std(r, ddof=1))

    rng = np.random.default_rng(seed)
    sims = rng.normal(loc=mu, scale=sigma, size=(n_sims, n_days))

    # Convert to equity paths starting at 1.0
    eq = np.cumprod(1.0 + sims, axis=1)
    eq = np.column_stack([np.ones(n_sims), eq])  # include day 0
    return eq, {"mu": mu, "sigma": sigma}


def simulate_paths_bootstrap(
    returns: pd.Series,
    n_sims: int = 2000,
    n_days: int = 252,
    seed: int = 42,
):
    """
    Monte Carlo (Bootstrap):
    Resamples historical daily returns with replacement.
    Preserves empirical distribution (fat tails / skew) better than Normal.
    """
    r = returns.dropna().values
    if len(r) < 30:
        raise ValueError("Not enough returns to run Monte Carlo. Need at least ~30 data points.")

    rng = np.random.default_rng(seed)
    idx = rng.integers(low=0, high=len(r), size=(n_sims, n_days))
    sims = r[idx]

    eq = np.cumprod(1.0 + sims, axis=1)
    eq = np.column_stack([np.ones(n_sims), eq])
    return eq, {"sample_size": len(r)}


def summarize_terminal(eq_paths: np.ndarray):
    """
    eq_paths shape: (n_sims, n_days+1), starting at 1.0.
    Returns useful summary stats for quant-style reporting.
    """
    terminal = eq_paths[:, -1]
    stats = {
        "p01": float(np.percentile(terminal, 1)),
        "p05": float(np.percentile(terminal, 5)),
        "p50": float(np.percentile(terminal, 50)),
        "p95": float(np.percentile(terminal, 95)),
        "p99": float(np.percentile(terminal, 99)),
        "mean": float(np.mean(terminal)),
        "prob_loss": float(np.mean(terminal < 1.0)),
    }
    return stats, terminal
