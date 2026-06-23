"""Trade-sequence Monte Carlo simulation."""

import numpy as np

from config import MONTE_CARLO_SIMULATIONS, RANDOM_SEED


def run_trade_monte_carlo(
    r_multiples: np.ndarray,
    risk_per_trade: float = 0.01,
    simulations: int = MONTE_CARLO_SIMULATIONS,
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    if len(r_multiples) == 0:
        raise ValueError("No trades available for Monte Carlo simulation.")

    account_returns = r_multiples * risk_per_trade
    rng = np.random.default_rng(seed)
    n_trades = len(account_returns)
    results = np.empty((simulations, n_trades + 1))

    for i in range(simulations):
        sampled = rng.choice(account_returns, size=n_trades, replace=True)
        equity = np.empty(n_trades + 1)
        equity[0] = 1.0
        equity[1:] = np.cumprod(1 + sampled)
        results[i] = equity

    return results


def monte_carlo_summary(mc_results: np.ndarray) -> dict:
    final = mc_results[:, -1]
    return {
        "worst": float(np.min(final)),
        "p5": float(np.percentile(final, 5)),
        "median": float(np.percentile(final, 50)),
        "p95": float(np.percentile(final, 95)),
        "best": float(np.max(final)),
        "prob_profit": float(np.mean(final > 1.0)),
        "prob_ruin_50pct": float(np.mean(final < 0.5)),
    }
