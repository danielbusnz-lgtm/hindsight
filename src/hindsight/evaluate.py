"""Leakage-safe leaderboard for trading strategies.

Every candidate is graded only on the post-cutoff window, the stretch of time
no model in the lineup could have memorised. Inside that window three layers
add statistical rigor:

  1. Breadth      - pool returns across many assets, not one. (Power.)
  2. Bootstrap    - a confidence interval on each Sharpe.     (Uncertainty.)
  3. Permutation  - a p-value vs a no-edge null, Bonferroni-adjusted for how
                    many candidates were tried.               (Honesty tax.)

`sharpe_pre` is reported only to expose leakage: a candidate that looks great
before the cutoff and dies after was remembering, not predicting.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from hindsight import backtest


@dataclass
class Score:
    name: str
    sharpe_pre: float   # in-sample; inflated by any leakage
    sharpe_post: float  # out-of-sample; the number you trust
    ci_low: float       # bootstrap 2.5% on sharpe_post
    ci_high: float      # bootstrap 97.5% on sharpe_post
    p_value: float      # permutation p, Bonferroni-adjusted


def _sharpe(returns: np.ndarray) -> float:
    sd = returns.std(ddof=1)
    return float(returns.mean() / sd) if sd else 0.0


def _pooled_returns(panel: pd.DataFrame, strategy, lo: int, hi) -> np.ndarray:
    """Honest per-bar strategy returns over [lo:hi], pooled across all assets."""
    chunks = []
    for col in panel.columns:
        prices = panel[col]
        strat = backtest.strategy_returns(prices, strategy(prices))
        chunks.append(strat.iloc[lo:hi].to_numpy())
    return np.concatenate(chunks)


def _bootstrap_ci(returns: np.ndarray, n_boot: int, rng) -> tuple[float, float]:
    n = len(returns)
    sharpes = [_sharpe(returns[rng.integers(0, n, n)]) for _ in range(n_boot)]
    lo, hi = np.percentile(sharpes, [2.5, 97.5])
    return float(lo), float(hi)


def _permutation_p(returns: np.ndarray, n_perm: int, rng) -> float:
    # Null: no real edge, just symmetric fluctuations. Flip each bar's sign.
    obs = _sharpe(returns)
    n = len(returns)
    hits = sum(_sharpe(returns * rng.choice([-1.0, 1.0], n)) >= obs for _ in range(n_perm))
    return (hits + 1) / (n_perm + 1)


def evaluate(panel, strategies, cutoff, n_boot=2000, n_perm=2000, seed=0):
    """Rank strategies on the post-cutoff window. Returns Scores, best first."""
    rng = np.random.default_rng(seed)
    n = len(strategies)
    scores = []
    for name, fn in strategies.items():
        pre = _pooled_returns(panel, fn, 0, cutoff)
        post = _pooled_returns(panel, fn, cutoff, None)
        ci_low, ci_high = _bootstrap_ci(post, n_boot, rng)
        p = min(1.0, _permutation_p(post, n_perm, rng) * n)  # Bonferroni
        scores.append(Score(name, _sharpe(pre), _sharpe(post), ci_low, ci_high, p))
    scores.sort(key=lambda s: s.sharpe_post, reverse=True)
    return scores
