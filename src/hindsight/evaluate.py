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
    # Leakage = excess pre->post drop vs a no-memory control (diff-in-differences).
    # Set only when evaluate(... control=...) is given. >0 and CI excluding 0 = real
    # leakage, not a market-regime artifact. None for the control itself.
    leakage: float | None = None
    leakage_ci_low: float | None = None
    leakage_ci_high: float | None = None


def _sharpe(returns: np.ndarray) -> float:
    sd = returns.std(ddof=1)
    return float(returns.mean() / sd) if sd else 0.0


def _pooled_pre_post(panel: pd.DataFrame, strategy, cutoff: int):
    """Run the strategy once per asset; return pooled returns split at the cutoff.

    The strategy can be expensive (an LLM called per bar), so it runs exactly
    once per asset and the single result is sliced into pre and post windows.
    """
    pre_chunks, post_chunks = [], []
    for col in panel.columns:
        prices = panel[col]
        strat = backtest.strategy_returns(prices, strategy(prices))
        pre_chunks.append(strat.iloc[:cutoff].to_numpy())
        post_chunks.append(strat.iloc[cutoff:].to_numpy())
    return np.concatenate(pre_chunks), np.concatenate(post_chunks)


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


def _bootstrap_did(cand_pre, cand_post, ctrl_pre, ctrl_post, n_boot, rng):
    """Diff-in-differences leakage: the candidate's pre->post Sharpe drop minus the
    control's. Subtracts the market regime out, leaving only the excess decline a
    candidate's memory could explain. Returns (point, ci_low, ci_high)."""
    def gap(pre, post):
        return _sharpe(pre) - _sharpe(post)

    def resample(a):
        return a[rng.integers(0, len(a), len(a))]

    point = gap(cand_pre, cand_post) - gap(ctrl_pre, ctrl_post)
    stats = [
        gap(resample(cand_pre), resample(cand_post)) - gap(resample(ctrl_pre), resample(ctrl_post))
        for _ in range(n_boot)
    ]
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def evaluate(panel, strategies, cutoff, n_boot=2000, n_perm=2000, seed=0, control=None):
    """Rank strategies on the post-cutoff window. Returns Scores, best first.

    Pass control=<strategy name> (a no-memory baseline, e.g. "momentum") to also
    compute each candidate's leakage: its excess pre->post drop over the control,
    with a bootstrap CI. The control's own positions run once and are reused, so
    this adds no extra work for an expensive (LLM) strategy.
    """
    rng = np.random.default_rng(seed)
    n = len(strategies)
    if control is not None and control not in strategies:
        raise ValueError(f"control {control!r} is not one of the strategies")

    # Pool every strategy's returns once (this is the expensive step for an LLM),
    # then reuse the arrays for the post-CI, the permutation test, and leakage.
    pooled = {name: _pooled_pre_post(panel, fn, cutoff) for name, fn in strategies.items()}

    scores = []
    for name, (pre, post) in pooled.items():
        ci_low, ci_high = _bootstrap_ci(post, n_boot, rng)
        p = min(1.0, _permutation_p(post, n_perm, rng) * n)  # Bonferroni
        score = Score(name, _sharpe(pre), _sharpe(post), ci_low, ci_high, p)
        if control is not None and name != control:
            ctrl_pre, ctrl_post = pooled[control]
            score.leakage, score.leakage_ci_low, score.leakage_ci_high = _bootstrap_did(
                pre, post, ctrl_pre, ctrl_post, n_boot, rng
            )
        scores.append(score)
    scores.sort(key=lambda s: s.sharpe_post, reverse=True)
    return scores
