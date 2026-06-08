"""Run the leaderboard on synthetic data with stand-in strategies.

No LLMs, no API cost. The point is to prove the machine works: the memorizer
should look brilliant before the cutoff and collapse after, while momentum
stays honest across both windows.

    python examples/leaderboard_demo.py
"""

import numpy as np
import pandas as pd

from hindsight import evaluate, strategies

N_ASSETS = 20
N_BARS = 500
CUTOFF = 400  # 400 bars "in training", 100 bars after the model's knowledge ends


def make_panel(n_assets, n_bars, phi=0.15, seed=7):
    """Prices with mild return momentum, so a causal momentum edge is real."""
    rng = np.random.default_rng(seed)
    cols = {}
    for a in range(n_assets):
        noise = rng.normal(0.0003, 0.02, n_bars)
        r = np.zeros(n_bars)
        for t in range(1, n_bars):
            r[t] = phi * r[t - 1] + noise[t]  # AR(1): yesterday's move bleeds in
        cols[f"asset_{a}"] = 100 * np.exp(np.cumsum(r))
    return pd.DataFrame(cols)


def main():
    panel = make_panel(N_ASSETS, N_BARS)

    candidates = {
        "memorizer": strategies.make_memorizer(CUTOFF),
        "momentum": strategies.momentum,
        "noise": strategies.make_noise(seed=1),
    }

    board = evaluate.evaluate(panel, candidates, CUTOFF)

    print(f"{'strategy':10s} {'sharpe_pre':>11s} {'sharpe_post':>12s} "
          f"{'post_95%_CI':>20s} {'p(adj)':>8s}")
    print("-" * 65)
    for s in board:
        ci = f"[{s.ci_low:+.2f}, {s.ci_high:+.2f}]"
        print(f"{s.name:10s} {s.sharpe_pre:>11.3f} {s.sharpe_post:>12.3f} "
              f"{ci:>20s} {s.p_value:>8.3f}")


if __name__ == "__main__":
    main()
