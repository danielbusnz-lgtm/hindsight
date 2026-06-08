"""Reference strategies: ground-truth stand-ins for real (model, prompt) traders.

cheater   - peeks at the future every bar. Pure code-level look-ahead.
momentum  - decides from the past only. No foresight, but a real edge if the
            data has momentum.
memorizer - simulates a model that trained on data up to a cutoff: it peeks
            before the cutoff (it "remembers") and goes causal after. The
            cutoff split is supposed to catch exactly this.
noise     - random positions. The floor; should look like luck in every window.
"""

import numpy as np
import pandas as pd


def cheater(prices: pd.Series) -> pd.Series:
    # Sets today's position from tomorrow's return: pure look-ahead.
    future_return = prices.pct_change().shift(-1)
    return np.sign(future_return).fillna(0)


def momentum(prices: pd.Series) -> pd.Series:
    # Long if the last bar was up. Uses only information already realized.
    past_return = prices.pct_change()
    return np.sign(past_return).fillna(0)


def make_memorizer(cutoff: int):
    """A model that 'remembers' the future up to `cutoff`, then flies blind."""

    def memorizer(prices: pd.Series) -> pd.Series:
        future = np.sign(prices.pct_change().shift(-1)).fillna(0)  # peek
        past = np.sign(prices.pct_change()).fillna(0)              # causal
        positions = past.copy()
        positions.iloc[:cutoff] = future.iloc[:cutoff]            # memory before cutoff
        return positions

    return memorizer


def make_noise(seed: int):
    """Random long/short, no information at all."""

    def noise(prices: pd.Series) -> pd.Series:
        rng = np.random.default_rng(seed)
        return pd.Series(rng.choice([-1.0, 1.0], size=len(prices)), index=prices.index)

    return noise
