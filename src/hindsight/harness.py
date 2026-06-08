"""Walk-forward harness: turn a bar-by-bar decider into a positions series.

The decider only ever sees the past. At each bar t it receives prices[:t+1]
(everything up to and including today) and returns a position: +1 long, 0 flat,
-1 short. The future is never passed in, so it cannot leak. A real LLM plugs in
as the `decide` callable:

    def decide(history):                 # history = prices up to and including now
        prompt = build_prompt(history)   # only the past goes into the prompt
        answer = call_model(prompt)      # "long" / "short" / "flat"
        return {"long": 1.0, "flat": 0.0, "short": -1.0}[answer]

    positions = walk_forward(prices, decide)
"""

from typing import Callable

import pandas as pd

Decider = Callable[[pd.Series], float]


def walk_forward(prices: pd.Series, decide: Decider) -> pd.Series:
    positions = []
    for t in range(len(prices)):
        history = prices.iloc[: t + 1]   # only bars 0..t; the future is not in scope
        positions.append(decide(history))
    return pd.Series(positions, index=prices.index, dtype=float)
