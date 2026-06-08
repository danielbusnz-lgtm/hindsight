"""Smoke-test the OpenAI decider on a tiny series (a dozen API calls).

    OPENAI_API_KEY=... python examples/llm_decider_demo.py
"""

import numpy as np
import pandas as pd

from hindsight import backtest, harness, llm

rng = np.random.default_rng(0)
prices = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.001, 0.02, 12))))

decide = llm.make_openai_decider(model="gpt-4o-mini", lookback=10)
positions = harness.walk_forward(prices, decide)

print("prices:   ", [round(p, 1) for p in prices])
print("positions:", list(positions))

r = backtest.run(prices, positions)
print(f"total_return={r.total_return:.3f}  sharpe={r.sharpe:.3f}  max_dd={r.max_drawdown:.3f}")
