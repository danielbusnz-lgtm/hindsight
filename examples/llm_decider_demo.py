"""Smoke-test LLM deciders on a tiny series (a dozen API calls per model).

Runs every model named on the command line, or a cross-provider default pair.
Models whose provider key is missing from the environment are skipped, so the
demo runs with either key alone.

    OPENAI_API_KEY=... ANTHROPIC_API_KEY=... python examples/llm_decider_demo.py
    python examples/llm_decider_demo.py gpt-4o-mini claude-haiku-4-5
"""

import os
import sys

import numpy as np
import pandas as pd

from hindsight import backtest, cutoffs, harness, llm

DEFAULT_MODELS = ["gpt-4o-mini", "claude-opus-4-8"]
KEY_FOR = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}

rng = np.random.default_rng(0)
prices = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.001, 0.02, 12))))
print("prices:", [round(p, 1) for p in prices])

models = sys.argv[1:] or DEFAULT_MODELS
for model in models:
    key = KEY_FOR[cutoffs.provider(model)]
    if not os.environ.get(key):
        print(f"\n{model}: skipped ({key} not set)")
        continue

    decide = llm.make_decider(model, lookback=10)
    positions = harness.walk_forward(prices, decide)
    r = backtest.run(prices, positions)
    print(f"\n{model} (cutoff {cutoffs.cutoff(model).date()})")
    print(f"  positions: {list(positions)}")
    print(f"  total_return={r.total_return:.3f}  sharpe={r.sharpe:.3f}  max_dd={r.max_drawdown:.3f}")
