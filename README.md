# hindsight

> How much is your LLM trading model *predicting* versus *remembering*?

When you backtest an LLM on past price data, the model may have trained on that
period. It isn't predicting, it's remembering. The backtest looks great and means
nothing. `hindsight` grades strategies only on data after the model's training
cutoff, where there is nothing to remember.

**Status:** working research tool. The backtester, walk-forward harness,
leaderboard, and OpenAI decider all run end to end. The `llm` extra is optional
(see install below). First real result is in. See [DESIGN.md](DESIGN.md) for
methodology and known limits.

## First real finding

gpt-4o-mini on weekly direction (AAPL, MSFT, NVDA, SPY, TSLA), split at its
October 2023 training cutoff:

```
strategy     sharpe_pre  sharpe_post  leakage (vs momentum)
gpt-4o-mini       0.197        0.101  +0.029 [-0.19, +0.28]
momentum          0.103        0.036  (control)
```

Leakage CI includes zero. The model's post-cutoff drop is no larger than
momentum's, so there is no detectable memorization signal in this run. That's an
honest null result, not a failure. The diff-in-differences design subtracts the
market regime out, so the CI is actually saying something.

## How it works

Three pieces:

**Honest backtest.** Positions earn the *next* bar's return, never a bar the
strategy could have already seen. That single shift is the whole defense against
look-ahead.

**Walk-forward harness.** Calls a decider with `prices[:t+1]` at each bar. The
future is never in scope. An LLM decider plugs in here.

**Leakage-safe leaderboard.** Splits each strategy's returns at the model's
training cutoff, pools across assets (breadth), bootstraps a 95% CI on each
Sharpe, and runs a permutation test (Bonferroni-adjusted). The diff-in-differences
leakage column subtracts a no-memory control (momentum) to strip out the market
regime.

## Backtest one strategy

```python
import pandas as pd
from hindsight import backtest

prices = pd.Series([100, 110, 121, 121, 133.1])
positions = pd.Series([1, 1, 0, 1, 1])  # +1 long, 0 flat, -1 short

r = backtest.run(prices, positions)
print(r.total_return, r.sharpe, r.max_drawdown)
```

## Run the leaderboard (controls only, no API needed)

```bash
python examples/leaderboard_demo.py
```

```
strategy    sharpe_pre  sharpe_post      post_95%_CI    p(adj)
memorizer        1.318        0.112   [+0.07, +0.16]    0.001   <- caught
momentum         0.131        0.112   [+0.07, +0.16]    0.001
noise            0.019        0.010   [-0.04, +0.06]    1.000
```

The memorizer looks brilliant before the cutoff and collapses after. That gap is
the leakage signal.

## Run with a real model

```bash
OPENAI_API_KEY=... python examples/real_leaderboard.py
```

Results append to `data/runs/leaderboard.csv`.

## Install

```bash
python -m venv .venv
source .venv/bin/activate

# Core only (no LLM calls)
pip install -e ".[dev]"

# With OpenAI decider
pip install -e ".[dev,llm]"

pytest   # 14 tests
```

## License

MIT
