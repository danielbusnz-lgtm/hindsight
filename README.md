# hindsight

> Can an LLM actually trade stocks, or does it just look like it because it has already seen the future?

When you backtest an LLM trading strategy on past price data, the model may have trained on that period. It isn't predicting, it's remembering, so the backtest looks great and means nothing. `hindsight` answers the question honestly: it grades every strategy only on data *after* the model's training cutoff, where there is nothing to remember. What survives is real skill, if any.

**Status:** working research tool. The backtester, walk-forward harness, leaderboard, and OpenAI decider all run end to end. First real result is in. See [DESIGN.md](DESIGN.md) for methodology and known limits.

## First finding

Does gpt-4o-mini have real trading skill? On weekly direction (AAPL, MSFT, NVDA, SPY, TSLA), graded across its October 2023 training cutoff:

```
strategy     sharpe_pre  sharpe_post  leakage (vs momentum)
gpt-4o-mini       0.197        0.101  +0.029 [-0.19, +0.28]
momentum          0.103        0.036  (control)
```

**Verdict: no detectable edge.** The post-cutoff Sharpe (0.101) is barely above the momentum baseline and well inside the noise, no evidence gpt-4o-mini can actually trade in this run.

It also shows **no detectable leakage**: the post-cutoff drop is no larger than momentum's and the leakage CI includes zero. That's an honest null, not a failure, and the diff-in-differences design subtracts the market regime out, so the CI is actually saying something. (It never looked impressive even pre-cutoff, so there was little memorization to find in the first place.)

## How it works

The question is **performance**. Leakage is the thing that makes naive backtests lie, so the method is built to be leakage-proof, and the leakage gap is reported as a diagnostic.

**Grade only on post-cutoff data.** You can't *detect* leakage from a single backtest, real skill and memorization look identical in the numbers. So don't try; make it impossible. Grade every strategy on the common window after the latest training cutoff in the lineup. Same blind exam for everyone.

**Honest backtest.** Positions earn the *next* bar's return, never a bar the strategy could have already seen. That single shift is the whole defense against look-ahead.

**Walk-forward harness.** Calls a decider with `prices[:t+1]` at each bar, so the future is never in scope. An LLM decider plugs in here.

**Fair leaderboard.** Splits each strategy's returns at the cutoff, pools across assets (breadth), bootstraps a 95% CI on each Sharpe, and runs a permutation test (Bonferroni-adjusted). The `sharpe_pre` vs `sharpe_post` gap is the **leakage diagnostic**: a strategy that collapses across the cutoff was remembering, which explains why its naive backtest looked good. The diff-in-differences column subtracts a no-memory control (momentum) to strip out the market regime.

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

The memorizer looks brilliant before the cutoff and collapses after. That gap is the leakage signal the method is designed to expose.

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
