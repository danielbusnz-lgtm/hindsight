# hindsight

> How much is your LLM trading model *predicting* versus *remembering*?

When you backtest an LLM on past data, it may have trained on that period. It
isn't predicting, it's remembering, and the backtest looks great for no real
reason. `hindsight` grades strategies only on data after the model's training
cutoff, where there's nothing to remember, so the score you get is honest.

**Status:** working backtester and leaderboard. Strategies are stand-ins for
now; the real LLM harness is next. See [DESIGN.md](DESIGN.md).

## Backtest one strategy

```python
import pandas as pd
from hindsight import backtest

prices = pd.Series([100, 110, 121, 121, 133.1])
positions = pd.Series([1, 1, 0, 1, 1])  # +1 long, 0 flat, -1 short

r = backtest.run(prices, positions)
print(r.total_return, r.sharpe, r.max_drawdown)
```

Positions earn the *next* bar's return, never a move they could have already
seen. That shift is the whole defense against look-ahead.

## Rank strategies (leakage-safe)

Grades every strategy on the post-cutoff window, pools across many assets, and
adds a bootstrap CI and a permutation p-value.

```bash
python examples/leaderboard_demo.py
```

```
strategy    sharpe_pre  sharpe_post      post_95%_CI    p(adj)
memorizer        1.318        0.112   [+0.07, +0.16]    0.001   <- caught
momentum         0.131        0.112   [+0.07, +0.16]    0.001
noise            0.019        0.010   [-0.04, +0.06]    1.000
```

The memorizer looks brilliant before the cutoff and collapses after. That gap
is the leakage.

## Develop

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest
```

## License

MIT
