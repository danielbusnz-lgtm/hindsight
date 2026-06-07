# hindsight

> How much is your LLM trading model *predicting* versus *remembering*?

When you backtest an LLM on past data, it already knows what happened. The backtest looks great and means nothing. `hindsight` measures that leakage and discounts the result.

**Status:** scaffold only. Nothing implemented yet.

## Planned API

```python
import pandas as pd
from hindsight import backtest

prices = pd.Series([100, 110, 121, 121, 133.1])
positions = pd.Series([1, 1, 0, 1, 1])  # +1 long, 0 flat, -1 short

r = backtest.run(prices, positions)
print(r.total_return, r.sharpe, r.max_drawdown)
```

## License

MIT
