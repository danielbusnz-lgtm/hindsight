import pandas as pd
import pytest

from hindsight import backtest


def test_readme_example():
    # Long the whole way up, flat for one bar. Captures the full 100 -> 133.1 move.
    prices = pd.Series([100, 110, 121, 121, 133.1])
    positions = pd.Series([1, 1, 0, 1, 1])

    r = backtest.run(prices, positions)

    assert r.total_return == pytest.approx(0.331)
    assert r.sharpe == pytest.approx(1.5)
    assert r.max_drawdown == pytest.approx(0.0)


def test_no_lookahead():
    # Go long only on the bar that already spiked. An honest backtest captures
    # nothing: the position was set too late to earn the move. A leaking one
    # would report +100%.
    prices = pd.Series([100, 100, 200])
    positions = pd.Series([0, 0, 1])

    r = backtest.run(prices, positions)

    assert r.total_return == pytest.approx(0.0)
