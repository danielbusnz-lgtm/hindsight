"""Minimal honest backtest.

The one rule that matters: a position held at time t-1 earns the return
realized at t. Positions are shifted forward by one bar so a position can
never earn a move it could have already seen. That shift is the whole
defense against look-ahead leakage.
"""

from dataclasses import dataclass

import pandas as pd


@dataclass
class Result:
    total_return: float
    sharpe: float
    max_drawdown: float


def strategy_returns(prices: pd.Series, positions: pd.Series) -> pd.Series:
    """Per-bar strategy returns under honest alignment.

    The position from the previous bar earns this bar's return, so a position
    can never earn a move it could have already seen. This shift is the whole
    defense against look-ahead leakage.
    """
    returns = prices.pct_change()
    return (positions.shift(1) * returns).dropna()


def run(prices: pd.Series, positions: pd.Series) -> Result:
    strat = strategy_returns(prices, positions)

    total_return = (1 + strat).prod() - 1
    sharpe = strat.mean() / strat.std() if strat.std() else 0.0

    equity = (1 + strat).cumprod()
    drawdown = equity / equity.cummax() - 1
    max_drawdown = drawdown.min()

    return Result(
        total_return=float(total_return),
        sharpe=float(sharpe),
        max_drawdown=float(max_drawdown),
    )
