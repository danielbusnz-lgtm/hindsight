import pandas as pd

from hindsight import harness


def test_walk_forward_only_sees_the_past():
    # The cage test: at bar t the decider must see exactly bars 0..t, never more.
    prices = pd.Series([10, 11, 12, 13, 14], dtype=float)

    seen_lengths = []
    last_values = []

    def spy(history):
        seen_lengths.append(len(history))
        last_values.append(history.iloc[-1])
        return 0.0

    harness.walk_forward(prices, spy)

    # One row at bar 0, two at bar 1, ... never a peek past the current bar.
    assert seen_lengths == [1, 2, 3, 4, 5]
    # The newest value it ever sees is the current bar, never a future one.
    assert last_values == [10, 11, 12, 13, 14]


def test_walk_forward_returns_one_position_per_bar():
    prices = pd.Series([10, 11, 12], dtype=float)

    # Trivial decider: long if the last bar rose, else flat.
    def decide(history):
        if len(history) < 2:
            return 0.0
        return 1.0 if history.iloc[-1] > history.iloc[-2] else 0.0

    positions = harness.walk_forward(prices, decide)

    assert list(positions) == [0.0, 1.0, 1.0]
    assert list(positions.index) == [0, 1, 2]
