import pandas as pd
import pytest

from hindsight import cutoffs


def test_cutoff_returns_timestamp():
    assert cutoffs.cutoff("gpt-4o-mini") == pd.Timestamp("2023-10-01")


def test_unknown_model_raises():
    with pytest.raises(KeyError):
        cutoffs.cutoff("gpt-9000")


def test_common_cutoff_is_the_latest():
    # The shared blind window starts at the newest model's cutoff.
    models = ["gpt-4o-mini", "claude-3-7-sonnet", "claude-sonnet-4-5"]
    assert cutoffs.common_cutoff(models) == pd.Timestamp("2025-01-01")
