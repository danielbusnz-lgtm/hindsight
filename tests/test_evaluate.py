import numpy as np
import pandas as pd

from hindsight import evaluate, strategies


def _panel(n_assets=8, n_bars=300, seed=0):
    rng = np.random.default_rng(seed)
    cols = {}
    for a in range(n_assets):
        r = rng.normal(0.0003, 0.02, n_bars)
        cols[f"a{a}"] = 100 * np.exp(np.cumsum(r))
    return pd.DataFrame(cols)


def test_leakage_flags_the_memorizer_not_the_noise():
    panel = _panel()
    cutoff = 200
    candidates = {
        "memorizer": strategies.make_memorizer(cutoff),  # peeks only before the cutoff
        "momentum": strategies.momentum,                 # the no-memory control
        "noise": strategies.make_noise(1),               # no information at all
    }
    board = {s.name: s for s in evaluate.evaluate(panel, candidates, cutoff, control="momentum")}

    # The memorizer's pre-cutoff peeking shows up as a large positive leakage...
    assert board["memorizer"].leakage > 0.5
    # ...whose CI clears zero (a real excess drop, not regime noise)...
    assert board["memorizer"].leakage_ci_low > 0
    # ...and it dwarfs the noise strategy, which has nothing to leak.
    assert board["memorizer"].leakage > board["noise"].leakage
    # The control itself gets no leakage score.
    assert board["momentum"].leakage is None


def test_no_control_leaves_leakage_unset():
    panel = _panel()
    board = evaluate.evaluate(panel, {"momentum": strategies.momentum}, 200)
    assert board[0].leakage is None
