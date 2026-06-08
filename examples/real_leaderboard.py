"""First real leakage reading: gpt-4o-mini vs a momentum baseline on AAPL, weekly.

Both are graded on the same windows, split at the model's training cutoff. If
the model was 'remembering' pre-cutoff, its pre-Sharpe towers over its
post-Sharpe; momentum (no memory) should look about the same on both sides.

Scoped small (one ticker, weekly bars) to keep the API bill tiny.

    OPENAI_API_KEY=... python examples/real_leaderboard.py
"""

import pandas as pd

from hindsight import cutoffs, evaluate, harness, llm, results, strategies

MODEL = "gpt-4o-mini"

df = pd.read_csv("data/prices.csv", index_col=0, parse_dates=True)
weekly = df.resample("W").last().dropna().loc["2023-01-01":"2025-01-01"]

cut = cutoffs.cutoff(MODEL)
cutoff_idx = int(weekly.index.searchsorted(cut))
print(f"{list(weekly.columns)}: {len(weekly)} weekly bars x {weekly.shape[1]} tickers; "
      f"cutoff {cut.date()} -> {cutoff_idx} pre / {len(weekly) - cutoff_idx} post")


def model_strategy(prices):
    decide = llm.make_openai_decider(model=MODEL, lookback=12, identify=True)
    return harness.walk_forward(prices, decide)


candidates = {MODEL: model_strategy, "momentum": strategies.momentum}
board = evaluate.evaluate(weekly, candidates, cutoff_idx, n_boot=1000, n_perm=1000, control="momentum")

print(f"\n{'strategy':12s} {'sharpe_pre':>11s} {'sharpe_post':>12s} {'leakage (vs momentum)':>28s}")
print("-" * 66)
for s in board:
    if s.leakage is None:
        lk = "(control)"
    else:
        lk = f"{s.leakage:+.3f} [{s.leakage_ci_low:+.2f}, {s.leakage_ci_high:+.2f}]"
    print(f"{s.name:12s} {s.sharpe_pre:>11.3f} {s.sharpe_post:>12.3f} {lk:>28s}")

out = results.append(
    "data/runs/leaderboard.csv",
    board,
    model=MODEL,
    ticker=",".join(weekly.columns),
    window=f"{weekly.index[0].date()}..{weekly.index[-1].date()}",
    cutoff=str(cut.date()),
)
print(f"\nappended {len(board)} rows to {out}")
