# Design

## Problem

When you backtest an LLM trading strategy on past data, the model may have
trained on that period. It isn't predicting, it's remembering. The backtest
looks great and means nothing.

## Idea

Don't try to *detect* leakage from one backtest (you can't: skill and peeking
look identical in the numbers). Make it impossible instead: grade every
strategy only on data *after the model's training cutoff*, where there is
nothing to remember.

For a (model, prompt) bake-off, grade everyone on a **common window**: the
period after the latest cutoff in the lineup. Same blind exam for all.

## How a strategy is scored

Three layers on top of the cutoff split:

1. **Breadth** - pool returns across many assets, not one. More bets = less luck.
2. **Bootstrap** - a 95% confidence interval on each Sharpe.
3. **Permutation + Bonferroni** - a p-value vs a no-edge null, adjusted for how
   many candidates were tried.

`sharpe_pre` vs `sharpe_post` exposes leakage: a big drop across the cutoff
means the strategy was remembering.

## Pieces

- `backtest.py` - honest backtest. Positions earn the *next* bar's return.
- `strategies.py` - stand-in strategies (cheater, momentum, memorizer, noise)
  used as ground truth while there are no real models yet.
- `evaluate.py` - the leaderboard: cutoff split + the three layers above.
- `harness.py` - walk-forward loop. Hands a decider only `prices[:t+1]` each
  bar, so the future can't leak. An LLM plugs in as the decider.
- `llm.py` - OpenAI decider. Forces a long/flat/short answer, maps to +1/0/-1.
- `examples/` - leaderboard demo (no LLMs) and a live decider smoke test.

## Next

- Wire the LLM decider into `evaluate.py` so real models run the leaderboard
  (expensive: many bars x assets x models = thousands of calls).
- Surface the leakage gap (`sharpe_pre - sharpe_post`) as its own column.
- Richer prompts (dates/news) where memorization leakage actually lives.
- Stress-test a short post-cutoff window to see when the CIs blow up.

## Known limits

- Post-cutoff data is shorter and noisier; the newest model caps the window.
- A short window means the ranking can't tell a real winner from a lucky one.
