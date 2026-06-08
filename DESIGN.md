# Design

## Question

Can an LLM actually trade stocks? Naive backtests say yes, but they can't be
trusted: when you backtest an LLM on past price data, the model may have trained
on that period. It isn't predicting, it's remembering. The backtest looks great
and means nothing.

The goal is an honest answer about **performance**. Leakage is the contamination
that gets in the way, so the method is built to defeat it, and the leakage gap is
reported as a diagnostic, not the headline.

## Method

Don't try to *detect* leakage from one backtest (you can't: skill and peeking
look identical in the numbers). Make it impossible instead: grade every strategy
only on data *after the model's training cutoff*, where there is nothing to
remember. What survives is real skill.

For a (model, prompt) bake-off, grade everyone on a **common window**: the period
after the latest cutoff in the lineup. Same blind exam for all.

## How a strategy is scored

Performance on the post-cutoff window, made trustworthy by three layers:

1. **Breadth** - pool returns across many assets, not one. More bets = less luck.
2. **Bootstrap** - a 95% confidence interval on each Sharpe.
3. **Permutation + Bonferroni** - a p-value vs a no-edge null, adjusted for how
   many candidates were tried.

**Leakage is the diagnostic, not the verdict.** `sharpe_pre` vs `sharpe_post`
exposes it: a big drop across the cutoff means the strategy was remembering, and
explains why its naive backtest looked good. It only matters when a strategy
looked good pre-cutoff in the first place; a strategy with no apparent skill has
nothing to inflate.

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
- The verdict is **evidence, scoped to what was tested**, not proof. "No edge in
  these models, prompts, and instruments" is the honest claim, not "LLMs can
  never trade."
- hindsight measures *model* memorization. It does not measure *researcher*
  hindsight (picking the instrument or period because you already knew it worked).
  That's a different bias, out of scope here.
