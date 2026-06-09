# hindsight: math and results

A one-page reference for the formulas and the numbers so far.

## Setup

- Prices $p_t$, positions $x_t \in \{-1, 0, +1\}$ (short / flat / long).
- A position is held one bar before it earns, so it can never earn a move it already saw:

$$r_t = x_{t-1}\left(\frac{p_t}{p_{t-1}} - 1\right)$$

That one-bar shift is the whole defense against look-ahead.

## Metrics

**Sharpe** (per period, not annualized):

$$S = \frac{\operatorname{mean}(r)}{\operatorname{std}(r)}$$

**Total return:**

$$\prod_t (1 + r_t) - 1$$

**Max drawdown** (worst peak-to-trough of the equity curve $E_t = \prod_{i \le t}(1+r_i)$):

$$\text{MDD} = \min_t\left(\frac{E_t}{\max_{i \le t} E_i} - 1\right)$$

## Leaderboard

Every candidate is graded **only on the post-cutoff window** (data no model could have memorized). Returns are pooled across assets for breadth, then three layers:

**Bootstrap 95% CI** on the post-cutoff Sharpe. Resample the returns with replacement $B = 2000$ times, recompute Sharpe each time, take the 2.5% and 97.5% percentiles.

**Permutation p-value** vs a no-edge null. Randomly flip the sign of each bar's return ($s_t \in \{-1, +1\}$) and recompute Sharpe; count how often the shuffled result beats the real one:

$$p = \frac{\#\{\, S(s \cdot r) \ge S(r) \,\} + 1}{P + 1}, \quad P = 2000$$

Then Bonferroni-adjust for $N$ candidates tried: $p_{\text{adj}} = \min(1,\ p \cdot N)$.

**Leakage (diff-in-differences).** The candidate's pre-to-post Sharpe drop minus a no-memory control's (momentum), which subtracts the market regime out and leaves only the excess decline the candidate's memory could explain:

$$L = \big(S_{\text{pre}}^{\text{cand}} - S_{\text{post}}^{\text{cand}}\big) - \big(S_{\text{pre}}^{\text{ctrl}} - S_{\text{post}}^{\text{ctrl}}\big)$$

with a bootstrap 95% CI. $L > 0$ with a CI excluding zero = real leakage, not a regime artifact.

## Results

### Real model: gpt-4o-mini

Weekly direction, AAPL/MSFT/NVDA/SPY/TSLA, split at the model's October 2023 cutoff.

| strategy | sharpe_pre | sharpe_post | leakage (vs momentum) | 95% CI |
| --- | --- | --- | --- | --- |
| gpt-4o-mini | 0.197 | 0.101 | +0.029 | [-0.19, +0.28] |
| momentum (control) | 0.103 | 0.036 | (control) | |

No detectable edge: post-cutoff Sharpe barely above the momentum baseline. No detectable leakage: the leakage CI includes zero. An honest null.

### Synthetic controls (validates the method catches leakage when it exists)

| strategy | sharpe_pre | sharpe_post | 95% CI | p(adj) |
| --- | --- | --- | --- | --- |
| memorizer | 1.318 | 0.112 | [+0.07, +0.16] | 0.001 |
| momentum | 0.131 | 0.112 | [+0.07, +0.16] | 0.001 |
| noise | 0.019 | 0.010 | [-0.04, +0.06] | 1.000 |

The memorizer looks brilliant before the cutoff (1.318) and collapses to baseline after (0.112). That gap is exactly the leakage signal the method is built to expose.

## Takeaway

The method works (it catches the synthetic memorizer). The first real model shows no trading edge once it can't cheat. Evidence, scoped to what was tested, not proof.
