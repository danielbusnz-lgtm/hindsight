# hindsight: math and results

The formulas and the numbers, on one page.

## Setup

- Prices $p_t$, position $x_t \in \{-1, 0, +1\}$ (short / flat / long).
- A position is held one bar before it earns, so it can never earn a move it already saw:

$$r_t = x_{t-1}\left(\frac{p_t}{p_{t-1}} - 1\right)$$

That one-bar shift is the whole defense against look-ahead.

## Metrics

$\bar r$ is the mean return, $\sigma$ its standard deviation.

**Sharpe** (per period, not annualized):

$$S = \frac{\bar r}{\sigma}$$

**Total return:**

$$\prod_t (1 + r_t) - 1$$

**Max drawdown.** With equity $E_t = \prod_{i \le t}(1 + r_i)$:

$$\text{MDD} = \min_t\left(\frac{E_t}{\max_{i \le t} E_i} - 1\right)$$

## Leaderboard

Grade every candidate only on the **post-cutoff window** (data no model could have memorized). Pool returns across assets for breadth, then three layers.

**Bootstrap 95% CI.** Resample the post-cutoff returns with replacement $B = 2000$ times, recompute Sharpe each time, take the 2.5% and 97.5% percentiles.

**Permutation p-value** vs a no-edge null. Flip the sign of each bar's return at random, $P = 2000$ times. Let $k$ be the number of flips whose Sharpe is at least the real one:

$$p = \frac{k + 1}{P + 1}$$

Then Bonferroni-adjust for $N$ candidates: $p_{\text{adj}} = \min(1,\ p \cdot N)$.

**Leakage (diff-in-differences).** The candidate's pre-to-post Sharpe drop minus a no-memory control's (momentum). Subtracts the market regime, leaving only the drop the candidate's memory could explain:

$$L = (S^{\text{cand}}_{\text{pre}} - S^{\text{cand}}_{\text{post}}) - (S^{\text{ctrl}}_{\text{pre}} - S^{\text{ctrl}}_{\text{post}})$$

with a bootstrap 95% CI. $L > 0$ and CI excludes zero means real leakage, not a regime artifact.

## Results

### gpt-4o-mini (real model)

Weekly direction, AAPL/MSFT/NVDA/SPY/TSLA, split at the Oct 2023 cutoff.

| strategy | sharpe_pre | sharpe_post | leakage | 95% CI |
| --- | --- | --- | --- | --- |
| gpt-4o-mini | 0.197 | 0.101 | +0.029 | [-0.19, +0.28] |
| momentum (control) | 0.103 | 0.036 | (control) | |

No edge (post Sharpe barely above momentum) and no leakage (CI includes zero). An honest null.

### Synthetic controls (method check)

| strategy | sharpe_pre | sharpe_post | 95% CI | p(adj) |
| --- | --- | --- | --- | --- |
| memorizer | 1.318 | 0.112 | [+0.07, +0.16] | 0.001 |
| momentum | 0.131 | 0.112 | [+0.07, +0.16] | 0.001 |
| noise | 0.019 | 0.010 | [-0.04, +0.06] | 1.000 |

The memorizer looks brilliant pre-cutoff (1.318) and collapses after (0.112). That gap is the leakage signal the method is built to catch.

## Takeaway

The method works: it catches the synthetic memorizer. The first real model shows no trading edge once it can't cheat. Evidence, scoped to what was tested, not proof.
