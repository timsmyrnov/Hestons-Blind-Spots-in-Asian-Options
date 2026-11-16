# Does the neural surrogate cost PnL? (synthetic experiment)

**Synthetic data only.** No real market data was used. Every error below is
surrogate approximation error or Monte Carlo sampling error under the Heston
pricer. None of it is Heston model error.

## Setup

- 27 synthetic AO calls (Aug-Oct 2024 averages, strikes $60-100).
- Narrow parameter box: v0 and theta 0.01-0.36 (10-60% volatility), kappa 0.1-5,
  sigma 0.05-1.5, rho -0.95 to 0.5.
- Network: the default manifold architecture, trained on 16,384 parameter draws priced
  with 16,384-path Monte Carlo (6,000 epochs).
- Truth: the average of two independent 262,144-path Monte Carlo runs at
  512 held-out test points.
- Market: truth plus random noise of 0.07 USD/bbl per
  contract. Each trader sells when the market is above its model price by more than
  0.03 USD/bbl and buys when below, paying
  0.01 USD/bbl per trade. PnL is measured against
  truth, 1,000 barrels per contract.

## Accuracy against truth (USD/bbl)

| Pricer | RMSE (truth noise removed) | Median error | Worst error |
|---|---:|---:|---:|
| Neural network | 0.0393 | 0.0170 | 0.209 |
| Monte Carlo, 16,384 paths | 0.0385 | 0.0155 | 0.205 |
| Monte Carlo, 4,096 paths | 0.0498 | 0.0204 | 0.277 |

## Trading PnL (512 scenarios x 27 contracts)

| Trader | Total PnL (USD) | % of oracle [95% CI] | Trades | Wrong-way trades |
|---|---:|---:|---:|---:|
| Oracle (knows truth) | 610,904 | 100.0% [100.0, 100.0] | 9,231 | 0 |
| Neural network | 540,583 | 88.5% [87.3, 89.6] | 9,730 | 627 |
| Monte Carlo, 16,384 paths | 542,713 | 88.8% [87.6, 90.0] | 9,746 | 631 |
| Monte Carlo, 4,096 paths | 508,491 | 83.2% [81.7, 84.7] | 9,837 | 990 |

The difference between the network and Monte Carlo at 16,384 paths is not statistically distinguishable from zero: network minus MC-16k is -2,130 USD
(-0.3 percentage points of oracle PnL), 95% interval
-4,435 to +175 USD. The network gives up
70,321 USD to the oracle (95% interval
63,438 to 77,579 USD).

Training ran the full 6,000 epochs without meeting the early-stopping rule, so the network may not have reached its best accuracy.

## Training length: runs side by side

The 3,000-epoch run hit its cap, so the experiment was repeated with a
6,000-epoch cap. Everything else is identical: same data, settings, training seed, truth, market and
Monte Carlo traders. Only the network differs. Both runs are reported whichever looks
better. The tables and charts above show the 6,000-epoch run, which was chosen as the reported run before it was trained.

| | 3,000-epoch cap | 6,000-epoch cap (shown above) |
|---|---:|---:|
| Epochs trained | 3,000 (hit cap) | 6,000 (hit cap) |
| Best validation loss (standardized MSE) | 9.52e-05 | 8.18e-05 |
| Network RMSE vs truth (USD/bbl) | 0.0394 | 0.0393 |
| Network median error (USD/bbl) | 0.0175 | 0.0170 |
| Network worst error (USD/bbl) | 0.2096 | 0.2089 |
| Network wrong-way trades | 626 | 627 |
| Network PnL, % of oracle [95% CI] | 88.2% [87.0, 89.4] | 88.5% [87.3, 89.6] |
| MC-16k PnL, % of oracle [95% CI] | 88.8% [87.6, 90.0] | 88.8% [87.6, 90.0] |
| Network minus MC-16k, USD [95% CI] | -3,592 [-6,015, -1,108] | -2,130 [-4,435, +175] |
| Network minus MC-16k, points [95% CI] | -0.59 pts [-0.98, -0.18] | -0.35 pts [-0.72, +0.03] |
| Network minus MC-4k, points [95% CI] | +5.01 pts [+3.83, +6.25] | +5.25 pts [+4.08, +6.48] |
| Training (core-hours) | 0.12 | 0.24 |
| Break-even calls vs MC-16k | 33,129 | 46,599 |

## Cost (one CPU thread)

- One pricing call: network 102 µs,
  MC 4,096 paths 9.2 ms,
  MC 16,384 paths 33.3 ms.
- Batch of 1,000: network 3.5 ms in total,
  MC 16,384 paths 33.25 s.
- Upfront cost of the network: 0.43
  core-hours (training data and training).
- It pays for itself after about 46,599 calls compared
  with 16,384-path Monte Carlo, or 169,551 compared with
  4,096 paths. Adding the held-out test set raises these to
  49,246 and
  179,184.

## Files

`accuracy.json`, `pnl.json`, `cost.json`, `setup.json` and five PNG charts for the reported
run; `runs.json` puts every training run side by side, and `runs/epochs_<cap>/` keeps each
run's own summaries. Regenerate
with `python -m hestonlab.surrogate_value`; the full outputs, model and cached data stay
in `outputs/surrogate/value/` (git-ignored).
