# Can Heston follow the market's moves?

Code: `src/hestonlab/tangent_moves.py`. Tests: `tests/test_tangent_moves.py`.

## What is measured

For each day *t*, with the day's best Heston fit *p* on the Heston manifold *M_h*:

- **move**: change in the 21 market grid prices from day *t* to day *t+h*
  (USD/bbl per option, equal weights).
- **followed**: the part of the move inside the tangent plane *T_pM_h*. Heston can
  make it, to first order, by changing its five parameters.
- **left over**: the rest. Heston cannot make it at all.
- **share followed** = |followed|² / |move|². Pooled over days, big moves count more.

Two readings:

- **5 directions**: the full tangent plane (generous to Heston).
- **4 directions**: without the near-flat kappa–theta direction (strict). The
  surrogate's slope there is 45–57° off Monte Carlo, and moving along it takes very
  large parameter changes. The other four agree with Monte Carlo within 1–10°.

References, chosen with the same information as Heston's plane (nothing after
day *t*):

- **market's own plane**: the top principal directions of the market's moves over
  the previous 120 days (moves that ended by day *t*). This is the fair rival.
- **chance**: a random 5- (4-) dimensional plane catches 5/21 (4/21) of a move.

## Results, Jan 2021 – Dec 2025 (1,223 days)

| | 1-day moves | 5-day moves |
|---|---:|---:|
| Typical move (median, RMS per option) | 6.2¢ | 15.3¢ |
| Left over by Heston, typical day (5 / 4 directions) | 0.7¢ / 0.7¢ | 1.5¢ / 1.6¢ |
| **Heston follows** (4 directions; 5 in brackets) | **98.1%** (98.6%) | **99.2%** (99.3%) |
| **Market's own plane follows** (same days) | **99.1%** (99.4%) | **99.5%** (99.6%) |
| Typical day: Heston / market's own plane | 98.3% / 98.4% | 98.8% / 99.0% |
| Chance (4 / 5 directions) | 19% / 24% | 19% / 24% |

1,163 days with a 120-day market reference for 1-day moves (1,155 for 5-day).

By year (1-day moves, 4 directions, pooled):

| Year | Typical move | Heston follows | Market's own plane | Heston left over |
|---|---:|---:|---:|---:|
| 2021 | 7.5¢ | 99.7% | 99.6% | 0.8¢ |
| 2022 | 9.6¢ | 97.6% | 99.2% | 1.3¢ |
| 2023 | 7.0¢ | 94.2% | 96.6% | 0.7¢ |
| 2024 | 4.7¢ | 98.6% | 98.8% | 0.6¢ |
| 2025 | 4.4¢ | 97.0% | 98.7% | 0.6¢ |

Per day (1-day moves, 4 directions): 10% of days below 85% followed by Heston.

## Reading

- **Heston follows the market's day-to-day moves almost fully.** On a typical day
  it does as well as the market's own recent directions (98.3% vs 98.4%), using the
  same information.
- **It falls behind on big moves.** Pooled over days (big moves count more) the
  market's own plane follows 99.1% and Heston 98.1%; the gap is widest in 2022
  (97.6% vs 99.2%).
- **Heston's error is a persistent offset, not the moves.** The 4–5¢ pattern in the
  gap heatmap (market above Heston near the money, below it for far calls) stays
  while the market moves.
- **Part of the 0.7¢ daily leftover is not Heston's fault.** Settlement prices are
  rounded to 1¢. Rounding alone adds about 0.4¢ RMS to each daily price change
  (two roundings of σ = 1¢/√12), and 16 of 21 directions of that noise land in the
  left-over part. Interpolation adds more.
- **2022 has the largest leftover (1.3¢ a day).** 2023's lower pooled share comes
  from a few large moves Heston could not follow.

## Files

`outputs/market_manifold/archive/tangent/moves_2021-01-01_to_2025-12-31.json`
(summary) and `.csv` (one row per day and horizon).

```bash
python -m hestonlab.tangent_moves --run outputs/market_manifold/archive \
  --root "$HESTONLAB_DATA_ROOT" --start 2021-01-01 --end 2025-12-31
```
