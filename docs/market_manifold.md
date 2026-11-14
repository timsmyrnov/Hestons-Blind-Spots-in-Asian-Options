# Market manifold vs Heston manifold

Code: `src/hestonlab/market_manifold.py`. Tests: `tests/test_market_manifold.py`.

## What it measures

For each day: **the distance from that day's point on the market manifold to the
nearest point on the Heston manifold.** This is `manifold_distance` in the report.

1. Every day becomes one vector on a fixed grid.
2. One Heston manifold is trained on that grid from Monte Carlo prices.
3. A market manifold is learned from the training days.
4. Each day is projected onto the market manifold. That point is then projected
   onto the Heston manifold. The second distance is the result.

Also reported per day: distance from the raw day to the market manifold (how well
the market manifold fits that day) and from the raw day to the Heston manifold.

## Fixed grid

- Coordinates: years to last fixing × log(strike / average forward).
  Default tenors `0.2, 0.3, 0.4`; log-moneyness `-0.2 … 0.2` (7 points); 21 values.
- Value: out-of-the-money price / (discount × average forward). Puts below the
  forward, calls at or above.
- Unseasoned months only. Interpolation runs in Black-76 implied volatility (a
  smooth coordinate; time = time to the last fixing). Within a month: linear in
  log-moneyness on volatility. Across months: linear in time on total variance.
  **No extrapolation.** Days
  that don't cover the whole grid are dropped and counted.

Why volatility, not price: out-of-the-money prices peak at the forward, so straight
lines between listed strikes cut under the peak. The earlier price interpolation
understated at-the-money grid prices by up to 38¢ (RMS 7.3¢ over the grid; checked
against Monte Carlo Heston prices on real 2025 strike sets). In volatility: worst
2.9¢, RMS 0.5¢. Results made before this change (2026-11-13) carry that bias.

On this grid Heston gives **one** manifold: forward 1, discount 1, flat curve
within the averaging month. That flat-curve step is an approximation.

## Market manifold

Same injective architecture as the Heston manifold, with a chosen dimension
(`--latent-dim`, default 5). Each training day gets its own latent point,
learned together with the network. PCA of the training days sets the start.
Training days fill the middle half of the latent box, leaving room for test days
more extreme than any training day.

## Held-out months

`--test-months` (e.g. three `YYYY-MM` values) are removed before anything is
fitted: the market manifold, its scaling and its PCA start. `--embargo-days N`
also drops training days within N calendar days of a test day. The report
records `test_days_in_training` (always false) and the model stores its training
dates.

## Distance metric

Default `--weighting noise`: each grid value is divided by how precisely the
Heston manifold prices it (held-out Monte Carlo test draws), floored at 1e-4
(about one $0.01 tick at $80). `--weighting equal` uses plain normalized prices.

## 3-D pictures

Pictures are for intuition only. Distances are always computed in the full
21-D grid space; the exact per-day breakdown is the 3 × 7 difference grid.

Each day is shown as **two slices** through both manifolds:

| Slice | Directions varied | Roughly (Heston) |
|---|---|---|
| 1 | 1 and 2 | overall volatility level (v0/theta), skew (rho, sigma) |
| 2 | 3 and 4 | term structure (v0 vs theta, kappa), smile curvature (sigma) |

- Directions are ranked by how much they move prices: singular vectors of the
  Heston manifold's Jacobian at the day's nearest Heston point. Parameters stuck
  at a bound are left out (they cannot move).
- Direction 5 is skipped: it is the near-flat kappa–theta direction, and its
  sheet would collapse to a line.
- Parameter meanings are rough; each direction mixes parameters, and the actual
  loadings are printed with each picture.
- Each day has its own frame. Origin: the day's nearest Heston point. Up: the
  day's gap (so Heston is level there; that is what "nearest" means). Floor:
  Heston's two slice directions.
- Both sheets sit over the same floor square. Over each floor point: the Heston
  point and the market point that make exactly that price move (the market's own
  move closest to it, measured in prices).
- Each sheet point is found by Newton steps that make exactly its floor move with
  the least other price change. Heston stays inside its parameter box: a parameter
  that reaches its limit stops there and the others carry on (a real edge of the
  model). The market sheet keeps one smooth branch and ends 0.25 past its latent
  box; further out the network only extrapolates and its sheet would fold.
- Lost in a slice: 3 of 5 manifold directions, 18 of 21 price directions, the
  rest of each manifold, and which prices make up the gap.

### Movies

Movies show **several consecutive days, each with its own manifolds**:

- Market manifold for day d: learned only from the 120 trading days before d
  (`--window`), with a smaller network (width 32, depth 2) suited to that little data.
- Heston manifold for day d: the grid's Heston manifold in that day's dollars
  (discount × average forward per tenor, `grid_scale`). Units are RMS USD/bbl.

```bash
python -m hestonlab.manifold_movie --run outputs/market_manifold/archive \
  --root "$HESTONLAB_DATA_ROOT" \
  --dates 2021-11-19 2021-11-22 2021-11-23 2021-11-24 2021-11-26

# Every complete day in a range; --check first: per-day health, no page, no video.
python -m hestonlab.manifold_movie --run outputs/market_manifold/archive \
  --root "$HESTONLAB_DATA_ROOT" --start 2021-10-01 --end 2021-11-30 --check

# A long range in a fixed time (here two years in 20 s; see "Long movies").
python -m hestonlab.manifold_movie --run outputs/market_manifold/archive \
  --root "$HESTONLAB_DATA_ROOT" --start 2024-01-01 --end 2025-12-31 --seconds 20

# Four years in 26 s; the last 7 s zoom onto p and its tangent plane.
python -m hestonlab.manifold_movie --run outputs/market_manifold/archive \
  --root "$HESTONLAB_DATA_ROOT" --start 2022-01-01 --end 2025-12-31 --seconds 26 \
  --finale 7
```

Output: `movies/<first>_to_<last>_slice<k>.mp4` (1920×1080, 60 fps, H.264), the WebGL
page it is recorded from (`.html`) and a `.json` with each day's numbers.

- The movie holds on each day (2.2 s) and glides to the next (1 s); the sheets,
  lines and numbers morph smoothly. The camera orbits slowly and diagonally.
- Every day is drawn in its own frame (above), so the sheets stay in place and
  keep their size; only their heights change. Small labelled axes in USD/bbl. The vertical axis
  has its own scale; read values off its ticks.
- One pair of floor directions for the whole movie: the two directions in which
  the days' slices move prices most on average, taken on each day as Heston can
  make them. Their meaning never changes from day to day (a day's own strongest
  directions swap when Heston hits a parameter limit).
- One square for the whole movie: 3× the median gap, but no larger than what both
  models can cover on 9 days in 10. On the other days, the part a model cannot
  reach fades out with a soft edge; the square never shrinks.
- A day that lies past its own market model (its point sits on the edge of the
  model's latent box, like Nov 26, 2021) shows no blue sheet: it would be pure
  extrapolation. The day's gap line stays.
- Daily market models are cached in `movies/markets/`.
- Lines run straight down from the blue sheet to the purple one.
- Tiny labels *M_h* (Heston) and *M_m* (market) ride on a corner of each sheet, in
  Computer Modern math italic.
- `--finale N`: the last N seconds stop on the last day and zoom onto a point *p*
  of the Heston sheet. Its tangent plane *T_pM_h* fades in, with the two tangent
  vectors *v_1*, *v_2* along the floor directions (estimated from the drawn sheet,
  so the plane is tangent to what is on screen). This shows why the structures
  must be manifolds: at every point the sheet has a tangent plane, the local
  linear picture that nearest points, slice directions and gaps are built on.
- Lines start grey. At a quarter of the way they turn green → yellow → red by
  the gap itself: one fixed log scale for the whole movie, from the 5th to the 95th
  percentile of its gaps (a tiny scale bar next to the legend gives the numbers).
  Equal gaps, equal colours. Lines draw after the see-through market sheet so it
  never tints them, and each shows its number: the
  true distance from that blue point to the whole Heston manifold. The thick
  line is the day itself. On screen: date, "avg error" (mean of the line numbers)
  and a small legend. Font: Computer Modern (embedded from matplotlib).
- Recording: headless Chrome renders every frame deterministically in chunks of 120
  frames (a stalled chunk is retried), and ffmpeg (`imageio-ffmpeg`) encodes them.
  Needs Chrome (`CHROME_PATH` to override) and internet for three.js 0.160.0.
  Recording takes about 0.2 s per frame (5 days: 15 s of video, about 3 minutes). `--no-video` writes the page only; it plays in a browser
  (click to replay), and `?t=0.6` in the URL freezes it at 60%.

Example: [`docs/media/manifolds_2022-2025.mp4`](media/manifolds_2022-2025.mp4), Jan 2022 to
Dec 2025 in 26 s (slice 1, 2-week smoothing, gap axis zooms in at 4 s, tangent-plane
finale).

Long movies (`--seconds`): too many days per second to show each one, so keyframes
every 4 trading days, each a Gaussian average (sd 4 trading days) of the nearby
days' sheets, fades, lines and numbers. They play as a steady glide with no holds
(no easing, so nothing stutters). The date shows month and year; the average line
says "2-week smoothing". Sheets use a 41 × 41 grid to keep the page small.

Nov 19–26, 2021 (slice 1, each day's own directions, before fixed directions and
fading): the square was ±0.12 USD/bbl. Inside it Heston bends under 1¢ and the
market tilts up to 7¢, so both sheets are close to flat; the gap grows from 8¢ to
29¢. Nov 26 itself is 1.25 USD/bbl from its trailing market manifold: the crash is
unlike the 120 days before it.

## Per-option gap heatmap

Code: `src/hestonlab/gap_heatmap.py`. The most direct per-day number: for each of
the 21 grid options, market price minus Heston price (USD/bbl, signed).

- Heston parameters: the day's best Heston fit on the grid.
- Heston prices: Monte Carlo (262,144 paths) at those parameters on the options
  actually listed that day, so the neural surrogate's error does not enter.
- Market and Heston go through the same grid interpolation from the same listed
  options, so any interpolation error cancels.
- Noise: each cell's Monte Carlo standard error carried through the interpolation
  (0.3–1.2¢ in 2025). Grey numbers are within 2σ of it.
- Animated over one year with light smoothing; a strip below shows the day's RMS
  over the 21 options.

```bash
python -m hestonlab.gap_heatmap --run outputs/market_manifold/archive \
  --root "$HESTONLAB_DATA_ROOT" --year 2025 --seconds 10
```

2025 (243 days, before the interpolation fix; the heatmap itself is not affected by
it): median RMS 5.4¢. Heston prices near-the-money options too cheap (market 5–10¢
above, about 20¢ in April) and far out-of-the-money calls at 4.8 months too
expensive (−11¢ on average at 122%). 79% of option-days exceed 2σ of the noise.

## Running

```bash
# Synthetic only (tests the method; never market results):
python -m hestonlab.market_manifold synthetic --quick
python -m hestonlab.market_manifold synthetic --quick --blind-spot 0

# Real data (not yet run):
python -m hestonlab.market_manifold archive --root "$HESTONLAB_DATA_ROOT" \
  --test-months 2022-03 2023-11 2025-06 --embargo-days 5
```

Outputs go to `outputs/market_manifold/<source>/`: `report.json`, `days.csv`,
`market_model.pt`, and `heston/` (Heston grid model and its Monte Carlo data).

## Synthetic check, 2026-11-13

500 synthetic days, 3 held-out months, quick sizes. Median manifold distance
(noise units):

| Synthetic market | Training days | Held-out months |
|---|---:|---:|
| Pure Heston | 0.012 | 0.010–0.022 |
| Heston + short-tenor wing bump (0.003) | 0.20 | 0.059–0.36 |

The bump's size follows a cycle, so held-out months differ. The archive loader
is tested only on small synthetic tables; it has not yet been run on real data.
