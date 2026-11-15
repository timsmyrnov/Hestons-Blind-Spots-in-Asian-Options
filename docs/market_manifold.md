# Non-neural market and Heston surfaces

Implementation: `market_grid.py`, `market_manifold.py`, `cpp_pricer.py`,
`weekly_pipeline.py`, `weekly_geometry.py`, and `manifold_movie.py`.
See the [design](weekly_cpp_design.md) and [runbook](weekly_cpp_runbook.md).

## Fixed price grid

Preserve the existing grid: time to last fixing 0.2, 0.3, 0.4 years, with
log(strike / average forward) −0.2, −0.1, −0.05, 0, 0.05, 0.1, 0.2. Use puts
below the forward and calls at or above it, normalized by discount × forward.
Interpolate within each averaging month in Black implied volatility; interpolate
across months in total variance. Do not extrapolate. Drop incomplete days.

The normalized Heston grid uses forward and discount 1, 21 equally spaced
fixings across 29 calendar days. Applying each day's forward/discount scale is
an approximation to the varying listed-contract curves and calendars. A small
numerical error does not eliminate this convention error.

## Market model

Each weekly date uses only the preceding 120 complete daily observations. PCA
provides up to five resolved coordinates. Fit a regularized quadratic graph
from these coordinates into their orthogonal complement. Its explicit PCA
coordinates make the graph injective in its parameter domain, even though the
fit is statistical and does not establish a true market-generating manifold.

Select ridge strength on the last 20 prior observations, using the preceding
100 for candidate estimation. Keep the PCA plane unless quadratic reconstruction
improves by at least 5%. Then refit the selected model on all 120 prior days.
This validation measures reconstruction with the same PCA encoder; it does not
certify a global nonlinear projection. Record the chosen rank and errors.

Market projection uses bounded least squares. Support is restricted by observed
coordinate ranges and radius; a boundary or unsupported target is flagged and
its market sheet hidden. This is a support screen, not proof of dense data
coverage in every part of a five-dimensional domain. Quadratic graphs are not
arbitrage-constrained; negative projected prices are flagged and hidden.

## Heston model and the measured gap

C++ directly prices each candidate parameter vector. Calibrate locally in a
normalized box (logs for the four positive parameters), with common random
numbers, bounded finite differences, several starts, and prior-week warm starts.
Every native price call counts against the budget, including Jacobian calls.

The measured gap is the RMS USD/barrel distance from the weekly observation's
projected market point to its locally fitted Heston point. It is neither a
certified global nearest point nor the minimum separation between the entirety
of two manifolds. Also export the raw market residual and all 21 price errors.

## Display

Use two resolved free Heston parameter directions to draw an 11×11 direct-Monte-
Carlo patch. Balance their first-order price displacements and shrink the patch
inside the parameter box. Align successive axes to prevent arbitrary SVD sign
flips. Market patch directions approximate those same local price moves, without
expensive iterative equal-floor construction. There is one central optimized
gap; there are no decorative lines claiming additional global optimizations.

The Heston display uses common-random-number price differences relative to the
patch center, anchored at the independently repriced central point. This keeps
finite-sample offsets from separating the central gap line from its patch.
Raw directly simulated vertices remain in the numerical cache.

The sheets and gap are projected from 21-D to 3-D, so a line's geometric screen
length is not generally its full-dimensional numerical distance. The gap axis
uses its own visual scale. Weekly snapshots are smoothed in trading-day time
(Gaussian standard deviation four trading days); missing days do not silently
compress the timeline. Interpolation and centered smoothing are presentation
only and must not be used as causal daily signals.
