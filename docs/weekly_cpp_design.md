# Weekly Heston/market animation without neural networks

Agreed design, 2026-11-15. This replaces the neural Heston surrogate and neural
market manifold. The four-year production run requires Tim's green light; setup,
compilation, input preparation, tests and small smoke runs are authorized now.

## What is being computed

- Four years (2022–2025), approximately 208 weekly snapshots. Keep daily market
  observations in the preceding 120-observation estimation window.
- Heston prices come directly from C++ Monte Carlo: common-factor futures,
  full-truncation Euler variance, log futures update, discrete arithmetic fixings,
  antithetic pairs, and common random numbers across candidate parameters.
- Preserve the existing 21-coordinate grid: three tenors (0.2, 0.3, 0.4 years)
  and seven log-moneyness coordinates. The normalized flat-forward Heston grid
  is an approximation to listed contracts, not an exact representation of their
  individual forward curves. Existing data eligibility and interpolation remain.
- Fit five Heston parameters directly to the selected market-surface point in
  RMS USD/barrel. The reported gap is point-to-Heston, not the minimum distance
  between the entire two manifolds, and a bounded local search is not a proof
  of a global minimum.
- Market surface: PCA coordinates plus a regularized quadratic graph in the
  orthogonal complement, selected against a PCA plane on a chronological
  validation portion of the trailing window. No neural networks. Future weeks
  and the current target never enter model estimation or hyperparameter choice.

## Where the speedup comes from

1. Compute weekly, rather than daily; retain the daily data for estimation.
2. Fit with 4,096 paths. Warm-start from the previous week, check alternate
   starts, and count every actual pricing call, including finite differences.
   Default budget: 1,200 pricing calls per weekly fit.
3. Draw an 11 x 11 directly priced parameter slice with 16,384 paths per point.
   Do not iteratively force thousands of vertices onto a common floor square.
   Parameter slices and displayed 3-D projections are not the full 5-D model.
4. Keep one central optimized gap. Remove the ten extra optimized gap labels;
   a distance to a drawn 2-D patch is not a full-model distance.
5. Independently reprice fitted points using four batches of 65,536 paths,
   refine suspicious fits, and spot-check step size, derivatives, surface
   interpolation and weekly versus daily sampling.
6. Cache numerical results with data, source and configuration fingerprints.
   Re-rendering cached geometry must not trigger pricing or refitting.

## Runtime target and approval boundary

The short local C++ prototype measured approximately 0.016 seconds per 21-price
vector at 4,096 paths, 0.245 seconds at 65,536, and 1 second at 262,144. These are
short benchmarks, not production guarantees. The production implementation must
benchmark itself and report a workload forecast before approval.

| Work | Planning allowance |
|---|---:|
| Weekly fits | 70 minutes |
| Surfaces and derivatives | 30 minutes |
| Independent checks and selective refinement | 30 minutes |
| Market models, I/O and rendering | 15 minutes |
| Reserve | 35 minutes |
| **Maximum production wall time** | **180 minutes** |

The deadline is a hard stop, not permission to claim unresolved results are
accurate. Save completed weeks, mark incomplete/unresolved weeks, and report a
partial run if the time budget expires. Do not quietly lower path counts or
fill missing numerical results with made-up observations. Preparation and smoke
tests are separate from the authorized production-run budget.

## Accuracy and interpretation

- Independent sampling checks measure Monte Carlo noise; they do not prove
  optimizer convergence or cover discretization, interpolation or model bias.
- Common random numbers stabilize comparisons; they do not make finite-path
  prices exact or globally smooth. Derivatives require a step-size check.
- Compare coarse and half-time-step pricing on selected weeks. Escalate path
  counts and refit when the fit changes materially. Report failures explicitly.
- Smooth in trading-day time (Gaussian standard deviation four trading days,
  the existing approximately two-week display convention), not in weekly array
  indices. Interpolated animation frames and smoothed gaps are presentation,
  not new daily estimates. Centered smoothing uses future snapshots and is
  retrospective, not a real-time signal.
- Check selected intermediate days against the weekly interpolation, especially
  during volatile periods. Weekly sampling can miss short-lived shocks.
- Limit the market patch to observed support; flag unsupported targets rather
  than extrapolating a convincing-looking sheet. Do not promise that five
  market factors or all five Heston directions are identifiable.
- Historical neural outputs and the licensed archive remain historical files;
  they must not be reused as current numerical results.

## Delivery

See `README.md` for the prepared commands and `docs/weekly_cpp_runbook.md` for
configuration, cache identity, validation flags, deadlines and evidence.
The production command is prepared but must not run until explicitly approved.
