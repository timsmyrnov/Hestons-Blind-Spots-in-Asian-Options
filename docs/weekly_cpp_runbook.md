# Weekly C++ runbook

The design agreement is in [weekly_cpp_design.md](weekly_cpp_design.md). The
production run has **not** been launched as part of implementation. Only tests,
synthetic export and isolated pilot checks are permitted before the green light.

## Prepare and launch

From the repository root:

```bash
python3 scripts/prepare_weekly.py --root HestonLab_Asian_Data
```

Preparation reads two existing archive inputs, preserves their hashes, builds
the 21-price daily panel with a history buffer before 2022, selects weekly
anchors, compiles C++, benchmarks the actual engine, and writes a workload
forecast. It also checks Python packages, Chrome and FFmpeg. Three.js 0.160.0
and its label renderer/license are downloaded once, SHA256 checked, and copied
alongside exported HTML. Rendering thereafter needs no internet connection.

Review `outputs/weekly_cpp/preflight.json` and `environment.json`. The first
local preparation found 209 weekly snapshots. Use the latest generated forecast,
not a fixed advertised speed. A forecast is not a guarantee; a parent watchdog
bounds production execution. If the forecast exceeds 180 minutes, the run can
still stop safely, but it should not be represented as likely to finish.

**Only after approval:**

```bash
python3 scripts/run_weekly.py
```

This launches a child process and writes progress to `run.log`. The default
180-minute budget covers worker startup, input identity checks, calculations
and rendering. Reserve the last ten minutes for rendering. The watchdog
terminates the worker's process group, including Chrome and FFmpeg, at the hard
deadline; termination/reaping can add a few seconds of cleanup. An exclusive
`run.lock` prevents concurrent writers. After a machine crash, verify that no
process remains before removing a stale lock.

Re-running that command resumes valid completed weeks, with a fresh invocation's
time budget. A stopped run is not silently extended. Partial results stay marked
partial. No existing archive, neural output directory or historical movie is
used as a destination.

## Run in two stages

The first review stage is the first `ceil(N/4)` weekly snapshots: **53 of 209**
for the current prepared archive. It computes those weeks and their within-stage
checks, exports a review video/report, and exits. It never automatically starts
the remaining 156 weeks.

```bash
python3 scripts/run_weekly.py --stage first-quarter
# Only after reviewing and approving the pilot:
python3 scripts/run_weekly.py --stage remaining
```

Both stages use the same numerical configuration and cache identity. The second
stage verifies the first-quarter checkpoints and restores their warm start,
aligned axes and refinement count before computing subsequent weeks. Missing or
changed first-quarter checkpoints block the remaining stage rather than silently
replacing reviewed results. The pilot report, HTML and MP4 are preserved under
`outputs/weekly_cpp/stages/first-quarter/` when the final outputs are generated.

The review should cover gap size versus Monte Carlo uncertainty, optimizer and
market-support flags, time-step/derivative/mesh checks, intermediate daily checks,
and whether the two-dimensional slices are readable. Slices are not the entire
five-dimensional model. Centered smoothing near the pilot's right edge is
provisional: it will change when later weeks become available, without changing
the stored unsmoothed prices or gaps. Numerical-method changes require a new
prepared identity; cosmetic renderer changes do not require new fits.

## Configuration

`configs/weekly_cpp.json` is copied to `outputs/weekly_cpp/config.json` by
preparation. Change the source configuration and prepare again before a new run.

| Setting | Default |
|---|---:|
| Range | 2022-01-01 through 2025-12-31 |
| Anchor | Last available complete day in each Friday-ending week |
| Prior history / chronological validation | 120 / 20 observations |
| Market latent rank | Up to 5 numerically resolved directions |
| Fit paths / actual-call cap / starts | 4,096 / 1,200 / 3 |
| Surface paths / vertices | 16,384 / 11×11 |
| Independent validation | 4 batches × 65,536 paths |
| Refinement paths / call cap | 16,384 / 180 |
| Refinement allowance | At most 24 weeks per complete computation |
| Detailed step/mesh/derivative checks | Every 13th weekly snapshot |
| Intermediate daily checks | Up to 12 high-move intervals |
| Maximum time step | 1 calendar day |
| Absolute diagnostic threshold | $0.02/barrel RMS |
| Relative fit/weekly diagnostic threshold | 15% of the measured gap |
| Derivative step-change threshold | 25% on displayed directions |
| Smoothing standard deviation | 4 trading days |
| Video | 26 seconds, 60 fps, 1920×1080 |
| Hard production budget / render reserve | 180 / 10 minutes |

These thresholds are explicit development criteria, not a claim of two-cent
accuracy in all market regimes. Sampling error, optimizer error, time-step bias,
market-model error and market-grid approximation are different quantities.

## Checks and failure handling

- Native covariance uses independent antithetic pairs. Cross-option covariance
  is included in the delta-method error of the RMS gap. Near zero gap, the RMS
  price-noise diagnostic is more informative than that linearized error.
- Compare the fitted finite-path prices against independent higher-path prices.
  If unstable or unconverged, use a higher-path refinement within its budget,
  compare candidates on common independent draws, and report with a fresh seed.
  If the refinement allowance is exhausted, flag remaining suspicious weeks.
- Detailed weeks compare the base and half time steps with sampling uncertainty,
  compare two finite-difference step sizes, and directly price triangle-edge
  midpoints to check the coarse display mesh. A failed check remains a flag;
  there is no silent expansion past the runtime budget.
- Select intermediate-day checks in the highest-move intervals. Fit those days
  using strictly prior history, then compare with weekly-gap interpolation.
  This is an unsmoothed spot check, not a proof that weekly and daily smoothed
  movies coincide. Failed or unconverged checks are not marked passed.
- Unsupported market targets, negative projected market prices, unresolved
  directions, noisy gaps, unstable prices and failed local searches are recorded
  individually. The stable display keeps model continuations visible with support
  warnings; missing research results are not replaced with invented daily fits.

`screened` means the implemented checks passed for that snapshot, not proven
accuracy or a global minimum. Detailed checks are sampled, not run on every week.
All reports retain `fully_validated: false` to avoid overstating these guarantees.

## Artifacts and caching

| File | Meaning |
|---|---|
| `panel.npz` | Daily normalized prices, dollar scales, dates and weekly indices |
| `preflight.json` | Inputs, hashes, native benchmark, workload forecast, identity |
| `environment.json` | Interpreter, package versions, browser and encoder |
| `weeks/<identity>/<date>.npz` | Parameters, validated prices/covariance, raw market point, direct surface samples and tangents |
| `weeks/<identity>/<date>.json` | Fit diagnostics, prior training dates, quality flags and NPZ hash |
| `progress.json` | Completed-week count and last completed date |
| `daily_checks.json` | Independent intermediate-day comparison results |
| `report.json` | Run status, weekly results, checks and rendering status |
| `execution.json` / `deadline.json` | Supervisor exit code and hard-timeout evidence |
| `animation.html` / `animation.mp4` | Presentation outputs |
| `animation_manifest.json` | Smoothing, interpolation and display interpretation |

A cache identity includes the panel hash, numerical configuration, numerical
Python/C++ source hashes, compiler version and flags. Source/configuration/input
changes invalidate the prepared run; prepare again. Each week is written to
an atomic NPZ followed by its JSON commit marker, and its checksum is verified
before reuse. Warm starts and axes are restored from the preceding cache.
Renderer changes do not require refitting numerical results.

Re-render only; the current video always goes to `outputs/animation.mp4`:

```bash
PYTHONPATH=src python3 -m hestonlab.manifold_movie --out outputs/weekly_cpp
# Browser version only, at outputs/animation.html:
PYTHONPATH=src python3 -m hestonlab.manifold_movie --out outputs/weekly_cpp --no-video
```

Superseded experimental videos are consolidated in `outputs/animation-history.zip`.
Preview renders belong in a temporary directory. Numerical caches and first-quarter
review tables remain under `outputs/weekly_cpp`; they are not video variants.

The renderer shows the actual cached Heston slice and the reconstructed market
polynomial, checking its prices against the cache. No Monte Carlo or Heston
calibration is run. The blue market centre is anchored at zero and its linear tilt
is removed from both sheets. A shared vertical reflection places purple Heston
above it. Purple moves relative to the market; both retain their residual curvature.
The central gap and every absolute paired vertical separation are preserved.
There is no curvature amplification or RMS-offset replacement.

The camera orbits at constant distance. Zoom operates through the z-axis: its
range widens during the first half and tightens during the second. Tick labels
update accordingly, while market zero stays at a fixed height in the scene.
Prefix/suffix bounds enclose every mesh vertex, including
between keyframes. The manifest records all ranges and presentation settings.

The footprint stays fixed on screen. Horizontal axes retain their slice-direction
names; their coordinates are fractions of the local section, as selected for the
wider view. Dollar widths are recorded per week. `--window fixed` uses one shared
dollar-coordinate window instead. `--raw-patches` retains the original diagnostic
patches, and `--fixed-camera` stops the orbit. `--seconds` and `--fps` control export.
If a local projection overlaps away from its center, its window is reduced until
the displayed sheets remain separated. Both models are evaluated at that smaller
window; no heights or central gaps are changed. Cropped weeks are recorded in the
manifest. This is a display restriction, not proof that the full manifolds are disjoint.

Sixteen local lines, including four near the corners, label paired full-price RMS differences, not new minimum-error
calibrations. The central line retains the validated central gap. All numerical
labels remain unchanged under display transforms. Two-week smoothing is display
only. Market-support and numerical accuracy flags remain in the research report
and manifest; neither appears as a label in the video. Both manifold names sit
outside the same corresponding corner of their sheets.

Error values are hidden for the first quarter of playback. At 6.5 seconds in the
26-second export, the line labels begin fading in and the headline gap appears.
The fade completes at 8.06 seconds. Lines are grey before the reveal; all vectors
then use the same fixed error-colour scale. Both sheets remain visible throughout,
with subtle blue/purple diagonal gradients. Surface opacity does not control the
number reveal.

Two translucent green tangent patches on the purple slice fade in from 13 to
14.56 seconds. Their green callouts report **bend**: the maximum absolute vertical
departure from the local central-difference tangent over each patch, in mUSD/bbl
(one thousandth of a dollar per barrel). This is a local slice statistic, not
intrinsic curvature of the full manifold or a market–Heston error. Patches span
eight grid intervals per side; local-window dollar widths vary between weeks.
Signed departures are interpolated before taking the absolute maximum. Camera
motion and vertical zoom do not affect the values, and no bending is amplified.

Chrome export uses a temporary localhost receiver and an isolated temporary
browser profile. A restricted execution environment may require local process /
loopback permission. It does not send market data to an external service.

## Heatmap

```bash
PYTHONPATH=src python3 -m hestonlab.weekly_heatmap
```

Writes `outputs/heatmap.mp4` (15 seconds, 1080p/60fps), `heatmap.png`,
`heatmap.pdf`, `heatmap.html`, `heatmap_data.csv`, and `heatmap_manifest.json`.
The video reveals all 209 weekly columns over 14 seconds, then holds the complete
view for one second. All text uses Computer Modern, including the date and browser
controls. Three panels retain all 21 canonical contract coordinates. Moneyness is
displayed as strike / average forward in percent: 100% means strike equals the
average forward. Each panel contains seven levels at one of three maturities.

Cells show signed Heston-minus-market errors in USD/bbl, without smoothing.
The default reference is the fitted market manifold, matching the cached gap;
the interactive view can switch to the original interpolated market grid using
the same Heston fit. Its hover includes both prices and their signed difference.
The video calls these references market estimates and market data, respectively.
One symmetric asinh color scale is fixed across dates, panels,
and reference modes, includes every value without clipping, and expands small
errors near zero. The CSV retains full precision and numerical flags.

Rendering verifies cache checksums and checks that each week's 21 residuals
reproduce its saved RMS gap. It does not run pricing or calibration. Use
`--no-video` for the figure and interactive exports, or `--seconds` / `--fps`
to change the video length and frame rate.

## Tests and retirement of the neural implementation

```bash
PYTHONPATH=src python3 -m pytest -q
PYTHONPATH=src python3 -m hestonlab.cpp_pricer
PYTHONPATH=src python3 -m hestonlab.weekly_pipeline smoke --video
```

Synthetic smoke output is labeled synthetic and deliberately uses tiny budgets;
accuracy flags are expected. Native tests include deterministic accrued payoffs,
a constant-variance Black limit, path-sharing/covariance invariants, independent
Python Monte Carlo agreement and input rejection. Workflow tests exercise
holdouts, exact-call budgets, cache corruption, temporal sampling and resumption.

Removed: neural surrogate generation/training, invertible neural manifold,
neural calibration, and their dependent old heatmap/tangent CLI implementations.
The preserved Python Heston engine and June baseline remain independent reference
implementations. Historical outputs, the licensed archive and checked-in past
video are preserved but never consumed by the new numerical workflow.
