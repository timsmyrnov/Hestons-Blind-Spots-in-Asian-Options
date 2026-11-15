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
  individually. Geometry may be hidden or unavailable; missing research results
  are not replaced with invented daily fits.

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

Re-render only:

```bash
PYTHONPATH=src python3 -m hestonlab.manifold_movie --out outputs/weekly_cpp
# WebGL page only:
PYTHONPATH=src python3 -m hestonlab.manifold_movie --out outputs/weekly_cpp --no-video
```

Chrome export uses a temporary localhost receiver and an isolated temporary
browser profile. A restricted execution environment may require local process /
loopback permission. It does not send market data to an external service.

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
