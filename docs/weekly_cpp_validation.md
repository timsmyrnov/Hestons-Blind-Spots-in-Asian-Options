# Implementation validation — 2026-11-15

The four-year production computation has **not** started. The prepared numerical
identity was checked after the final source changes. There is no production
`report.json` or `weeks/` directory under `outputs/weekly_cpp/`.

## Ready-to-run inputs and engine

- Complete local archive preparation: **209 weekly snapshots**, from
  2022-01-07 through 2025-12-31, with prior daily history included.
- Current workload forecast: **161.0 minutes**, including the
  planning margin and rendering reserve; hard production budget: 180 minutes.
  This is a forecast, not a guarantee of complete validated output.
- C++17 engine compiles with Apple Clang, through a small ctypes C ABI.
- Offline Three.js assets are downloaded and checksum-verified; Chrome and FFmpeg
  are available. No neural or Torch imports remain in the active source tree.
- A wheel built successfully, includes `native/heston.cpp`, and contains no
  removed neural modules.

## Verification

- Full available test suite: **66 passed, 15 skipped**. Skips are optional or
  archive-dependent tests. Local test execution used the existing supplemental
  data packages without replacing the runtime NumPy/SciPy environment.
- Ruff checks and `git diff --check` pass.
- Native prices checked against an independent Python implementation, the
  constant-variance Black limit, exact accrued payoffs and shared-path covariance.
- Tests also cover training-window isolation, actual pricing-call caps, the hard
  parent watchdog, cache corruption/resumption, and trading-day smoothing.
- Final synthetic end-to-end run: **2 snapshots**, status
  `computed_with_flags`, video export `complete`. Tiny synthetic path
  counts intentionally produce accuracy flags; the video is visibly labeled
  synthetic. HTML and MP4 are in `outputs/weekly_cpp_smoke/`.

## Isolated real-data pilots

Two separate weekly calculations were run outside the production cache using
production path counts, including higher-path repricing and detailed checks.
They each took about 24 seconds. These are spot checks, not a four-year run or a
runtime guarantee.

- 2022-01-07 passed the implemented checks.
- 2022-03-04 remained flagged for a market point outside the trailing model's
  supported interior and Monte Carlo noise above the configured two-cent RMS
  threshold. The pipeline preserves these flags; it does not call that week
  accurate or draw an extrapolated market sheet.

Detailed local evidence: `outputs/weekly_cpp_pilot/`. The complete forecast and
input/source hashes are in `outputs/weekly_cpp/preflight.json`; environment
versions are in `outputs/weekly_cpp/environment.json`.

## Approval handoff

Read [the design](weekly_cpp_design.md) and [the runbook](weekly_cpp_runbook.md).
The only remaining action for the production run is explicit approval, followed by:

```bash
python3 scripts/run_weekly.py
```

This command starts the bounded, resumable four-year computation and exports
its animation. Re-rendering saved geometry is a separate command and does not
refit or resimulate any week.
