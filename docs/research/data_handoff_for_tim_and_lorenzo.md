# Data handoff for Tim and Lorenzo

## Tim: the June root now works

The full existing archive and the new local June-only package both produce exactly the same 19 `CalibrationSnapshot` objects: 613 observed prices on the June-selected 34-coordinate grid. Use the root layout and command in `run_june_calibration.md`. Contract prices and errors are USD/barrel, not contract dollars; the multiplier is 1,000 only for contract-dollar reporting. No values were fabricated or normalized.

Each contract supplies ID, strictly increasing ACT/365F remaining fixing times, a positive final forward for every fixing, total fixing count, zero accrued sum, strike, side and an external discount factor. August–October options are unseasoned during June. The common-factor stochastic paths and payoff are entirely Tim's existing implementation. Source quality is retained separately: all selected marks carry actual/final flags, mostly zero trading volume, so they are not executable quotes.

The new June-only package contains the needed root-level fixing file, contract master subset, rate source subset, exact training grid, and June observation quality/source rows. Its manifest pins parent hashes and exact slices. It is a private local package, not the full historical archive. No transfer or upload was performed. Raw provenance remains in Ben's authoritative archive.

Review decisions for Tim: local-search budget versus convergence; multiple starts/profiles; Monte Carlo and discretization accuracy; Feller diagnostic; pooled June/daily-v0-reset interpretation; flat-SOFR and payment-date sensitivity. These were not changed to improve the fit.

## Lorenzo: interface, not an architecture decision

The grid is 34 fixed IDs, with a **19 × 34** June observed-price matrix and boolean mask: 613 present, 33 absent. No missing price is zero-filled. July's **22 × 34** grid has 698 observed cells; July labels were not priced or used for setting selection/tuning. Coordinate selection uses June only.

Use contract features (strike, side, maturity, currency/unit), observation masks, and the conditioning state (required CL curve values, time/fixing schedule, total/remaining counts, accrued sum, discount factor). Tim's pricer takes a known five-parameter vector and returns conditional prices and MC sampling SE. Synthetic training examples must be generated from independently specified parameter draws with that tested pricer; the one fitted June vector is not a comprehensive supervised training set. No network or parameter-draw distribution was chosen here.

Baseline uses raw USD/barrel loss with equal observation weights. No scaler was trained. Any normalization must be fixed using training material only, saved and inverted consistently. Benchmark surrogate error against direct Monte Carlo separately from market/pricer residuals; use independent random streams and report MC uncertainty. Do not label surrogate approximation error as Heston model failure.

## July: two later experiments

A: Freeze approved June parameters/settings and score July with July's observed market inputs. B: Obtain the closest permitted July fit/projection under its fixed market inputs. Define separate functions/reports after team approval; the June-only calibrator intentionally rejects July dates. Poor transfer is not proof of prices outside the model family. Data schemas and forward/fixing coverage were checked for compatibility only. No July score, performance plot, surrogate training or financial backtest was run.
