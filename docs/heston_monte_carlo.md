# Experimental June Heston Monte Carlo calibration

This first implementation fits the five Heston parameters to observed AO
settlements using Monte Carlo Asian prices. June 2024 is the calibration month;
July is held out and its option files are never opened by this workflow. This
page describes the independent Python Monte Carlo baseline. The current
[weekly C++ workflow](weekly_cpp_design.md) uses direct simulation and a
non-neural market surface with independent numerical checks. No market parameter estimates
can be produced from the code-only checkout: the 50-row public sample does not
include June curves, discount factors or fixing schedules.

The implementation follows the existing proposed common-factor futures model
in [model defaults](data/model_and_validation_defaults.md). All CL maturities
share the same proportional shock, initialized from their individual observed
forward prices. Variance uses full-truncation Euler and futures use a log update
with zero futures drift. See [Lord, Koekkoek and van Dijk](https://repub.eur.nl/pub/7738/)
for the variance scheme. Simulation lands exactly on each discrete fixing date,
with intermediate steps no longer than the configured maximum. Time is ACT/365F.
The arithmetic average includes the known fixing sum. Discounting is supplied
by the data handoff; output prices and fitting errors are USD/barrel.

The loader consumes `pilot_june_july_2024/train_fixed_coordinate_observations.parquet`
and June rows of `ao_model_fixing_inputs.parquet` under the archive root. It
requires prototype-complete unseasoned options, final positive forwards,
consistent schedules, and the existing terminal diagnostic. Missing labels
are never imputed. July rows mistakenly placed in the training file cause an
error. The canonical research-eligibility masks remain unchanged.

With the full archive installed, prepare the existing handoff if needed:

```python
from hestonlab.data.model_inputs import build
from hestonlab.data.splits import build_pilot_split
build(root)
build_pilot_split(root)
```

Then run from the repository with the project environment activated:

```powershell
python -m hestonlab.calibration --root 'D:\HestonLab_Asian_Data' --out outputs/heston_june.json --paths 4096 --max-evaluations 100
python -m pytest tests/test_heston.py -q
```

The starter optimizer is a bounded local coordinate search, using log scales
for positive parameters and a linear scale for rho. It minimizes equal-weight
squared observed-price errors with common random numbers across candidates.
It fits one shared June vector and resets v0 at each daily snapshot, an explicit
benchmark assumption rather than a sequential volatility filter. Defaults and
search bounds are development choices. The result records parameters, bounds,
simulation seed, initial and fitted RMSE, independent-seed RMSE, antithetic-pair
sampling errors, search termination and the Feller margin. Feller is diagnostic,
not imposed. Sparse observed coordinates alone enter the loss.

These outputs remain experimental and `fully_validated=false`. Monte Carlo
standard errors do not cover discretization bias. Before interpreting fitted
parameters, increase path counts, halve maximum step size, compare independent
seeds and starting points, and check identifiability and sensitivity to the
fixed pricing conventions. A local minimum or a small price error does not
establish unique parameters. The tests check a constant-variance Black-price
limit, arithmetic call/put parity, common-factor consistency, fully accrued
payoffs, input failures, synthetic calibration improvement and month separation.
Synthetic fixtures are tests only; they are never presented as market results.

The weekly C++ workflow has its own native engine, independently checked against
this Python implementation. It prices the normalized fixed grid directly; it
does not generate neural training pairs. See [the runbook](weekly_cpp_runbook.md).
