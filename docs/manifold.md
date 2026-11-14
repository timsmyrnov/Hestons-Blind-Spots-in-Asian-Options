# Heston price manifold and parameter fitting

The default learned model is now an **embedded five-dimensional manifold by
construction**. This is a property of the learned map. Its fidelity to Heston
prices, and Heston's fidelity to market prices, are separate measured questions.
The legacy unconstrained MLP remains available with `--architecture mlp`.

## Construction and guarantee

For one fixed observation date and ordered contract set of size N >= 5, let
`u` be the five Heston parameters scaled into the unit box. The first four use
logarithmic scaling; rho uses linear scaling. These are the original Heston
coordinates, not unidentified latent variables. Define

```
E(u) = (u - 1/2, 0, ..., 0)                 # N-dimensional vector
F(u) = mean + diag(std) [A G(E(u)) + offset] # N prices, USD/barrel
```

`G` is a composition of alternating affine coupling layers. Each splits its
input into fixed and transformed coordinates and applies

```
y_fixed = x_fixed
y_changed = x_changed * exp(s(x_fixed)) + t(x_fixed)
```

The functions `s` and `t` use smooth tanh networks. Log scales are bounded by
`1.5 * tanh(raw_scale)`; inverse layers subtract the same shift and multiply by
`exp(-s)`. No clipping, sorting, ReLU, dimension-reducing output layer, or
postprocessing of prices is used.

For finite weights in exact arithmetic, every coupling is a global smooth
diffeomorphism. `A` is a fixed nonsingular matrix and every output standard
deviation is strictly positive. Therefore the ambient transformation is a
diffeomorphism. Its composition with the five-plane embedding `E` is injective,
has Jacobian rank five everywhere, and has a continuous smooth inverse on its
image. This establishes an **embedded smooth 5-manifold for the open parameter
box**, not merely a numerical observation of rank at sampled points. The image
of the closed box is a manifold with corners, not a boundary-free manifold.

The inverse chart is obtained by undoing output standardization, `A`, and the
couplings, then taking the first five latent coordinates plus 1/2. On the image,
the other N-5 latent coordinates vanish. **Inverting an off-manifold price
vector and dropping coordinates is not Euclidean nearest-point projection.**
Fitting instead optimizes the original Heston coordinates using the price metric.

Initialization fits a linear map using training rows only. SVD extends its five
columns to an invertible ambient map, with an initial singular-value floor of
0.001 in standardized coordinates. The floor is initialization, not an assertion
that Heston's five directions are identifiable or well conditioned. Flow
training can compress these directions. For N < 5 this architecture rejects
the contract set rather than claiming an impossible five-dimensional embedding.

This construction is in the family of injective flows described by
[Puthawala et al., ICML 2022](https://proceedings.mlr.press/v162/puthawala22a.html).
It is a supervised price-map implementation, not a density estimator or a
reproduction of that paper's experiments.

## Why geometry is part of the research

The manifold represents price combinations available by changing Heston
parameters, conditional on fixed curves, schedules, discounting, contracts and
valuation date. A single evaluation returns one price vector; the image of the
parameter domain is the manifold. Tangent spaces describe local parameter-driven
price movements. Their orthogonal complements describe departures from those
movements in the selected price metric. The guarantee gives these geometric
calculations a defined object; pricing-error statistics alone do not establish it.

The hypothesis of a five-dimensional embedded approximation must be checked
against the simulator. Genuine exact-price degeneracies, topology differing from
a parameter box, or a lower-dimensional image cannot be made faithful merely by
imposing an embedding. The code does not assert that the true Heston price image
has the same geometry. No architecture here guarantees economic inequalities,
positive predictions, unique nearest points, or a global calibration optimum.
Numerical conditioning can still be poor despite the exact-arithmetic guarantee.

## Training and diagnostics

Train directly on parameter/Monte Carlo price pairs, not on the old network's
predictions. No market prices are used to train the manifold. The existing Sobol
parameter bounds and simulation conventions remain in use. Validation rows drive
early stopping; independent test draws are used for evaluation. A linear
initialization and output scaling use only the fitting portion of training data.

```bash
# With the project and its surrogate extra installed:
python -m hestonlab.surrogate_pipeline synthetic --quick \
  --out outputs/manifold/synthetic-quick

# Real June contract sets require the private archive:
python -m hestonlab.surrogate_pipeline june --root "$HESTONLAB_DATA_ROOT" \
  --out outputs/manifold/june
```

`--quick` is a development smoke run, not research-quality accuracy certification.
For source-tree execution without installation, prefix commands with
`PYTHONPATH=src`. Optional dependencies are in the `surrogate` project extra.

Each run writes `model.pt`, `report.json`, parameter/price datasets, Jacobian
arrays and plots. Reports include:

- The architecture and scope of its structural guarantee.
- Numerical inverse recovery and rank/conditioning checks, including box corners.
- Held-out approximation errors and independent two-stream Monte Carlo benchmarks.
- Monte Carlo repricing along the network's flattest direction.

Checkpoints use format 2 and record architecture explicitly. Format-1 MLP
checkpoints still load as MLPs; they do not acquire a manifold guarantee.
Manifold fitting rejects those checkpoints. Changing the architecture requires
retraining; cached Monte Carlo data may be reused when its contract/config keys
match. Use distinct output directories for distinct experiments.

## Fitting after manifoldization

```bash
# Train/diagnose all June manifolds, then fit each date on its own:
python -m hestonlab.surrogate_pipeline june --root "$HESTONLAB_DATA_ROOT" \
  --out outputs/manifold/june --fit

# Or fit already trained manifolds without retraining:
python -m hestonlab.manifold_calibration --root "$HESTONLAB_DATA_ROOT" \
  --models outputs/manifold/june --out outputs/manifold/june/fit-new.json

# The old pooled fit: one shared vector, v0 included, for all dates:
python -m hestonlab.manifold_calibration --root "$HESTONLAB_DATA_ROOT" \
  --models outputs/manifold/june --pooled \
  --out outputs/manifold/june/fit-pooled.json
```

There is one trained manifold per date and fixed ordered contract set. Models
must match the snapshot's complete contract fingerprint and coordinate order.

**Per date (default).** Each date gets its own five parameters: the nearest
point on that date's manifold. `fit.json` has a `summary` row per date and the
full report under `dates`. `--pooled` instead fits one shared vector, v0
included, across all dates. Different dates' manifolds have different
conditioning and must not be pooled as one unconditioned price cloud.

**Weighting (default `noise`).** Each residual is divided by that contract's
pricing error scale: `sqrt(surrogate test MSE + mean label MC variance)`, from
the held-out test draws, floored at the $0.01 tick. Training stores it in
`model.pt` as `price_error_scale_usd_per_bbl`. Distances are then in noise
units, and cheap contracts are no longer swamped by expensive ones. The scale is
a whole-box average, not local to the fit. `--weighting usd` restores equal
USD weights. Older checkpoints without scales need `usd` or retraining.

**Heston control.** After the market fit, Heston's own prices at the fitted
parameters (independent MC) are fitted exactly the same way. Their distance,
`heston_control.weighted_distance`, is pure approximation error: the zero
level. `market_to_control_ratio` compares the market against it. A ratio near
or below 1 means the market gap is within approximation error. The control also
starts from the market fit, so it is never worse than the surrogate's error there.

Bounded least squares uses automatic derivatives and eight deterministic starts
by default. Reports preserve every start's termination and error; global optimality
is not certified. Numerical geometry is checked before optimization and again
at the result. Training stream metadata is mandatory. Two independent Monte Carlo
streams then reprice the fitted parameters; neither uses the training stream.
July market labels are never used by this workflow.

`fit.json` separates:

- Market distance in the chosen weighting, and market-to-manifold USD RMSE.
- Heston control distance and the market-to-control ratio.
- Market-to-independent-Monte-Carlo RMSE at the fitted parameters.
- Manifold-to-independent-Monte-Carlo approximation RMSE and stream disagreement.
- Local tangent/normal residual decomposition in the chosen weighting.
- Active parameter boundaries, conditioning, optimizer status and model hashes.

The local normal residual is not another global projection or an automatic
diagnosis of Heston failure. At a boundary, the feasible tangent cone differs
from the full tangent space, and a nearest-point residual can have a tangent
component. Raw MC verification errors include sampling noise; time-discretization
bias requires separate resolution checks. No application-specific acceptance
threshold is assumed, and fits remain `fully_validated=false`.

The separate manifold learned from real market data and the final comparison
between market and Heston geometries remain subsequent research work.

## Local implementation validation, 2026-11-11

Update 2026-11-13: after per-date fitting, noise weighting and the Heston
control were added, the suite completed with **80 passed, 15 skipped**. A quick
smoke run on one real June date (2024-06-14, 32 contracts) gave market distance
0.36 vs Heston control 0.60 (ratio 0.60). At quick-mode accuracy the market gap
is within approximation error. The rest of this section is the original 11-11 record.

The local suite completed with **77 passed, 15 skipped** (private-archive checks).
Ruff and `git diff --check` passed. Geometry tests exercise nonidentity couplings
after an optimizer update, ambient inverse recovery, analytic versus numerical
derivatives, degenerate linear initialization, checkpoint compatibility, known
parameter recovery, boundary residuals, and the combined training/fitting CLI.

A saved synthetic quick run used 27 contracts, 512 training draws, 256 test draws,
and 300 training epochs. Results are local ignored artifacts under
`outputs/manifold/synthetic-quick/`:

| Check | Observed result |
|---|---:|
| Minimum sampled numerical rank | 5 |
| Maximum inverse error in 259 diagnostic points | 7.77e-14 |
| Test-label pricing RMSE | 0.1866 USD/barrel |
| Independent MC noise-corrected RMSE, 16 parameter points | 0.2538 USD/barrel |

`synthetic-fit.json` fits a separate synthetic target generated with 65,536 paths
and seed 712 at `(v0, kappa, theta, sigma, rho) = (0.16, 2, 0.12, 0.5, -0.5)`.
Eight fitting starts reduced initial RMSE from 1.9784 to 0.0960 USD/barrel;
independent MC repricing of the fitted parameters gave market-target RMSE 0.1312
USD/barrel and surrogate-versus-MC RMSE 0.1803 USD/barrel. The fit reached the rho
lower bound and did **not** reliably recover the generating parameters. These
are synthetic development results, not market findings or accuracy acceptance.

Training and testing used the available Python 3.13.0, PyTorch 2.8.0, NumPy 2.3.5,
SciPy 1.15.2 and pandas 2.2.3 environment, rather than the complete pinned project
environment. Missing data-test dependencies were installed under ignored
`outputs/test-deps` and appended as a fallback import path. The full suite command
for that local environment was:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python3 - <<'PY'
import sys
sys.path.append('outputs/test-deps')
import pytest
raise SystemExit(pytest.main(['-q', '--disable-warnings']))
PY
```

No private June archive was configured, so real-market training and fitting were
not run. A run in the pinned project environment remains part of reproducibility
validation.
