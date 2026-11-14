# Heston-Blind-Spots-in-Asian-Options

Heston has five parameters, so for a fixed set of strikes and maturities, the Asian option prices it can produce form an at most 5-dim set in price space, with one coordinate per option. We use a neural network to fit a manifold to this set. We then measure how real prices deviate from it, and compare it with a manifold learned from real data. Each day's error is the distance between the real prices and the closest point on the fitted manifold.

## AO data workstream

The research instrument is CME WTI Average Price Options: clearing code **AO**, Globex root **AAO**. The local historical archive contains 5,221,594 option/day observations, including 5,153,872 positive final settlement marks, across the 2020–2026 capture window. These are exchange marks, not automatically executable quotes.

- [Data workflow and Python interface](docs/data/README.md)
- [Units, contract identity, state and quality masks](docs/data/ao_data_dictionary.md)
- [Actual public CME sample records](data/samples/ao_historical_sample.csv)
- [Experimental June-only Heston Monte Carlo calibration](docs/heston_monte_carlo.md)
- [Manifold-constrained neural pricing and parameter fitting](docs/manifold.md)
- [Market manifold vs Heston manifold](docs/market_manifold.md)
- [Friday readiness brief](docs/data/HestonLab_Friday_October_30_AO_Data_Readiness.pdf)

```bash
python -m pip install -e '.[dev]'
export HESTONLAB_DATA_ROOT=/absolute/path/to/HestonLab_Asian_Data
python -m hestonlab.data --help
python -m pytest -q
```

```python
from hestonlab.data import load_ao_panel, build_ao_snapshot
panel = load_ao_panel('2024-06-03', '2024-06-07')
snapshot = build_ao_snapshot('2024-06-14')
```

Full licensed raw and Parquet archives are kept outside Git. Ben maintains the local archive and provenance package. The checked-in sample uses public CME source documents and is not the full multiyear dataset. Real-file integration tests require that archive and explicitly skip without it. No missing or zero prices are imputed into research vectors.

## Learned Heston manifold

The default neural architecture embeds the five Heston parameters into price
space through smooth invertible coupling layers. Its image is an embedded
five-dimensional manifold in the parameter-domain interior (with corners at the
box boundaries). This construction guarantees the learned geometry, not accuracy
against Heston or the market. Independent Monte Carlo benchmarks measure pricing
fidelity; numerical inverse and Jacobian checks report conditioning.

```bash
python -m pip install -e '.[dev,surrogate]'
python -m hestonlab.surrogate_pipeline synthetic --quick
python -m hestonlab.surrogate_pipeline june --root "$HESTONLAB_DATA_ROOT" --fit
```

The June command trains one manifold per date, fits each date on its own, and
independently reprices the fit with Monte Carlo. Residuals are weighted by each
contract's pricing error. A Heston control shows how far Heston's own prices
land, so market gaps can be compared against it. See the manifold guide for
pooled fitting, weighting, the mathematical guarantee and its limits. The legacy
MLP is available with `--architecture mlp`; it has no manifold guarantee.

## Market manifold vs Heston manifold

Every day is put on one fixed (time × moneyness) grid. One Heston manifold and
one market manifold are learned on that grid, with chosen months held out. Per
day, the result is the distance between the two manifolds. See
[docs/market_manifold.md](docs/market_manifold.md).

```bash
python -m hestonlab.market_manifold synthetic --quick
python -m hestonlab.market_manifold archive --root "$HESTONLAB_DATA_ROOT" \
  --test-months 2022-03 2023-11 2025-06
```
