# Heston-Blind-Spots-in-Asian-Options

Heston has five parameters, so for a fixed set of strikes and maturities, the Asian option prices it can produce form an at most 5-dim set in price space, with one coordinate per option. We use a neural network to fit a manifold to this set. We then measure how real prices deviate from it, and compare it with a manifold learned from real data. Each day's error is the distance between the real prices and the closest point on the fitted manifold.

## AO data workstream

The research instrument is CME WTI Average Price Options: clearing code **AO**, Globex root **AAO**. The local historical archive contains 5,221,594 option/day observations, including 5,153,872 positive final settlement marks, across the 2020–2026 capture window. These are exchange marks, not automatically executable quotes.

- [Data workflow and Python interface](docs/data/README.md)
- [Units, contract identity, state and quality masks](docs/data/ao_data_dictionary.md)
- [Actual public CME sample records](data/samples/ao_historical_sample.csv)
- [Experimental June-only Heston Monte Carlo calibration](docs/heston_monte_carlo.md)
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
