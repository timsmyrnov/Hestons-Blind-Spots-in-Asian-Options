# HestonLab AO data workflow

The source package extends the previously working AO extractor. `parsers.py` and `acquisition.py` are its refactored implementation; the original script is preserved under `work/ao_acquisition/ao_data_extractor_pre_refactor.py`. The public `ao_data_extractor.py` is now a compatibility entry point to the package CLI. There is one maintained implementation.

Python 3.12+:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
export HESTONLAB_DATA_ROOT=/absolute/path/to/HestonLab_Asian_Data
python -m hestonlab.data --help
python -m hestonlab.data validate
python -m hestonlab.data coverage
```

No credentials are required to use the downloaded panel. For authorized acquisition only, set `DATABENTO_API_KEY` locally without adding it to files or version control. `backfill` defaults to no approved spend. Reuse existing receipts with `download --job-id ...`; completed historical requests must not be repurchased. `collect-underlyings` quotes missing CL/CSX work and reuses existing receipts. Only pass `--approve --max-cost ...` after a human authorizes that specific spend. Failed/resumable download is different from failed purchase. Do not delete receipts to retry.

Rebuild from originals, in order:

```python
from pathlib import Path
from hestonlab.data.acquisition import normalize, audit
root = Path('/absolute/path/to/HestonLab_Asian_Data')
normalize(root)  # 71m raw AO statistics messages; slower than using Parquet
#audit(root)     # optional dataset-wide degraded capture flags
```

```bash
python -m hestonlab.data underlyings
python -m hestonlab.data fixings
python -m hestonlab.data build
python -m hestonlab.data validate
python -m hestonlab.data export --start 2024-06-03 --end 2024-06-07 --out research_sample.parquet
python -m pytest -q
ruff check src tests
ruff format --check src tests
```

`fixings` requires `ao_observations.parquet`; run `build` once before it if building from a clean normalized AO archive. This first build intentionally returns zero research eligibility until futures/state inputs exist. Monthly underlying revision Parquet checkpoints support a resumed ingestion without paying again. Output writes are atomic per artifact; source originals are immutable. This is an explicit local batch workflow, not a scheduled service. Preserve the full sources folder and receipts to rebuild. A compact delivery bundle can load normalized data but cannot reproduce the raw ingestion unless originals are also copied.

```python
from hestonlab.data import load_ao_panel, build_ao_snapshot
panel = load_ao_panel('2024-06-03', '2024-06-07',
                      filters={'call_put':'C','eligible_only':True})
observations = panel['observations']
snapshot = build_ao_snapshot('2024-06-14',
                             contract_filter={'eligible_only':True})
ids = snapshot['coordinate_ids']
y = snapshot['price_vector']
```

Tables are deterministically sorted by date, expiration, side, strike, contract ID. Coordinate IDs include maturity and side. Missing coordinates are not imputed. Native coordinate populations change daily. Dense fixed-coordinate experiments need a declared universe, out-of-sample calendar selection rule and missingness policy agreed with Tim and Lorenzo. Fixed state variables include curve, accrued average, remaining schedule, discounting and time; only conditional on these does the five-Heston-parameter dimension statement apply.

A zero-row default export is a fail-closed signal when no records pass underlying/calendar/quality masks. Inspect `eligible_only=False` and the masks; do not replace nulls or override exclusions without a documented research decision. The loader screens the returned futures curve at the publication cutoff for each selected date/month. The raw retrospective revision archive remains separate for audit.

Source lineage, failed retrievals, observed coverage, regime changes and exact charge receipts are separate evidence artifacts. Licensed data must remain outside the Git review bundle. Existing repository MIT code licensing does not grant market-data redistribution rights. Verify each recipient's entitlement before sending proprietary records to teammates or publishing derived files.

`build_fixing_schedule(contract_id, observation_date)` returns the contract's schedule with only prices published by the stored cutoff; future and unpublished fixings stay null. The loader's `futures` table ranks final revisions at each selected date/month cutoff, and its `fixing_schedule` table contains structural dates/identities without realized future prices. Raw retrospective futures remain available separately for audit, not as unconstrained causal model inputs.

`hestonlab.duckdb` contains local views over the canonical Parquet files. If the data directory moves, run `from hestonlab.data.pipeline import create_catalog; create_catalog(new_root)` to rebuild the view paths. The Parquet loader itself is portable through `HESTONLAB_DATA_ROOT`.

Nine real-file integration tests explicitly skip if the full local archive is absent. This makes the code-only repository usable without redistributing licensed sources, while retaining visible skip reasons. Set `HESTONLAB_DATA_ROOT` to the preserved full archive to run those tests; the measured local test report uses the actual archive, not the skipped mode. Other tests use clearly labeled synthetic fixtures to verify failure behavior, deletion handling, roll and publication cutoffs.

## Actual underlying delivery and remaining blocker

Both approved statistics jobs are complete: 223,233 CL and 178,168 CSX date/contract records on 1,736 reference dates, November 29, 2019 through October 26, 2026. All 130 overlapping CL prices match CME's April 3, 2023 exchange file. All 166 monthly definition snapshots were downloaded. The combined quoted/receipted underlying acquisition is $4.702629 of existing credits, below the $6 approval. No repeat purchase is needed.

Expiry-day CL and CSX settlement messages lack the vendor final bit. The final-only fixing policy therefore leaves zero rows in the strict research-eligible subset. This is an explicit quality-policy restriction, not an absence of AO prices. Use `eligible_only=False` to inspect the downloaded native panel and masks. The updated clock/holiday-screened diagnostic reconciles 80 of 81 available CSX terminal prices within $0.00501/barrel. One stale-reference CL publication is quarantined, and two copied Good Friday prices are excluded as extra fixing dates. February 2020 remains unresolved; June 2020 lacks a terminal comparator. Raw originals and the original canonical outputs are preserved. The separate June/July 2024 pilot uses unseasoned August–October options: its required forward inputs carry final flags and no accrued fixing is required. See model_and_validation_defaults.md for the distinction between complete prototype inputs and broader validation.

The dataset supports descriptive settlement-surface work now. It is not yet an approved fully conditioned Heston calibration panel or a causal backtest. A further data purchase is not established as necessary; resolve the observed contract/flag differences first. Exact raw UTC nanosecond timestamps and source sequences are retained and used for publication ordering.


## June/July pilot and proposed model defaults

See [model and validation defaults](model_and_validation_defaults.md), [split manifest](split_manifest.json), and [input requirements](input_requirements.csv). The local package now provides `hestonlab.data.model_inputs.build`, `load_model_snapshot`, and `hestonlab.data.splits.build_pilot_split`. June 2024 supplies training/calibration observations; July is held out. Coordinates are selected using June only. All pricing defaults remain explicit assumptions, and a Heston pricer/calibrator is not included.

The local input pilot has 16,453 June and 18,254 July prototype-complete rows. The fixed 34-coordinate grid has 1,311 observed prices. Its required forward records carry final flags and no accrued fixing is required. The wider canonical strict research mask remains zero while terminal finality is unresolved. Current test evidence reports 39 passing tests against the preserved local archive; archive-dependent tests skip explicitly on a code-only checkout.

The proprietary pilot and full vendor archive are retained locally and excluded from this Git repository. The repository includes collection/assembly code, public CME sample records, aggregate coverage and validation evidence.
