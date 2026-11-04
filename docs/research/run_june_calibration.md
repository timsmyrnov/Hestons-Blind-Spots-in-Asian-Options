# Reproduce the June calibration

Use the `ao-june-integration` branch based on main commit `6f76fcf619aa0137028480c6e1b057336db1be20`. Install the pinned dependencies with `python -m pip install -e ".[dev]"` from the checkout. Tim's model/optimizer files are unmodified. Python 3.12+ and the pinned project dependencies are required. The wrapper is `hestonlab.june_workflow`; the unchanged original entry point remains `hestonlab.calibration`.

## Data root

Set `HESTONLAB_DATA_ROOT` to the complete private archive or the new **June_Private_Data_Handoff.zip** extracted root. Do not use the older compact pilot ZIP as that root. Required files:

- `pilot_june_july_2024/train_fixed_coordinate_observations.parquet`
- `ao_model_fixing_inputs.parquet`
- `ao_contracts.parquet`
- `sofr_history.parquet`
- `ao_observations.parquet` (June source-quality enrichment for the wrapper)

The new package uses exact June slicing and was tested for snapshot equality against the full archive. Manifest hashes describe its intentionally reduced tables. The 50-row public CSV is insufficient. Do not run archive builders merely to repair a path: they write derived data, and this experiment did not overwrite the historical archive.

## One command

From the matching repository environment:

```bash
export HESTONLAB_DATA_ROOT=/path/to/private_june_root
python -m hestonlab.june_workflow --out outputs/june-2024-new-run --paths 4096 --max-evaluations 100 --seed 2024 --max-step-days 1 --sensitivity
```

In PowerShell set `$env:HESTONLAB_DATA_ROOT = 'D:\path\to\private_june_root'` and run the same Python command. In a genuine authenticated Git checkout `--source-commit` can be omitted and HEAD is recorded automatically. An exported source snapshot requires the verified full commit SHA via `--source-commit`. Use a fresh output directory; the wrapper rejects archive-contained outputs and nonempty existing experiment directories.

The original command is:

```bash
python -m hestonlab.calibration --root /path/to/private_june_root --out outputs/june-original/calibration.json --paths 4096 --max-evaluations 100 --seed 2024 --max-step-days 1
```

## Expected outputs and checks

`calibration.json`, `manifest.json`, `actual_vs_model.csv`, `residual_summary.json`, grouped error CSVs and optional fixed-parameter sensitivity CSVs. Redirect console output to a log if desired. The manifest records source/model hashes, input hashes, environment, seeds, bounds and runtime. No mid-search resume is provided. A repeated unchanged run reproduced the original calibration JSON exactly.

Expected this version: 19 June dates, 613 prices, 34 coordinates. Fit RMSE about 0.211321 and independent-seed RMSE 0.217086 USD/barrel. The 100-evaluation budget is a development limit, not convergence. Missing files/columns, July training dates, realized future fixings or inconsistent schedules must fail, not be imputed. Compare hashes to the relevant full-root or reduced-package manifest, not interchangeably.

Tests: `python -m pytest -q`; measured results are 60 passed with the full archive. Archive-dependent tests skip without authorized local data. Ruff: `ruff check src tests scripts`.

Charts were generated separately using project-local Matplotlib 3.10.8 from the saved actual-versus-model CSV; no plotting dependency was added to Tim's pricing requirements. All experiment/private handoff files stay under ignored outputs or outside the checkout. No paid API request is part of this command.
