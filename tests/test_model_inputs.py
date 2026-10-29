from pathlib import Path
import math
import os
import duckdb
import pytest
from hestonlab.data.model_inputs import discount_factor, arithmetic_payoff, load_model_snapshot

ROOT = Path(os.environ.get("HESTONLAB_DATA_ROOT", "outputs/HestonLab_Asian_Data"))
requires_inputs = pytest.mark.skipif(
    not (ROOT / "ao_calibration_inputs_provisional.parquet").exists(),
    reason="Acquired provisional input archive absent",
)


def test_flat_sofr_units_and_zero_horizon():
    assert discount_factor(5.31, 0) == 1
    assert discount_factor(5.31, 360) == pytest.approx((1 + 0.0531 / 360) ** -360)
    assert discount_factor(5.31, 90) > discount_factor(5.31, 180) > 0
    assert discount_factor(-0.1, 30) > 1
    with pytest.raises(ValueError):
        discount_factor(5, -1)
    with pytest.raises(ValueError):
        discount_factor(math.nan, 30)


def test_discrete_arithmetic_asian_uses_accrued_sum_and_equal_weights():
    assert arithmetic_payoff(140, [80, 100], 4, 75, "C") == 5
    assert arithmetic_payoff(140, [80, 100], 4, 85, "P") == 5
    with pytest.raises(ValueError):
        arithmetic_payoff(0, [], 0, 75, "C")


@requires_inputs
def test_model_snapshot_contains_no_realized_future_fixing():
    x = load_model_snapshot("2024-06-14", ROOT)
    assert len(x["options"]) >= 20
    f = x["fixings"]
    future = f.fixing_date.astype(str) > "2024-06-14"
    assert f.loc[future, "known_fixing"].isna().all()
    assert "realized_fixing_retrospective" not in f
    assert f.loc[future, "current_forward"].gt(0).all()
    p = x["options"]
    assert (p.known_fixing_count + p.remaining_fixings == p.total_fixings).all()
    assert p.discount_factor.notna().all()
    assert (p.rate_effective_date.astype(str) < "2024-06-14").all()
    assert not p.fully_validated_calibration_ready.any()
    assert x["model_defaults"]["free_parameters"] == ["v0", "kappa", "theta", "sigma", "rho"]


@requires_inputs
def test_quarantined_stale_reference_preserves_original_and_correct_candidate():
    c = duckdb.connect()
    original = c.execute(
        "SELECT settlement_price FROM read_parquet(?) WHERE contract_id='CL:2020-05' AND trade_date='2020-04-17'",
        [str(ROOT / "cl_futures_settlements.parquet")],
    ).fetchone()[0]
    candidate = c.execute(
        "SELECT settlement_price FROM read_parquet(?) WHERE contract_id='CL:2020-05' AND trade_date='2020-04-17'",
        [str(ROOT / "underlying_clock_screened_candidates.parquet")],
    ).fetchone()[0]
    c.close()
    assert original == -37.63  # Immutable evidence of the original selection issue.
    assert candidate == 18.27


@requires_inputs
def test_copied_holiday_marks_do_not_add_averaging_fixings():
    c = duckdb.connect()
    assert (
        c.execute(
            "SELECT count(*) FROM read_parquet(?) WHERE fixing_date IN (DATE '2021-04-02',DATE '2023-04-07')",
            [str(ROOT / "ao_candidate_fixing_calendar.parquet")],
        ).fetchone()[0]
        == 0
    )
    d = c.execute(
        "SELECT contract_month,difference,terminal_diagnostic_pass FROM read_csv_auto(?) WHERE contract_month IN ('2020-04','2021-04','2023-04')",
        [str(ROOT / "evidence/clock_screened_monthly_reconciliation.csv")],
    ).fetchall()
    c.close()
    assert len(d) == 3
    assert all(abs(x[1]) <= 0.00501 and x[2] for x in d)


@requires_inputs
def test_pilot_selection_uses_only_training_dates_and_keeps_missing_prices_null():
    import json
    import pandas as pd

    folder = ROOT / "pilot_june_july_2024"
    m = json.loads((folder / "split_manifest.json").read_text())
    assert m["date_overlap"] == 0
    assert m["train"]["last_date"] < m["test"]["first_date"]
    assert m["coordinate_count"] == 34
    for name in ["train", "test"]:
        grid = pd.read_csv(folder / f"{name}_price_grid.csv")
        assert grid.loc[~grid.observed_price, "settlement_price"].isna().all()
    selected = pd.read_csv(folder / "training_selected_coordinates.csv")
    assert selected.training_fraction.ge(0.8).all()
    assert selected.contract_month.isin(["2024-08", "2024-09", "2024-10"]).all()


@requires_inputs
def test_fixed_grid_avoids_unflagged_terminal_fixings():
    c = duckdb.connect()
    folder = ROOT / "pilot_june_july_2024"
    c.read_parquet(
        [
            str(folder / "train_fixed_coordinate_observations.parquet"),
            str(folder / "test_fixed_coordinate_observations.parquet"),
        ]
    ).create_view("grid")
    bad = c.sql(
        "SELECT count(*) FROM grid WHERE known_fixing_count!=0 OR provisional_or_missing_elapsed!=0 OR provisional_remaining_forwards!=0 OR discount_factor IS NULL OR NOT terminal_diagnostic_pass"
    ).fetchone()[0]
    assert c.sql("SELECT count(*) FROM grid").fetchone()[0] > 0
    assert bad == 0
    c.close()
