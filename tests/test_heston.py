"""Numerical checks use labeled synthetic inputs, never substitute market data."""

from dataclasses import replace
import math
import numpy as np
import pandas as pd
import pytest
from hestonlab.heston import (
    AsianContract,
    HestonParameters,
    MonteCarloConfig,
    price_asian_options,
)
from hestonlab.calibration import CalibrationSnapshot, calibrate_june, load_june_training


def contract(strike=80, side="C"):
    return AsianContract(
        f"synthetic:{side}:{strike}", (0.1, 0.2), (80.0, 82.0), 3, 78.0, strike, side, 0.99
    )


def test_fully_accrued_payoff_is_exact():
    c = AsianContract("accrued", (), (), 3, 270, 80, "C", 0.98)
    result = price_asian_options([c], HestonParameters())[0]
    assert result.price == pytest.approx(9.8)
    assert result.standard_error == 0


def test_single_fixing_constant_variance_matches_black_price():
    # sigma -> 0 and v0=theta: the futures marginal has a known Black price.
    c = replace(
        contract(), fixing_times=(0.5,), forwards=(80.0,), total_fixings=1, known_fixing_sum=0
    )
    p = HestonParameters(v0=0.04, theta=0.04, sigma=1e-10)
    result = price_asian_options([c], p, MonteCarloConfig(paths=40000))[0]

    def normal_cdf(x):
        return (1 + math.erf(x / math.sqrt(2))) / 2

    d1 = 0.5 * math.sqrt(0.04 * 0.5)
    exact = 0.99 * 80 * (normal_cdf(d1) - normal_cdf(-d1))
    assert abs(result.price - exact) < 5 * result.standard_error


def test_discrete_asian_parity_with_curve_and_accrued_fixing():
    call, put = contract(), contract(side="P")
    a, b = price_asian_options([call, put], HestonParameters(), MonteCarloConfig(paths=20000))
    expected = 0.99 * ((78 + 80 + 82) / 3 - 80)
    assert abs(a.price - b.price - expected) < 5 * (a.standard_error + b.standard_error)


def test_common_factor_coordinates_and_seed_reproducibility():
    c = contract()
    scaled = replace(
        c, coordinate_id="scaled", forwards=(160.0, 164.0), known_fixing_sum=156, strike=160
    )
    cfg = MonteCarloConfig(paths=1000)
    a = price_asian_options([c, scaled], HestonParameters(), cfg)
    assert a == price_asian_options([c, scaled], HestonParameters(), cfg)
    assert a[1].price == pytest.approx(2 * a[0].price)
    assert a[1].standard_error == pytest.approx(2 * a[0].standard_error)
    other = price_asian_options([c], HestonParameters(), replace(cfg, seed=42))
    assert a[0].price != other[0].price


def test_full_truncation_survives_feller_violation():
    p = HestonParameters(v0=0.001, kappa=0.2, theta=0.001, sigma=2.0, rho=-0.9)
    assert p.feller_margin < 0
    result = price_asian_options([contract()], p, MonteCarloConfig(paths=2000))[0]
    assert np.isfinite([result.price, result.standard_error]).all()
    assert result.price >= 0 and result.standard_error >= 0


@pytest.mark.parametrize(
    "changes",
    [
        {"forwards": (0, 82)},
        {"fixing_times": (0.1, 0.1)},
        {"discount_factor": float("nan")},
        {"total_fixings": 1},
        {"side": "X"},
    ],
)
def test_bad_contract_inputs_fail(changes):
    with pytest.raises(ValueError):
        replace(contract(), **changes)


def test_bad_simulation_and_parameters_fail():
    with pytest.raises(ValueError):
        HestonParameters(rho=1)
    with pytest.raises(ValueError):
        MonteCarloConfig(paths=3)
    with pytest.raises(ValueError):
        MonteCarloConfig(max_step_years=0)
    with pytest.raises(ValueError):
        price_asian_options([], HestonParameters())


def test_calibration_improves_synthetic_fit_and_keeps_july_out():
    contracts = tuple(contract(k) for k in (70, 75, 80, 85, 90, 95))
    truth = HestonParameters(v0=0.16, theta=0.16)
    config = MonteCarloConfig(paths=1000, max_step_years=1 / 52)
    prices = tuple(e.price for e in price_asian_options(contracts, truth, config))
    snapshot = CalibrationSnapshot("2024-06-14", contracts, prices)
    result = calibrate_june([snapshot], config=config, max_evaluations=31)
    assert result["training_rmse"] < result["initial_rmse"] * 0.8
    assert result["evaluations"] <= 31
    assert not result["july_labels_used"] and not result["fully_validated"]
    assert np.isfinite(result["independent_seed_rmse"])
    assert set(result["parameters"]) == {"v0", "kappa", "theta", "sigma", "rho"}
    with pytest.raises(ValueError, match="July is held out"):
        replace(snapshot, observation_date="2024-07-01")


def write_pilot_fixture(root):
    folder = root / "pilot_june_july_2024"
    folder.mkdir()
    rows = [
        dict(
            observation_date=pd.Timestamp("2024-06-14").date(),
            contract_id=f"synthetic:{k}",
            contract_month="2024-08",
            strike=k,
            call_put="C",
            settlement_price=2.0,
            total_fixings=2,
            remaining_fixings=2,
            known_fixing_count=0,
            known_fixing_sum=0,
            prototype_input_complete=True,
            provisional_or_missing_elapsed=0,
            provisional_remaining_forwards=0,
            missing_or_nonpositive_forwards=0,
            terminal_diagnostic_pass=True,
            discount_factor=0.99,
        )
        for k in (70, 75, 80, 85, 90, 95)
    ]
    p = folder / "train_fixed_coordinate_observations.parquet"
    pd.DataFrame(rows).to_parquet(p)
    f = root / "ao_model_fixing_inputs.parquet"
    pd.DataFrame(
        [
            dict(
                observation_date=pd.Timestamp("2024-06-14").date(),
                contract_month="2024-08",
                fixing_date=pd.Timestamp(day).date(),
                current_forward=80.0,
                fixing_time_years=(pd.Timestamp(day) - pd.Timestamp("2024-06-14")).days / 365,
                forward_final_flag=True,
                known_fixing=None,
            )
            for day in ("2024-08-01", "2024-08-02")
        ]
    ).to_parquet(f)
    return p, f


def test_archive_adapter_without_any_july_option_file(tmp_path):
    write_pilot_fixture(tmp_path)
    snapshots = load_june_training(tmp_path)
    assert len(snapshots) == 1
    assert len(snapshots[0].contracts) == 6
    assert snapshots[0].contracts[0].fixing_times == pytest.approx((48 / 365, 49 / 365))


def test_archive_adapter_rejects_heldout_dates(tmp_path):
    p, _ = write_pilot_fixture(tmp_path)
    rows = pd.read_parquet(p)
    rows.loc[0, "observation_date"] = pd.Timestamp("2024-07-01").date()
    rows.to_parquet(p)
    with pytest.raises(ValueError, match="outside June"):
        load_june_training(tmp_path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("current_forward", None),
        ("forward_final_flag", False),
        ("known_fixing", 80.0),
    ],
)
def test_archive_adapter_rejects_missing_or_realized_inputs(tmp_path, field, value):
    _, f = write_pilot_fixture(tmp_path)
    rows = pd.read_parquet(f)
    rows.loc[0, field] = value
    rows.to_parquet(f)
    with pytest.raises(ValueError):
        load_june_training(tmp_path)


def test_missing_archive_fails_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="public CSV is insufficient"):
        load_june_training(tmp_path)
