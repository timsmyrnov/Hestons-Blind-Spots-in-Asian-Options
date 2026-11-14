"""Market-vs-Heston manifold comparison; every input here is synthetic."""

from datetime import date
import json
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from hestonlab.heston import AsianContract, MonteCarloConfig  # noqa: E402
from hestonlab.manifold import ManifoldNetwork, geometry_diagnostics  # noqa: E402
from hestonlab import market_manifold as mm  # noqa: E402
from hestonlab.surrogate import OutputScaler, Surrogate, TrainConfig  # noqa: E402

SMALL = mm.GridSpec(tenors=(0.2, 0.3), log_moneyness=(-0.1, 0.0, 0.1))


def month(tau, xs, value, forward=80.0, discount=0.98, extra=()):
    """Contracts on one unseasoned averaging month whose normalized OTM price is value(x)."""
    times = tuple(np.linspace(tau - 29 / 365, tau, 21).tolist())
    contracts, prices = [], []
    for x in xs:
        strike = forward * np.exp(x)
        side = "P" if strike < forward else "C"
        contracts.append(AsianContract(f"{tau}:{x}", times, (forward,) * 21, 21, 0.0,
                                       float(strike), side, discount))
        prices.append(discount * forward * value(x))
    for contract, price in extra:
        contracts.append(contract)
        prices.append(price)
    return contracts, prices


def test_grid_contracts_are_canonical_out_of_the_money():
    contracts = mm.heston_grid_contracts(SMALL)
    assert [c.coordinate_id for c in contracts] == list(SMALL.coordinate_ids)
    assert [c.side for c in contracts] == ["P", "C", "C"] * 2
    np.testing.assert_allclose([c.strike for c in contracts], np.exp([-0.1, 0, 0.1] * 2))
    assert {c.fixing_times[-1] for c in contracts} == {0.2, 0.3}
    assert all(c.forwards == (1.0,) * 21 and c.discount_factor == 1.0 for c in contracts)


def test_day_vector_interpolates_exactly_and_never_extrapolates():
    def smile(x):
        return 0.35 + 0.4 * x ** 2

    xs = (-0.15, -0.1, 0.0, 0.1, 0.15)
    times = tuple(np.linspace(0.15 - 29 / 365, 0.15, 21).tolist())
    itm = AsianContract("itm", times, (80.0,) * 21, 21, 0.0, 72.0, "C", 0.98)
    seasoned = AsianContract("old", times[5:], (80.0,) * 16, 21, 400.0, 80.0, "C", 0.98)
    near = month(0.15, xs, lambda x: mm.black_otm(x, smile(x), 0.15),
                 extra=[(itm, 999.0), (seasoned, 5.0)])
    far = month(0.45, xs, lambda x: mm.black_otm(x, smile(x), 0.45))
    contracts, prices = near[0] + far[0], near[1] + far[1]
    vector = mm.day_vector(contracts, prices, SMALL)
    # Quotes sit on the grid strikes and both months share one smile, so total variance
    # is linear in time and the volatility interpolation is exact.
    expected = [mm.black_otm(x, smile(x), t) for t in SMALL.tenors for x in SMALL.log_moneyness]
    np.testing.assert_allclose(vector, expected, rtol=1e-7)
    wide = mm.GridSpec(tenors=(0.3, 0.5), log_moneyness=(0.0, 0.2))
    out = mm.day_vector(contracts, prices, wide).reshape(2, 2)
    assert np.isfinite(out[0, 0]) and np.isnan(out[0, 1]) and np.isnan(out[1]).all()
    panel = mm.build_panel([("2021-03-02", contracts, prices),
                            ("2021-03-01", near[0], near[1])], SMALL)
    assert panel.dates == ("2021-03-02",) and panel.incomplete == ("2021-03-01",)


def test_month_split_keeps_test_months_and_embargo_out_of_training():
    dates = [str(d.date()) for d in pd.bdate_range("2021-01-04", "2021-04-30")]
    train, test, embargo = mm.split_months(dates, ["2021-02"], embargo_days=3)
    picked = lambda mask: {d for d, k in zip(dates, mask) if k}  # noqa: E731
    assert picked(test) == {d for d in dates if d.startswith("2021-02")}
    assert picked(embargo) == {"2021-01-29", "2021-03-01"}
    assert not picked(train) & (picked(test) | picked(embargo))
    assert len(picked(train)) == len(dates) - len(picked(test)) - 2
    with pytest.raises(ValueError, match="No days in test months"):
        mm.split_months(dates, ["2021-02", "2022-01"])
    with pytest.raises(ValueError, match="YYYY-MM"):
        mm.split_months(dates, ["2021-2"])


def test_market_manifold_learns_curved_data_and_round_trips(tmp_path):
    z = np.random.default_rng(0).uniform(size=(80, 2))
    y = np.column_stack([z[:, 0], z[:, 1], z[:, 0] ** 2, z[:, 1] ** 2, z[:, 0] * z[:, 1],
                         np.sin(z[:, 0])])
    model = mm.train_market_manifold(y, [f"c{i}" for i in range(6)],
                                     mm.MarketConfig(latent_dim=2, width=16, depth=2,
                                                     epochs=300, lr_patience=30))
    assert model.metadata["best_train_mse_standardized"] < 0.05
    assert geometry_diagnostics(model, model.train_units)["numerically_resolved"]
    model.save(tmp_path / "m.pt")
    loaded = mm.MarketManifold.load(tmp_path / "m.pt")
    np.testing.assert_array_equal(mm.evaluate_units(loaded, loaded.train_units),
                                  mm.evaluate_units(model, model.train_units))
    with pytest.raises(ValueError, match="training days"):
        mm.train_market_manifold(y[:3], [f"c{i}" for i in range(6)], mm.MarketConfig())


def plane_heston():
    """A known 5-plane standing in for the Heston manifold on a 6-point grid."""
    return Surrogate(ManifoldNetwork(6, width=8, depth=2), OutputScaler(np.full(6, 20.0),
                     np.ones(6)), SMALL.coordinate_ids)


@pytest.mark.parametrize("lift", [0.0, 0.3])
def test_manifold_distance_measures_the_non_heston_direction(lift):
    heston = plane_heston()
    t = np.arange(120)
    blind = lift * (0.5 + 0.5 * np.sin(t / 7))
    units = np.column_stack([0.5 + 0.3 * np.sin(t / 10), 0.5 + 0.3 * np.cos(t / 17),
                             np.full((120, 3), 0.5)])
    y = mm.evaluate_units(heston, units)
    y[:, 5] += blind  # the plane never moves coordinate 5
    dates = tuple(str(d.date()) for d in pd.bdate_range("2021-01-04", periods=120))
    panel = mm.MarketPanel(dates, y, SMALL)
    masks = mm.split_months(dates, ["2021-03", "2021-05"])
    market = mm.train_market_manifold(y[masks[0]], SMALL.coordinate_ids,
                                      mm.MarketConfig(latent_dim=3, width=8, depth=2,
                                                      epochs=200, lr_patience=20))
    cloud = np.random.default_rng(1).uniform(size=(256, 5))
    rows = mm.compare_manifolds(panel, masks, heston, market, np.ones(6),
                                (cloud, mm.evaluate_units(heston, cloud)), starts=2)
    test_rows = [(r, b) for r, b in zip(rows, blind) if r["split"] == "test"]
    assert {r["month"] for r, _ in test_rows} == {"2021-03", "2021-05"}
    for row, b in test_rows:
        assert row["manifold_distance"] == pytest.approx(b / np.sqrt(6), abs=0.01)
        assert row["market_to_heston"] == pytest.approx(b / np.sqrt(6), abs=0.01)
        assert row["market_to_market_manifold"] < 0.01
    summary = mm.summarize(rows)
    assert summary["test"]["days"] == len(test_rows)
    json.dumps(summary, allow_nan=False)


def test_archive_loader_keeps_only_complete_unseasoned_final_inputs(tmp_path):
    day = date(2024, 3, 1)
    fixing_days = pd.bdate_range("2024-05-01", "2024-05-31")
    fixings = pd.DataFrame({
        "observation_date": day, "contract_month": "2024-05",
        "fixing_date": [d.date() for d in fixing_days],
        "current_forward": np.linspace(78, 80, len(fixing_days)),
        "forward_final_flag": True,
        "fixing_time_years": [(d.date() - day).days / 365 for d in fixing_days],
    })
    provisional = fixings.assign(contract_month="2024-06", forward_final_flag=False)
    pd.concat([fixings, provisional]).to_parquet(tmp_path / "ao_model_fixing_inputs.parquet")
    n = len(fixing_days)
    base = {"observation_date": day, "contract_month": "2024-05", "call_put": "C",
            "settlement_price": 2.5, "discount_factor": 0.99, "prototype_input_complete": True,
            "elapsed_fixings": 0, "known_fixing_sum": 0.0, "total_fixings": n,
            "remaining_fixings": n, "provisional_remaining_forwards": 0}
    rows = [
        {**base, "contract_id": "keep-1", "strike": 80.0},
        {**base, "contract_id": "keep-2", "strike": 85.0, "settlement_price": 1.0},
        {**base, "contract_id": "incomplete", "strike": 90.0, "prototype_input_complete": False},
        {**base, "contract_id": "seasoned", "strike": 95.0, "elapsed_fixings": 2},
        {**base, "contract_id": "provisional", "strike": 80.0, "contract_month": "2024-06"},
    ]
    pd.DataFrame(rows).to_parquet(tmp_path / "ao_calibration_inputs_provisional.parquet")
    days = mm.load_archive_days(tmp_path)
    assert [d for d, _, _ in days] == ["2024-03-01"]
    _, contracts, prices = days[0]
    assert [c.coordinate_id for c in contracts] == ["keep-1", "keep-2"] and prices == (2.5, 1.0)
    np.testing.assert_allclose(contracts[0].forwards, fixings.current_forward)
    np.testing.assert_allclose(contracts[0].fixing_times, fixings.fixing_time_years)
    assert contracts[0].discount_factor == 0.99
    assert mm.load_archive_days(tmp_path, start="2024-03-02") == []
    with pytest.raises(FileNotFoundError, match="full local archive"):
        mm.load_archive_days(tmp_path / "missing")


def test_synthetic_cli_holds_out_three_months_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(mm, "QUICK", mm.RunConfig(
        grid=SMALL, heston_draws=64, heston_test_draws=32,
        heston_mc=MonteCarloConfig(paths=128, max_step_years=1 / 52),
        heston_train=TrainConfig(width=12, depth=2, max_epochs=5),
        market=mm.MarketConfig(width=8, depth=2, epochs=50, lr_patience=10),
        heston_cloud=128, starts=1, max_evaluations=30,
    ))
    months = ["2021-02", "2021-04", "2021-06"]
    out = tmp_path / "run"
    report = mm.main(["synthetic", "--quick", "--days", "130", "--test-months", *months,
                      "--out", str(out), "--embargo-days", "2"])
    assert report["primary"].startswith("manifold_distance")
    assert not report["test_days_in_training"]
    assert list(report["summary"]["by_test_month"]) == months
    assert all(np.isfinite(r["manifold_distance"]) for r in report["rows"])
    trained_on = mm.MarketManifold.load(out / "market_model.pt").metadata["training_dates"]
    assert not {d[:7] for d in trained_on} & set(months)
    assert (out / "report.json").exists() and (out / "days.csv").exists()
    assert (out / "heston" / "model.pt").exists()


def test_black_transform_round_trips_and_is_smooth_across_the_money():
    x = np.linspace(-0.3, 0.3, 13)
    vol = 0.4 + 0.5 * x ** 2 - 0.1 * x
    price = mm.black_otm(x, vol, 0.3)
    np.testing.assert_allclose(mm.implied_vol(x, price, 0.3), vol, atol=1e-9)
    assert np.isnan(mm.implied_vol(0.1, 2.0, 0.3))  # above any Black price


def test_grid_interpolates_in_volatility_so_the_money_peak_is_kept():
    """Uneven listed strikes around a smooth smile: the grid matches exact Black prices
    (straight lines on prices would cut under the at-the-money peak)."""
    grid = mm.GridSpec(tenors=(0.2, 0.3), log_moneyness=(-0.1, 0.0, 0.1))
    listed_x = np.array([-0.25, -0.17, -0.04, 0.05, 0.15, 0.27])

    def smile(x):
        return 0.4 + 0.3 * x ** 2 - 0.05 * x

    contracts, prices = [], []
    for tau, forward, discount in ((0.15, 70.0, 0.99), (0.45, 68.0, 0.97)):
        times = tuple(np.linspace(tau - 29 / 365, tau, 21).tolist())
        for x in listed_x:
            side = "P" if x < 0 else "C"
            contracts.append(AsianContract(f"{tau}{x}", times, (forward,) * 21, 21, 0.0,
                                           forward * np.exp(x), side, discount))
            prices.append(discount * forward * mm.black_otm(x, smile(x), tau))
    got = mm.day_vector(contracts, np.array(prices), grid).reshape(2, 3)
    for row, tau in zip(got, grid.tenors):
        x = np.array(grid.log_moneyness)
        lo, hi = smile(x) ** 2 * 0.15, smile(x) ** 2 * 0.45
        total = lo + (hi - lo) * (tau - 0.15) / 0.3  # both months share the smile
        exact = mm.black_otm(x, np.sqrt(total / tau), tau)
        # Off the listed strikes only the smile's curvature between them is lost.
        np.testing.assert_allclose(row, exact, rtol=1e-2)
        assert row[1] == pytest.approx(exact[1], rel=3e-3)  # at the money
