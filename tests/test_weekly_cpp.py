"""Budgets, leakage, market geometry, cache integrity and presentation semantics."""

from dataclasses import replace
import json
import time
import numpy as np
import pandas as pd
import pytest
from hestonlab.cpp_pricer import Estimate
from hestonlab.direct_fit import DeadlineExceeded, encode, fit
from hestonlab.market_manifold import fit_market
from hestonlab.manifold_movie import smoothing_weights
from hestonlab.weekly_store import Config, weekly_indices, output_root, write_npz
from hestonlab.weekly_geometry import gap_statistics
from hestonlab.weekly_pipeline import cached_week, save_week, model_for_day, smoke


class AnalyticPricer:
    def __init__(self):
        self.calls = 0

    def price(self, parameters, **kwargs):
        self.calls += 1
        unit = encode(parameters)
        return Estimate(np.r_[unit, unit.sum()], np.zeros(6), np.zeros((6, 6)))


def test_optimizer_counts_derivatives_restarts_and_respects_deadline():
    pricer = AnalyticPricer()
    target = np.r_[np.full(5, 0.4), 2.0]
    result = fit(pricer, target, np.ones(6), [np.full(5, 0.1), np.full(5, 0.8)], max_calls=30)
    assert pricer.calls == result["calls"] <= 30
    assert len(result["starts"]) == 2
    assert result["rmse_usd"] < 0.1
    with pytest.raises(DeadlineExceeded):
        fit(pricer, target, np.ones(6), [np.full(5, 0.5)], deadline=time.monotonic() - 1)


def test_quadratic_graph_has_exact_pca_coordinates_and_no_future_leakage():
    rng = np.random.default_rng(42)
    z = rng.uniform(-1, 1, (160, 2))
    values = np.column_stack(
        [
            z,
            0.3 * z[:, 0] ** 2,
            0.25 * z[:, 1] ** 2,
            0.2 * z[:, 0] * z[:, 1],
            0.1 * z[:, 0] ** 2 + 0.1 * z[:, 1] ** 2,
        ]
    )
    dates = [str(d.date()) for d in pd.bdate_range("2023-01-02", periods=160)]
    model = fit_market(values[:120], dates[:120], rank=2, validation=20)
    assert model.metadata["method"] == "quadratic_graph"
    points = rng.uniform(-0.5, 0.5, (10, 2))
    np.testing.assert_allclose(model.coordinates(model.prices(points)), points, atol=1e-12)
    assert model.metadata["validation_rmse_usd"] < model.metadata["pca_validation_rmse_usd"]
    cfg = replace(Config(), market_rank=2)
    panel = {"dates": np.array(dates), "vectors": values.copy(), "scales": np.ones_like(values)}
    first, _ = model_for_day(panel, 125, cfg)
    panel["vectors"][125:] += 10000
    second, _ = model_for_day(panel, 125, cfg)
    np.testing.assert_array_equal(first.coefficients, second.coefficients)
    assert max(second.metadata["training_dates"]) < dates[125]
    assert not model.supported(np.full(2, 100.0))


def test_weekly_selection_uses_last_available_day_and_full_prior_history():
    dates = [
        str(d.date())
        for d in pd.bdate_range("2021-06-01", "2022-01-31")
        if str(d.date()) != "2022-01-14"
    ]
    cfg = replace(Config(), end="2022-01-31")
    indices = weekly_indices(dates, cfg)
    chosen = [dates[i] for i in indices]
    assert "2022-01-13" in chosen and "2022-01-14" not in chosen
    assert len(chosen) == 5 and min(indices) >= 120


def test_smoothing_uses_actual_trading_day_distances():
    weights = smoothing_weights([0, 5, 10], 4)
    assert weights[0, 1] / weights[0, 0] == pytest.approx(np.exp(-0.5 * (5 / 4) ** 2))
    np.testing.assert_allclose(weights.sum(axis=1), 1)
    with pytest.raises(ValueError):
        smoothing_weights([0, 0], 4)


def test_gap_uncertainty_uses_full_cross_contract_covariance():
    estimate = Estimate(np.zeros(2), np.ones(2) * 0.1, np.full((2, 2), 0.01))
    got = gap_statistics(np.ones(2), estimate, np.ones(2))
    assert got["gap_usd"] == 1
    assert got["gap_delta_se_usd"] == pytest.approx(0.1)


def test_cache_requires_matching_identity_and_intact_arrays(tmp_path):
    summary = {"date": "2024-01-05"}
    save_week(tmp_path, "abc", summary, {"unit": np.ones(5)})
    assert cached_week(tmp_path, "abc", "2024-01-05") is not None
    assert cached_week(tmp_path, "def", "2024-01-05") is None
    write_npz(tmp_path / "weeks/abc/2024-01-05.npz", unit=np.zeros(5))
    assert cached_week(tmp_path, "abc", "2024-01-05") is None
    with pytest.raises(ValueError, match="archive"):
        output_root(tmp_path, tmp_path / "experiment")


def test_synthetic_pipeline_is_resumable_and_records_accuracy_flags(tmp_path):
    out = tmp_path / "smoke"
    assert smoke(out) == 0
    report = json.loads((out / "report.json").read_text())
    assert report["synthetic"] and report["completed_weeks"] == 2
    assert report["status"] == "computed_with_flags"
    assert report["rows"][0]["checks"]["detailed_checks_performed"]
    assert (out / "animation.html").exists()
    assert "cdn.jsdelivr.net" not in (out / "animation.html").read_text()
    from hestonlab.weekly_pipeline import run

    assert run(out, no_video=True) == 0
    assert "verified cache" in (out / "run.log").read_text()


def test_parent_watchdog_marks_timeout_and_releases_lock(tmp_path):
    from dataclasses import asdict
    from hestonlab.weekly_store import write_json
    from hestonlab.weekly_pipeline import run

    config = replace(Config(), budget_seconds=0.001, render_reserve_seconds=0.0001)
    write_json(tmp_path / "config.json", asdict(config))
    assert run(tmp_path, no_video=True) == 124
    assert not (tmp_path / "run.lock").exists()
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["status"] == "partial_hard_deadline"
    assert not report["fully_validated"]


def test_configuration_rejects_silent_budget_and_geometry_changes():
    with pytest.raises(ValueError, match="three hours"):
        replace(Config(), budget_seconds=11000)
    with pytest.raises(ValueError, match="odd"):
        replace(Config(), surface_grid=10)
    with pytest.raises(ValueError, match="chronological"):
        replace(Config(), history=20)
