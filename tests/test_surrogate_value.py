"""Surrogate-value experiment checks. SYNTHETIC grid only; small and fast."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hestonlab.heston import MonteCarloConfig  # noqa: E402
from hestonlab.surrogate import TrainConfig  # noqa: E402
from hestonlab.surrogate_box import train_in_box, VALUE_EXPERIMENT_BOX  # noqa: E402
from hestonlab.surrogate_data import synthetic_pilot_grid  # noqa: E402
from hestonlab.surrogate_value import (  # noqa: E402
    LABEL_SEED,
    ValueConfig,
    accuracy_report,
    build_data,
    evaluation_prices,
)

GRID = synthetic_pilot_grid(strikes=(70, 80, 90))
TINY = ValueConfig(
    train_draws=128, test_draws=32, eval_points=16, label_paths=512, cheap_paths=256,
    truth_paths=2048, bootstrap_resamples=200, timing_batch=10, timing_repeats=2, workers=1,
    train=TrainConfig(max_epochs=30, batch_size=32, learning_rate=1e-2),
)


@pytest.fixture(scope="module")
def stage2(tmp_path_factory):
    out = tmp_path_factory.mktemp("value")
    train, test = build_data(GRID, TINY, out, log=lambda *_: None)
    model, _ = train_in_box(train, GRID, VALUE_EXPERIMENT_BOX, TINY.train)
    params, a, b, prices = evaluation_prices(model, GRID, test, TINY, out, log=lambda *_: None)
    return out, train, test, model, params, a, b, prices


def test_config_keeps_truth_streams_separate_from_labels():
    assert LABEL_SEED == MonteCarloConfig().seed
    assert LABEL_SEED not in ValueConfig().truth_seeds
    # The experiment trains longer; the package default is untouched.
    assert ValueConfig().train == TrainConfig(max_epochs=6000)
    assert TrainConfig().max_epochs == 3000
    with pytest.raises(ValueError, match="different"):
        ValueConfig(truth_seeds=(LABEL_SEED, 1))
    with pytest.raises(ValueError, match="different"):
        ValueConfig(truth_seeds=(5, 5))
    with pytest.raises(ValueError, match="test draws"):
        ValueConfig(eval_points=4096, test_draws=2048)


def test_evaluation_prices_line_up_and_truth_is_cached(stage2):
    out, _, test, model, params, a, b, prices = stage2
    assert set(prices) == {"oracle", "network", "mc_16k", "mc_4k"}
    for p in prices.values():
        assert p.shape == (TINY.eval_points, len(GRID))
    np.testing.assert_array_equal(prices["oracle"], (a + b) / 2)
    np.testing.assert_array_equal(prices["mc_16k"], test["prices"][:TINY.eval_points])
    assert VALUE_EXPERIMENT_BOX.contains(params)
    log = []
    again = evaluation_prices(model, GRID, test, TINY, out, log=log.append)
    assert sum("reusing" in m for m in log) == 3  # both truth streams and MC-4k
    np.testing.assert_array_equal(again[1], a)


def test_accuracy_report_covers_every_model_trader(stage2):
    *_, params, a, b, prices = stage2
    report = accuracy_report(prices, a, b, params)
    for name in ("network", "mc_16k", "mc_4k"):
        r = report[name]
        assert r["points"] == TINY.eval_points
        assert r["rmse_noise_corrected_usd_per_bbl"] <= r["rmse_raw_usd_per_bbl"]
        assert len(r["by_v0_decile"]) == 10
    # More paths, less error: 4k-path labels are noisier than the training stream here.
    assert report["mc_4k"]["rmse_raw_usd_per_bbl"] > report["mc_16k"]["rmse_raw_usd_per_bbl"]


def test_trading_rule_on_a_hand_worked_example():
    from hestonlab.surrogate_value import trade

    truth = np.array([[10.0, 10.0, 10.0, 10.0]])
    market = np.array([[10.10, 9.95, 10.02, 10.10]])
    model = np.array([[10.00, 10.00, 10.00, 10.20]])
    position, pnl, wrong = trade(model, market, truth, 0.03, 0.01, 1000)
    # Sell (market rich), buy (market cheap), no trade (edge 0.02), buy (model too high).
    np.testing.assert_array_equal(position, [[-1, 1, 0, 1]])
    # (10.00-10.10)x-1 = +0.10 -> 100 - 10 cost; +0.05 -> 50 - 10; 0; -0.10 -> -100 - 10.
    np.testing.assert_allclose(pnl, [[90.0, 40.0, 0.0, -110.0]])
    np.testing.assert_array_equal(wrong, [[False, False, False, True]])


def test_bootstrap_is_paired_and_exact_for_a_constant_gap():
    from hestonlab.surrogate_value import bootstrap_totals, interval

    base = np.random.default_rng(1).normal(100, 30, size=50)
    boot = bootstrap_totals({"a": base + 2.0, "b": base}, resamples=300, seed=0)
    # Same resampled scenarios for both traders: the gap is exactly 50 x 2 every time.
    np.testing.assert_allclose(boot["a"] - boot["b"], 100.0)
    assert interval(boot["a"] - boot["b"]) == pytest.approx([100.0, 100.0])


def test_pnl_report_oracle_is_the_benchmark(stage2):
    from hestonlab.surrogate_value import pnl_report

    *_, prices, = stage2
    truth = prices["oracle"]
    report = pnl_report(prices, truth, TINY)
    oracle = report["traders"]["oracle"]
    assert oracle["pct_of_oracle"] == pytest.approx(100.0)
    assert oracle["wrong_way_trades"] == 0  # knows the truth, so never trades the wrong way
    assert report["opportunities"] == TINY.eval_points * len(GRID)
    for name in ("network", "mc_16k", "mc_4k"):
        t = report["traders"][name]
        assert t["trades"] == t["buys"] + t["sells"]
        low, high = t["pct_of_oracle_ci95"]
        assert low <= high
    d = report["differences"]["network_minus_oracle"]
    assert d["point_usd"] == pytest.approx(
        report["traders"]["network"]["total_pnl_usd"] - oracle["total_pnl_usd"])
    # Same market for every trader, fixed seed: rerun gives identical numbers.
    assert pnl_report(prices, truth, TINY) == report


def test_break_even_is_upfront_cost_over_saving_per_call():
    from hestonlab.surrogate_value import break_even

    timing = {"network": {"single_call_seconds": 0.001},
              "mc_16k": {"single_call_seconds": 0.031},
              "mc_4k": {"single_call_seconds": 0.011}}
    upfront = {"training_data": 500.0, "training": 100.0, "test_data": 60.0}
    out = break_even(timing, upfront)
    assert out["mc_16k"]["break_even_calls"] == pytest.approx(600 / 0.030)
    assert out["mc_4k"]["break_even_calls"] == pytest.approx(600 / 0.010)
    assert out["mc_16k"]["break_even_calls_including_test_set"] == pytest.approx(660 / 0.030)
    slow = {**timing, "network": {"single_call_seconds": 0.05}}
    assert break_even(slow, upfront)["mc_4k"]["break_even_calls"] is None  # never pays off


def test_timing_reports_every_method_on_one_thread(stage2):
    from hestonlab.surrogate_value import time_pricing

    _, _, _, model, params, *_ = stage2
    threads = torch.get_num_threads()
    out = time_pricing(model, GRID, params, TINY)
    assert torch.get_num_threads() == threads  # restored
    for name in ("network", "mc_16k", "mc_4k"):
        assert out[name]["single_call_seconds"] > 0
        assert out[name]["batch_seconds_per_pricing"] > 0
    assert out["mc_16k"]["single_call_slowdown_vs_network"] > 1


def test_run_experiment_writes_summaries_plots_and_docs_copy(tmp_path):
    import json
    from hestonlab.surrogate_value import run_experiment

    out, docs = tmp_path / "outputs" / "value", tmp_path / "docs"
    summaries = run_experiment(TINY, out, docs, GRID, log=lambda *_: None)
    pngs = ["pnl_pct_of_oracle.png", "accuracy_vs_truth.png", "error_by_v0_decile.png",
            "break_even.png", "time_per_pricing.png"]
    jsons = ["accuracy.json", "pnl.json", "cost.json", "setup.json"]
    for name in pngs + jsons + ["README.md", "model.pt", "data/train.npz", "data/truth_0.npz"]:
        assert (out / name).is_file(), name
    for name in pngs + jsons + ["README.md"]:
        assert (docs / name).is_file(), name
    assert not (docs / "model.pt").exists() and not (docs / "data").exists()
    setup = json.loads((docs / "setup.json").read_text())
    assert setup["architecture"] == "manifold"
    assert "SYNTHETIC" in setup["status"] and "not Heston model error" in setup["status"]
    readme = (docs / "README.md").read_text()
    assert "Synthetic data only" in readme and "Heston model error" in readme
    assert summaries["pnl.json"]["traders"]["oracle"]["pct_of_oracle"] == pytest.approx(100)
    # Charts and README can be redrawn from the saved summaries alone.
    from hestonlab.surrogate_value import replot

    for name in pngs:
        (out / name).unlink()
    assert replot(out, None, log=lambda *_: None)["pnl.json"] == summaries["pnl.json"]
    assert all((out / name).is_file() for name in pngs)
    # Each run is archived; one run gives no comparison section yet.
    assert (out / "runs" / "epochs_30" / "model.pt").is_file()
    assert (docs / "runs" / "epochs_30" / "pnl.json").is_file()
    assert not (docs / "runs" / "epochs_30" / "model.pt").exists()
    assert "runs side by side" not in readme


def test_a_longer_run_is_reported_next_to_the_first(tmp_path):
    import json
    from dataclasses import replace
    from hestonlab.surrogate_value import run_experiment

    out, docs = tmp_path / "outputs" / "value", tmp_path / "docs"
    quiet = dict(contracts=GRID, log=lambda *_: None)
    first = run_experiment(TINY, out, docs, **quiet)
    longer = replace(TINY, train=replace(TINY.train, max_epochs=40))
    second = run_experiment(longer, out, docs, **quiet)
    runs = json.loads((docs / "runs.json").read_text())
    assert [r["run"] for r in runs["runs"]] == ["epochs_30", "epochs_40"]
    assert runs["primary_run"] == "epochs_40"
    a, b = runs["runs"]
    # Only the network changes: the Monte Carlo traders are identical in both runs.
    assert a["pct_of_oracle"]["mc_16k"] == b["pct_of_oracle"]["mc_16k"]
    assert a["total_pnl_usd"]["mc_4k"] == b["total_pnl_usd"]["mc_4k"]
    assert a["network_minus_mc_16k"] == first["pnl.json"]["differences"]["network_minus_mc_16k"]
    assert b["network_minus_mc_16k"] == second["pnl.json"]["differences"]["network_minus_mc_16k"]
    assert b["best_validation_mse_standardized"] > 0
    readme = (docs / "README.md").read_text()
    assert "runs side by side" in readme and "40-epoch cap (shown above)" in readme
    assert readme == (out / "README.md").read_text()
