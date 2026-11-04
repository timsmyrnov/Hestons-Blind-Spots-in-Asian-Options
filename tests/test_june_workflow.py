"""Data-side integration checks around Tim's unchanged pricing and optimizer."""

from dataclasses import replace
from pathlib import Path
import os
import numpy as np
import pytest
from hestonlab.calibration import calibrate_june, load_june_training
from hestonlab.heston import HestonParameters, MonteCarloConfig
from hestonlab.june_workflow import validate_output_root, price_snapshots, error_metrics, main


def test_experiment_cannot_write_into_archive(tmp_path):
    root = tmp_path / "archive"
    for out in [root, root / "experiment"]:
        with pytest.raises(ValueError, match="outside"):
            validate_output_root(root, out)
    assert validate_output_root(root, tmp_path / "results")[1] == tmp_path / "results"


def test_missing_root_has_actionable_cli_error(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("HESTONLAB_DATA_ROOT", raising=False)
    monkeypatch.setattr("sys.argv", ["june_workflow", "--out", str(tmp_path / "results")])
    with pytest.raises(SystemExit) as e:
        main()
    assert e.value.code == 2
    assert "HESTONLAB_DATA_ROOT" in capsys.readouterr().err
    assert not (tmp_path / "results").exists()


ROOT = Path(os.environ.get("HESTONLAB_DATA_ROOT", "outputs/HestonLab_Asian_Data"))


@pytest.mark.skipif(
    not (ROOT / "pilot_june_july_2024/train_fixed_coordinate_observations.parquet").exists(),
    reason="Full authorized June archive required",
)
def test_real_june_export_reproduces_optimizer_streams():
    s = load_june_training(ROOT)[0]
    s = replace(s, contracts=s.contracts[:6], market_prices=s.market_prices[:6])
    cfg = MonteCarloConfig(paths=128)
    result = calibrate_june([s], config=cfg, max_evaluations=11)
    p = HestonParameters(**result["parameters"])
    same = price_snapshots([s], p, cfg)
    assert error_metrics(same)["rmse"] == pytest.approx(result["training_rmse"], abs=1e-12)
    fresh = price_snapshots([s], p, replace(cfg, seed=cfg.seed + 1000000))
    assert error_metrics(fresh)["rmse"] == pytest.approx(result["independent_seed_rmse"], abs=1e-12)
    assert np.allclose(same.residual, same.observed_price - same.model_price)
    assert same.observation_date.str.startswith("2024-06").all()
