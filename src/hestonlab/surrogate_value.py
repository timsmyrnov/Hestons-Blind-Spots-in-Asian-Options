"""Does the neural surrogate cost PnL compared with Monte Carlo? SYNTHETIC data only.

A trading-desk thought experiment on the synthetic contract grid, inside a narrow,
realistic parameter box:

* "Truth" is the average of two independent 262,144-path Monte Carlo runs.
* Synthetic market prices are truth plus N(0, 0.07) USD/bbl noise.
* Four traders price the same contracts: an oracle that knows truth, the neural
  surrogate, Monte Carlo at 16,384 paths (the surrogate's own training stream) and
  Monte Carlo at 4,096 paths (a cheap baseline).
* Each trades against mispricings beyond a threshold; PnL is measured against truth.

Everything here measures SURROGATE APPROXIMATION ERROR and its trading cost relative
to Monte Carlo, under the Heston pricer. Nothing here is a statement about Heston's
fit to real markets, and no real data is read.

    python -m hestonlab.surrogate_value              # full run, about 10 minutes
    python -m hestonlab.surrogate_value --quick      # smoke run
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
import hashlib
import json
import time
import numpy as np
from .heston import MonteCarloConfig
from .surrogate import TrainConfig, compare_to_benchmark, contract_fingerprint
from .surrogate_box import (
    VALUE_EXPERIMENT_BOX,
    cached_generate_in_box,
    price_points_parallel,
    train_in_box,
)
from .surrogate_data import synthetic_pilot_grid

OUT = Path("outputs/surrogate/value")
DOCS = Path("docs/results/surrogate_value")
LABEL_SEED = MonteCarloConfig().seed  # 2024: the training stream
TRADERS = ("oracle", "network", "mc_16k", "mc_4k")
MODEL_TRADERS = ("network", "mc_16k", "mc_4k")
# This experiment's training: the default network and settings, with a 6,000-epoch cap
# instead of 3,000 (the first run hit the 3,000 cap). Tim's TrainConfig default is unchanged.
EXPERIMENT_TRAIN = TrainConfig(max_epochs=6000)
RUNS = "runs"  # one sub-folder per training run: runs/epochs_<cap>/


@dataclass(frozen=True)
class ValueConfig:
    train_draws: int = 16384
    test_draws: int = 2048
    eval_points: int = 512  # accuracy points = PnL scenarios = first test draws
    label_paths: int = 16384
    cheap_paths: int = 4096
    truth_paths: int = 262144
    # Unused anywhere else (benchmarks use +1..4 million).
    truth_seeds: tuple[int, int] = (LABEL_SEED + 5_000_000, LABEL_SEED + 6_000_000)
    train_sobol_seed: int = 0
    test_sobol_seed: int = 1
    market_noise: float = 0.07  # USD/bbl, standard deviation
    threshold: float = 0.03  # USD/bbl mispricing needed to trade
    cost_per_trade: float = 0.01  # USD/bbl
    multiplier: float = 1000.0  # barrels per contract
    market_seed: int = 20240614
    bootstrap_resamples: int = 10000
    bootstrap_seed: int = 7
    timing_batch: int = 1000
    timing_repeats: int = 20
    workers: int | None = None
    train: TrainConfig = field(default_factory=lambda: EXPERIMENT_TRAIN)

    def __post_init__(self):
        if len(set(self.truth_seeds)) != 2 or LABEL_SEED in self.truth_seeds:
            raise ValueError("Truth needs two seeds, different from each other and the labels")
        if not 0 < self.eval_points <= self.test_draws:
            raise ValueError("Evaluation points must come from the test draws")

    @property
    def label_mc(self):
        return MonteCarloConfig(paths=self.label_paths, seed=LABEL_SEED)

    @property
    def cheap_mc(self):
        return MonteCarloConfig(paths=self.cheap_paths, seed=LABEL_SEED)

    @property
    def truth_mc(self):
        return [MonteCarloConfig(paths=self.truth_paths, seed=s) for s in self.truth_seeds]


QUICK = ValueConfig(
    train_draws=512, test_draws=128, eval_points=64, label_paths=2048, cheap_paths=512,
    truth_paths=8192, bootstrap_resamples=500, timing_batch=50, timing_repeats=3,
    train=TrainConfig(max_epochs=200, lr_patience=20, batch_size=32, learning_rate=1e-2),
)


# ---- stage 2: data, truth, accuracy ---------------------------------------------------


def build_data(contracts, cfg, out, log=print):
    data_dir = Path(out) / "data"
    train = cached_generate_in_box(
        data_dir / "train.npz", contracts, cfg.train_draws, cfg.label_mc,
        cfg.train_sobol_seed, VALUE_EXPERIMENT_BOX, cfg.workers, log,
    )
    test = cached_generate_in_box(
        data_dir / "test.npz", contracts, cfg.test_draws, cfg.label_mc,
        cfg.test_sobol_seed, VALUE_EXPERIMENT_BOX, cfg.workers, log,
    )
    return train, test


def cached_prices(path, contracts, params, mc, workers=None, log=print):
    """Monte Carlo at given points, saved; reused only for identical inputs."""
    path = Path(path)
    key = hashlib.sha256(
        (contract_fingerprint(contracts) + str(asdict(mc))
         + np.asarray(params, dtype=float).tobytes().hex()).encode()
    ).hexdigest()
    if path.is_file():
        saved = dict(np.load(path))
        if str(saved.get("key")) == key:
            log(f"reusing {path}")
            return saved
    prices, errors, timing = price_points_parallel(contracts, params, mc, workers)
    saved = {"prices": prices, "standard_errors": errors, "key": np.array(key),
             "core_seconds": np.array(timing["core_seconds"]),
             "wall_seconds": np.array(timing["wall_seconds"])}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **saved)
    log(f"priced {len(prices)} points at {mc.paths} paths (seed {mc.seed}): "
        f"{timing['wall_seconds']:.0f}s wall -> {path}")
    return saved


def evaluation_prices(model, contracts, test, cfg, out, log=print):
    """Truth (both streams) and every trader's prices at the evaluation points."""
    data_dir = Path(out) / "data"
    params = np.asarray(test["parameters"])[:cfg.eval_points]
    a, b = (
        cached_prices(data_dir / f"truth_{i}.npz", contracts, params, mc, cfg.workers, log)
        for i, mc in enumerate(cfg.truth_mc)
    )
    cheap = cached_prices(data_dir / "mc_4k.npz", contracts, params, cfg.cheap_mc,
                          cfg.workers, log)
    truth = (a["prices"] + b["prices"]) / 2
    prices = {
        "oracle": truth,
        "network": model(params),
        "mc_16k": np.asarray(test["prices"])[:cfg.eval_points],  # training stream, unseen points
        "mc_4k": cheap["prices"],
    }
    return params, a["prices"], b["prices"], prices


def accuracy_report(prices, stream_a, stream_b, params):
    """Each model trader's error against two-stream truth, benchmark noise removed."""
    return {
        "what": "error vs average of two independent truth runs, truth noise subtracted; "
        "surrogate approximation and Monte Carlo sampling error, not model error",
        "truth_stream_to_stream_rmse_usd_per_bbl": float(
            np.sqrt(np.mean((stream_a - stream_b) ** 2))
        ),
        **{name: compare_to_benchmark(prices[name], stream_a, stream_b, params[:, 0])
           for name in MODEL_TRADERS},
    }


# ---- stage 3: synthetic PnL ------------------------------------------------------------


def market_prices(truth, cfg):
    """Synthetic market: truth plus independent N(0, market_noise) per contract."""
    rng = np.random.default_rng(cfg.market_seed)
    return truth + rng.normal(0.0, cfg.market_noise, size=np.shape(truth))


def trade(model_prices, market, truth, threshold, cost, multiplier):
    """One contract per mispricing beyond the threshold, judged against truth.

    Sell when the market is above the model by more than the threshold, buy when it
    is below. PnL = position x (truth - market) x multiplier, minus the trading cost.
    A wrong-way trade is one whose direction is opposite to the true mispricing.
    Returns (positions, pnl in USD, wrong-way flags), all shaped like the prices.
    """
    edge = np.asarray(market) - np.asarray(model_prices)
    position = np.where(edge > threshold, -1, np.where(edge < -threshold, 1, 0))
    true_gain = position * (np.asarray(truth) - np.asarray(market))
    pnl = true_gain * multiplier - np.abs(position) * cost * multiplier
    return position, pnl, true_gain < 0


def bootstrap_totals(per_scenario, resamples, seed):
    """Paired bootstrap over scenarios: the same resampled scenarios for every trader,
    because one scenario's contracts share a parameter vector and its model errors."""
    names = list(per_scenario)
    n = len(per_scenario[names[0]])
    index = np.random.default_rng(seed).integers(0, n, size=(resamples, n))
    return {name: per_scenario[name][index].sum(axis=1) for name in names}


def interval(samples):
    low, high = np.percentile(samples, [2.5, 97.5])
    return [float(low), float(high)]


def pnl_report(prices, truth, cfg):
    """Every trader's PnL on the same synthetic market, with bootstrap 95% intervals."""
    market = market_prices(truth, cfg)
    traders, per_scenario = {}, {}
    for name in TRADERS:
        position, pnl, wrong = trade(prices[name], market, truth, cfg.threshold,
                                     cfg.cost_per_trade, cfg.multiplier)
        trades = int(np.abs(position).sum())
        per_scenario[name] = pnl.sum(axis=1)
        traders[name] = {
            "total_pnl_usd": float(pnl.sum()),
            "trades": trades,
            "buys": int((position > 0).sum()),
            "sells": int((position < 0).sum()),
            "wrong_way_trades": int(wrong.sum()),
            "wrong_way_share": float(wrong.sum() / trades) if trades else 0.0,
            "pnl_per_trade_usd": float(pnl.sum() / trades) if trades else 0.0,
        }
    oracle_total = traders["oracle"]["total_pnl_usd"]
    boot = bootstrap_totals(per_scenario, cfg.bootstrap_resamples, cfg.bootstrap_seed)
    for name in TRADERS:
        traders[name]["pct_of_oracle"] = 100 * traders[name]["total_pnl_usd"] / oracle_total
        traders[name]["pct_of_oracle_ci95"] = interval(100 * boot[name] / boot["oracle"])
        traders[name]["total_pnl_ci95_usd"] = interval(boot[name])

    def difference(a, b):
        samples = boot[a] - boot[b]
        return {
            "point_usd": traders[a]["total_pnl_usd"] - traders[b]["total_pnl_usd"],
            "ci95_usd": interval(samples),
            "point_pct_of_oracle": traders[a]["pct_of_oracle"] - traders[b]["pct_of_oracle"],
            "ci95_pct_of_oracle": interval(100 * samples / boot["oracle"]),
            "share_of_resamples_a_at_least_b": float((samples >= 0).mean()),
        }

    return {
        "what": "synthetic trading test: market = truth + N(0, noise); PnL judged against "
        "truth. Measures the trading cost of surrogate approximation error vs Monte "
        "Carlo, not Heston model error",
        "scenarios": int(truth.shape[0]),
        "contracts_per_scenario": int(truth.shape[1]),
        "opportunities": int(truth.size),
        "rules": {
            "market_noise_usd_per_bbl": cfg.market_noise,
            "threshold_usd_per_bbl": cfg.threshold,
            "cost_per_trade_usd_per_bbl": cfg.cost_per_trade,
            "barrels_per_contract": cfg.multiplier,
            "market_seed": cfg.market_seed,
            "bootstrap": f"{cfg.bootstrap_resamples} paired resamples of scenarios, "
                         f"seed {cfg.bootstrap_seed}",
        },
        "traders": traders,
        "differences": {
            "network_minus_mc_16k": difference("network", "mc_16k"),
            "network_minus_oracle": difference("network", "oracle"),
            "network_minus_mc_4k": difference("network", "mc_4k"),
        },
    }


# ---- stage 4: cost ---------------------------------------------------------------------


def _median_seconds(fn, repeats):
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return float(np.median(times))


def time_pricing(model, contracts, params, cfg):
    """Seconds per pricing call (one parameter vector -> all N prices), single-threaded,
    so times are core-seconds. Batch = timing_batch different parameter vectors."""
    import torch

    from .heston import HestonParameters, price_asian_options
    from .calibration import PARAMETER_NAMES

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        params = np.asarray(params, dtype=float)
        batch = np.resize(params, (cfg.timing_batch, 5))
        model(params[0])  # warm-up
        net_single = _median_seconds(lambda: model(params[0]), 10 * cfg.timing_repeats)
        net_batch = _median_seconds(lambda: model(batch), 5)
        out = {"network": {"single_call_seconds": net_single, "batch_seconds": net_batch}}
        for name, mc in (("mc_16k", cfg.label_mc), ("mc_4k", cfg.cheap_mc)):
            def one(row):
                price_asian_options(
                    contracts, HestonParameters(**dict(zip(PARAMETER_NAMES, row))), mc)
            single = _median_seconds(lambda: one(params[0]), cfg.timing_repeats)
            start = time.perf_counter()
            for row in batch:
                one(row)
            out[name] = {"single_call_seconds": single,
                         "batch_seconds": time.perf_counter() - start}
    finally:
        torch.set_num_threads(threads)
    for v in out.values():
        v["batch_seconds_per_pricing"] = v["batch_seconds"] / cfg.timing_batch
    for name in ("mc_16k", "mc_4k"):
        out[name]["single_call_slowdown_vs_network"] = (
            out[name]["single_call_seconds"] / out["network"]["single_call_seconds"])
        out[name]["batch_slowdown_vs_network"] = (
            out[name]["batch_seconds"] / out["network"]["batch_seconds"])
    return {
        "what": f"one pricing = one parameter vector priced on all {len(contracts)} contracts; "
        f"single thread; batch = {cfg.timing_batch} different parameter vectors",
        **out,
    }


def break_even(timing, upfront):
    """Calls after which the surrogate's upfront compute is repaid, per MC baseline.

    upfront: core-seconds per item. 'data + training' is the cost asked about; the
    held-out test set is listed separately because a careful user also pays for it.
    """
    net = timing["network"]["single_call_seconds"]
    fixed = upfront["training_data"] + upfront["training"]
    with_test = fixed + upfront["test_data"]
    out = {
        "what": "core-seconds; per-call costs are single-call, single-thread times",
        "upfront_core_seconds": upfront,
        "fixed_core_seconds_data_plus_training": fixed,
        "fixed_core_seconds_including_test_set": with_test,
        "network_seconds_per_call": net,
    }
    for name in ("mc_16k", "mc_4k"):
        saving = timing[name]["single_call_seconds"] - net
        out[name] = {
            "seconds_per_call": timing[name]["single_call_seconds"],
            "break_even_calls": float(fixed / saving) if saving > 0 else None,
            "break_even_calls_including_test_set":
                float(with_test / saving) if saving > 0 else None,
        }
    return out


# ---- stage 5: slide-ready plots ---------------------------------------------------------

# One colour per trader on every chart (validated categorical slots 1-3; oracle neutral).
COLORS = {"network": "#2a78d6", "mc_16k": "#eb6834", "mc_4k": "#1baf7a", "oracle": "#52514e"}
LABELS = {"oracle": "Oracle (knows truth)", "network": "Neural network",
          "mc_16k": "Monte Carlo, 16,384 paths", "mc_4k": "Monte Carlo, 4,096 paths"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _slide_axes(plt, ncols=1, figsize=(12, 6.75)):
    plt.rcParams.update({
        "font.size": 16, "axes.titlesize": 20, "axes.labelsize": 17,
        "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 15,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
        "axes.spines.right": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    })
    fig, axes = plt.subplots(1, ncols, figsize=figsize)
    for ax in np.atleast_1d(axes):
        ax.grid(True, color=GRID, linewidth=1)
        ax.set_axisbelow(True)
    return fig, axes


def _finish(fig, path, title, subtitle):
    lines = subtitle.count("\n") + 1
    fig.suptitle(title, fontsize=22, fontweight="bold", x=0.02, ha="left", y=0.985)
    fig.text(0.02, 0.93, subtitle, fontsize=15, color=MUTED, ha="left", va="top",
             linespacing=1.4)
    fig.tight_layout(rect=(0, 0, 1, 0.9 - 0.04 * (lines - 1)))
    fig.savefig(path, dpi=150)
    fig.clf()


def _usd(x):
    return f"{x:.3f}" if x < 0.1 else f"{x:.2f}"


def plot_pnl(pnl, path):
    from .surrogate import _pyplot

    plt = _pyplot()
    fig, ax = _slide_axes(plt)
    names = list(TRADERS)[::-1]
    t = pnl["traders"]
    values = [t[n]["pct_of_oracle"] for n in names]
    low = [t[n]["pct_of_oracle"] - t[n]["pct_of_oracle_ci95"][0] for n in names]
    high = [t[n]["pct_of_oracle_ci95"][1] - t[n]["pct_of_oracle"] for n in names]
    y = np.arange(len(names))
    ax.barh(y, values, height=0.55, color=[COLORS[n] for n in names])
    ax.errorbar(values, y, xerr=[low, high], fmt="none", ecolor=INK, capsize=6, linewidth=2)
    for yi, n, v in zip(y, names, values):
        hi = t[n]["pct_of_oracle_ci95"][1]
        ax.text(max(v, hi) + 2, yi, f"{v:.1f}%  (${t[n]['total_pnl_usd'] / 1e3:,.0f}k)",
                va="center", fontsize=15)
    ax.set_yticks(y, [LABELS[n] for n in names])
    ax.set_xlim(0, 130)
    ax.set_xlabel("Total PnL as % of the oracle's PnL")
    d = pnl["differences"]["network_minus_mc_16k"]
    _finish(fig, path, "Trading PnL by pricing model",
            f"{pnl['scenarios']} synthetic scenarios x {pnl['contracts_per_scenario']} contracts; "
            f"error bars are 95% bootstrap intervals\n"
            f"Network minus MC-16k: {d['point_pct_of_oracle']:+.1f} percentage points "
            f"[{d['ci95_pct_of_oracle'][0]:+.1f}, {d['ci95_pct_of_oracle'][1]:+.1f}]")


def plot_accuracy(accuracy, path):
    from .surrogate import _pyplot

    plt = _pyplot()
    fig, (typical, worst) = _slide_axes(plt, ncols=2)
    names = list(MODEL_TRADERS)
    x = np.arange(len(names))
    width = 0.38
    for offset, key, label, hatch in (
        (-width / 2, "rmse_noise_corrected_usd_per_bbl", "RMSE", None),
        (width / 2, "median_abs_error_usd_per_bbl", "Median", "//"),
    ):
        vals = [accuracy[n][key] for n in names]
        bars = typical.bar(x + offset, vals, width - 0.03, color=[COLORS[n] for n in names],
                           hatch=hatch, edgecolor=SURFACE, label=label)
        typical.bar_label(bars, [_usd(v) for v in vals], padding=3, fontsize=14)
    typical.set_xticks(x, ["Network", "MC 16k", "MC 4k"])
    typical.set_ylabel("USD/bbl")
    typical.set_title("Typical error: RMSE (solid), median (hatched)", fontsize=17, loc="left")
    vals = [accuracy[n]["max_abs_error_usd_per_bbl"] for n in names]
    bars = worst.bar(x, vals, 0.55, color=[COLORS[n] for n in names])
    worst.bar_label(bars, [_usd(v) for v in vals], padding=3, fontsize=14)
    worst.set_xticks(x, ["Network", "MC 16k", "MC 4k"])
    worst.set_ylabel("USD/bbl")
    worst.set_title("Worst single price", fontsize=17, loc="left")
    for ax in (typical, worst):
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    _finish(fig, path, "Pricing accuracy against truth",
            f"Error vs two independent 262,144-path runs (their noise removed from RMSE); "
            f"{accuracy['network']['points']} test points")


def plot_error_by_region(accuracy, path, threshold):
    from .surrogate import _pyplot

    plt = _pyplot()
    fig, ax = _slide_axes(plt)
    for name, marker in (("network", "o"), ("mc_16k", "s"), ("mc_4k", "^")):
        rows = accuracy[name]["by_v0_decile"]
        vol = [100 * np.sqrt(np.mean(r["v0_range"])) for r in rows]
        ax.plot(vol, [r["rmse_noise_corrected_usd_per_bbl"] for r in rows], marker=marker,
                markersize=9, linewidth=2.5, color=COLORS[name], label=LABELS[name])
    ax.axhline(threshold, color=MUTED, linestyle="--", linewidth=1.5)
    ax.text(ax.get_xlim()[1], threshold - 0.002, f"trading threshold {threshold:.2f} ",
            ha="right", va="top", color=MUTED, fontsize=14)
    ax.set_xlabel("Starting volatility, sqrt(v0), % (middle of each v0 decile)")
    ax.set_ylabel("RMSE vs truth, USD/bbl")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, loc="upper left")
    _finish(fig, path, "Error by volatility region",
            "Noise-corrected RMSE in each tenth of the test points, sorted by v0 "
            f"(about {accuracy['network']['points'] // 10} points each)")


def plot_break_even(cost, path):
    from .surrogate import _pyplot

    plt = _pyplot()
    fig, ax = _slide_axes(plt)
    fixed = cost["fixed_core_seconds_data_plus_training"]
    net = cost["network_seconds_per_call"]
    reach = max(cost[m]["break_even_calls"] or 1 for m in ("mc_16k", "mc_4k"))
    calls = np.logspace(0, np.log10(reach * 30), 300)
    ax.loglog(calls, (fixed + net * calls) / 3600, color=COLORS["network"], linewidth=3,
              label="Neural network (data + training, then per call)")
    offsets = {"mc_16k": (-20, 30), "mc_4k": (20, -60)}
    aligns = {"mc_16k": "right", "mc_4k": "left"}
    for m in ("mc_16k", "mc_4k"):
        per = cost[m]["seconds_per_call"]
        ax.loglog(calls, per * calls / 3600, color=COLORS[m], linewidth=3,
                  label=LABELS[m])
        n = cost[m]["break_even_calls"]
        if n:
            ax.plot(n, per * n / 3600, "o", markersize=12, color=COLORS[m],
                    markeredgecolor=INK, zorder=5)
            ax.annotate(f"break-even\n{n:,.0f} calls", (n, per * n / 3600),
                        textcoords="offset points", xytext=offsets[m], ha=aligns[m],
                        fontsize=14)
    ax.set_xlabel("Number of pricing calls (one call = all contracts, one parameter vector)")
    ax.set_ylabel("Cumulative compute, core-hours")
    ax.legend(frameon=False, loc="upper left")
    _finish(fig, path, "When does the surrogate pay for itself?",
            f"Upfront: {fixed / 3600:.2f} core-hours of training data and training; "
            f"then {net * 1e3:.2f} ms per call")


def plot_timing(timing, path):
    from .surrogate import _pyplot

    plt = _pyplot()
    fig, ax = _slide_axes(plt)
    names = ["network", "mc_4k", "mc_16k"]
    y = np.arange(len(names))
    h = 0.38
    single = [timing[n]["single_call_seconds"] for n in names]
    batch = [timing[n]["batch_seconds_per_pricing"] for n in names]
    b1 = ax.barh(y + h / 2, single, h - 0.03, color=[COLORS[n] for n in names],
                 label="single call")
    b2 = ax.barh(y - h / 2, batch, h - 0.03, color=[COLORS[n] for n in names], hatch="//",
                 edgecolor=SURFACE, label="per pricing, batch of 1,000")
    ax.bar_label(b1, [_duration(v) for v in single], padding=6, fontsize=14)
    ax.bar_label(b2, [_duration(v) for v in batch], padding=6, fontsize=14)
    ax.set_xscale("log")
    ax.set_xlim(min(batch) / 3, max(single) * 20)
    ax.set_yticks(y, [LABELS[n] for n in names])
    ax.set_xlabel("Seconds per pricing (log scale), one CPU thread")
    ax.legend(frameon=False, loc="lower right")
    _finish(fig, path, "Time per pricing",
            "Solid: one call. Hatched: average over a batch of 1,000 parameter vectors")


def _duration(seconds):
    if seconds >= 1:
        return f"{seconds:.2f} s"
    if seconds >= 1e-3:
        return f"{seconds * 1e3:.1f} ms"
    return f"{seconds * 1e6:.0f} µs"


# ---- run ----------------------------------------------------------------------------------


def run_experiment(cfg=ValueConfig(), out=OUT, docs=DOCS, contracts=None, log=print):
    """Whole experiment on the SYNTHETIC grid; writes outputs and the docs copy."""
    from .surrogate_pipeline import check_output_location

    contracts = tuple(contracts or synthetic_pilot_grid())
    out = check_output_location(out)
    log(f"surrogate value experiment: {len(contracts)} synthetic contracts -> {out}")
    train_data, test = build_data(contracts, cfg, out, log)
    log("training the default (manifold) network ...")
    start = time.process_time()
    model, history = train_in_box(train_data, contracts, VALUE_EXPERIMENT_BOX, cfg.train)
    training_core_seconds = time.process_time() - start
    model.save(out / "model.pt")
    params, a, b, prices = evaluation_prices(model, contracts, test, cfg, out, log)
    accuracy = accuracy_report(prices, a, b, params)
    pnl = pnl_report(prices, prices["oracle"], cfg)
    log("timing ...")
    timing = time_pricing(model, contracts, params, cfg)
    upfront = {
        "training_data": float(train_data["core_seconds"]),
        "training": training_core_seconds,
        "test_data": float(test["core_seconds"]),
    }
    cost = {"timing": timing, "break_even": break_even(timing, upfront),
            "wall_seconds": {"training_data": float(train_data["wall_seconds"]),
                             "test_data": float(test["wall_seconds"])}}
    summaries = {
        "accuracy.json": accuracy,
        "pnl.json": pnl,
        "cost.json": cost,
        "setup.json": {
            "status": "SYNTHETIC experiment; surrogate approximation error only, "
                      "not Heston model error; no real data used",
            "contracts": len(contracts),
            "architecture": model.network.architecture,
            "parameter_box": VALUE_EXPERIMENT_BOX.as_dict(),
            "config": {k: v for k, v in asdict(cfg).items() if k != "train"},
            "train_config": asdict(cfg.train),
            "training_epochs": len(history),
            "best_validation_mse_standardized":
                float(model.metadata["best_validation_mse_standardized"]),
            "final_learning_rate": float(history[-1]["lr"]),
            "label_monte_carlo": asdict(cfg.label_mc),
            "truth_monte_carlo": [asdict(m) for m in cfg.truth_mc],
            "cheap_monte_carlo": asdict(cfg.cheap_mc),
        },
    }
    for name, content in summaries.items():
        (out / name).write_text(json.dumps(content, indent=2), encoding="utf-8")
    archive_run(out, summaries)
    summaries["runs.json"] = compare_runs(out, primary=run_name(cfg.train))
    (out / "runs.json").write_text(json.dumps(summaries["runs.json"], indent=2), encoding="utf-8")
    publish(summaries, out, docs, log)
    return summaries


# ---- several training runs side by side --------------------------------------------------

SUMMARIES = ("accuracy.json", "pnl.json", "cost.json", "setup.json")


def run_name(train_config):
    return f"epochs_{train_config.max_epochs}"


def archive_run(out, summaries):
    """Keep this run's summaries and model in runs/epochs_<cap>/ so later runs with a
    different cap can be compared with it."""
    run = Path(out) / RUNS / run_name(TrainConfig(**summaries["setup.json"]["train_config"]))
    run.mkdir(parents=True, exist_ok=True)
    for name in SUMMARIES:
        (run / name).write_text(json.dumps(summaries[name], indent=2), encoding="utf-8")
    (run / "model.pt").write_bytes((Path(out) / "model.pt").read_bytes())
    return run


def _best_validation(run, setup):
    if "best_validation_mse_standardized" in setup:
        return setup["best_validation_mse_standardized"]
    model = run / "model.pt"  # runs saved before setup.json recorded it
    if not model.is_file():
        return None
    from .surrogate_box import load_in_box

    return float(load_in_box(model, VALUE_EXPERIMENT_BOX)
                 .metadata["best_validation_mse_standardized"])


def run_row(run):
    """One run's headline numbers, from its saved summaries."""
    run = Path(run)
    acc, pnl, cost, setup = (json.loads((run / n).read_text(encoding="utf-8"))
                             for n in SUMMARIES)
    be, up = cost["break_even"], cost["break_even"]["upfront_core_seconds"]
    t, d = pnl["traders"], pnl["differences"]
    cap = setup["train_config"]["max_epochs"]
    return {
        "run": run.name,
        "max_epochs": cap,
        "epochs_trained": setup["training_epochs"],
        "hit_epoch_cap": setup["training_epochs"] >= cap,
        "best_validation_mse_standardized": _best_validation(run, setup),
        "training_core_seconds": up["training"],
        "accuracy_usd_per_bbl": {
            n: {"rmse_noise_corrected": acc[n]["rmse_noise_corrected_usd_per_bbl"],
                "median": acc[n]["median_abs_error_usd_per_bbl"],
                "max": acc[n]["max_abs_error_usd_per_bbl"]} for n in MODEL_TRADERS},
        "pct_of_oracle": {n: {"point": t[n]["pct_of_oracle"],
                              "ci95": t[n]["pct_of_oracle_ci95"]} for n in TRADERS},
        "total_pnl_usd": {n: t[n]["total_pnl_usd"] for n in TRADERS},
        "wrong_way_trades": {n: t[n]["wrong_way_trades"] for n in MODEL_TRADERS},
        "network_minus_mc_16k": d["network_minus_mc_16k"],
        "network_minus_mc_4k": d["network_minus_mc_4k"],
        "break_even_calls": {n: be[n]["break_even_calls"] for n in ("mc_16k", "mc_4k")},
    }


def compare_runs(out, primary):
    rows = sorted((run_row(r) for r in (Path(out) / RUNS).iterdir()
                   if (r / "setup.json").is_file()), key=lambda r: r["max_epochs"])
    return {
        "what": "the same experiment with different training-epoch caps; data, truth, "
                "market and Monte Carlo traders are identical, only the network differs",
        "primary_run": primary,
        "primary_rule": "the run with the experiment's configured epoch cap, fixed before "
                        "it was trained (not the best-looking run)",
        "runs": rows,
    }


PLOTS = ("pnl_pct_of_oracle.png", "accuracy_vs_truth.png", "error_by_v0_decile.png",
         "break_even.png", "time_per_pricing.png")


def publish(summaries, out, docs=None, log=print):
    """Draw the five charts and the README from summaries; copy to docs if asked."""
    out = Path(out)
    accuracy, pnl, cost = (summaries[k] for k in ("accuracy.json", "pnl.json", "cost.json"))
    threshold = pnl["rules"]["threshold_usd_per_bbl"]
    draw = dict(zip(PLOTS, (
        lambda p: plot_pnl(pnl, p),
        lambda p: plot_accuracy(accuracy, p),
        lambda p: plot_error_by_region(accuracy, p, threshold),
        lambda p: plot_break_even(cost["break_even"], p),
        lambda p: plot_timing(cost["timing"], p),
    )))
    for name, fn in draw.items():
        fn(out / name)
    readme = render_readme(summaries)
    (out / "README.md").write_text(readme, encoding="utf-8")
    if docs is not None:
        docs = Path(docs)
        docs.mkdir(parents=True, exist_ok=True)
        for name in [*summaries, *PLOTS]:
            (docs / name).write_bytes((out / name).read_bytes())
        for run in sorted((out / RUNS).glob("*/setup.json")):  # summaries only, no model
            target = docs / RUNS / run.parent.name
            target.mkdir(parents=True, exist_ok=True)
            for name in SUMMARIES:
                (target / name).write_bytes((run.parent / name).read_bytes())
        (docs / "README.md").write_text(readme, encoding="utf-8")
        log(f"copied summaries, plots and README to {docs}")
    log(readme)


def replot(out=OUT, docs=DOCS, log=print):
    """Redraw charts and README from saved JSON summaries, without recomputing."""
    out = Path(out)
    names = (*SUMMARIES, "runs.json")
    summaries = {n: json.loads((out / n).read_text(encoding="utf-8"))
                 for n in names if (out / n).is_file()}
    publish(summaries, out, docs, log)
    return summaries


def render_readme(s):
    acc, pnl, cost = s["accuracy.json"], s["pnl.json"], s["cost.json"]
    t, d = pnl["traders"], pnl["differences"]
    be, tm = cost["break_even"], cost["timing"]
    nm = d["network_minus_mc_16k"]
    lo, hi = nm["ci95_pct_of_oracle"]
    if lo <= 0 <= hi:
        verdict = ("The difference between the network and Monte Carlo at 16,384 paths is "
                   "not statistically distinguishable from zero")
    elif lo > 0:
        verdict = "The network earned significantly more than Monte Carlo at 16,384 paths"
    else:
        verdict = "The network earned significantly less than Monte Carlo at 16,384 paths"

    def row(name):
        a = acc[name]
        return (f"| {LABELS[name]} | {a['rmse_noise_corrected_usd_per_bbl']:.4f} | "
                f"{a['median_abs_error_usd_per_bbl']:.4f} | {a['max_abs_error_usd_per_bbl']:.3f} |")

    def prow(name):
        x = t[name]
        ci = x["pct_of_oracle_ci95"]
        return (f"| {LABELS[name]} | {x['total_pnl_usd']:,.0f} | {x['pct_of_oracle']:.1f}% "
                f"[{ci[0]:.1f}, {ci[1]:.1f}] | {x['trades']:,} | {x['wrong_way_trades']:,} |")

    rules = pnl["rules"]
    setup = s["setup.json"]
    capped = setup["training_epochs"] >= setup["train_config"]["max_epochs"]
    training_note = (
        f"\n\nTraining ran the full {setup['training_epochs']:,} epochs without meeting the "
        "early-stopping rule, so the network may not have reached its best accuracy."
        if capped else "")
    return f"""# Does the neural surrogate cost PnL? (synthetic experiment)

**Synthetic data only.** No real market data was used. Every error below is
surrogate approximation error or Monte Carlo sampling error under the Heston
pricer. None of it is Heston model error.

## Setup

- {s['setup.json']['contracts']} synthetic AO calls (Aug-Oct 2024 averages, strikes $60-100).
- Narrow parameter box: v0 and theta 0.01-0.36 (10-60% volatility), kappa 0.1-5,
  sigma 0.05-1.5, rho -0.95 to 0.5.
- Network: the default manifold architecture, trained on 16,384 parameter draws priced
  with 16,384-path Monte Carlo ({setup['training_epochs']:,} epochs).
- Truth: the average of two independent 262,144-path Monte Carlo runs at
  {acc['network']['points']} held-out test points.
- Market: truth plus random noise of {rules['market_noise_usd_per_bbl']} USD/bbl per
  contract. Each trader sells when the market is above its model price by more than
  {rules['threshold_usd_per_bbl']} USD/bbl and buys when below, paying
  {rules['cost_per_trade_usd_per_bbl']} USD/bbl per trade. PnL is measured against
  truth, 1,000 barrels per contract.

## Accuracy against truth (USD/bbl)

| Pricer | RMSE (truth noise removed) | Median error | Worst error |
|---|---:|---:|---:|
{row('network')}
{row('mc_16k')}
{row('mc_4k')}

## Trading PnL ({pnl['scenarios']} scenarios x {pnl['contracts_per_scenario']} contracts)

| Trader | Total PnL (USD) | % of oracle [95% CI] | Trades | Wrong-way trades |
|---|---:|---:|---:|---:|
{prow('oracle')}
{prow('network')}
{prow('mc_16k')}
{prow('mc_4k')}

{verdict}: network minus MC-16k is {nm['point_usd']:+,.0f} USD
({nm['point_pct_of_oracle']:+.1f} percentage points of oracle PnL), 95% interval
{nm['ci95_usd'][0]:+,.0f} to {nm['ci95_usd'][1]:+,.0f} USD. The network gives up
{-d['network_minus_oracle']['point_usd']:,.0f} USD to the oracle (95% interval
{-d['network_minus_oracle']['ci95_usd'][1]:,.0f} to {-d['network_minus_oracle']['ci95_usd'][0]:,.0f} USD).{training_note}

{render_runs(s.get("runs.json"))}## Cost (one CPU thread)

- One pricing call: network {_duration(tm['network']['single_call_seconds'])},
  MC 4,096 paths {_duration(tm['mc_4k']['single_call_seconds'])},
  MC 16,384 paths {_duration(tm['mc_16k']['single_call_seconds'])}.
- Batch of 1,000: network {_duration(tm['network']['batch_seconds'])} in total,
  MC 16,384 paths {_duration(tm['mc_16k']['batch_seconds'])}.
- Upfront cost of the network: {be['fixed_core_seconds_data_plus_training'] / 3600:.2f}
  core-hours (training data and training).
- It pays for itself after about {be['mc_16k']['break_even_calls']:,.0f} calls compared
  with 16,384-path Monte Carlo, or {be['mc_4k']['break_even_calls']:,.0f} compared with
  4,096 paths. Adding the held-out test set raises these to
  {be['mc_16k']['break_even_calls_including_test_set']:,.0f} and
  {be['mc_4k']['break_even_calls_including_test_set']:,.0f}.

## Files

`accuracy.json`, `pnl.json`, `cost.json`, `setup.json` and five PNG charts for the reported
run; `runs.json` puts every training run side by side, and `runs/epochs_<cap>/` keeps each
run's own summaries. Regenerate
with `python -m hestonlab.surrogate_value`; the full outputs, model and cached data stay
in `outputs/surrogate/value/` (git-ignored).
"""


def render_runs(runs):
    """README section putting every training run side by side (empty for one run)."""
    if not runs or len(runs["runs"]) < 2:
        return ""
    rows = runs["runs"]

    def line(label, fn):
        return f"| {label} | " + " | ".join(fn(r) for r in rows) + " |"

    def pct(r, name):
        x = r["pct_of_oracle"][name]
        return f"{x['point']:.1f}% [{x['ci95'][0]:.1f}, {x['ci95'][1]:.1f}]"

    def gap(r, key):
        g = r[key]
        lo, hi = g["ci95_pct_of_oracle"]
        return f"{g['point_pct_of_oracle']:+.2f} pts [{lo:+.2f}, {hi:+.2f}]"

    def gap_usd(r):
        g = r["network_minus_mc_16k"]
        return f"{g['point_usd']:+,.0f} [{g['ci95_usd'][0]:+,.0f}, {g['ci95_usd'][1]:+,.0f}]"

    def net(r, key):
        return f"{r['accuracy_usd_per_bbl']['network'][key]:.4f}"

    def val(r):
        v = r["best_validation_mse_standardized"]
        return "n/a" if v is None else f"{v:.2e}"

    def calls(r):
        c = r["break_even_calls"]["mc_16k"]
        return "never" if c is None else f"{c:,.0f}"

    header = "| | " + " | ".join(
        f"{r['max_epochs']:,}-epoch cap" + (" (shown above)" if r["run"] == runs["primary_run"]
                                            else "") for r in rows) + " |"
    table = "\n".join([
        header,
        "|---|" + "---:|" * len(rows),
        line("Epochs trained", lambda r: f"{r['epochs_trained']:,}"
             + (" (hit cap)" if r["hit_epoch_cap"] else " (stopped early)")),
        line("Best validation loss (standardized MSE)", val),
        line("Network RMSE vs truth (USD/bbl)", lambda r: net(r, "rmse_noise_corrected")),
        line("Network median error (USD/bbl)", lambda r: net(r, "median")),
        line("Network worst error (USD/bbl)", lambda r: net(r, "max")),
        line("Network wrong-way trades", lambda r: f"{r['wrong_way_trades']['network']:,}"),
        line("Network PnL, % of oracle [95% CI]", lambda r: pct(r, "network")),
        line("MC-16k PnL, % of oracle [95% CI]", lambda r: pct(r, "mc_16k")),
        line("Network minus MC-16k, USD [95% CI]", gap_usd),
        line("Network minus MC-16k, points [95% CI]", lambda r: gap(r, "network_minus_mc_16k")),
        line("Network minus MC-4k, points [95% CI]", lambda r: gap(r, "network_minus_mc_4k")),
        line("Training (core-hours)", lambda r: f"{r['training_core_seconds'] / 3600:.2f}"),
        line("Break-even calls vs MC-16k", calls),
    ])
    return f"""## Training length: runs side by side

The {rows[0]['max_epochs']:,}-epoch run hit its cap, so the experiment was repeated with a
{rows[-1]['max_epochs']:,}-epoch cap. Everything else is identical: same data, settings, training seed, truth, market and
Monte Carlo traders. Only the network differs. Both runs are reported whichever looks
better. The tables and charts above show the {int(runs['primary_run'].replace('epochs_', '')):,}-epoch run, which was chosen as the reported run before it was trained.

{table}

"""


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--quick", action="store_true",
                        help="tiny smoke run; does not touch docs/")
    parser.add_argument("--no-docs", action="store_true", help="skip the docs/ copy")
    parser.add_argument("--replot", action="store_true",
                        help="redraw charts and README from saved summaries only")
    args = parser.parse_args(argv)
    cfg = QUICK if args.quick else ValueConfig()
    out = args.out if not args.quick or args.out != OUT else OUT.parent / "value-quick"
    docs = None if args.quick or args.no_docs else DOCS
    if args.replot:
        return replot(out, docs)
    return run_experiment(cfg, out, docs)


if __name__ == "__main__":
    main()
