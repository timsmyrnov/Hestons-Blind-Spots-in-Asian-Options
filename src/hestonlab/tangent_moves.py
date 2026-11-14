"""How much of the market's day-to-day move Heston can follow, using its tangent plane.

For each day t, with the day's best Heston fit p_t on the Heston manifold:

    move      = market grid prices at t+h minus at t          (21 options, USD/bbl)
    tangent   = the part of the move lying in T_{p_t} M_h      (Heston can make it by
                changing its five parameters, to first order)
    normal    = the rest                                        (Heston cannot)
    followed  = |tangent|^2 / |move|^2

References, using the same information as Heston's plane (nothing after day t): the
market's own trailing plane, the top principal directions of its moves over the
previous 120 days; and chance, 5/21 for a random 5-dimensional plane. The best fixed
plane fitted on all moves (future included) is kept only as a hindsight figure. Longer horizons
h dilute day-to-day quote noise, which always lands mostly in the normal part.

    python -m hestonlab.tangent_moves --run outputs/market_manifold/archive \
        --root "$HESTONLAB_DATA_ROOT" --start 2022-01-01 --end 2025-12-31
"""

from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

from .manifold_movie import _jacobian, grid_scale, heston_cloud
from .market_manifold import GridSpec, build_panel, load_archive_days, nearest_point
from .surrogate import Surrogate


def tangent_basis(heston, unit, weight, rank=5):
    """Orthonormal basis (21 x rank) of Heston's tangent plane at `unit`, in USD/bbl,
    keeping its `rank` strongest directions. The fifth is the near-flat kappa-theta
    direction: the surrogate's slope there disagrees with Monte Carlo, and moving along
    it takes very large parameter changes, so rank 4 is the strict reading."""
    jac = _jacobian(heston, unit) * np.asarray(weight, dtype=float)[:, None]
    u, s, _ = np.linalg.svd(jac, full_matrices=False)
    keep = (s > 1e-10 * s.max()) & (np.arange(len(s)) < rank)
    return u[:, keep]


def split_move(basis, move):
    """(tangent part, normal part) of a move, given an orthonormal tangent basis."""
    move = np.asarray(move, dtype=float)
    tangent = basis @ (basis.T @ move)
    return tangent, move - tangent


def fit_days(heston, panel, contracts, dates, starts=6, log=print):
    """Best Heston fit (unit parameters) and USD weights for each date."""
    cloud = heston_cloud(heston, 2048)
    units, weights = [], []
    for i, date in enumerate(dates):
        weight = grid_scale(contracts[date], panel.grid)
        y = panel.vectors[panel.dates.index(date)]
        units.append(nearest_point(heston, y, weight, *cloud, starts)["unit"])
        weights.append(weight)
        if log and (i + 1) % 200 == 0:
            log(f"  fitted {i + 1}/{len(dates)} days")
    return np.array(units), np.array(weights)


def trailing_basis(moves, t, horizon, window, rank):
    """Top `rank` directions of the market's own moves known by day t (moves that ended
    on or before t, the last `window` of them): the market's reference plane, chosen
    with the same information as Heston's (nothing after day t). None if too few."""
    stop = t - horizon + 1  # moves[s] ends on day s + horizon <= t
    past = moves[max(0, stop - window):max(0, stop)]
    if len(past) < max(2 * rank, window // 2):
        return None
    return np.linalg.svd(past, full_matrices=False)[2][:rank].T


def moves_table(heston, panel, dates, units, weights, horizon=1, window=120):
    """One row per day t with a day t+horizon in `dates`: sizes, the share Heston's
    tangent plane follows, and the share the market's trailing plane follows."""
    rows = []
    index = {d: panel.dates.index(d) for d in dates}
    moves = np.array([(panel.vectors[index[dates[t + horizon]]] - panel.vectors[index[dates[t]]])
                      * weights[t] for t in range(len(dates) - horizon)])
    for t in range(len(dates) - horizon):
        a, b = dates[t], dates[t + horizon]
        move = moves[t]
        total = float(move @ move)
        row = {"date": a, "to": b, "horizon": horizon, "move_rms": np.sqrt(total / len(move)),
               "at_bound": bool(np.any((units[t] < 1e-6) | (units[t] > 1 - 1e-6)))}
        for rank in (5, 4):
            tangent, normal = split_move(tangent_basis(heston, units[t], weights[t], rank), move)
            row[f"tangent_rms_{rank}"] = float(np.linalg.norm(tangent) / np.sqrt(len(move)))
            row[f"normal_rms_{rank}"] = float(np.linalg.norm(normal) / np.sqrt(len(move)))
            row[f"followed_{rank}"] = float(tangent @ tangent / total) if total > 0 else np.nan
            ref = trailing_basis(moves, t, horizon, window, rank)
            if ref is None:
                row[f"trailing_tangent_rms_{rank}"] = row[f"trailing_{rank}"] = np.nan
            else:
                tangent, _ = split_move(ref, move)
                row[f"trailing_tangent_rms_{rank}"] = float(np.linalg.norm(tangent)
                                                            / np.sqrt(len(move)))
                row[f"trailing_{rank}"] = float(tangent @ tangent / total) if total > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def best_fixed_share(moves, rank=5):
    """Share of all moves' squared size caught by their own top `rank` principal
    components (in-sample, uncentred): the best any fixed `rank`-dim subspace can do."""
    m = np.asarray(moves, dtype=float)
    s = np.linalg.svd(m, compute_uv=False) ** 2
    return float(s[:rank].sum() / s.sum())


def summarize(table, moves, n_prices):
    """Pooled shares (sum of squares, so big moves count more) and day medians, for the
    full tangent plane (5) and its four reliable directions (4)."""
    out = {"days": int(len(table)),
           "move_rms_cents_median": round(100 * float(table.move_rms.median()), 3),
           "days_at_parameter_limit": int(table.at_bound.sum())}
    for rank in (5, 4):
        pooled = float((table[f"tangent_rms_{rank}"] ** 2).sum() / (table.move_rms ** 2).sum())
        fair = table[table[f"trailing_{rank}"].notna()]  # days the market reference exists
        size = (fair.move_rms ** 2).sum()
        out[f"rank_{rank}"] = {
            "followed_pooled": round(pooled, 4),
            "followed_median_day": round(float(table[f"followed_{rank}"].median()), 4),
            "chance": round(rank / n_prices, 4),
            "unfollowed_rms_cents_median":
                round(100 * float(table[f"normal_rms_{rank}"].median()), 3),
            # Same days, same information (nothing after day t), side by side:
            "same_days": int(len(fair)),
            "heston_pooled_same_days": round(float((fair[f"tangent_rms_{rank}"] ** 2).sum()
                                                   / size), 4) if size else None,
            "market_trailing_pooled": round(float((fair[f"trailing_tangent_rms_{rank}"] ** 2).sum()
                                                  / size), 4) if size else None,
            "market_trailing_median_day": round(float(fair[f"trailing_{rank}"].median()), 4)
            if len(fair) else None,
            # Hindsight only (fitted on all moves, future included): not a fair rival.
            "hindsight_best_fixed": round(best_fixed_share(moves, rank), 4)}
    return out


def run(run_dir, root, start, end, horizons=(1, 5), log=print):
    run_dir = Path(run_dir)
    report = json.loads((run_dir / "report.json").read_text())
    grid = GridSpec(**{k: tuple(v) if isinstance(v, list) else v
                       for k, v in report["grid"].items()})
    heston = Surrogate.load(run_dir / "heston" / "model.pt")
    days = load_archive_days(root, start=start, end=end)
    contracts = {str(pd.Timestamp(d).date()): c for d, c, _ in days}
    panel = build_panel(days, grid)
    dates = [d for d in panel.dates if start <= d <= end]
    folder = run_dir / "tangent"
    folder.mkdir(parents=True, exist_ok=True)
    cache = folder / f"fits_{start}_to_{end}.npz"
    if cache.exists() and list(np.load(cache)["dates"]) == dates:
        units, weights = np.load(cache)["units"], np.load(cache)["weights"]
    else:
        log(f"{len(dates)} complete days; fitting Heston to each ...")
        units, weights = fit_days(heston, panel, contracts, dates, log=log)
        np.savez(cache, dates=np.array(dates), units=units, weights=weights)
    out = {"units": "USD/bbl per option; move = change in the 21 grid prices",
           "weights": "equal USD per option", "start": start, "end": end, "horizons": {}}
    tables = []
    for h in horizons:
        table = moves_table(heston, panel, dates, units, weights, h)
        index = [panel.dates.index(d) for d in dates]
        moves = np.array([(panel.vectors[index[t + h]] - panel.vectors[index[t]]) * weights[t]
                          for t in range(len(dates) - h)])
        summary = {"all": summarize(table, moves, grid.size)}
        for year, part in table.groupby(table.date.str[:4]):
            keep = table.date.str[:4] == year
            summary[year] = summarize(part, moves[keep.to_numpy()], grid.size)
        out["horizons"][str(h)] = summary
        tables.append(table)
        for rank in (5, 4):
            a = summary["all"][f"rank_{rank}"]
            log(f"h={h} rank {rank}: Heston {a['heston_pooled_same_days']:.1%} vs market's own "
                f"trailing plane {a['market_trailing_pooled']:.1%} on {a['same_days']} days "
                f"(chance {a['chance']:.0%})")
    pd.concat(tables).to_csv(folder / f"moves_{start}_to_{end}.csv", index=False)
    path = folder / f"moves_{start}_to_{end}.json"
    path.write_text(json.dumps(out, indent=2))
    return path, out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="market_manifold output folder")
    parser.add_argument("--root", type=Path, required=True, help="private AO archive")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 5])
    args = parser.parse_args(argv)
    try:
        path, _ = run(args.run, args.root, args.start, args.end, tuple(args.horizons))
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"Tangent moves unavailable: {exc}\n")
    print(path)
    return path


if __name__ == "__main__":
    main()
