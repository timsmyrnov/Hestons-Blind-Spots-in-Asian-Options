"""Calendar-separated pilot; contract universe is selected using training data only."""

from pathlib import Path
import json
import duckdb
from .pipeline import write, literal


def build_pilot_split(root):
    root = Path(root)
    out = root / "pilot_june_july_2024"
    out.mkdir(exist_ok=True)
    c = duckdb.connect()
    c.read_parquet(str(root / "ao_calibration_inputs_provisional.parquet")).create_view("p")
    c.execute(
        "CREATE TEMP TABLE train AS SELECT * FROM p WHERE observation_date BETWEEN DATE '2024-06-01' AND DATE '2024-06-30' AND prototype_input_complete"
    )
    c.execute(
        "CREATE TEMP TABLE test AS SELECT * FROM p WHERE observation_date BETWEEN DATE '2024-07-01' AND DATE '2024-07-31' AND prototype_input_complete"
    )
    # Maturities are fixed in advance and survive the entire test period.
    c.execute("""CREATE TEMP TABLE coordinates AS SELECT contract_id,contract_month,strike,call_put,
      count(DISTINCT observation_date) training_dates,
      count(DISTINCT observation_date)::DOUBLE/(SELECT count(DISTINCT observation_date) FROM train) training_fraction
      FROM train WHERE contract_month IN ('2024-08','2024-09','2024-10') AND strike BETWEEN 60 AND 100
      AND strike%5=0 GROUP BY ALL HAVING training_fraction>=0.8""")
    c.execute(
        f"COPY (SELECT * FROM coordinates ORDER BY contract_month,call_put,strike) TO {literal(out / 'training_selected_coordinates.csv')} (HEADER)"
    )
    stats = {}
    for name in ["train", "test"]:
        write(
            c,
            f"SELECT * FROM {name} ORDER BY observation_date,contract_id",
            out / f"{name}_native_inputs.parquet",
        )
        write(
            c,
            f"SELECT p.* FROM {name} p SEMI JOIN coordinates USING(contract_id) ORDER BY observation_date,contract_id",
            out / f"{name}_fixed_coordinate_observations.parquet",
        )
        # Long grid makes every missing coordinate explicit, never a zero price.
        grid = f"""SELECT d.observation_date,k.contract_id,k.contract_month,k.strike,k.call_put,
          p.settlement_price,p.contract_id IS NOT NULL observed_price
          FROM (SELECT DISTINCT observation_date FROM {name}) d CROSS JOIN coordinates k
          LEFT JOIN {name} p ON p.observation_date=d.observation_date AND p.contract_id=k.contract_id
          ORDER BY d.observation_date,k.contract_id"""
        c.execute(f"COPY ({grid}) TO {literal(out / f'{name}_price_grid.csv')} (HEADER)")
        c.execute(
            f"COPY (SELECT observation_date,count(*) observed_coordinates FROM {name} SEMI JOIN coordinates USING(contract_id) GROUP BY observation_date ORDER BY 1) TO {literal(out / f'{name}_daily_coverage.csv')} (HEADER)"
        )
        stats[name] = (
            c.sql(f"""SELECT count(*) records,count(DISTINCT observation_date) n_dates,
          min(observation_date)::VARCHAR first_date,max(observation_date)::VARCHAR last_date FROM {name}""")
            .df()
            .to_dict("records")[0]
        )
        stats[name]["grid_observed_rows"] = c.sql(
            f"SELECT count(*) FROM {name} SEMI JOIN coordinates USING(contract_id)"
        ).fetchone()[0]
        stats[name]["grid_expected_rows"] = (
            stats[name]["n_dates"] * c.sql("SELECT count(*) FROM coordinates").fetchone()[0]
        )
    overlap = c.sql("SELECT count(*) FROM train SEMI JOIN test USING(observation_date)").fetchone()[
        0
    ]
    if overlap:
        raise ValueError("Training/test dates overlap")
    stats.update(
        date_overlap=overlap,
        coordinate_count=c.sql("SELECT count(*) FROM coordinates").fetchone()[0],
        selection="August/September/October 2024 contracts; strikes 60 to 100 in $5 steps; >=80% June coverage; no July selection",
        price_imputation="none",
        purpose="retrospective held-out market-price pilot, not a causal trading backtest",
        fully_validated=False,
        test_policy="Freeze pricing convention, coordinate selection, surrogate weights, parameter search bounds, scaling and hyperparameters before scoring July. Frozen-parameter transfer and per-day manifold projection must be reported separately.",
        caveats=[
            "Terminal feed final flags unresolved.",
            "Historical metadata/calendar and current revised rates are retrospective.",
            "Some underlying fixing periods and contract IDs span both months; dates are disjoint, but economic exposures are not independent.",
            "No Heston pricer, parameter estimates or synthetic training prices are fabricated.",
        ],
    )
    (out / "split_manifest.json").write_text(json.dumps(stats, indent=2))
    c.close()
    return stats
