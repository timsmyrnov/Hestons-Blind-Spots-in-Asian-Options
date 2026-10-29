"""Stable researcher interface. No dense imputation and no silent look-ahead."""

from pathlib import Path
import os
from datetime import date
import duckdb
import pyarrow as pa

DEFAULT_ROOT = Path(
    os.environ.get(
        "HESTONLAB_DATA_ROOT",
        str(Path(__file__).resolve().parents[3] / "outputs/HestonLab_Asian_Data"),
    )
)
ALLOWED = {
    "call_put",
    "contract_month",
    "min_strike",
    "max_strike",
    "eligible_only",
    "require_liquidity",
    "price_type",
}


def load_ao_panel(start_date, end_date, filters=None, root=None):
    """Inclusive dates; returns structured pandas tables, never filled arrays.

    Settlement labels are corrected retrospective observations. Their publication
    timestamps are exposed. Fixing state is separately screened at that timestamp.
    OI/volume availability time is not established and is not a causal covariate.
    """
    start_date = date.fromisoformat(str(start_date)).isoformat()
    end_date = date.fromisoformat(str(end_date)).isoformat()
    if start_date > end_date:
        raise ValueError("start_date after end_date")
    root = Path(root or DEFAULT_ROOT)
    filters = dict(filters or {})
    unknown = set(filters) - ALLOWED
    if unknown:
        raise ValueError(f"Unknown filters: {sorted(unknown)}")
    where = ["observation_date BETWEEN ?::DATE AND ?::DATE"]
    values = [start_date, end_date]
    for key in ["call_put", "contract_month", "price_type"]:
        if key in filters:
            where.append(f"{key} = ?")
            values.append(filters[key])
    for key, op in [("min_strike", ">="), ("max_strike", "<=")]:
        if key in filters:
            where.append(f"strike {op} ?")
            values.append(filters[key])
    if filters.get("eligible_only", True):
        where.append("research_eligible")
    if filters.get("require_liquidity", False):
        where.append("(coalesce(open_interest,0)>0 OR coalesce(trade_volume,0)>0)")
    con = duckdb.connect()
    try:
        con.read_parquet(str(root / "ao_daily_research_panel.parquet")).create_view("panel")
        panel = con.execute(
            "SELECT * FROM panel WHERE "
            + " AND ".join(where)
            + " ORDER BY observation_date, expiration_date, call_put, strike, contract_id",
            values,
        ).df()
        con.read_parquet(str(root / "ao_contracts.parquet")).create_view("contracts")
        con.register("selected", pa.Table.from_pandas(panel[["contract_id"]].drop_duplicates()))
        contracts = con.sql(
            "SELECT c.* FROM contracts c SEMI JOIN selected s USING(contract_id) ORDER BY contract_id"
        ).df()
        futures = None
        revision_file = root / "cl_futures_settlement_revisions.parquet"
        if revision_file.exists() and not panel.empty:
            con.read_parquet(str(revision_file)).create_view("revisions")
            con.register(
                "selected_cutoffs",
                pa.Table.from_pandas(
                    panel[
                        [
                            "observation_date",
                            "contract_month",
                            "cutoff_timestamp",
                            "cutoff_timestamp_raw",
                        ]
                    ].drop_duplicates()
                ),
            )
            futures = con.sql("""SELECT * EXCLUDE(rn) FROM (SELECT c.observation_date,
               c.contract_month averaging_month,r.*,
               row_number() OVER(PARTITION BY c.observation_date,c.contract_month,r.contract_id
               ORDER BY r.publication_timestamp_raw DESC,r.source_sequence DESC) rn
               FROM selected_cutoffs c JOIN revisions r ON r.trade_date=c.observation_date
               AND r.publication_timestamp_raw<=c.cutoff_timestamp_raw AND r.is_final AND r.root='CL')
               WHERE rn=1 AND update_action=1 ORDER BY observation_date,averaging_month,expiration""").df()
        schedule = None
        if (root / "ao_fixing_schedule.parquet").exists():
            schedule = (
                con.read_parquet(str(root / "ao_fixing_schedule.parquet"))
                .project("* EXCLUDE(fixing_price,publication_timestamp,source_file)")
                .df()
            )
        return dict(
            observations=panel,
            contracts=contracts,
            futures=futures,
            fixing_schedule=schedule,
            conventions={
                "price_unit": "USD/barrel",
                "multiplier": 1000,
                "coordinate_order": ["expiration_date", "call_put", "strike", "contract_id"],
                "labels": "latest final retrospective; not causal bid/ask",
                "rates": "not supplied; model owner must specify discounting",
            },
        )
    finally:
        con.close()


def build_ao_snapshot(observation_date, contract_filter=None, root=None):
    """Deterministic native-contract vector plus masks; retains provenance."""
    result = load_ao_panel(observation_date, observation_date, contract_filter, root)
    table = result["observations"]
    result["coordinate_ids"] = table.contract_id.tolist()
    result["price_vector"] = table.settlement_price.to_numpy()
    return result
