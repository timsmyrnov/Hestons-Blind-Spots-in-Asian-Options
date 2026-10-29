"""Measured artifact checks; raises on identity, units or price corruption."""

from pathlib import Path
import json
from datetime import datetime, timezone
import duckdb


def validate(root):
    root = Path(root)
    con = duckdb.connect()
    con.read_parquet(str(root / "ao_daily_research_panel.parquet")).create_view("p")
    checks = {
        "duplicate_coordinates": """SELECT count(*) FROM (SELECT observation_date,contract_id,
          count(*) n FROM p GROUP BY ALL HAVING n>1)""",
        "bad_units": "SELECT count(*) FROM p WHERE price_unit!='USD/barrel' OR currency!='USD'",
        "bad_eligible_prices": """SELECT count(*) FROM p WHERE research_eligible AND
         (settlement_price IS NULL OR settlement_price<=0 OR settlement_status!='final')""",
        "invalid_sides": "SELECT count(*) FROM p WHERE call_put NOT IN ('C','P')",
        "missing_expiration": "SELECT count(*) FROM p WHERE expiration_date IS NULL",
        "future_cutoff_fixings": """SELECT count(*) FROM p WHERE known_fixing_count<0
          OR missing_elapsed_fixings<0""",
        "wrong_eligible_state": """SELECT count(*) FROM p WHERE research_eligible
          AND NOT underlying_state_complete""",
    }
    result = {name: con.sql(sql).fetchone()[0] for name, sql in checks.items()}
    result["run_utc"] = datetime.now(timezone.utc).isoformat()
    result["passed"] = all(result[name] == 0 for name in checks)
    (root / "evidence/research_validation.json").write_text(json.dumps(result, indent=2))
    con.close()
    if not result["passed"]:
        raise ValueError(result)
    return result
