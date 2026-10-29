"""Empirical coordinate-population and terminal payoff diagnostics."""

from pathlib import Path
import duckdb
from .pipeline import literal


def run(root):
    root = Path(root)
    con = duckdb.connect()
    con.execute("SET enable_progress_bar=false")
    con.execute("SET memory_limit='2GB'")
    con.read_parquet(str(root / "ao_daily_research_panel.parquet")).create_view("p")
    con.execute("""CREATE TEMP TABLE days AS SELECT observation_date,count(*) coordinates,
       lag(observation_date) OVER(ORDER BY observation_date) previous_date,
       lag(count(*)) OVER(ORDER BY observation_date) previous_coordinates
       FROM p WHERE mark_eligible GROUP BY observation_date""")
    query = """SELECT d.*,count(prev.contract_id) common_coordinates,
       count(prev.contract_id)::DOUBLE/nullif(d.coordinates+d.previous_coordinates-count(prev.contract_id),0) coordinate_jaccard
       FROM days d JOIN p cur ON cur.observation_date=d.observation_date AND cur.mark_eligible
       LEFT JOIN p prev ON prev.observation_date=d.previous_date
       AND prev.contract_id=cur.contract_id AND prev.mark_eligible
       GROUP BY d.observation_date,d.coordinates,d.previous_date,d.previous_coordinates
       ORDER BY d.observation_date"""
    con.execute(
        f'COPY ({query}) TO {literal(root / "ao_coordinate_population.csv")} (HEADER,DELIMITER ",")'
    )
    recon = root / "evidence/monthly_average_reconciliation.csv"
    if recon.exists():
        con.read_csv(str(recon)).create_view("recon")
        payoff = """SELECT p.observation_date,p.contract_id,p.contract_month,p.strike,p.call_put,
          p.settlement_price,r.arithmetic_average,
          greatest(CASE WHEN p.call_put='C' THEN r.arithmetic_average-p.strike
          ELSE p.strike-r.arithmetic_average END,0) terminal_payoff,
          p.settlement_price-terminal_payoff difference
          FROM p JOIN recon r ON p.contract_month=r.averaging_month
          WHERE p.observation_date=p.expiration_date AND r.verified
          AND p.settlement_status IN ('final','zero_price')"""
        con.execute(
            f'COPY ({payoff}) TO {literal(root / "evidence/ao_expiry_payoff_checks.csv")} (HEADER,DELIMITER ",")'
        )
    con.close()


def reconcile_missing_dates(root):
    """Distinguish absent AO on observed CL days from unverified closure days."""
    import pandas as pd

    root = Path(root)
    file = root / "ao_missing_dates.csv"
    original = root / "ao_missing_dates_weekday_screen_original.csv"
    if not original.exists():
        original.write_bytes(file.read_bytes())
    screen = pd.read_csv(original)
    con = duckdb.connect()
    dates = con.execute(
        "SELECT DISTINCT trade_date::VARCHAR FROM read_parquet(?) WHERE root='CL' AND is_final AND update_action=1",
        [str(root / "cl_futures_settlements.parquet")],
    ).fetchall()
    observed = {d[0] for d in dates}
    screen["cl_reference_date_present"] = screen.trade_date.isin(observed)
    screen["classification"] = screen.cl_reference_date_present.map(
        {
            True: "AO absent despite observed CL settlement date",
            False: "No observed CL settlement date; holiday or capture gap unresolved",
        }
    )
    screen.to_csv(file, index=False)
    con.close()
