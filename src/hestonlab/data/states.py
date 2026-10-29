"""Observed fixing calendar, roll reconciliation, and publication-screened states."""

from pathlib import Path
import json
import duckdb
from .pipeline import write, literal


def build_states(root):
    root = Path(root)
    con = duckdb.connect()
    con.execute("SET enable_progress_bar=false")
    con.execute("SET memory_limit='2GB'")
    for name, file in [
        ("futures", "cl_futures_settlements.parquet"),
        ("revisions", "cl_futures_settlement_revisions.parquet"),
        ("contracts", "underlying_contracts.parquet"),
        ("ao", "ao_observations.parquet"),
    ]:
        con.read_parquet(str(root / file)).create_view(name)
    # Use actual exchange reference dates on the CL curve, not a US bank calendar.
    con.execute("""CREATE TEMP TABLE fixing_days AS SELECT DISTINCT trade_date fixing_date
      FROM futures WHERE root='CL' AND is_final AND update_action=1
      AND settlement_price IS NOT NULL AND extract(isodow FROM trade_date)<=5""")
    con.execute("""CREATE TEMP TABLE nearby AS SELECT fixing_date,contract_id reference_contract_id,
      contract_month reference_contract_month,expiration FROM (
        SELECT d.*,c.contract_id,c.contract_month,c.expiration,
        row_number() OVER(PARTITION BY fixing_date ORDER BY expiration,contract_month) rn
        FROM fixing_days d JOIN (SELECT DISTINCT contract_id,contract_month,expiration
        FROM contracts WHERE root='CL') c ON c.expiration>=d.fixing_date) WHERE rn=1""")
    con.execute("""CREATE TEMP TABLE fixed AS SELECT n.*,f.settlement_price fixing_price,
      f.publication_timestamp,f.source_file,strftime(fixing_date,'%Y-%m') averaging_month
      FROM nearby n LEFT JOIN futures f ON n.reference_contract_id=f.contract_id
      AND n.fixing_date=f.trade_date AND f.is_final AND f.update_action=1""")
    con.execute("""CREATE TEMP TABLE recon AS SELECT x.*,
       c.settlement_price csx_final,c.trade_date csx_final_date,
       x.arithmetic_average-c.settlement_price difference,
       abs(x.arithmetic_average-c.settlement_price)<=0.00501 AND missing=0 verified
       FROM (SELECT averaging_month,count(*) fixing_count,
       count_if(fixing_price IS NULL) missing,avg(fixing_price) arithmetic_average,
       min(fixing_date) first_fixing,max(fixing_date) last_fixing
       FROM fixed GROUP BY averaging_month) x LEFT JOIN (
       SELECT * FROM futures WHERE root='CSX' AND trade_date=expiration
       AND is_final AND update_action=1) c ON x.averaging_month=c.contract_month""")
    con.execute(
        f'COPY (SELECT * FROM recon ORDER BY averaging_month) TO {literal(root / "evidence/monthly_average_reconciliation.csv")} (HEADER,DELIMITER ",")'
    )
    # Diagnose the other common roll convention independently.
    alternative = """SELECT strftime(d.fixing_date,'%Y-%m') averaging_month,
       avg(f.settlement_price) average_roll_before_expiry FROM (
       SELECT fixing_date,contract_id FROM (SELECT d.fixing_date,c.contract_id,
       row_number() OVER(PARTITION BY fixing_date ORDER BY c.expiration) rn
       FROM fixing_days d JOIN (SELECT DISTINCT contract_id,expiration FROM contracts WHERE root='CL') c
       ON c.expiration>d.fixing_date) WHERE rn=1) d
       LEFT JOIN futures f ON d.fixing_date=f.trade_date AND d.contract_id=f.contract_id
       AND f.is_final AND f.update_action=1 GROUP BY 1"""
    con.execute(
        f'COPY ({alternative}) TO {literal(root / "evidence/alternative_roll_diagnostic.csv")} (HEADER,DELIMITER ",")'
    )
    # Observed months have a source-derived calendar. Future dates are explicitly
    # provisional weekdays, never fabricated prices or verified holidays.
    con.execute("""CREATE TEMP TABLE schedule AS SELECT f.averaging_month,f.fixing_date,
       f.reference_contract_id,f.reference_contract_month,f.fixing_price,
       f.publication_timestamp,f.source_file,coalesce(r.verified,false) calendar_verified,
       'observed_CL_reference_days; month_average_checked_against_CSX' calendar_basis,
       'expiry_day_included' roll_convention FROM fixed f
       JOIN (SELECT DISTINCT contract_month FROM ao) m ON f.averaging_month=m.contract_month
       LEFT JOIN recon r USING(averaging_month)""")
    maxdate = con.sql("SELECT max(fixing_date) FROM fixing_days").fetchone()[0]
    maxmonth = con.sql("SELECT max(contract_month) FROM ao").fetchone()[0]
    con.execute(f"""INSERT INTO schedule SELECT strftime(d,'%Y-%m'),d::DATE,
        NULL,NULL,NULL,NULL,NULL,false,'future_weekday_provisional_holidays_unverified',
        'unresolved_future_calendar' FROM generate_series(DATE {literal(maxdate)}+INTERVAL 1 DAY,
        last_day(DATE {literal(maxmonth + "-01")}),INTERVAL 1 DAY) t(d)
        WHERE extract(isodow FROM d)<=5""")
    write(
        con,
        "SELECT * FROM schedule ORDER BY averaging_month,fixing_date",
        root / "ao_fixing_schedule.parquet",
    )
    # Normalized month calendar prevents replicating the same schedule per strike.
    # The explicit bridge supplies each contract_id -> averaging_month mapping.
    write(
        con,
        "SELECT DISTINCT contract_id,contract_month averaging_month FROM ao",
        root / "ao_contract_fixing_bridge.parquet",
    )
    con.execute("""CREATE TEMP TABLE cutoffs AS SELECT observation_date,contract_month,
       min(publication_timestamp) cutoff_timestamp,min(stat_ts_recv) cutoff_timestamp_raw FROM ao
       GROUP BY observation_date,contract_month""")
    # Revisions are screened BEFORE ranking. Deleted records cannot resurrect a
    # prior mark: latest final record wins, then the action is evaluated.
    con.execute("""CREATE TEMP TABLE asof_fixings AS SELECT observation_date,contract_month,
        cutoff_timestamp,fixing_date,reference_contract_id,settlement_price,
        update_action FROM (SELECT o.*,s.fixing_date,s.reference_contract_id,
        r.settlement_price,r.update_action,
        row_number() OVER(PARTITION BY o.observation_date,o.contract_month,s.fixing_date
        ORDER BY r.publication_timestamp_raw DESC NULLS LAST,r.source_sequence DESC) rn
        FROM cutoffs o JOIN schedule s ON o.contract_month=s.averaging_month
        AND s.fixing_date<=o.observation_date
        LEFT JOIN revisions r ON r.contract_id=s.reference_contract_id AND r.trade_date=s.fixing_date
        AND r.is_final AND r.publication_timestamp_raw<=o.cutoff_timestamp_raw)
        WHERE rn=1""")
    con.execute("""CREATE TEMP TABLE accrued AS SELECT observation_date,contract_month,
        count_if(settlement_price IS NOT NULL AND update_action=1) known_fixing_count,
        coalesce(sum(CASE WHEN update_action=1 THEN settlement_price END),0) accrued_fixing_sum,
        count_if(settlement_price IS NULL OR coalesce(update_action,0)!=1) missing_elapsed_fixings
        FROM asof_fixings GROUP BY observation_date,contract_month""")
    con.execute("""CREATE TEMP TABLE asof_curves AS SELECT * EXCLUDE(rn) FROM (
       SELECT o.observation_date,o.contract_month,r.contract_id,r.settlement_price,
       r.expiration,r.update_action,
       row_number() OVER(PARTITION BY o.observation_date,o.contract_month,r.contract_id
       ORDER BY r.publication_timestamp_raw DESC,r.source_sequence DESC) rn FROM cutoffs o JOIN revisions r
       ON r.trade_date=o.observation_date AND r.root='CL' AND r.is_final
       AND r.publication_timestamp_raw<=o.cutoff_timestamp_raw AND r.expiration>=o.observation_date)
       WHERE rn=1 AND update_action=1 AND settlement_price IS NOT NULL""")
    con.execute("""CREATE TEMP TABLE curves AS SELECT observation_date,contract_month,
       contract_id front_cl_contract_id,settlement_price front_cl_settlement FROM (
       SELECT *,row_number() OVER(PARTITION BY observation_date,contract_month
       ORDER BY expiration) near_n FROM asof_curves) WHERE near_n=1""")
    con.execute("""CREATE TEMP TABLE curve_masks AS SELECT o.observation_date,o.contract_month,
       count_if(s.fixing_date>o.observation_date AND
         (s.reference_contract_id IS NULL OR c.contract_id IS NULL))=0 remaining_curve_complete
       FROM cutoffs o JOIN schedule s ON s.averaging_month=o.contract_month
       LEFT JOIN asof_curves c ON c.observation_date=o.observation_date
       AND c.contract_month=o.contract_month AND c.contract_id=s.reference_contract_id
       GROUP BY o.observation_date,o.contract_month""")
    write(
        con,
        """SELECT o.*,coalesce(a.known_fixing_count,0) known_fixing_count,
       coalesce(a.accrued_fixing_sum,0) accrued_fixing_sum,
       coalesce(a.missing_elapsed_fixings,0) missing_elapsed_fixings,
       (SELECT count(*) FROM schedule s WHERE s.averaging_month=o.contract_month
        AND s.fixing_date>o.observation_date) remaining_fixing_count,
       coalesce((SELECT bool_and(calendar_verified) FROM schedule s WHERE
        s.averaging_month=o.contract_month),false) fixing_calendar_verified,
       c.front_cl_contract_id,c.front_cl_settlement,c.front_cl_contract_id IS NOT NULL curve_available,
       coalesce(cm.remaining_curve_complete,false) remaining_curve_complete,
       coalesce(a.missing_elapsed_fixings,0)=0 AND c.front_cl_contract_id IS NOT NULL
       AND coalesce(cm.remaining_curve_complete,false)
       AND coalesce((SELECT bool_and(calendar_verified) FROM schedule s WHERE
       s.averaging_month=o.contract_month),false) underlying_state_complete
       FROM cutoffs o LEFT JOIN accrued a USING(observation_date,contract_month)
       LEFT JOIN curves c USING(observation_date,contract_month)
       LEFT JOIN curve_masks cm USING(observation_date,contract_month)""",
        root / "ao_fixing_state.parquet",
    )
    stats = con.sql("""SELECT count(*) months_compared,count_if(verified) verified_months,
       count_if(csx_final IS NULL) no_final_csx,count_if(missing>0) missing_cl,
       max(abs(difference)) max_absolute_difference FROM recon""").fetchone()
    (root / "evidence/fixing_validation.json").write_text(
        json.dumps(
            dict(
                months_compared=stats[0],
                verified_months=stats[1],
                no_final_csx=stats[2],
                months_missing_cl=stats[3],
                max_absolute_difference=stats[4],
                calendar_method="Empirical CL reference days validated by CSX terminal average; future weekdays unverified",
                price_asof="Latest final publication at/before earliest AO publication per day/month; reference dates through observation date only",
                limitations=[
                    "Monthly mean agreement is not an independent holiday listing census",
                    "Definitions use historical monthly snapshots, not exhaustive daily validity records",
                    "Future-month calendars remain provisional and fail research_eligible",
                    "OI and volume are retrospective descriptive fields; publication timing not reconstructed",
                ],
            ),
            indent=2,
        )
    )
    con.close()
