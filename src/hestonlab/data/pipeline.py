"""Deterministic local canonicalization, diagnostics and research exports."""

from pathlib import Path
from datetime import datetime, timezone
import json
import duckdb
import pyarrow as pa


def literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def write(con, query, path):
    """Atomic artifact replacement. Raw sources are never written here."""
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    con.execute(f"COPY ({query}) TO {literal(temp)} (FORMAT PARQUET, COMPRESSION ZSTD)")
    temp.replace(path)


def build(root):
    root = Path(root)
    con = duckdb.connect()
    con.execute("SET enable_progress_bar=false")
    con.execute("SET memory_limit='2GB'")
    stamp = datetime.now(timezone.utc).isoformat()
    con.read_parquet(str(root / "ao_historical_panel.parquet")).create_view("legacy")
    condition_file = root / "evidence/databento_dataset_condition_exceptions.csv"
    if condition_file.exists():
        con.read_csv(str(condition_file), all_varchar=True).create_view("bad_capture_days")
        degraded = "EXISTS(SELECT 1 FROM bad_capture_days b WHERE b.date=substr(stat_ts_recv,1,10))"
    else:
        degraded = "false"
    con.execute(f"""CREATE TEMP TABLE obs AS SELECT *,
        trade_date::DATE observation_date,
        'NYMEX:AO:'||contract_month||':'||call_put||':'||printf('%.8f',strike) contract_id,
        option_expiration::DATE expiration_date,
        try_cast(stat_ts_recv AS TIMESTAMPTZ) publication_timestamp,
        {literal(stamp)}::TIMESTAMPTZ ingested_at,
        volume trade_volume, NULL::DOUBLE bid_price, NULL::DOUBLE ask_price,
        NULL::DOUBLE last_trade_price, NULL::BOOLEAN nominal_indicator,
        'latest_final_retrospective' revision_policy,
        {degraded} source_capture_degraded,
        expiration_status='explicit_definition' explicit_contract_metadata,
        CASE WHEN trade_date<'2020-04-22' THEN 'pre_change_model_unverified'
        WHEN trade_date<'2020-05-18' THEN 'Bachelier_notice20-171'
        ELSE 'Curran_notice20-198_later_history_not_exhaustive' END methodology_regime,
        CASE WHEN trade_date>option_expiration THEN 'post_expiration;'
             ELSE '' END ||
        CASE WHEN volume IS NULL OR open_interest IS NULL THEN 'liquidity_fields_missing;' ELSE '' END ||
        CASE WHEN NOT usable_settlement THEN 'invalid_or_nonfinal_price;' ELSE '' END quality_flags
        FROM legacy""")
    con.read_csv(str(root / "ao_exchange_samples.csv"), all_varchar=True).create_view(
        "exchange_samples"
    )
    con.execute("""CREATE TEMP TABLE enriched AS SELECT o.* EXCLUDE(open_interest,volume,trade_volume),
       o.open_interest vendor_open_interest,o.volume vendor_volume,
       coalesce(o.open_interest,try_cast(e.open_interest AS DOUBLE)) open_interest,
       coalesce(o.volume,try_cast(e.volume AS DOUBLE)) volume,
       coalesce(o.volume,try_cast(e.volume AS DOUBLE)) trade_volume,
       CASE WHEN o.open_interest IS NOT NULL THEN o.source
            WHEN try_cast(e.open_interest AS DOUBLE) IS NOT NULL THEN e.source END open_interest_source,
       CASE WHEN o.volume IS NOT NULL THEN o.source
            WHEN try_cast(e.volume AS DOUBLE) IS NOT NULL THEN e.source END trade_volume_source,
       e.source_file exchange_comparison_source_file,e.source_row exchange_comparison_source_row,
       try_cast(e.delta_published AS DOUBLE) exchange_delta,
       try_cast(e.implied_volatility AS DOUBLE) exchange_implied_volatility,
       NULL::TIMESTAMPTZ liquidity_publication_timestamp
       FROM obs o LEFT JOIN exchange_samples e ON o.trade_date=e.trade_date
       AND o.contract_month=e.contract_month AND o.call_put=e.call_put
       AND o.strike=try_cast(e.strike AS DOUBLE)""")
    con.execute("DROP TABLE obs")
    con.execute("ALTER TABLE enriched RENAME TO obs")
    if con.sql(
        "SELECT count(*) FROM (SELECT observation_date,contract_id,count(*) n FROM obs GROUP BY ALL HAVING n>1)"
    ).fetchone()[0]:
        raise ValueError("Duplicate canonical option/day identity")
    write(
        con,
        "SELECT * FROM obs ORDER BY observation_date,contract_id",
        root / "ao_observations.parquet",
    )
    write(
        con,
        """SELECT contract_id, any_value(exchange) exchange,
       any_value(clearing_code) clearing_root,any_value(globex_root) globex_root,
       any_value(raw_symbol) vendor_symbol,contract_month,call_put,strike,
       min(expiration_date) expiration_date, NULL::TIMESTAMPTZ expiration_timestamp,
       0.01 strike_scale, 'USD' currency,'USD/barrel' premium_quote_unit,
       1000 contract_multiplier,'cash' settlement_type,'European' exercise_style,
       'CME_ch341_current_spec_historical_changes_not_exhaustive' specification_version,
       'CSX:'||contract_month related_underlying_id,
       string_agg(DISTINCT instrument_id, '|') vendor_instrument_ids,
       string_agg(DISTINCT expiration_status, '|') expiration_evidence,
       min(observation_date) first_observed,max(observation_date) last_observed,
       'Observation bounds are not listing/activation dates' effective_date_note
       FROM obs GROUP BY contract_id,contract_month,call_put,strike""",
        root / "ao_contracts.parquet",
    )
    con.execute("""CREATE TEMP TABLE quality AS SELECT *,
       lag(settlement_price) OVER(PARTITION BY observation_date,contract_month,call_put ORDER BY strike) previous_strike_price,
       lag(strike) OVER(PARTITION BY observation_date,contract_month,call_put ORDER BY strike) previous_strike
       FROM obs""")
    con.execute("""CREATE TEMP TABLE q2 AS SELECT *,
        coalesce((call_put='C' AND settlement_price>previous_strike_price+0.011)
          OR (call_put='P' AND settlement_price<previous_strike_price-0.011),false) monotonicity_flag,
        CASE WHEN previous_strike IS NOT NULL THEN
          (settlement_price-previous_strike_price)/(strike-previous_strike) END strike_slope
        FROM quality""")
    con.execute("""CREATE TEMP TABLE q3 AS SELECT *,
      coalesce(strike_slope < lag(strike_slope) OVER(PARTITION BY observation_date,contract_month,call_put ORDER BY strike)-0.02,false) convexity_flag
      FROM q2""")
    con.execute("""CREATE TEMP TABLE q4 AS SELECT *,
       coalesce(settlement_price=lag(settlement_price) OVER(PARTITION BY contract_id ORDER BY observation_date),false)
       same_price_previous_observed_day FROM q3""")
    state = root / "ao_fixing_state.parquet"
    if state.exists():
        con.read_parquet(str(state)).create_view("states")
        join = """LEFT JOIN states s ON q.observation_date=s.observation_date AND q.contract_month=s.contract_month"""
        extra = """s.known_fixing_count,s.accrued_fixing_sum,s.remaining_fixing_count,
          s.missing_elapsed_fixings,s.fixing_calendar_verified,s.cutoff_timestamp,s.cutoff_timestamp_raw,
          s.curve_available,s.remaining_curve_complete,s.front_cl_contract_id,s.front_cl_settlement,
          s.underlying_state_complete"""
        eligible = "coalesce(s.underlying_state_complete,false)"
    else:
        join = ""
        extra = """NULL::BIGINT known_fixing_count,NULL::DOUBLE accrued_fixing_sum,
          NULL::BIGINT remaining_fixing_count,NULL::BIGINT missing_elapsed_fixings,
          false fixing_calendar_verified,NULL::TIMESTAMPTZ cutoff_timestamp,NULL::VARCHAR cutoff_timestamp_raw,
          false curve_available,false remaining_curve_complete,NULL::VARCHAR front_cl_contract_id,
          NULL::DOUBLE front_cl_settlement,false underlying_state_complete"""
        eligible = "false"
    write(
        con,
        f"""SELECT q.*, {extra}, NULL::DOUBLE interest_rate,
        date_diff('day',q.observation_date,q.expiration_date)/365.0 time_to_expiry_years,
        usable_settlement AND q.observation_date<=q.expiration_date AND NOT monotonicity_flag
        AND NOT convexity_flag AND NOT source_capture_degraded AND explicit_contract_metadata
        AND {eligible} research_eligible,
        usable_settlement AND q.observation_date<=q.expiration_date mark_eligible,
        'Settlements are exchange marks; actual flag does not prove a trade' price_interpretation
        FROM q4 q {join} ORDER BY q.observation_date,q.expiration_date,q.call_put,q.strike""",
        root / "ao_daily_research_panel.parquet",
    )
    con.read_parquet(str(root / "ao_daily_research_panel.parquet")).create_view("panel")
    coverage(con, root)
    con.close()
    from .contracts import build_reference_versions

    reference_checks = build_reference_versions(root)
    (root / "evidence/reference_validation.json").write_text(json.dumps(reference_checks, indent=2))
    create_catalog(root)
    return json.loads((root / "ao_coverage_summary.json").read_text())


def coverage(con, root):
    daily = """SELECT observation_date trade_date,count(*) records,
      count_if(usable_settlement) positive_final_marks,
      count_if(mark_eligible) unexpired_positive_final,
      count_if(research_eligible) research_eligible,
      count_if(call_put='C') calls,count_if(call_put='P') puts,
      count(DISTINCT contract_id) contracts,count(DISTINCT contract_month) contract_months,
      count(DISTINCT expiration_date) expirations,count(DISTINCT strike) strikes,
      count_if(settlement_price=0) zero_prices,
      count_if(settlement_price<0) negative_option_premiums,
      count_if(settlement_status='preliminary') preliminary,
      count_if(open_interest IS NULL) missing_open_interest,
      count_if(trade_volume IS NULL) missing_volume,
      count_if(monotonicity_flag) monotonicity_flags,
      count_if(convexity_flag) convexity_flags,
      count_if(source_capture_degraded) degraded_capture_rows,
      count_if(same_price_previous_observed_day) unchanged_price_rows,
      count_if(NOT explicit_contract_metadata) same_month_metadata_rows,
      count_if(observation_date>expiration_date) expired_observations,
      count_if(price_type='official_final_theoretical') theoretical_final_marks,
      count_if(mark_eligible AND (open_interest>0 OR trade_volume>0)) liquidity_supported_marks,
      count_if(research_eligible)>=6 has_6_coordinates,
      count_if(research_eligible)>=10 has_10_coordinates,
      count_if(research_eligible)>=20 has_20_coordinates
      FROM panel GROUP BY observation_date ORDER BY observation_date"""
    con.execute(
        f'COPY ({daily}) TO {literal(root / "ao_coverage_daily.csv")} (HEADER,DELIMITER ",")'
    )
    byexp = """SELECT observation_date trade_date,contract_month,expiration_date,
      count(*) records,count(DISTINCT strike) strikes,
      count_if(call_put='C') calls,count_if(call_put='P') puts,
      count_if(mark_eligible) usable_marks,count_if(research_eligible) research_eligible,
      count_if(trade_volume>0 OR open_interest>0) liquidity_supported,
      count_if(open_interest IS NULL) missing_open_interest
      FROM panel GROUP BY ALL ORDER BY 1,2"""
    con.execute(
        f'COPY ({byexp}) TO {literal(root / "ao_coverage_by_expiry.csv")} (HEADER,DELIMITER ",")'
    )
    summary = (
        con.sql("""SELECT count(*) observations,count_if(usable_settlement) positive_final_marks,
      count_if(mark_eligible) unexpired_positive_final_marks,count_if(research_eligible) research_eligible,
      count(DISTINCT observation_date) observation_dates,count(DISTINCT contract_id) unique_contracts,
      count(DISTINCT contract_month) contract_months,count(DISTINCT strike) strikes,
      min(observation_date)::VARCHAR first_date,max(observation_date)::VARCHAR last_date,
      count_if(monotonicity_flag) monotonicity_flags,count_if(convexity_flag) convexity_flags,
      count_if(source_capture_degraded) degraded_capture_rows,
      count_if(same_price_previous_observed_day) unchanged_price_rows,
      count_if(NOT explicit_contract_metadata) same_month_metadata_rows,
      count_if(observation_date>expiration_date) expired_observations,
      count_if(price_type='official_final_theoretical') theoretical_final_marks,
      count_if(bid_price IS NOT NULL OR ask_price IS NOT NULL) quoted_observations,
      count_if(trade_volume>0) reported_positive_volume,
      count_if(underlying_state_complete) underlying_state_complete
      FROM panel""")
        .df()
        .iloc[0]
        .to_dict()
    )
    feasibility = (
        con.sql(f"""SELECT count_if(has_6_coordinates) days_ge6,
      count_if(has_10_coordinates) days_ge10,count_if(has_20_coordinates) days_ge20
      FROM ({daily})""")
        .df()
        .iloc[0]
        .to_dict()
    )
    summary.update(feasibility)
    summary["generated_utc"] = datetime.now(timezone.utc).isoformat()
    summary["listing_completeness"] = (
        "No exhaustive listing-day census; observed-record denominator only"
    )
    summary["liquidity_warning"] = (
        "Zero volume and theoretical marks are retained and flagged; no bid/ask spread evidence"
    )
    (root / "ao_coverage_summary.json").write_text(json.dumps(summary, indent=2, default=int))
    # Preserve the old daily/source report, then make the requested summary high-level.
    summary_csv = root / "ao_coverage_summary.csv"
    if summary_csv.exists() and summary_csv.read_text().splitlines()[0].startswith("trade_date,"):
        archive = root / "ao_coverage_summary_by_source_legacy.csv"
        if not archive.exists():
            summary_csv.rename(archive)
    con.register("summary_row", pa.Table.from_pylist([summary]))
    con.execute(
        f'COPY summary_row TO {literal(root / "ao_coverage_summary.csv")} (HEADER,DELIMITER ",")'
    )


def create_catalog(root):
    """Build a lightweight DuckDB view catalog; rerun after moving the directory."""
    root = Path(root).resolve()
    con = duckdb.connect(str(root / "hestonlab.duckdb"))
    for name in [
        "ao_contracts",
        "ao_contract_reference_versions",
        "ao_observations",
        "cl_futures_settlements",
        "cl_futures_settlement_revisions",
        "underlying_contracts",
        "ao_fixing_schedule",
        "ao_contract_fixing_bridge",
        "ao_fixing_state",
        "ao_daily_research_panel",
    ]:
        file = root / (name + ".parquet")
        if file.exists():
            con.execute(
                f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet({literal(file)})"
            )
    con.close()
