"""Explicit research defaults and a separate, provisional calibration handoff.

Never overrides canonical observations or the strict research-eligible mask.
"""

from pathlib import Path
import json
import math
from datetime import datetime, timezone
import duckdb
import pandas as pd
import pyarrow as pa
from .pipeline import write, literal

MODEL_DEFAULTS = {
    "version": "ao_common_factor_heston_flat_sofr_v1",
    "status": "proposed_research_baseline_not_team_ratified",
    "pricing_measure": "risk_neutral_deterministic_rates",
    "futures_dynamics": "dF(u,T)/F(u,T)=sqrt(v(u))*dW_F(u), common factor across T",
    "variance_dynamics": "dv=kappa*(theta-v)*du+sigma*sqrt(v)*dW_v; corr(dW_F,dW_v)=rho",
    "free_parameters": ["v0", "kappa", "theta", "sigma", "rho"],
    "fixed_conditioning_inputs": [
        "CL_curve",
        "strike",
        "fixing_schedule",
        "known_fixing_sum",
        "known_fixing_count",
        "discount_factor",
        "valuation_time",
        "expiry_time",
    ],
    "discount_convention": "flat last-available SOFR, ACT/360 daily compounding; not an OIS term curve",
    "discount_formula": "D=(1+SOFR_percent/100/360)^(-calendar_days)",
    "variance_time_basis": "ACT/365F",
    "valuation": "retrospective reference-date EOD; not a tradable point-in-time backtest",
    "payment_default": "expiry; premium/payment settlement lag not independently verified",
    "price_unit": "USD/barrel",
    "contract_multiplier": 1000,
    "baseline_dates": ["2021-01-01", "2025-12-31"],
    "maximum_calendar_days_to_expiry": 183,
    "negative_forward_policy": "exclude from positive multiplicative baseline; preserve raw stress data",
    "parameter_constraints": "v0,theta,kappa,sigma>0; -1<rho<1; Feller is a reported diagnostic, not an automatic calibration constraint",
    "simulation": "joint CL paths with common Brownian factors; discrete arithmetic fixings; convergence-tested variance simulation required",
    "limitations": [
        "Perfect proportional co-movement across CL maturities is a restrictive modeling assumption.",
        "Missing terminal final flags remain unresolved; prototype completeness is not research validation.",
        "SOFR flat-curve and payment-date sensitivities required before attributing residuals to Heston.",
        "Rates are revised historical downloads; original-vintage publication timestamps are not available.",
        "No synthetic labeled Heston training data or pricer is supplied by this data module.",
    ],
}


def discount_factor(sofr_percent, calendar_days):
    """A declared flat-rate assumption, not a bootstrapped market curve."""
    if calendar_days < 0 or not math.isfinite(sofr_percent):
        raise ValueError("Invalid horizon or rate")
    base = 1 + sofr_percent / 100 / 360
    if base <= 0:
        raise ValueError("Nonpositive compounding base")
    return math.exp(-calendar_days * math.log1p(sofr_percent / 100 / 360))


def arithmetic_payoff(known_sum, simulated_remaining_fixings, total_fixings, strike, side):
    """Per-barrel terminal payoff; a hypothetical path is not an observation."""
    if side not in ("C", "P") or total_fixings <= 0:
        raise ValueError("Invalid contract")
    average = (known_sum + sum(simulated_remaining_fixings)) / total_fixings
    return max((average - strike) if side == "C" else (strike - average), 0.0)


def build(root):
    root = Path(root)
    (root / "model_defaults.json").write_text(json.dumps(MODEL_DEFAULTS, indent=2))
    raw = json.loads((root / "sources/nyfed_sofr_20191101_20261026.json").read_text())
    rates = pd.DataFrame(raw["refRates"]).sort_values("effectiveDate")
    rates["effective_date"] = pd.to_datetime(rates.effectiveDate).dt.date
    rates["sofr_percent"] = rates.percentRate.astype(float)
    # The next SOFR observation date identifies the next Fed publication business day.
    # This is a documented calendar proxy, not an actual original-vintage timestamp.
    rates["assumed_available_date"] = rates.effective_date.shift(-1)
    rates["availability_basis"] = (
        "next_SOFR_effective_date_calendar_proxy_08ET; original_vintage_unverified"
    )
    rates[
        [
            "effective_date",
            "sofr_percent",
            "assumed_available_date",
            "revisionIndicator",
            "availability_basis",
        ]
    ].to_parquet(root / "sofr_history.parquet", index=False)
    rates[
        [
            "effective_date",
            "sofr_percent",
            "assumed_available_date",
            "revisionIndicator",
            "availability_basis",
        ]
    ].to_csv(root / "sofr_history.csv", index=False)
    c = duckdb.connect()
    c.execute("SET enable_progress_bar=false")
    c.execute("SET memory_limit='2GB'")
    for name, file in [
        ("p", "ao_daily_research_panel.parquet"),
        ("r", "cl_futures_settlement_revisions.parquet"),
        ("contracts", "underlying_contracts.parquet"),
        ("rates", "sofr_history.parquet"),
    ]:
        c.read_parquet(str(root / file)).create_view(name)
    # Quarantine stale reference-date publications after a newer date has started.
    c.execute(
        """CREATE TEMP TABLE starts AS SELECT contract_id,trade_date,min(publication_timestamp_raw) first_capture FROM r GROUP BY 1,2"""
    )
    c.execute(
        """CREATE TEMP TABLE windows AS SELECT *,lead(first_capture) OVER(PARTITION BY contract_id ORDER BY trade_date) next_reference_first_capture FROM starts"""
    )
    c.execute("""CREATE TEMP TABLE screened AS SELECT r.*,w.next_reference_first_capture,
      coalesce(r.publication_timestamp_raw>=w.next_reference_first_capture,false) reference_overlap
      FROM r LEFT JOIN windows w USING(contract_id,trade_date)""")
    write(
        c,
        "SELECT * FROM screened WHERE reference_overlap",
        root / "evidence/underlying_reference_overlap.parquet",
    )
    c.execute("""CREATE TEMP TABLE f AS SELECT * EXCLUDE(n) FROM (SELECT *,row_number() OVER(
      PARTITION BY contract_id,trade_date ORDER BY is_final DESC,publication_timestamp_raw DESC,source_sequence DESC) n
      FROM screened WHERE NOT reference_overlap) WHERE n=1 AND update_action=1""")
    write(c, "SELECT * FROM f", root / "underlying_clock_screened_candidates.parquet")
    # Calendar and reconstructed terminal averages remain diagnostic while final flags are unresolved.
    c.execute("""CREATE TEMP TABLE fixed AS SELECT n.trade_date fixing_date,n.contract_id,f.settlement_price,f.is_final,
      f.publication_timestamp_raw,f.source_file,strftime(n.trade_date,'%Y-%m') contract_month FROM (
       SELECT * EXCLUDE(n) FROM (SELECT d.trade_date,c.contract_id,row_number() OVER(PARTITION BY d.trade_date ORDER BY c.expiration,c.contract_id) n
       FROM (SELECT DISTINCT trade_date FROM f WHERE root='CL' AND extract(isodow FROM trade_date)<=5 AND trade_date NOT IN (DATE '2021-04-02',DATE '2023-04-07')) d
       JOIN (SELECT DISTINCT contract_id,expiration FROM contracts WHERE root='CL') c ON c.expiration>=d.trade_date) WHERE n=1) n
       LEFT JOIN f USING(trade_date,contract_id)""")
    c.execute("""CREATE TEMP TABLE recon AS SELECT a.*,f.settlement_price csx_terminal,f.is_final csx_final_flag,
       a.mean_cl-f.settlement_price difference,coalesce(abs(a.mean_cl-f.settlement_price)<=0.00501 AND a.missing=0,false) terminal_diagnostic_pass
       FROM (SELECT contract_month,count(*) fixing_count,count_if(settlement_price IS NULL) missing,
       avg(settlement_price) mean_cl,count_if(NOT is_final) nonfinal_cl FROM fixed GROUP BY 1) a
       LEFT JOIN f ON a.contract_month=f.contract_month AND f.root='CSX' AND f.trade_date=f.expiration""")
    c.execute(
        f"COPY (SELECT * FROM recon ORDER BY contract_month) TO {literal(root / 'evidence/clock_screened_monthly_reconciliation.csv')} (HEADER)"
    )
    write(
        c, "SELECT * FROM fixed ORDER BY fixing_date", root / "ao_candidate_fixing_calendar.parquet"
    )
    c.execute("""CREATE TEMP TABLE months AS SELECT DISTINCT observation_date,contract_month,expiration_date FROM p
      WHERE observation_date BETWEEN DATE '2021-01-01' AND DATE '2025-12-31' AND mark_eligible
      AND date_diff('day',observation_date,expiration_date) BETWEEN 1 AND 183""")
    c.execute("""CREATE TEMP TABLE state AS SELECT o.*,
      count(*) total_fixings,count_if(s.fixing_date<=o.observation_date) elapsed_fixings,
      count_if(s.fixing_date<=o.observation_date AND s.settlement_price IS NOT NULL) known_fixing_count,
      coalesce(sum(s.settlement_price) FILTER(WHERE s.fixing_date<=o.observation_date),0) known_fixing_sum,
      sum(CASE WHEN s.fixing_date<=o.observation_date THEN s.settlement_price ELSE curve.settlement_price END)/count(*) expected_average_from_cl,
      count_if(s.fixing_date>o.observation_date) remaining_fixings,
      count_if(s.fixing_date<=o.observation_date AND (s.settlement_price IS NULL OR NOT s.is_final)) provisional_or_missing_elapsed,
      count_if(s.fixing_date>o.observation_date AND (curve.settlement_price IS NULL OR curve.settlement_price<=0)) missing_or_nonpositive_forwards,
      count_if(s.fixing_date>o.observation_date AND NOT coalesce(curve.is_final,false)) provisional_remaining_forwards,
      bool_and(recon.terminal_diagnostic_pass) terminal_diagnostic_pass
      FROM months o JOIN fixed s ON s.contract_month=o.contract_month
      LEFT JOIN f curve ON curve.trade_date=o.observation_date AND curve.contract_id=s.contract_id AND curve.root='CL'
      LEFT JOIN recon ON o.contract_month=recon.contract_month GROUP BY ALL""")
    write(
        c,
        """SELECT o.observation_date,o.contract_month,s.fixing_date,s.contract_id,
       
       CASE WHEN s.fixing_date<=o.observation_date THEN s.settlement_price END known_fixing,
       CASE WHEN s.fixing_date>o.observation_date THEN curve.settlement_price END current_forward,
       s.is_final fixing_final_flag,curve.is_final forward_final_flag,
       1.0/count(*) OVER(PARTITION BY o.observation_date,o.contract_month) arithmetic_weight,
       date_diff('day',o.observation_date,s.fixing_date)/365.0 fixing_time_years,
       s.publication_timestamp_raw fixing_capture,curve.publication_timestamp_raw forward_capture,
       'retrospective_EOD; not publication-screened causal input' time_policy
       FROM months o JOIN fixed s ON s.contract_month=o.contract_month
       LEFT JOIN f curve ON curve.trade_date=o.observation_date AND curve.contract_id=s.contract_id AND curve.root='CL'
       ORDER BY o.observation_date,o.contract_month,s.fixing_date""",
        root / "ao_model_fixing_inputs.parquet",
    )
    # Paired-call/put parity is a diagnostic, never the default discount input.
    c.execute("""CREATE TEMP TABLE pairs AS SELECT observation_date,contract_month,expiration_date,strike,
      max(settlement_price) FILTER(WHERE call_put='C') c,max(settlement_price) FILTER(WHERE call_put='P') p
      FROM p WHERE mark_eligible AND NOT source_capture_degraded GROUP BY ALL""")
    c.execute("""CREATE TEMP TABLE parity AS SELECT observation_date,contract_month,expiration_date,count(*) pairs,
      max(strike)-min(strike) strike_span,-regr_slope(c-p,strike) parity_df,
      regr_intercept(c-p,strike)/nullif(-regr_slope(c-p,strike),0) parity_expected_average,
      regr_r2(c-p,strike) parity_r_squared FROM pairs WHERE c IS NOT NULL AND p IS NOT NULL
      GROUP BY ALL HAVING count(*)>=5 AND max(strike)-min(strike)>=5""")
    c.execute(
        f"COPY (SELECT * FROM parity ORDER BY observation_date,contract_month) TO {literal(root / 'ao_parity_discount_diagnostic.csv')} (HEADER)"
    )
    c.execute("""CREATE TEMP TABLE rate_days AS SELECT * EXCLUDE(n) FROM (
      SELECT o.observation_date,rates.*,row_number() OVER(PARTITION BY o.observation_date ORDER BY assumed_available_date DESC,effective_date DESC) n
      FROM (SELECT DISTINCT observation_date FROM months) o LEFT JOIN rates ON rates.assumed_available_date<=o.observation_date
      AND rates.effective_date<o.observation_date) WHERE n=1""")
    write(
        c,
        """SELECT p.observation_date,p.contract_id,p.contract_month,p.expiration_date,p.strike,p.call_put,
      p.settlement_price,p.price_type,p.stat_ts_recv option_capture,p.source option_source,p.source_file option_source_file,p.source_row option_source_row,p.open_interest,p.trade_volume,
      s.* EXCLUDE(observation_date,contract_month,expiration_date),
      d.sofr_percent,d.effective_date rate_effective_date,d.assumed_available_date rate_assumed_available_date,
      date_diff('day',p.observation_date,p.expiration_date) calendar_days,
      date_diff('day',p.observation_date,p.expiration_date)/365.0 variance_years,
      exp(-date_diff('day',p.observation_date,p.expiration_date)*ln(1+d.sofr_percent/100/360)) discount_factor,
      parity.parity_df,parity.parity_expected_average,parity.pairs parity_pairs,
      csx.settlement_price current_csx_future,csx.is_final current_csx_final_flag,
      p.observation_date IN (DATE '2021-04-02',DATE '2023-04-07') copied_holiday_observation,
      d.sofr_percent IS NOT NULL AND s.known_fixing_count=s.elapsed_fixings
       AND s.total_fixings>0 AND s.missing_or_nonpositive_forwards=0 AND s.terminal_diagnostic_pass
       AND p.explicit_contract_metadata AND NOT p.source_capture_degraded
       AND NOT p.monotonicity_flag AND NOT p.convexity_flag
       AND p.observation_date NOT IN (DATE '2021-04-02',DATE '2023-04-07') prototype_input_complete,
      false fully_validated_calibration_ready,
      'terminal_finality_and_independent_calendar_pending; flat_SOFR_and_payment_defaults; retrospective_EOD' limitations,
      'ao_common_factor_heston_flat_sofr_v1' model_version
      FROM p JOIN state s USING(observation_date,contract_month,expiration_date)
      LEFT JOIN rate_days d USING(observation_date)
      LEFT JOIN f csx ON csx.trade_date=p.observation_date AND csx.contract_month=p.contract_month AND csx.root='CSX'
      LEFT JOIN parity USING(observation_date,contract_month,expiration_date)
      WHERE p.mark_eligible ORDER BY p.observation_date,p.expiration_date,p.call_put,p.strike""",
        root / "ao_calibration_inputs_provisional.parquet",
    )
    c.read_parquet(str(root / "ao_calibration_inputs_provisional.parquet")).create_view("out")
    result = (
        c.sql("""SELECT count(*) records,count_if(prototype_input_complete) prototype_complete_records,
      count(DISTINCT observation_date) dates,count_if(fully_validated_calibration_ready) fully_validated_records,
      min(observation_date)::VARCHAR first_date,max(observation_date)::VARCHAR last_date,
      count_if(sofr_percent IS NULL) missing_rate FROM out""")
        .df()
        .to_dict("records")[0]
    )
    result["rate_observations"] = len(rates)
    result["reference_overlap_quarantined"] = c.sql(
        "SELECT count(*) FROM screened WHERE reference_overlap"
    ).fetchone()[0]
    result["terminal_months_reconciled"] = c.sql(
        "SELECT count(*) FROM recon WHERE terminal_diagnostic_pass"
    ).fetchone()[0]
    result["generated_utc"] = datetime.now(timezone.utc).isoformat()
    result["prototype_vector_counts"] = dict(
        zip(
            ["days_ge6", "days_ge10", "days_ge20"],
            c.sql(
                """SELECT count_if(n>=6),count_if(n>=10),count_if(n>=20) FROM (SELECT observation_date,count(*) n FROM out WHERE prototype_input_complete GROUP BY observation_date)"""
            ).fetchone(),
        )
    )
    # Concrete review snapshot, not a claim of clean validation.
    c.execute(
        f"COPY (SELECT * FROM out WHERE observation_date=DATE '2024-06-14' AND prototype_input_complete) TO {literal(root / 'ao_model_input_sample_20240614.csv')} (HEADER)"
    )
    (root / "evidence/model_input_readiness.json").write_text(json.dumps(result, indent=2))
    c.close()
    return result


def load_model_snapshot(observation_date, root, complete_only=True):
    """Prototype inputs with explicit unvalidated status; no future realized fixings."""
    from datetime import date

    day = date.fromisoformat(str(observation_date))
    root = Path(root)
    con = duckdb.connect()
    try:
        predicate = " AND prototype_input_complete" if complete_only else ""
        options = con.execute(
            "SELECT * FROM read_parquet(?) WHERE observation_date=?::DATE"
            + predicate
            + " ORDER BY expiration_date,call_put,strike,contract_id",
            [str(root / "ao_calibration_inputs_provisional.parquet"), day],
        ).df()
        con.register(
            "selected_months", pa.Table.from_pandas(options[["contract_month"]].drop_duplicates())
        )
        fixings = con.execute(
            """SELECT f.* FROM read_parquet(?) f SEMI JOIN selected_months s USING(contract_month)
          WHERE observation_date=?::DATE ORDER BY contract_month,fixing_date""",
            [str(root / "ao_model_fixing_inputs.parquet"), day],
        ).df()
        return dict(
            options=options,
            fixings=fixings,
            model_defaults=json.loads((root / "model_defaults.json").read_text()),
            fully_validated=False,
        )
    finally:
        con.close()
