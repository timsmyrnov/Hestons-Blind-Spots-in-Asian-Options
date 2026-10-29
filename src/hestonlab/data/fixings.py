"""Explicit nearby roll and point-in-time accrued arithmetic averaging state."""

from datetime import date
import pandas as pd


def select_nearby(fixing_date, contracts):
    """Expiry day is included. Metadata expirations override calendar formulae."""
    d = pd.Timestamp(fixing_date).date()
    valid = contracts[pd.to_datetime(contracts.expiration).dt.date >= d]
    if valid.empty:
        return None
    return valid.sort_values(["expiration", "contract_month"]).iloc[0].contract_id


def accrued_state(schedule, revisions, observation_date, cutoff):
    """Only fixing dates through observation date and captures through cutoff count.

    A missing or unpublished fixing is never zero. Negative CL prices are valid.
    Schedule must contain a VERIFIED exchange settlement-day calendar; provisional
    weekdays are allowed for diagnostics but must retain their verification flag.
    """
    d = pd.Timestamp(observation_date).date()
    cutoff = pd.Timestamp(cutoff)
    if cutoff.tzinfo is None:
        raise ValueError("cutoff must be timezone-aware")
    s = schedule.copy()
    if s.empty:
        return dict(
            known_fixing_count=0,
            accrued_fixing_sum=0.0,
            remaining_fixing_count=0,
            missing_elapsed_fixings=0,
            fixing_calendar_verified=False,
        )
    r = revisions.copy()
    r["publication_timestamp"] = pd.to_datetime(r.publication_timestamp, utc=True)
    r["trade_date"] = pd.to_datetime(r.trade_date).dt.strftime("%Y-%m-%d")
    s["fixing_date"] = pd.to_datetime(s.fixing_date).dt.strftime("%Y-%m-%d")
    r = r[
        (pd.to_datetime(r.publication_timestamp, utc=True) <= cutoff)
        & (pd.to_datetime(r.trade_date).dt.date <= d)
        & r.is_final
    ]
    r = r.sort_values("publication_timestamp").drop_duplicates(
        ["trade_date", "contract_id"], keep="last"
    )
    r = r[r.update_action == 1]
    s = s.merge(
        r[["trade_date", "contract_id", "settlement_price"]],
        how="left",
        left_on=["fixing_date", "reference_contract_id"],
        right_on=["trade_date", "contract_id"],
        validate="many_to_one",
    )
    elapsed = pd.to_datetime(s.fixing_date).dt.date <= d
    known = elapsed & s.settlement_price.notna()
    return dict(
        known_fixing_count=int(known.sum()),
        accrued_fixing_sum=float(s.loc[known, "settlement_price"].sum()),
        remaining_fixing_count=int((~elapsed).sum()),
        missing_elapsed_fixings=int((elapsed & ~known).sum()),
        fixing_calendar_verified=bool(s.calendar_verified.all()),
    )


def methodology_regime(trade_date):
    d = date.fromisoformat(str(trade_date)[:10])
    if d < date(2020, 4, 22):
        return "pre_2020_04_22_model_not_verified"
    if d < date(2020, 5, 18):
        return "documented_Bachelier"
    return "Curran_from_2020_05_18_later_changes_not_exhaustively_audited"


def build_fixing_schedule(ao_contract, observation_date, root=None, cutoff=None):
    """Return one contract's calendar with only contemporaneously known prices.

    ao_contract can be a canonical ID string or a mapping with contract_month.
    A structural schedule is not a forecast; unverified future calendar flags
    remain explicit. No final realized average is exposed as an early input.
    """
    from pathlib import Path
    import duckdb
    from .research import DEFAULT_ROOT

    root = Path(root or DEFAULT_ROOT)
    d = date.fromisoformat(str(observation_date))
    con = duckdb.connect()
    try:
        con.read_parquet(str(root / "ao_contracts.parquet")).create_view("contracts")
        if isinstance(ao_contract, str):
            match = con.execute(
                "SELECT contract_month FROM contracts WHERE contract_id=?", [ao_contract]
            ).fetchone()
            if match is None:
                raise ValueError("Unknown AO contract_id")
            month = match[0]
        else:
            month = ao_contract["contract_month"]
        if cutoff is None:
            con.read_parquet(str(root / "ao_fixing_state.parquet")).create_view("states")
            match = con.execute(
                "SELECT cutoff_timestamp,cutoff_timestamp_raw FROM states WHERE observation_date=?::DATE AND contract_month=?",
                [d, month],
            ).fetchone()
            if not match or match[0] is None:
                raise ValueError(
                    "No observed AO publication cutoff; supply an explicit aware cutoff"
                )
            cutoff = match[1] or match[0]
        cutoff = pd.Timestamp(cutoff)
        if cutoff.tzinfo is None:
            raise ValueError("cutoff must be timezone-aware")
        cutoff = cutoff.tz_convert("UTC")
        cutoff_raw = (
            cutoff.strftime("%Y-%m-%dT%H:%M:%S")
            + f".{cutoff.microsecond * 1000 + cutoff.nanosecond:09d}Z"
        )
        con.read_parquet(str(root / "ao_fixing_schedule.parquet")).create_view("schedule")
        con.read_parquet(str(root / "cl_futures_settlement_revisions.parquet")).create_view(
            "revisions"
        )
        return con.execute(
            """SELECT s.* EXCLUDE(fixing_price,publication_timestamp,source_file),
          CASE WHEN s.fixing_date<=?::DATE AND r.update_action=1 THEN r.settlement_price END known_fixing_price,
          CASE WHEN s.fixing_date<=?::DATE THEN r.publication_timestamp END known_publication_timestamp,
          coalesce(s.fixing_date<=?::DATE AND r.update_action=1 AND r.settlement_price IS NOT NULL,false) known_fixing,
          s.fixing_date>?::DATE future_fixing
          FROM schedule s LEFT JOIN LATERAL (
          SELECT * FROM revisions r WHERE r.contract_id=s.reference_contract_id
          AND r.trade_date=s.fixing_date AND r.is_final AND r.publication_timestamp_raw<=?
          ORDER BY publication_timestamp_raw DESC,source_sequence DESC LIMIT 1) r ON true
          WHERE s.averaging_month=? ORDER BY s.fixing_date""",
            [d, d, d, d, cutoff_raw, month],
        ).df()
    finally:
        con.close()
