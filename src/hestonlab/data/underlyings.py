"""Expiry-specific CL/CSX settlements. All revisions retained for as-of fixing joins."""

from pathlib import Path
import json
import hashlib
import re
import pandas as pd
import duckdb
import pyarrow as pa
from .pipeline import write, literal
from .csvio import csv_relation
from .parsers import month_name, num


def parse_cl_exchange(path):
    lines = Path(path).read_text().splitlines()
    match = re.search(r"AS OF (\d{2}/\d{2}/\d{4})", lines[0])
    if not match:
        raise ValueError("Missing dated exchange header")
    d = pd.to_datetime(match[1]).date().isoformat()
    active = False
    rows = []
    for n, line in enumerate(lines, 1):
        if line.startswith("CL Light Sweet Crude Oil Futures"):
            active = True
            continue
        if active and re.match(r"^[A-Z]{3}\d{2}\s", line):
            rows.append(
                dict(
                    trade_date=d,
                    contract_month=month_name(line[:5]),
                    settlement_price=num(line[62:75]),
                    volume=num(line[87:100]),
                    prior_open_interest=num(line[126:140]),
                    source_record=n,
                    source_file=Path(path).name,
                    price_unit="USD/barrel",
                )
            )
        elif active:
            break
    if not rows:
        raise ValueError("No CL futures rows")
    return pd.DataFrame(rows)


def ingest(root):
    root = Path(root)
    con = duckdb.connect()
    con.execute("SET enable_progress_bar=false")
    con.execute("SET memory_limit='2GB'")
    definitions = list((root / "sources/underlying_definitions").glob("*.csv"))
    if not definitions:
        raise FileNotFoundError("Matching underlying definitions not downloaded")
    csv_relation(con, definitions).create_view("defs")
    con.execute("""CREATE TEMP TABLE contracts AS SELECT DISTINCT
        instrument_id,raw_symbol,
        CASE WHEN raw_symbol LIKE 'CSX%' THEN 'CSX' ELSE 'CL' END root,
        printf('%04d-%02d',maturity_year::INT,maturity_month::INT) contract_month,
        expiration::TIMESTAMPTZ expiration_timestamp,expiration::TIMESTAMPTZ::DATE expiration,
        (CASE WHEN raw_symbol LIKE 'CSX%' THEN 'CSX' ELSE 'CL' END)||':'||
        printf('%04d-%02d',maturity_year::INT,maturity_month::INT) contract_id
        FROM defs WHERE regexp_full_match(raw_symbol,'(CL|CSX)[FGHJKMNQUVXZ][0-9]{1,2}')""")
    conflicts = con.sql("""SELECT instrument_id FROM contracts GROUP BY instrument_id
       HAVING count(DISTINCT contract_id)>1 OR count(DISTINCT expiration)>1""").fetchall()
    if conflicts:
        raise ValueError(f"Conflicting underlying definitions: {conflicts[:3]}")
    write(con, "SELECT DISTINCT * FROM contracts", root / "underlying_contracts.parquet")
    files = sorted((root / "sources/underlying_statistics").glob("**/*.statistics.csv.zst"))
    if not files:
        raise FileNotFoundError("Underlying batch files not downloaded")
    # Individual monthly checkpoints allow resume without a second API request.
    parts = root / "evidence/underlying_revision_parts"
    parts.mkdir(exist_ok=True)
    for p in files:
        target = parts / (p.parent.name + "_" + p.name.replace(".csv.zst", ".parquet"))
        fingerprint = hashlib.sha256(p.read_bytes()).hexdigest()
        receipt = target.with_suffix(".source.json")
        if target.exists() and receipt.exists():
            old = json.loads(receipt.read_text())
            if old.get("sha256") == fingerprint and old.get("normalizer_version") == 2:
                continue
        csv_relation(con, p).create_view("raw", replace=True)
        write(
            con,
            f"""SELECT DISTINCT substr(ts_ref,1,10)::DATE trade_date,
            symbol raw_symbol,instrument_id,
            CASE WHEN isfinite(try_cast(price AS DOUBLE)) AND abs(try_cast(price AS DOUBLE))<9e9
            THEN try_cast(price AS DOUBLE) END settlement_price,
            raw_source_row source_record_reference,
            stat_flags::INT stat_flags,(stat_flags::INT & 1)=1 is_final,
            update_action::INT update_action,ts_recv::TIMESTAMPTZ publication_timestamp,
            ts_recv publication_timestamp_raw,try_cast(sequence AS BIGINT) source_sequence,
            {literal(str(p.relative_to(root)))} source_file,
            'GLBX.MDP3 statistics' AS "source",'USD/barrel' price_unit
            FROM (SELECT *,row_number() OVER ()+1 raw_source_row FROM raw) WHERE stat_type::INT=3 AND (stat_flags::INT & 12)=0
            AND regexp_full_match(symbol,'(CL|CSX)[FGHJKMNQUVXZ][0-9]{{1,2}}')
            AND try_cast(ts_ref AS TIMESTAMPTZ) IS NOT NULL""",
            target,
        )
        receipt.write_text(
            json.dumps(
                dict(
                    sha256=fingerprint, normalizer_version=2, source_file=str(p.relative_to(root))
                ),
                indent=2,
            )
        )
        print("Underlying revisions", p.name, flush=True)
    con.read_parquet(str(parts / "*.parquet")).create_view("parts")
    con.execute("""CREATE TEMP TABLE revisions AS SELECT p.*,c.root,c.contract_month,
        c.contract_id,c.expiration FROM parts p LEFT JOIN contracts c USING(instrument_id,raw_symbol)""")
    write(con, "SELECT DISTINCT * FROM revisions", root / "cl_futures_settlement_revisions.parquet")
    write(
        con,
        """SELECT * EXCLUDE(rn) FROM (SELECT *, row_number() OVER(
        PARTITION BY trade_date,contract_id ORDER BY is_final DESC,publication_timestamp_raw DESC,source_sequence DESC) rn
        FROM revisions WHERE contract_id IS NOT NULL) WHERE rn=1
        ORDER BY trade_date,contract_id""",
        root / "cl_futures_settlements.parquet",
    )
    missing = con.sql("SELECT count(*) FROM revisions WHERE contract_id IS NULL").fetchone()[0]
    stats = con.sql(
        """SELECT root,count(*) AS "rows",count(DISTINCT trade_date) dates,
       count(DISTINCT contract_id) contracts,min(trade_date)::VARCHAR first_date,
       max(trade_date)::VARCHAR last_date,count_if(settlement_price<0) negative_prices
       FROM read_parquet("""
        + literal(root / "cl_futures_settlements.parquet")
        + """) GROUP BY root"""
    ).df()
    stats.to_csv(root / "evidence/underlying_coverage.csv", index=False)
    sample = parse_cl_exchange(root / "sources/STLNYMEX_20230403")
    sample.to_csv(root / "cl_exchange_sample.csv", index=False)
    con.register("sample", pa.Table.from_pandas(sample))
    checks = con.sql(
        """SELECT s.*, f.settlement_price vendor_settlement,
        s.settlement_price-f.settlement_price difference
        FROM sample s LEFT JOIN read_parquet("""
        + literal(root / "cl_futures_settlements.parquet")
        + """) f
        ON s.trade_date::DATE=f.trade_date AND s.contract_month=f.contract_month AND f.root='CL' """
    ).df()
    checks.to_csv(root / "evidence/cl_exchange_crosscheck.csv", index=False)
    comparable = checks.vendor_settlement.notna()
    if (checks.loc[comparable, "difference"].abs() > 1e-8).any():
        raise ValueError(
            "CL settlement differs from original exchange file; inspect crosscheck before research joins"
        )
    (root / "evidence/underlying_ingestion.json").write_text(
        json.dumps(
            dict(
                unmapped_revision_records=missing,
                matched_exchange_sample=int(checks.difference.eq(0).sum()),
                exchange_sample_rows=len(checks),
                stats=stats.to_dict("records"),
            ),
            indent=2,
        )
    )
    con.close()
