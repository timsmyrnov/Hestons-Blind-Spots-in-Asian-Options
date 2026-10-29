"""Historical definition snapshots retained as dated metadata evidence."""

from pathlib import Path
import duckdb
from .pipeline import write
from .csvio import csv_relation


def build_reference_versions(root):
    root = Path(root)
    files = sorted((root / "sources/databento_definition_snapshots").glob("*.csv"))
    if not files:
        raise FileNotFoundError("AO definitions absent")
    con = duckdb.connect()
    con.execute("SET enable_progress_bar=false")
    csv_relation(con, files, filename=True).create_view("defs")
    write(
        con,
        """SELECT DISTINCT
       'NYMEX:AO:'||printf('%04d-%02d',maturity_year::INT,maturity_month::INT)||':'||
       substr(split_part(raw_symbol,' ',2),1,1)||':'||printf('%.8f',strike_price::DOUBLE) contract_id,
       raw_symbol,instrument_id,underlying,currency,
       unit_of_measure_qty::DOUBLE contract_size,
       display_factor::DOUBLE display_factor,
       try_cast(expiration AS TIMESTAMPTZ) expiration_timestamp,
       try_cast(activation AS TIMESTAMPTZ) activation_timestamp,
       try_cast(ts_event AS TIMESTAMPTZ) definition_event_timestamp,
       try_cast(ts_recv AS TIMESTAMPTZ) definition_capture_timestamp,
       regexp_extract(filename,'[0-9]{4}-[0-9]{2}-[0-9]{2}')::DATE metadata_snapshot_date,
       filename source_file,
       'monthly historical snapshot, not complete daily validity history' validity_basis
       FROM defs WHERE regexp_full_match(raw_symbol,'AAO[FGHJKMNQUVXZ][0-9]{1,2} [CP]-?[0-9]+(\\.[0-9]+)?')
       ORDER BY contract_id,metadata_snapshot_date""",
        root / "ao_contract_reference_versions.parquet",
    )
    con.read_parquet(str(root / "ao_contracts.parquet")).create_view("master")
    con.read_parquet(str(root / "ao_contract_reference_versions.parquet")).create_view("versions")
    write(
        con,
        """SELECT m.* EXCLUDE(expiration_timestamp),
       v.expiration_timestamp,v.activation_timestamp,
       v.metadata_snapshot_date latest_definition_snapshot_date,
       v.definition_event_timestamp,v.source_file definition_source_file,
       v.display_factor,v.contract_size explicit_definition_contract_size
       FROM master m LEFT JOIN (SELECT * EXCLUDE(rn) FROM (
       SELECT *,row_number() OVER(PARTITION BY contract_id ORDER BY metadata_snapshot_date DESC,
       definition_capture_timestamp DESC) rn FROM versions) WHERE rn=1) v USING(contract_id)""",
        root / "ao_contracts.parquet",
    )
    result = con.sql("""SELECT count(*) definitions,count(DISTINCT contract_id) contracts,
       count_if(contract_size!=1000) wrong_contract_size,
       count_if(display_factor!=0.01) other_display_factor FROM versions""").fetchone()
    if result[2]:
        raise ValueError("Unexpected AO contract size")
    con.close()
    return dict(
        definition_records=result[0],
        explicit_contracts=result[1],
        wrong_contract_size=result[2],
        other_display_factor=result[3],
    )
