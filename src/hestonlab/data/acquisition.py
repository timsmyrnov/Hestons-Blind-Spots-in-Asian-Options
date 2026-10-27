"""Existing AO collection and normalization implementation, retained for backfill."""

from pathlib import Path
from datetime import timedelta, date
from collections import defaultdict
import csv
import json
import re
import hashlib
import sqlite3
from .parsers import (
    BASE_FIELDS,
    EXTRA,
    MONTH_CODES,
    now,
    num,
    month_name,
    csvwrite,
    base,
    clearport,
    eod_sample,
)


def load_definitions(root):
    """Use explicit maturity fields and expiry metadata, with conflict detection."""
    import pandas as pd

    lookup = {}
    calendar = {}
    calendar_sources = defaultdict(set)
    checked = 0
    for p in sorted((root / "sources" / "databento_definition_snapshots").glob("*.csv")):
        for chunk in pd.read_csv(p, chunksize=100000):
            for r in chunk.to_dict("records"):
                raw = str(r["raw_symbol"])
                if not re.fullmatch(r"AAO[FGHJKMNQUVXZ]\d{1,2} [CP]-?\d+(?:\.\d+)?", raw):
                    continue
                mo = f"{int(r['maturity_year']):04d}-{int(r['maturity_month']):02d}"
                expiry = str(r["expiration"])[:10]
                strike = round(float(r["strike_price"]), 8)
                suffix = float(raw.split(" ")[1][1:])
                if abs(suffix / 100 - strike) > 1e-8:
                    raise ValueError(("strike scaling conflict", raw, strike))
                checked += 1
                # Same month can have exchange time changes, but calendar expiry must agree.
                if mo in calendar and calendar[mo] != expiry:
                    raise ValueError(("expiry conflict", mo, calendar[mo], expiry))
                calendar[mo] = expiry
                calendar_sources[mo].add(str(p.relative_to(root)))
                lookup[(mo, raw.split(" ")[1][0], strike)] = (expiry, str(r["underlying"]))
    csvwrite(
        root / "evidence/contract_calendar.csv",
        [
            dict(
                contract_month=mo,
                option_expiration=ex,
                definition_sources=" | ".join(sorted(calendar_sources[mo])),
            )
            for mo, ex in sorted(calendar.items())
        ],
        ["contract_month", "option_expiration", "definition_sources"],
    )
    return lookup, calendar, checked


def raw_identity(raw, trade_date):
    m = re.fullmatch(r"AAO([FGHJKMNQUVXZ])(\d{1,2}) ([CP])(-?\d+(?:\.\d+)?)", str(raw))
    if not m:
        return None
    yy = m[2]
    year = 2000 + int(yy) if len(yy) == 2 else int(trade_date[:3]) * 10 + int(yy)
    if len(yy) == 1 and year < int(trade_date[:4]) - 1:
        year += 10
    return f"{year:04d}-{MONTH_CODES[m[1]]:02d}", m[3], round(float(m[4]) / 100, 8)


def normalize(root):
    import pandas as pd

    root = Path(root)
    (root / "evidence").mkdir(exist_ok=True)
    receipt = root / "evidence/databento_download.json"
    if receipt.exists():
        expected = json.loads(receipt.read_text()).get("files", [])
        absent = [x for x in expected if not (root / x).is_file()]
        if absent:
            raise FileNotFoundError(
                f"{len(absent)} bulk source files are absent. Copy the preserved sources folder or download the existing batch before normalizing; existing outputs were not replaced."
            )
    lookup, calendar, checked = load_definitions(root)
    dbpath = root / "evidence" / "ao_normalization.sqlite"
    dbpath.unlink(missing_ok=True)
    con = sqlite3.connect(dbpath)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA cache_size=-128000")
    con.execute(
        "CREATE TABLE stats (d TEXT,s TEXT,t INTEGER,p REAL,q REAL,f INTEGER,ts TEXT,i TEXT,src TEXT,n INTEGER,action INTEGER,PRIMARY KEY(d,s,t))"
    )
    rawcounts = {}
    rejected = defaultdict(int)
    files = sorted(
        set((root / "sources").glob("AAO_statistics*.csv"))
        | set((root / "sources" / "databento_statistics").glob("**/*.csv.zst"))
        | set((root / "sources" / "databento_statistics").glob("**/*.csv"))
    )
    for p in files:
        n = 0
        for ch in pd.read_csv(p, chunksize=250000, compression="infer", dtype={"symbol": str}):
            n += len(ch)
            ch["source_row_number"] = ch.index + 2
            ch = ch[ch.stat_type.isin([3, 6, 9])].copy()
            ch["d"] = ch.ts_ref.astype(str).str[:10]
            ch = ch[
                ch.d.str.match(r"20\d\d-\d\d-\d\d")
                & ch.symbol.str.match(r"^AAO[FGHJKMNQUVXZ]\d{1,2} [CP]-?\d+(?:\.\d+)?$", na=False)
            ]
            alternate = (ch.stat_type == 3) & ((ch.stat_flags & 12) != 0)
            rejected["intraday_or_trading_tick"] += int(alternate.sum())
            ch = ch[~alternate]
            # Raw streams are capture-time ordered. Collapse repeated updates within a
            # chunk, keeping final and preliminary classes separate before the global merge.
            ch["finalclass"] = (ch.stat_type == 3).astype(int) * (ch.stat_flags & 1)
            if not ch.ts_recv.is_monotonic_increasing:
                ch = ch.sort_values("ts_recv", kind="stable")
            ch = ch.drop_duplicates(["d", "symbol", "stat_type", "finalclass"], keep="last")
            data = []
            for r in ch.itertuples(index=False):
                f = int(r.stat_flags)
                t = int(r.stat_type)
                data.append(
                    (
                        r.d,
                        r.symbol,
                        t,
                        num(r.price),
                        num(r.quantity),
                        f,
                        str(r.ts_recv),
                        str(r.instrument_id),
                        str(p.relative_to(root)),
                        int(r.source_row_number),
                        int(r.update_action),
                    )
                )
            # For settlements prefer final; within a status, latest capture timestamp wins.
            con.executemany(
                """INSERT INTO stats VALUES(?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(d,s,t) DO UPDATE SET p=excluded.p,q=excluded.q,f=excluded.f,ts=excluded.ts,i=excluded.i,src=excluded.src,n=excluded.n,action=excluded.action WHERE
   (excluded.t=3 AND (excluded.f&1)>(stats.f&1)) OR ((excluded.t!=3 OR (excluded.f&1)=(stats.f&1)) AND excluded.ts>=stats.ts)""",
                data,
            )
            con.commit()
        rawcounts[str(p.relative_to(root))] = n
        print("normalized source", p.name, n, flush=True)
    # Retain independent exchange observations with separate source identities.
    official = []
    for p in sorted((root / "sources").glob("STLCPC_*")):
        if p.is_file():
            official += clearport(p)
    p = root / "sources" / "CME_AO_official_sample_2026-10-08.csv"
    if p.exists():
        official += eod_sample(p)
    for p in sorted((root / "bulletin_extracts").glob("*.csv")):
        for row in csv.DictReader(p.open()):
            for k in ["strike", "settlement_price", "volume", "open_interest"]:
                row[k] = num(row.get(k))
            official.append(base(**{k: v for k, v in row.items() if k in BASE_FIELDS + EXTRA}))
    cached = root / "AO_actual_settlements_2026-10-26.csv"
    if cached.exists() and not (root / "bulletin_extracts").exists():
        for r in csv.DictReader(cached.open()):
            official.append(
                base(
                    trade_date=r["trade_date"],
                    contract_month=month_name(r["contract_month"]),
                    strike=num(r["strike_usd_per_barrel"]),
                    strike_raw=r.get("strike_raw"),
                    delta_published=num(r.get("delta_published")),
                    globex_volume=num(r.get("globex_volume_published")),
                    floor_volume=num(r.get("floor_volume_published")),
                    pnt_volume=num(r.get("pnt_volume_published")),
                    call_put=r["put_call"][0],
                    settlement_price=num(r["settlement_usd_per_barrel"]),
                    open_interest=num(r["open_interest"]),
                    underlying_reference="CS",
                    source="CME Daily Bulletin PG63",
                    price_type="official_exchange_settlement",
                    settlement_status="final",
                    source_file="Section63_Energy_Options_Products.pdf",
                    source_row=f"page {r['pdf_page']}: {r['raw_line']}",
                )
            )
    for r in official:
        ex, under = lookup.get(
            (r["contract_month"], r["call_put"], r["strike"]),
            (calendar.get(r["contract_month"]), None),
        )
        if not r["option_expiration"]:
            r["option_expiration"] = ex
            r["expiration_status"] = (
                "same_month_exchange_definition" if ex else "missing_definition"
            )
        if under:
            r["underlying_reference"] = under
    csvwrite(root / "ao_exchange_samples.csv", official, BASE_FIELDS + EXTRA)
    officialmap = {
        (r["trade_date"], r["contract_month"], r["strike"], r["call_put"]): r for r in official
    }
    cross = []
    coverage = []
    sample = []
    total = usable = 0
    dates = []

    def summarize(d, source, rr):
        return dict(
            trade_date=d,
            source=source,
            records=len(rr),
            usable_observations=sum(r["usable_settlement"] for r in rr),
            usable_calls=sum(r["call_put"] == "C" and r["usable_settlement"] for r in rr),
            usable_puts=sum(r["call_put"] == "P" and r["usable_settlement"] for r in rr),
            calls=sum(r["call_put"] == "C" for r in rr),
            puts=sum(r["call_put"] == "P" for r in rr),
            usable_strikes=len({r["strike"] for r in rr if r["usable_settlement"]}),
            distinct_strikes=len({r["strike"] for r in rr}),
            contract_months=len({r["contract_month"] for r in rr}),
            expirations=len({r["option_expiration"] for r in rr if r["option_expiration"]}),
            missing_expiration=sum(not r["option_expiration"] for r in rr),
            missing_volume=sum(r["volume"] is None for r in rr),
            missing_open_interest=sum(r["open_interest"] is None for r in rr),
            zero_prices=sum(r["settlement_price"] == 0 for r in rr),
            preliminary=sum(r["settlement_status"] == "preliminary" for r in rr),
            completeness="observed settlement subset; full listing denominator not established",
        )

    import pyarrow as pa
    import pyarrow.parquet as pq

    types = {k: pa.string() for k in BASE_FIELDS + EXTRA}
    for k in [
        "strike",
        "settlement_price",
        "volume",
        "open_interest",
        "prior_day_open_interest",
        "prior_day_volume",
        "delta_published",
        "implied_volatility",
        "floor_volume",
        "globex_volume",
        "pnt_volume",
        "prior_day_settlement_price",
    ]:
        types[k] = pa.float64()
    types["usable_settlement"] = pa.bool_()
    types["stat_flags"] = pa.int64()
    schema = pa.schema([(k, types[k]) for k in BASE_FIELDS + EXTRA])
    pw = pq.ParquetWriter(root / "ao_historical_panel.parquet", schema, compression="zstd")
    vector_schema = pa.schema(
        [
            (k, types[k])
            for k in ["trade_date", "contract_month", "strike", "call_put", "settlement_price"]
        ]
    )
    vw = pq.ParquetWriter(
        root / "ao_daily_price_vectors.parquet", vector_schema, compression="zstd"
    )
    f = (root / "ao_historical_panel.csv").open("w", newline="")
    writer = csv.DictWriter(f, fieldnames=BASE_FIELDS + EXTRA)
    writer.writeheader()

    def flush(rr):
        nonlocal total, usable
        if not rr:
            return
        if len({(r["contract_month"], r["strike"], r["call_put"]) for r in rr}) != len(rr):
            raise ValueError("duplicate contract identity on day")
        d = rr[0]["trade_date"]
        dates.append(d)
        total += len(rr)
        usable += sum(r["usable_settlement"] for r in rr)
        writer.writerows(rr)
        coverage.append(summarize(d, "Databento", rr))
        converted = [{**r, "source_row": str(r["source_row"])} for r in rr]
        pw.write_table(pa.Table.from_pylist(converted, schema=schema))
        vv = [{k: r[k] for k in vector_schema.names} for r in rr if r["usable_settlement"]]
        if vv:
            vw.write_table(pa.Table.from_pylist(vv, schema=vector_schema))
        # Include full stress-date and recent API observations, plus exchange documents.
        if d in ["2020-04-20", "2020-04-21", "2026-10-26"]:
            sample.extend(rr)
        for r in rr:
            z = officialmap.get((d, r["contract_month"], r["strike"], r["call_put"]))
            if z:
                cross.append(
                    dict(
                        trade_date=d,
                        contract_month=r["contract_month"],
                        strike=r["strike"],
                        call_put=r["call_put"],
                        databento_price=r["settlement_price"],
                        exchange_price=z["settlement_price"],
                        difference=None
                        if r["settlement_price"] is None
                        else round(r["settlement_price"] - z["settlement_price"], 10),
                        databento_status=r["settlement_status"],
                        exchange_source=z["source"],
                    )
                )

    rr = []
    last = None
    for d, s, p, fla, ts, i, src, n, action, qv, qo in con.execute(
        """SELECT a.d,a.s,a.p,a.f,a.ts,a.i,a.src,a.n,a.action,v.q,o.q FROM stats a LEFT JOIN stats v ON a.d=v.d AND a.s=v.s AND v.t=6 AND v.action=1 LEFT JOIN stats o ON a.d=o.d AND a.s=o.s AND o.t=9 AND o.action=1 WHERE a.t=3 ORDER BY a.d,a.s"""
    ):
        mo, cp, k = raw_identity(s, d)
        ex, under = lookup.get((mo, cp, k), (calendar.get(mo), None))
        status = "deleted" if action != 1 else "final" if fla & 1 else "preliminary"
        if p is None:
            status = "missing_price"
        elif p == 0:
            status = "zero_price"
        elif p < 0:
            status = "negative_price"
        r = base(
            trade_date=d,
            option_expiration=ex,
            contract_month=mo,
            strike=k,
            call_put=cp,
            settlement_price=p,
            volume=qv,
            open_interest=qo,
            underlying_reference=under or f"CSX/{mo}",
            source="CME MDP via Databento GLBX.MDP3",
            price_type="official_final_actual"
            if fla & 1 and fla & 2
            else "official_final_theoretical"
            if fla & 1
            else "official_preliminary",
            settlement_status=status,
            stat_flags=fla,
            stat_ts_recv=ts,
            raw_symbol=s,
            instrument_id=i,
            source_file=src,
            source_row=n,
            expiration_status="explicit_definition"
            if (mo, cp, k) in lookup
            else "same_month_exchange_definition"
            if ex
            else "missing_definition",
        )
        if last != d:
            flush(rr)
            rr = []
            last = d
        rr.append(r)
    flush(rr)
    f.close()
    pw.close()
    vw.close()
    con.close()
    dbpath.unlink(missing_ok=True)
    groups = defaultdict(list)
    for r in official:
        groups[r["trade_date"]].append(r)
    for d, rr in sorted(groups.items()):
        coverage.append(summarize(d, "CME free documents", rr))
    csvwrite(root / "ao_historical_sample.csv", official + sample, BASE_FIELDS + EXTRA)
    # Explicitly list requested weekdays with no observations; do not call them holidays.
    if dates:
        start = date.fromisoformat(min(dates))
        end = date.fromisoformat(max(dates))
        present = set(dates)
        cursor = start
        while cursor <= end:
            if cursor.weekday() < 5 and cursor.isoformat() not in present:
                empty = summarize(cursor.isoformat(), "Databento", [])
                empty["completeness"] = (
                    "no records: exchange holiday or missing observations; not inferred"
                )
                coverage.append(empty)
            cursor += timedelta(days=1)
    coverage.sort(key=lambda x: (x["trade_date"], x["source"]))
    csvwrite(root / "ao_coverage_summary.csv", coverage)
    csvwrite(
        root / "ao_missing_dates.csv", [r for r in coverage if not r["records"]], list(coverage[0])
    )
    if cross:
        csvwrite(root / "evidence" / "cross_source_validation.csv", cross)
    meta = dict(
        generated_utc=now(),
        raw_statistics_counts=rawcounts,
        panel_rows=total,
        usable_rows=usable,
        usable_dates=sum(
            r["usable_observations"] > 0 for r in coverage if r["source"] == "Databento"
        ),
        dates=len(dates),
        first_date=min(dates, default=None),
        last_date=max(dates, default=None),
        exchange_sample_rows=len(official),
        sample_rows=len(official) + len(sample),
        definition_scale_checks=checked,
        rejected=dict(rejected),
        cross_source_comparisons=len(cross),
        cross_source_matches=sum(r["difference"] == 0 for r in cross),
    )
    (root / "evidence" / "normalization_summary.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


def collect(args):
    import databento as db
    import pandas as pd

    receipt = Path(args.root) / "evidence/databento_batch_details.json"
    if receipt.exists():
        old = json.loads(receipt.read_text())
        if (
            old.get("state") == "done"
            and old.get("schema") == "statistics"
            and pd.to_datetime(old["start"], utc=True) <= pd.to_datetime(args.start, utc=True)
            and pd.to_datetime(old["end"], utc=True) >= pd.to_datetime(args.end, utc=True)
        ):
            raise SystemExit(
                "Existing completed job "
                + old["id"]
                + " already covers this capture window. Use download, not another paid collection."
            )
    c = db.Historical()
    kw = dict(
        dataset="GLBX.MDP3",
        symbols=["AAO.OPT"],
        stype_in="parent",
        schema="statistics",
        start=args.start,
        end=args.end,
    )
    cost = c.metadata.get_cost(**kw)
    print("Estimated cost USD", cost)
    if not args.approve or cost > args.max_cost:
        raise SystemExit("No collection: require explicit --approve and estimate <= --max-cost")
    if args.batch:
        evidence = Path(args.root) / "evidence"
        evidence.mkdir(parents=True, exist_ok=True)
        receipt = evidence / f"batch_{args.start}_{args.end}.json"
        if receipt.exists():
            raise SystemExit("Existing request receipt: download the existing job; do not resubmit")
        result = c.batch.submit_job(
            **kw,
            encoding="csv",
            compression="zstd",
            pretty_px=True,
            pretty_ts=True,
            map_symbols=True,
            split_duration="month",
        )
        result.pop("user_id", None)
        receipt.write_text(json.dumps(result, indent=2, default=str))
        print(json.dumps(result, indent=2, default=str))
        return
    p = Path(args.root) / "sources" / f"AAO_statistics_{args.start}_{args.end}.dbn.zst"
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        raise SystemExit("Refusing to overwrite existing source or charge for duplicate retrieval")
    store = c.timeseries.get_range(**kw, path=p)
    store.to_csv(
        p.with_suffix("").with_suffix(".csv"), map_symbols=True, pretty_px=True, pretty_ts=True
    )
    p.with_suffix(".collection.json").write_text(
        json.dumps(
            dict(
                parameters=kw,
                estimated_cost=cost,
                collected_utc=now(),
                sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
            ),
            indent=2,
        )
    )


def definitions(args):
    import databento as db

    c = db.Historical()
    first = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    dest = Path(args.root) / "sources/databento_definition_snapshots"
    dest.mkdir(parents=True, exist_ok=True)
    plans = []
    y, m = first.year, first.month
    while (y, m) <= (end.year, end.month):
        d = date(y, m, 8)
        while d.weekday() != 2:
            d += timedelta(days=1)
        p = dest / f"AAO_definition_{d}.dbn.zst"
        if first <= d < end and not p.exists():
            kw = dict(
                dataset="GLBX.MDP3",
                symbols=["AAO.OPT"],
                stype_in="parent",
                schema="definition",
                start=str(d),
                end=str(d + timedelta(days=1)),
            )
            plans.append((kw, p, c.metadata.get_cost(**kw)))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    cost = sum(x[2] for x in plans)
    print("Definition snapshot estimate USD", cost, "requests", len(plans))
    if not args.approve or cost > args.max_cost:
        raise SystemExit(
            "No collection: require explicit --approve and total estimate <= --max-cost"
        )
    log = []
    for kw, p, price in plans:
        c.timeseries.get_range(**kw, path=p)
        db.DBNStore.from_file(p).to_csv(
            p.with_suffix("").with_suffix(".csv"), map_symbols=True, pretty_px=True, pretty_ts=True
        )
        log.append(
            dict(
                parameters=kw,
                estimated_cost=price,
                collected_utc=now(),
                sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
            )
        )
    (dest / "collection.json").write_text(json.dumps(log, indent=2))


def download(args):
    import databento as db

    c = db.Historical()
    details = c.batch.get_job_details(args.job_id)
    details.pop("user_id", None)
    print("Batch state", details["state"])
    if details["state"] != "done":
        return
    paths = c.batch.download(
        args.job_id, output_dir=Path(args.root) / "sources/databento_statistics"
    )
    evidence = Path(args.root) / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / f"{args.job_id}.json").write_text(json.dumps(details, indent=2, default=str))
    print("Downloaded files", len(paths))


def audit(root):
    """Flag vendor dataset-level degraded capture dates without assuming AO damage."""
    import pyarrow.parquet as pq

    root = Path(root)
    bad = set()
    conditions = []
    for p in (root / "sources/databento_statistics").glob("**/condition.json"):
        conditions += json.loads(p.read_text())
    badrows = [x for x in conditions if x["condition"] != "available"]
    bad = {x["date"] for x in badrows}
    if badrows:
        csvwrite(root / "evidence/databento_dataset_condition_exceptions.csv", badrows)
    counts = defaultdict(int)
    flagged = 0
    path = root / "ao_source_quality_flags.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "trade_date",
                "raw_symbol",
                "selected_capture_date",
                "source_file",
                "source_row",
                "flag",
            ],
        )
        w.writeheader()
        for batch in pq.ParquetFile(root / "ao_historical_panel.parquet").iter_batches(
            batch_size=100000,
            columns=["trade_date", "raw_symbol", "stat_ts_recv", "source_file", "source_row"],
        ):
            cols = batch.to_pydict()
            for d, s, ts, src, n in zip(
                *(
                    cols[x]
                    for x in [
                        "trade_date",
                        "raw_symbol",
                        "stat_ts_recv",
                        "source_file",
                        "source_row",
                    ]
                )
            ):
                if ts[:10] in bad:
                    counts[d] += 1
                    flagged += 1
                    w.writerow(
                        dict(
                            trade_date=d,
                            raw_symbol=s,
                            selected_capture_date=ts[:10],
                            source_file=src,
                            source_row=n,
                            flag="dataset capture day degraded; AO impact not established",
                        )
                    )
    coverage_path = root / "ao_coverage_summary.csv"
    if not coverage_path.read_text().splitlines()[0].startswith("trade_date,"):
        coverage_path = root / "ao_coverage_summary_by_source_legacy.csv"
    cover = list(csv.DictReader(coverage_path.open()))
    for r in cover:
        r["selected_settlements_on_degraded_dataset_capture_day"] = (
            counts[r["trade_date"]] if r["source"] == "Databento" else ""
        )
    csvwrite(coverage_path, cover)
    quality = dict(
        dataset_degraded_capture_dates=len(bad),
        selected_settlement_records_on_those_capture_dates=flagged,
        note="Dataset-wide flags are not a claim of AO loss or error; reference dates can differ from capture dates.",
    )
    (root / "evidence/source_quality_summary.json").write_text(json.dumps(quality, indent=2))
    print(json.dumps(quality))
