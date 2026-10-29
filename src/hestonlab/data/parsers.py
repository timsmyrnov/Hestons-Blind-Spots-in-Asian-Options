"""Exchange-format parsers, refactored from the acquired-data extractor."""

from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
import csv
import re
import hashlib
import math

BASE_FIELDS = "trade_date option_expiration contract_month strike call_put settlement_price volume open_interest underlying_reference source price_type".split()
EXTRA = "exchange clearing_code globex_root raw_symbol instrument_id currency price_unit settlement_status usable_settlement stat_flags stat_ts_recv source_file source_row prior_day_open_interest prior_day_volume expiration_status delta_published implied_volatility floor_volume globex_volume pnt_volume strike_raw prior_day_settlement_price settlement_change_raw close_range_raw".split()
MONTHS = {x: i + 1 for i, x in enumerate("JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split())}
MONTH_CODES = {x: i + 1 for i, x in enumerate("FGHJKMNQUVXZ")}


def now():
    return datetime.now(timezone.utc).isoformat()


def num(s):
    if s is None or str(s).strip() in ("", "----", "NULL", "NaN", "nan", "None"):
        return None
    v = float(s)
    return v if math.isfinite(v) and abs(v) < 9e9 else None


def month_name(s):
    return f"{2000 + int(s[-2:]):04d}-{MONTHS[s[:3]]:02d}"


def csvwrite(p, rows, fields=None):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(p.suffix + ".tmp")
    with temp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields or list(rows[0]), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    temp.replace(p)


def base(**kw):
    d = dict.fromkeys(BASE_FIELDS + EXTRA)
    d.update(
        exchange="NYMEX",
        clearing_code="AO",
        globex_root="AAO",
        currency="USD",
        price_unit="USD/barrel",
    )
    d.update(kw)
    p = d.get("settlement_price")
    d["usable_settlement"] = bool(p is not None and p > 0 and d.get("settlement_status") == "final")
    return d


def bulletin(pdf_path, trade_date):
    """Column-aware PG63 parser. Fail closed if layout/price/OI reconciliation changes."""
    import pdfplumber

    out = []
    totals = []
    active = None
    month = None
    finished = False
    with pdfplumber.open(pdf_path) as pdf:
        # No assumed AO page range: product heading resets prevent adjacent product contamination.
        header = " ".join((pdf.pages[0].extract_text() or "").split())
        label = datetime.strptime(trade_date, "%Y-%m-%d").strftime("%b %d, %Y").upper()
        if label not in header.upper():
            raise ValueError(f"Provided date {trade_date} does not match bulletin header")
        for i, p in enumerate(pdf.pages):
            bands = []
            for w in sorted(p.extract_words(x_tolerance=1, y_tolerance=2), key=lambda w: w["top"]):
                if not bands or abs(w["top"] - bands[-1][0]) > 2:
                    bands.append([w["top"], [w]])
                else:
                    bands[-1][1].append(w)
            for y, ws in bands:
                if y < 175 or y > 970:
                    continue
                ws = sorted(ws, key=lambda w: w["x0"])
                line = " ".join(w["text"] for w in ws)
                if re.match(r"^[A-Z0-9]+ (CALL|PUT|OPT) ", line):
                    m = re.match(r"^AO (CALL|PUT) CRUDE OIL APO OPTIONS", line)
                    new = m.group(1) if m else None
                    if new != active:
                        month = None
                    active = new
                    if out and active is None:
                        finished = True
                        break
                if not active:
                    continue
                if re.fullmatch(r"[A-Z]{3}\d{2}", line):
                    month = month_name(line)
                    continue

                def fld(a, b):
                    return " ".join(w["text"] for w in ws if a <= w["x0"] < b)

                def integer(a, b):
                    s = fld(a, b).replace(",", "").strip()
                    if not s or set(s) == {"-"}:
                        return None
                    m = re.fullmatch(r"(\d+)(?:\s*[+-])?", s)
                    if not m:
                        raise ValueError((i + 1, line, s))
                    return int(m.group(1))

                if line.startswith("TOTAL"):
                    totals.append((active[0], month, integer(540, 570)))
                    continue
                if not (month and ws[0]["x0"] < 35 and re.fullmatch(r"-?\d{1,6}", ws[0]["text"])):
                    continue
                raw = fld(316, 352)
                if not re.fullmatch(r"\d+\.\d+", raw):
                    raise ValueError(("invalid settlement", i + 1, line))
                r = base(
                    trade_date=trade_date,
                    contract_month=month,
                    strike=int(ws[0]["text"]) / 100,
                    call_put=active[0],
                    settlement_price=num(raw),
                    open_interest=integer(540, 570),
                    underlying_reference="CS (WTI calendar-month average of CL settlements)",
                    source="CME Daily Bulletin PG63",
                    price_type="official_exchange_settlement",
                    settlement_status="final",
                    expiration_status="not_published_in_bulletin",
                    source_file=Path(pdf_path).name,
                    source_row=f"page {i + 1}: {line}",
                )
                r.update(
                    strike_raw=ws[0]["text"],
                    floor_volume=integer(440, 474),
                    globex_volume=integer(474, 505),
                    pnt_volume=integer(505, 540),
                    delta_published=num(fld(390, 417)),
                )
                # Venue blanks are missing; never assume total volume is zero.
                if all(r[k] is not None for k in ["floor_volume", "globex_volume", "pnt_volume"]):
                    r["volume"] = sum(r[k] for k in ["floor_volume", "globex_volume", "pnt_volume"])
                r["settlement_change_raw"] = fld(352, 390)
                r["close_range_raw"] = fld(265, 316)
                out.append(r)
            if finished:
                break
    sums = defaultdict(int)
    for r in out:
        sums[(r["call_put"], r["contract_month"])] += r["open_interest"] or 0
    if not out or not totals or not all(sums[(c, m)] == v for c, m, v in totals):
        raise ValueError("OI totals fail validation")
    if len({(r["contract_month"], r["call_put"], r["strike"]) for r in out}) != len(out):
        raise ValueError("duplicate rows")
    return out, {
        "printed_oi_total_checks": len(totals),
        "all_match": True,
        "source_sha256": hashlib.sha256(Path(pdf_path).read_bytes()).hexdigest(),
    }


def clearport(path):
    lines = Path(path).read_text().splitlines()
    m = re.search(r"AS OF (\d{2}/\d{2}/\d{4})", lines[0])
    if not m:
        raise ValueError("Missing dated final ClearPort header")
    d = datetime.strptime(m[1], "%m/%d/%Y").date().isoformat()
    out = []
    active = None
    for n, x in enumerate(lines, 1):
        m = re.fullmatch(r"AO ([A-Z]{3}\d{2}) WTI Average Price Option (CALL|PUT)", x)
        if m:
            active = m.groups()
            continue
        if x and x[0].isalpha() and not x.startswith("TOTAL"):
            active = None
        if active and re.match(r"^-?\d+\.\d+\s", x):
            out.append(
                base(
                    trade_date=d,
                    contract_month=month_name(active[0]),
                    strike=num(x[:12]),
                    call_put=active[1][0],
                    settlement_price=num(x[61:74]),
                    volume=num(x[87:100]),
                    strike_raw=x[:12].strip(),
                    settlement_change_raw=x[74:87].strip(),
                    prior_day_settlement_price=num(x[100:114]),
                    prior_day_volume=num(x[114:126]),
                    prior_day_open_interest=num(x[126:140]),
                    underlying_reference="CS (WTI calendar-month average of CL settlements)",
                    source="CME ClearPort final settlement file",
                    price_type="official_exchange_settlement",
                    settlement_status="final",
                    expiration_status="not_published_in_file",
                    source_file=Path(path).name,
                    source_row=n,
                )
            )
    return out


def eod_sample(path):
    out = []
    for n, r in enumerate(csv.DictReader(Path(path).open()), 2):
        if r["Product Code"] != "AO":
            continue
        out.append(
            base(
                trade_date=datetime.strptime(r["Trade Date"], "%Y%m%d").date().isoformat(),
                option_expiration=datetime.strptime(r["Last Trade Date"], "%Y%m%d")
                .date()
                .isoformat(),
                contract_month=f"{int(r['Contract Year']):04d}-{int(r['Contract Month']):02d}",
                strike=num(r["Strike Price"]) / 100,
                strike_raw=r["Strike Price"],
                delta_published=num(r.get("Delta")),
                implied_volatility=num(r.get("Implied Volatility")),
                globex_volume=num(r.get("Globex Volume")),
                floor_volume=num(r.get("Floor Volume")),
                pnt_volume=num(r.get("PNT Volume")),
                call_put=r["Put/Call"],
                settlement_price=num(r["Settlement"]),
                volume=num(r["Total Volume"]),
                open_interest=num(r["Open Interest"]),
                underlying_reference=r["Underlying Product Code"],
                source="CME DataMine AO free sample",
                price_type="official_exchange_settlement",
                settlement_status="final",
                expiration_status="explicit_last_trade_date",
                source_file=Path(path).name,
                source_row=n,
            )
        )
    return out
