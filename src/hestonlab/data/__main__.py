"""Local HestonLab data workflow; collection always requires a spending guard."""

import argparse
from pathlib import Path
import json
from .research import DEFAULT_ROOT, load_ao_panel


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    sub = ap.add_subparsers(dest="command", required=True)
    for command in ["build", "coverage", "validate", "audit", "underlyings", "fixings"]:
        sub.add_parser(command)
    p = sub.add_parser("parse", help="Parse an existing exchange original without network")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--format", choices=["bulletin", "clearport", "eod", "cl"], required=True)
    p.add_argument("--date")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("export")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--include-ineligible", action="store_true")
    p = sub.add_parser("backfill", help="Quote or explicitly approve guarded AO batch collection")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--max-cost", type=float, default=0)
    p.add_argument("--approve", action="store_true")
    p.add_argument("--batch", action="store_true", default=True)
    p = sub.add_parser(
        "collect-underlyings",
        help="Quote missing CL/CSX work, or explicitly approve a capped collection",
    )
    p.add_argument("--start", default="2019-12-01")
    p.add_argument("--end", default="2026-10-27")
    p.add_argument("--max-cost", type=float, default=0)
    p.add_argument("--approve", action="store_true")
    p = sub.add_parser("download", help="Resume an existing job; never submit another request")
    p.add_argument("--job-id", required=True)
    a = ap.parse_args()
    if a.command == "build":
        from .pipeline import build

        print(json.dumps(build(a.root), indent=2))
    elif a.command == "underlyings":
        from .underlyings import ingest

        ingest(a.root)
    elif a.command == "fixings":
        from .states import build_states

        build_states(a.root)
    elif a.command == "coverage":
        from .pipeline import coverage
        import duckdb

        con = duckdb.connect()
        con.read_parquet(str(a.root / "ao_daily_research_panel.parquet")).create_view("panel")
        coverage(con, a.root)
        print((a.root / "ao_coverage_summary.json").read_text())
    elif a.command in ["validate", "audit"]:
        from .validation import validate

        print(json.dumps(validate(a.root), indent=2))
    elif a.command == "parse":
        from .parsers import bulletin, clearport, eod_sample, csvwrite

        if a.format == "bulletin":
            if not a.date:
                ap.error("bulletin requires --date matching the printed header")
            rows, check = bulletin(a.input, a.date)
            print(json.dumps(check))
        elif a.format == "clearport":
            rows = clearport(a.input)
        elif a.format == "eod":
            rows = eod_sample(a.input)
        else:
            from .underlyings import parse_cl_exchange

            rows = parse_cl_exchange(a.input).to_dict("records")
        csvwrite(a.out, rows)
        print("Parsed records", len(rows))
    elif a.command == "export":
        result = load_ao_panel(a.start, a.end, {"eligible_only": not a.include_ineligible}, a.root)
        result["observations"].to_parquet(a.out, index=False)
        print("Exported records", len(result["observations"]))
    elif a.command == "collect-underlyings":
        from .providers import acquire_underlyings

        print(
            json.dumps(
                acquire_underlyings(a.root, a.start, a.end, a.approve, a.max_cost),
                indent=2,
                default=str,
            )
        )
    else:
        from .acquisition import collect, download

        (collect if a.command == "backfill" else download)(a)


if __name__ == "__main__":
    main()
