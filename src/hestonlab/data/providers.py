"""Guarded, resumable Databento collection; uses DATABENTO_API_KEY from environment."""

from pathlib import Path
from datetime import date, timedelta, datetime, timezone
import json
import hashlib
import databento as db


def acquire_underlyings(root, start="2019-12-01", end="2026-10-27", approve=False, max_cost=0):
    """Quote missing work first. Approved credit cap is for this call only.

    Existing request receipts are reused even if jobs are still queued. A failed
    download never causes a replacement paid request. One monthly definition
    snapshot per product is an explicit approximation to daily listing history.
    """
    root = Path(root)
    evidence = root / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    dest = root / "sources/underlying_definitions"
    dest.mkdir(parents=True, exist_ok=True)
    client = db.Historical()
    plans = []
    jobs = []
    for symbol in ["CL", "CSX"]:
        receipt = evidence / f"{symbol}_statistics_job.json"
        if receipt.exists():
            existing = json.loads(receipt.read_text())
            if existing["start"][:10] > start or existing["end"][:10] < end:
                raise ValueError(
                    "Existing underlying job does not cover requested window; explicit extension plan required"
                )
            jobs.append(existing)
        else:
            kw = dict(
                dataset="GLBX.MDP3",
                symbols=[symbol + ".FUT"],
                stype_in="parent",
                schema="statistics",
                start=start,
                end=end,
            )
            plans.append(dict(parameters=kw, cost=client.metadata.get_cost(**kw), receipt=receipt))
        first = date.fromisoformat(start)
        for month in range(
            first.year * 12 + first.month - 1,
            date.fromisoformat(end).year * 12 + date.fromisoformat(end).month,
        ):
            y, m = divmod(month, 12)
            d = date(y, m + 1, 8)
            while d.weekday() != 2:
                d += timedelta(days=1)
            if not start <= str(d) < end:
                continue
            target = dest / f"{symbol}_definition_{d}.dbn.zst"
            if target.exists():
                continue
            kw = dict(
                dataset="GLBX.MDP3",
                symbols=[symbol + ".FUT"],
                stype_in="parent",
                schema="definition",
                start=str(d),
                end=str(d + timedelta(days=1)),
            )
            plans.append(dict(parameters=kw, cost=client.metadata.get_cost(**kw), target=target))
    cost = sum(p["cost"] for p in plans)
    if plans and (not approve or cost > max_cost):
        return dict(
            status="quote_only", estimated_cost=cost, missing_requests=len(plans), jobs=jobs
        )
    for p in plans:
        kw = p["parameters"]
        if "receipt" in p:
            job = client.batch.submit_job(
                **kw,
                encoding="csv",
                compression="zstd",
                pretty_px=True,
                pretty_ts=True,
                map_symbols=True,
                split_duration="month",
            )
            job.pop("user_id", None)
            p["receipt"].write_text(json.dumps(job, indent=2, default=str))
            jobs.append(job)
        else:
            target = p["target"]
            client.timeseries.get_range(**kw, path=target)
            db.DBNStore.from_file(target).to_csv(
                target.with_suffix("").with_suffix(".csv"),
                map_symbols=True,
                pretty_px=True,
                pretty_ts=True,
            )
            target.with_suffix(".receipt.json").write_text(
                json.dumps(
                    dict(
                        parameters=kw,
                        estimated_cost=p["cost"],
                        collected_utc=datetime.now(timezone.utc).isoformat(),
                        sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                    ),
                    indent=2,
                )
            )
    for job in jobs:
        detail = client.batch.get_job_details(job["id"])
        if detail["state"] == "done":
            client.batch.download(job["id"], output_dir=root / "sources/underlying_statistics")
    return dict(status="submitted_or_resumed", estimated_cost=cost, jobs=jobs)
