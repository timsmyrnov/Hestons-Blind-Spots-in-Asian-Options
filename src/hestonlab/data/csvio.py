"""Strict Databento CSV dialect: validate headers, avoid costly format guessing."""

import csv
from pathlib import Path
import zstandard as zstd


def csv_relation(con, paths, filename=False):
    if isinstance(paths, (str, Path)):
        paths = [Path(paths)]
    paths = [Path(p) for p in paths]
    if not paths:
        raise ValueError("No source CSV files")
    expected = None
    for path in paths:
        opener = zstd.open if path.suffix == ".zst" else open
        with opener(path, "rt", encoding="utf-8", newline="") as f:
            header = next(csv.reader(f), None)
        if not header or len(set(header)) != len(header):
            raise ValueError(f"Missing or duplicate CSV header: {path.name}")
        if expected is None:
            expected = header
        elif header != expected:
            raise ValueError(f"CSV schema change requires explicit review: {path.name}")
    from .pipeline import literal

    locations = "[" + ",".join(literal(p) for p in paths) + "]"
    columns = "{" + ",".join(literal(k) + ":'VARCHAR'" for k in expected) + "}"
    return con.sql(f"""SELECT * FROM read_csv({locations},columns={columns},
       auto_detect=false,header=true,delim=',',quote='"',escape='"',nullstr='',
       filename={"true" if filename else "false"})""")
