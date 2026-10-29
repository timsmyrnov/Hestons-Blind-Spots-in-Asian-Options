from pathlib import Path
import json
import os
import pandas as pd
import pytest
import duckdb
from hestonlab.data.parsers import bulletin, clearport, eod_sample, num, base
from hestonlab.data.acquisition import raw_identity, collect
from hestonlab.data.fixings import select_nearby, accrued_state, methodology_regime
from hestonlab.data.research import load_ao_panel, build_ao_snapshot
from hestonlab.data.underlyings import parse_cl_exchange

ROOT = Path(
    os.environ.get(
        "HESTONLAB_DATA_ROOT",
        str(Path(__file__).resolve().parents[1] / "outputs/HestonLab_Asian_Data"),
    )
)


requires_archive = pytest.mark.skipif(
    not (ROOT / "ao_observations.parquet").exists(),
    reason="Full licensed source archive not installed; set HESTONLAB_DATA_ROOT to run real-file integration checks",
)


@pytest.mark.parametrize(
    "raw,expected",
    [("----", None), ("", None), ("NaN", None), ("0", 0), ("-37.63", -37.63), ("1.2", 1.2)],
)
def test_missing_and_zero_not_confused(raw, expected):
    assert num(raw) == expected


@pytest.mark.parametrize(
    "price,status,usable",
    [
        (0, "final", False),
        (-1, "final", False),
        (1, "preliminary", False),
        (1, "final", True),
        (None, "final", False),
    ],
)
def test_option_validity(price, status, usable):
    assert base(settlement_price=price, settlement_status=status)["usable_settlement"] == usable


@requires_archive
def test_real_clearport_units_and_prior_oi():
    rows = clearport(ROOT / "sources/STLCPC_20230403")
    assert len(rows) == 1579
    assert {r["call_put"] for r in rows} == {"C", "P"}
    assert len({r["contract_month"] for r in rows}) == 33
    assert all(r["open_interest"] is None for r in rows)
    assert any(r["prior_day_open_interest"] is not None for r in rows)
    assert all(r["source"] == "CME ClearPort final settlement file" for r in rows)


@requires_archive
def test_real_eod_scale_expiry():
    rows = eod_sample(ROOT / "sources/CME_AO_official_sample_2026-10-08.csv")
    assert len(rows) == 10
    assert rows[0]["strike"] == 100
    assert rows[0]["settlement_price"] == 0.09
    assert rows[0]["option_expiration"] == "2026-10-30"
    assert rows[0]["volume"] == 0


@requires_archive
def test_real_pdf_reconciles_printed_totals():
    rows, ev = bulletin(ROOT / "sources/Section63_Energy_Options_Products.pdf", "2026-10-26")
    assert len(rows) == 2397
    assert ev["printed_oi_total_checks"] == 66
    assert ev["all_match"]
    assert len({r["contract_month"] for r in rows}) == 33
    assert len({(r["contract_month"], r["call_put"], r["strike"]) for r in rows}) == 2397


@requires_archive
def test_pdf_wrong_date_fails():
    with pytest.raises(ValueError, match="does not match"):
        bulletin(ROOT / "sources/Section63_Energy_Options_Products.pdf", "2020-04-20")


def test_corrupt_clearport_fails(tmp_path):
    p = tmp_path / "bad"
    p.write_text("BROKEN HEADER\n")
    with pytest.raises(ValueError):
        clearport(p)


@requires_archive
def test_real_cl_exchange_price():
    frame = parse_cl_exchange(ROOT / "sources/STLNYMEX_20230403")
    assert frame.iloc[0].contract_month == "2023-05"
    assert frame.iloc[0].settlement_price == 80.42
    assert frame.iloc[0].volume == 547126


def test_identity_year_and_negative_strike():
    assert raw_identity("AAOK0 C-500", "2020-04-20") == ("2020-05", "C", -5)
    assert raw_identity("AAOF7 P6000", "2026-10-26") == ("2027-01", "P", 60)
    assert raw_identity("SPX", "2020-01-01") is None


def fixtures():
    s = pd.DataFrame(
        dict(
            fixing_date=["2020-04-20", "2020-04-21", "2020-04-22"],
            reference_contract_id=["CL:2020-05"] * 3,
            calendar_verified=[True] * 3,
        )
    )
    r = pd.DataFrame(
        [
            ("2020-04-20", "CL:2020-05", -37.63, "2020-04-20T19:00:00Z", True, 1),
            ("2020-04-20", "CL:2020-05", 999, "2020-04-23T19:00:00Z", True, 1),
            ("2020-04-21", "CL:2020-05", 10, "2020-04-21T19:00:00Z", True, 1),
            ("2020-04-22", "CL:2020-05", 12, "2020-04-22T19:00:00Z", True, 1),
        ],
        columns=[
            "trade_date",
            "contract_id",
            "settlement_price",
            "publication_timestamp",
            "is_final",
            "update_action",
        ],
    )
    return s, r


def test_fixings_exclude_future_days_and_future_revisions():
    s, r = fixtures()
    state = accrued_state(s, r, "2020-04-21", "2020-04-21T20:00:00Z")
    assert state["known_fixing_count"] == 2
    assert state["accrued_fixing_sum"] == pytest.approx(-27.63)
    assert state["remaining_fixing_count"] == 1
    assert state["missing_elapsed_fixings"] == 0


def test_unpublished_is_missing_not_zero():
    s, r = fixtures()
    state = accrued_state(s, r, "2020-04-21", "2020-04-21T18:00:00Z")
    assert state["known_fixing_count"] == 1
    assert state["missing_elapsed_fixings"] == 1
    with pytest.raises(ValueError):
        accrued_state(s, r, "2020-04-21", "2020-04-21")


def test_delete_does_not_resurrect_previous_price():
    s, r = fixtures()
    r.loc[len(r)] = ["2020-04-20", "CL:2020-05", -37.63, "2020-04-21T17:00:00Z", True, 2]
    result = accrued_state(s, r, "2020-04-21", "2020-04-21T20:00:00Z")
    assert result["known_fixing_count"] == 1
    assert result["accrued_fixing_sum"] == 10


def test_roll_includes_expiry_day():
    c = pd.DataFrame(
        dict(
            contract_id=["CL:2020-05", "CL:2020-06"],
            contract_month=["2020-05", "2020-06"],
            expiration=["2020-04-21", "2020-05-19"],
        )
    )
    assert select_nearby("2020-04-21", c) == "CL:2020-05"
    assert select_nearby("2020-04-22", c) == "CL:2020-06"
    assert select_nearby("2021-01-01", c) is None


def test_documented_model_changes():
    assert methodology_regime("2020-04-21").startswith("pre_")
    assert methodology_regime("2020-04-22") == "documented_Bachelier"
    assert methodology_regime("2020-05-18").startswith("Curran")


def test_recollection_guard_before_network(tmp_path):
    from argparse import Namespace

    (tmp_path / "evidence").mkdir()
    (tmp_path / "evidence/databento_batch_details.json").write_text(
        json.dumps(
            dict(
                state="done",
                schema="statistics",
                start="2020-01-01",
                end="2026-10-27",
                id="existing",
            )
        )
    )
    with pytest.raises(SystemExit, match="Existing completed"):
        collect(Namespace(root=tmp_path, start="2023-01-01", end="2024-01-01"))


@requires_archive
def test_full_ingestion_exchange_prices():
    checks = pd.read_csv(ROOT / "evidence/cross_source_validation.csv")
    assert len(checks) == 3986
    assert checks.difference.eq(0).all()
    c = duckdb.connect()
    p = str(ROOT / "ao_observations.parquet")
    assert c.execute("SELECT count(*) FROM read_parquet(?)", [p]).fetchone()[0] == 5221594
    assert (
        c.execute(
            "SELECT count(*) FROM read_parquet(?) WHERE call_put NOT IN ('C','P')", [p]
        ).fetchone()[0]
        == 0
    )


@requires_archive
def test_loader_native_contracts_and_determinism():
    a = build_ao_snapshot("2026-10-26", {"eligible_only": False}, ROOT)
    b = build_ao_snapshot("2026-10-26", {"eligible_only": False}, ROOT)
    assert len(a["price_vector"]) == 2399
    assert a["coordinate_ids"] == b["coordinate_ids"]
    assert len(set(a["coordinate_ids"])) == 2399
    assert a["observations"].bid_price.isna().all()
    assert len(a["contracts"]) == 2399
    with pytest.raises(ValueError):
        load_ao_panel("2026-10-27", "2026-10-26", root=ROOT)
    with pytest.raises(ValueError):
        load_ao_panel("2026-10-26", "2026-10-26", {"made_up": 1}, ROOT)


def test_state_pipeline_asof_end_to_end(tmp_path):
    from hestonlab.data.states import build_states
    from datetime import date

    (tmp_path / "evidence").mkdir()
    contracts = pd.DataFrame(
        [
            dict(
                contract_id="CL:2020-06",
                contract_month="2020-06",
                root="CL",
                expiration=date(2020, 5, 19),
            ),
            dict(
                contract_id="CSX:2020-04",
                contract_month="2020-04",
                root="CSX",
                expiration=date(2020, 4, 30),
            ),
        ]
    )
    contracts.to_parquet(tmp_path / "underlying_contracts.parquet")
    rows = []
    for day, price, root, cid, month, expiry in [
        (29, 10, "CL", "CL:2020-06", "2020-06", date(2020, 5, 19)),
        (30, 20, "CL", "CL:2020-06", "2020-06", date(2020, 5, 19)),
        (30, 15, "CSX", "CSX:2020-04", "2020-04", date(2020, 4, 30)),
    ]:
        rows.append(
            dict(
                trade_date=date(2020, 4, day),
                settlement_price=price,
                root=root,
                contract_id=cid,
                contract_month=month,
                expiration=expiry,
                is_final=True,
                update_action=1,
                publication_timestamp=pd.Timestamp(f"2020-04-{day}T20:00:00Z"),
                source_file="synthetic_test",
            )
        )
    f = pd.DataFrame(rows)
    f["publication_timestamp_raw"] = f.publication_timestamp.dt.strftime(
        "%Y-%m-%dT%H:%M:%S.000000000Z"
    )
    f["source_sequence"] = 0
    f.to_parquet(tmp_path / "cl_futures_settlements.parquet")
    f.to_parquet(tmp_path / "cl_futures_settlement_revisions.parquet")
    pd.DataFrame(
        [
            dict(
                observation_date=date(2020, 4, 29),
                contract_month="2020-04",
                contract_id="testAO",
                publication_timestamp=pd.Timestamp("2020-04-29T19:00:00Z"),
            ),
            dict(
                observation_date=date(2020, 4, 30),
                contract_month="2020-04",
                contract_id="testAO",
                publication_timestamp=pd.Timestamp("2020-04-30T21:00:00Z"),
            ),
        ]
    ).to_parquet(tmp_path / "ao_observations.parquet")
    ao_fixture = pd.read_parquet(tmp_path / "ao_observations.parquet")
    ao_fixture["stat_ts_recv"] = ao_fixture.publication_timestamp.dt.strftime(
        "%Y-%m-%dT%H:%M:%S.000000000Z"
    )
    ao_fixture.to_parquet(tmp_path / "ao_observations.parquet")
    build_states(tmp_path)
    state = pd.read_parquet(tmp_path / "ao_fixing_state.parquet").sort_values("observation_date")
    assert state.iloc[0].known_fixing_count == 0
    assert state.iloc[0].missing_elapsed_fixings == 1
    assert state.iloc[1].known_fixing_count == 2
    assert state.iloc[1].accrued_fixing_sum == 30
    assert state.iloc[1].underlying_state_complete
    assert (
        json.loads((tmp_path / "evidence/fixing_validation.json").read_text())["verified_months"]
        == 1
    )


@requires_archive
def test_liquidity_filter_cannot_escape_date_or_side():
    result = load_ao_panel(
        "2024-06-14",
        "2024-06-14",
        {"eligible_only": False, "require_liquidity": True, "call_put": "P"},
        ROOT,
    )
    frame = result["observations"]
    assert not frame.empty
    assert (frame.observation_date.dt.date.astype(str) == "2024-06-14").all()
    assert (frame.call_put == "P").all()


def test_csv_schema_change_is_not_silently_accepted(tmp_path):
    from hestonlab.data.csvio import csv_relation

    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    a.write_text("price,symbol\n-37.63,CLK0\n")
    b.write_text("price,wrong_column\n10,CLM0\n")
    c = duckdb.connect()
    assert csv_relation(c, a).fetchone() == ("-37.63", "CLK0")
    with pytest.raises(ValueError, match="schema change"):
        csv_relation(c, [a, b])


def test_missing_raw_archive_cannot_replace_valid_panel(tmp_path):
    from hestonlab.data.acquisition import normalize

    (tmp_path / "evidence").mkdir()
    (tmp_path / "evidence/databento_download.json").write_text(
        json.dumps({"files": ["sources/absent.csv.zst"]})
    )
    panel = tmp_path / "ao_historical_panel.parquet"
    panel.write_bytes(b"existing_valid_artifact")
    with pytest.raises(FileNotFoundError, match="bulk source files are absent"):
        normalize(tmp_path)
    assert panel.read_bytes() == b"existing_valid_artifact"


def test_nanosecond_cutoff_cannot_admit_later_revision():
    schedule = pd.DataFrame(
        dict(
            fixing_date=["2020-04-20"],
            reference_contract_id=["CL:2020-05"],
            calendar_verified=[True],
        )
    )
    revisions = pd.DataFrame(
        [
            ("2020-04-20", "CL:2020-05", -37.63, "2020-04-20T19:00:00.000000099Z", True, 1),
            ("2020-04-20", "CL:2020-05", 999, "2020-04-20T19:00:00.000000101Z", True, 1),
        ],
        columns=[
            "trade_date",
            "contract_id",
            "settlement_price",
            "publication_timestamp",
            "is_final",
            "update_action",
        ],
    )
    state = accrued_state(schedule, revisions, "2020-04-20", "2020-04-20T19:00:00.000000100Z")
    assert state["accrued_fixing_sum"] == -37.63


@requires_archive
def test_downloaded_underlyings_match_exchange_and_preserve_negative_price():
    import json
    import duckdb

    report = json.loads((ROOT / "evidence/underlying_ingestion.json").read_text())
    assert report["unmapped_revision_records"] == 0
    assert report["matched_exchange_sample"] == report["exchange_sample_rows"] == 130
    con = duckdb.connect()
    row = con.execute(
        "SELECT settlement_price,is_final FROM read_parquet(?) WHERE root='CL' AND contract_month='2020-05' AND trade_date='2020-04-20'",
        [str(ROOT / "cl_futures_settlements.parquet")],
    ).fetchone()
    con.close()
    assert row == (-37.63, False)  # Preserve the actual terminal value and vendor final flag.
