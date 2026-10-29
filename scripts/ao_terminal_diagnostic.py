from pathlib import Path
import duckdb
import json

r = Path("outputs/HestonLab_Asian_Data")
c = duckdb.connect()
c.read_parquet(str(r / "cl_futures_settlements.parquet")).create_view("f")
c.read_parquet(str(r / "underlying_contracts.parquet")).create_view("contracts")
q = """WITH days AS (SELECT DISTINCT trade_date FROM f WHERE root='CL' AND extract(isodow FROM trade_date)<=5), near AS (SELECT * EXCLUDE(n) FROM (SELECT d.trade_date,c.contract_id,row_number() OVER(PARTITION BY d.trade_date ORDER BY expiration,contract_id) n FROM days d JOIN (SELECT DISTINCT contract_id,expiration FROM contracts WHERE root='CL') c ON c.expiration>=d.trade_date) WHERE n=1), averages AS (SELECT strftime(n.trade_date,'%Y-%m') contract_month,count(*) fixing_count,count_if(f.settlement_price IS NULL) missing,avg(f.settlement_price) mean_cl,count_if(NOT f.is_final) nonfinal_cl FROM near n LEFT JOIN f ON n.trade_date=f.trade_date AND n.contract_id=f.contract_id AND f.update_action=1 GROUP BY 1) SELECT a.*,f.settlement_price csx_terminal,f.is_final csx_final_flag,a.mean_cl-f.settlement_price difference,abs(a.mean_cl-f.settlement_price)<=0.00501 AND a.missing=0 reconciles FROM averages a LEFT JOIN f ON a.contract_month=f.contract_month AND f.root='CSX' AND f.trade_date=f.expiration AND f.update_action=1 ORDER BY a.contract_month"""
d = c.sql(q).df()
d.to_csv(r / "evidence/terminal_unflagged_reconciliation.csv", index=False)
(r / "evidence/terminal_flag_limitation.json").write_text(
    json.dumps(
        dict(
            months=len(d),
            terminal_available=int(d.csx_terminal.notna().sum()),
            reconciles=int(d.reconciles.fillna(False).sum()),
            max_difference=float(d.difference.abs().max()),
            explanation="Expiry-day CL and CSX settlement records lack the statistics final bit. Their prices are preserved, but the strict final-only fixing policy rejects them. This separate reconciliation is diagnostic and does not silently relabel finality.",
        ),
        indent=2,
    )
)
print(d.to_string(index=False))
