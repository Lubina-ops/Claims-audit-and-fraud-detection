"""Step 2: product claim-count velocity and prior-month seasonality."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
INPUT_FILE = BASE / "data" / "01_claims_base.parquet"
OUTPUT_FILE = BASE / "data" / "02_claims_velocity.parquet"
EXCEL_OUTPUT = BASE / "data" / "02_claims_velocity.xlsx"


def pv(v):
    p=Path(v).expanduser(); return (p if p.is_absolute() else BASE/p).resolve()

def safe_ratio(a,b):
    a=pd.to_numeric(a,errors="coerce"); b=pd.to_numeric(b,errors="coerce")
    return (a/b.where(b.ne(0))).replace([np.inf,-np.inf],np.nan)

def read(path):
    p=pv(path); return pd.read_parquet(p) if p.suffix.lower()==".parquet" else pd.read_excel(p,engine="openpyxl")

def add_product_velocity(df, multiplier=2.0, min_prior_months=3):
    out=df.copy(); out["ClaimSubmitionDate"]=pd.to_datetime(out["ClaimSubmitionDate"],errors="coerce")
    out["product_claim_month"]=out["ClaimSubmitionDate"].dt.to_period("M").dt.to_timestamp()
    # Count each logical claim item once, even when multiple SalesTransactionID rows exist.
    unique=(out[["ClaimId","ProductID","product_claim_month"]]
            .dropna().drop_duplicates())
    pm=(unique.groupby(["ProductID","product_claim_month"],as_index=False)
        .agg(product_seasonal_month_claims=("ClaimId","nunique"))
        .sort_values(["ProductID","product_claim_month"]))
    pm["product_seasonal_avg_month_claims"]=(pm.groupby("ProductID")["product_seasonal_month_claims"]
        .transform(lambda s:s.shift(1).expanding(min_periods=1).mean()))
    pm["product_seasonal_active_months"]=pm.groupby("ProductID").cumcount()
    pm["product_seasonality_claim_count_ratio"]=safe_ratio(pm["product_seasonal_month_claims"],pm["product_seasonal_avg_month_claims"])
    pm["product_seasonality_claim_count_high_flag"]=(
        pm["product_seasonality_claim_count_ratio"].ge(multiplier) &
        pm["product_seasonal_active_months"].ge(min_prior_months)).astype("int8")
    return out.merge(pm,on=["ProductID","product_claim_month"],how="left",validate="many_to_one")

def run(input_path=INPUT_FILE,output_path=OUTPUT_FILE,excel_output=EXCEL_OUTPUT,save_excel=False):
    out=add_product_velocity(read(input_path)); op=pv(output_path); op.parent.mkdir(parents=True,exist_ok=True)
    out.to_parquet(op,index=False,engine="pyarrow"); print(f"[save] {op}")
    if save_excel: out.to_excel(pv(excel_output),index=False,engine="openpyxl")
    return out

def main():
    p=argparse.ArgumentParser(); p.add_argument("--input",default=INPUT_FILE); p.add_argument("--output",default=OUTPUT_FILE)
    p.add_argument("--excel-output",default=EXCEL_OUTPUT); p.add_argument("--save-excel",action="store_true")
    a=p.parse_args(); run(a.input,a.output,a.excel_output,a.save_excel)
if __name__=="__main__": main()
