"""Step 2 - Add product activity and seasonality features.

The historical average uses shift(1), so the current month is not allowed to
help define its own expected value. This is a point-in-time leakage control.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
BASE=Path(__file__).resolve().parent
DATE_CUTOFF=pd.Timestamp("2024-01-01")
INPUT_FILE=BASE/"data"/"01_claims_base_revised.parquet"
OUTPUT_FILE=BASE/"data"/"02_claims_velocity_revised.parquet"
EXCEL_FILE=BASE/"data"/"02_claims_velocity_revised.xlsx"

def ratio(a,b):
    a=pd.to_numeric(a,errors="coerce"); b=pd.to_numeric(b,errors="coerce")
    return (a/b.where(b.ne(0))).replace([np.inf,-np.inf],np.nan)

def add_velocity(df,min_prior_months=3,multiplier=2.0):
    out=df.copy(); out["ClaimSubmitionDate"]=pd.to_datetime(out["ClaimSubmitionDate"],errors="coerce")
    # Defensive check: pre-2024 claims cannot affect product seasonality.
    out=out.loc[out["ClaimSubmitionDate"].ge(DATE_CUTOFF)].copy()
    out["date_eligible"]=1
    if out.empty: raise ValueError("Step 2: no claims remain after the January 1, 2024 cutoff")
    if not out["ClaimSubmitionDate"].ge(DATE_CUTOFF).all(): raise ValueError("Step 2 date-cutoff validation failed")
    out["product_claim_month"]=out["ClaimSubmitionDate"].dt.to_period("M").dt.to_timestamp()
    unique=out[["ClaimId","ProductID","product_claim_month"]].dropna().drop_duplicates()
    monthly=(unique.groupby(["ProductID","product_claim_month"],as_index=False)
             .agg(product_seasonal_month_claims=("ClaimId","nunique"))
             .sort_values(["ProductID","product_claim_month"]))
    monthly["product_seasonal_avg_month_claims"]=(monthly.groupby("ProductID")["product_seasonal_month_claims"]
        .transform(lambda s:s.shift(1).expanding(min_periods=1).mean()))
    monthly["product_seasonal_active_months"]=monthly.groupby("ProductID").cumcount()
    monthly["product_seasonality_claim_count_ratio"]=ratio(monthly["product_seasonal_month_claims"],monthly["product_seasonal_avg_month_claims"])
    monthly["product_seasonality_claim_count_high_flag"]=(
        monthly["product_seasonality_claim_count_ratio"].ge(multiplier)&
        monthly["product_seasonal_active_months"].ge(min_prior_months)).astype("int8")
    result=out.merge(monthly,on=["ProductID","product_claim_month"],how="left",validate="many_to_one")
    result["velocity_history_excludes_current_month"]=1
    return result

def run(input_path=INPUT_FILE,output_path=OUTPUT_FILE,excel_path=EXCEL_FILE,save_excel=False):
    p=Path(input_path); df=pd.read_parquet(p) if p.suffix.lower()==".parquet" else pd.read_excel(p,engine="openpyxl")
    out=add_velocity(df); Path(output_path).parent.mkdir(parents=True,exist_ok=True); out.to_parquet(output_path,index=False)
    if save_excel: out.to_excel(excel_path,index=False,engine="openpyxl")
    print(f"[save] {output_path}"); return out

def main():
    p=argparse.ArgumentParser(); p.add_argument("--input",default=INPUT_FILE); p.add_argument("--output",default=OUTPUT_FILE); p.add_argument("--excel",default=EXCEL_FILE); p.add_argument("--save-excel",action="store_true")
    a=p.parse_args(); run(a.input,a.output,a.excel,a.save_excel)
if __name__=="__main__": main()
