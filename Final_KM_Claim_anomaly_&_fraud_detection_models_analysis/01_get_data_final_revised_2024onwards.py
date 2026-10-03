"""Step 1 - Clean the source and create transparent base fields.

Important design choice
-----------------------
All rows are retained here because older rows may describe prior behavior.
A separate scoring_eligible flag marks only Processed and Rejected claims.
Current outcomes are not model inputs; they are retained only to build prior history.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
BASE=Path(__file__).resolve().parent
DATE_CUTOFF=pd.Timestamp("2024-01-01")
INPUT_FILE=BASE/"Updated_KM_data_0726.xlsx"
OUTPUT_FILE=BASE/"data"/"01_claims_base_revised.parquet"
EXCEL_FILE=BASE/"data"/"01_claims_base_revised.xlsx"
REQUIRED=["ClaimFormID","ClaimId","ClaimantName","DealerName","ProductID","ProductSoldDate",
          "ClaimSubmitionDate","ItemQuantity","SaleAmount","ClaimItemStatus","ClaimItemStatusReason"]
TEXT_COLUMNS=["ClaimFormID","ClaimId","ClaimItemId","SalesTransactionID","ClaimantName","DealerName",
              "ClaimName","ProductID","ProductName","SerialNumber","ClaimItemStatus","ClaimItemStatusReason","AuditComment"]

def read_file(path):
    p=Path(path)
    if not p.exists(): raise FileNotFoundError(f"Input not found: {p}")
    return pd.read_parquet(p) if p.suffix.lower()==".parquet" else pd.read_excel(p,engine="openpyxl")

def prepare(df):
    out=df.copy(); out.columns=[str(c).strip() for c in out.columns]
    missing=[c for c in REQUIRED if c not in out.columns]
    if missing: raise ValueError(f"Missing required columns: {missing}")
    for c in TEXT_COLUMNS:
        if c in out.columns: out[c]=out[c].astype("string").str.strip()
    for c in ["ProductSoldDate","ClaimSubmitionDate","ClaimItemStatusDate","InvoiceDate","ClaimDate"]:
        if c in out.columns: out[c]=pd.to_datetime(out[c],errors="coerce")
    # Kelly-approved date window: use claims submitted on or after 2024-01-01.
    rows_before_cutoff=len(out)
    missing_submission_dates=int(out["ClaimSubmitionDate"].isna().sum())
    out=out.loc[out["ClaimSubmitionDate"].ge(DATE_CUTOFF)].copy()
    out["date_eligible"]=1
    print(f"[Step 1] 2024 cutoff: kept {len(out):,} of {rows_before_cutoff:,}; excluded missing/invalid dates={missing_submission_dates:,}")
    if out.empty: raise ValueError("Step 1: no claims remain after ClaimSubmitionDate >= 2024-01-01")
    for c in ["ItemQuantity","SaleAmount","SaleQuantity"]:
        if c in out.columns: out[c]=pd.to_numeric(out[c],errors="coerce")
    out["num_days"]=(out["ClaimSubmitionDate"]-out["ProductSoldDate"]).dt.days.astype("Int64")
    out["num_days_buckets"]=pd.cut(out["num_days"],[-np.inf,-1,29,59,89,180,np.inf],
        labels=["Negative","0-29 days","30-59 days","60-89 days","90-180 days","180+ days"],right=True)
    out["claim_days_since_sale"]=out["num_days"]
    out["claim_file_delay"]=out["num_days"].ge(90).fillna(False).astype("int8")
    out["claim_month"]=out["ClaimSubmitionDate"].dt.month.astype("Int64")
    out["claim_prod_qty"]=out["ItemQuantity"]
    out["claim_prod_amt"]=out["SaleAmount"]
    out["is_serialized_product"]=out.get("SerialNumber",pd.Series(pd.NA,index=out.index)).notna().astype("int8")
    status=out["ClaimItemStatus"].astype("string").str.casefold()
    out["rejected_claim_event"]=status.eq("rejected").fillna(False).astype("int8")
    out["returned_claim_event"]=status.eq("returned").fillna(False).astype("int8")
    out["cancelled_claim_event"]=status.eq("cancelled").fillna(False).astype("int8")
    out["processed_claim_event"]=status.eq("processed").fillna(False).astype("int8")
    out["scoring_eligible"]=status.isin(["processed","rejected"]).astype("int8")
    # Rejected is deliberately NOT part of this business behavior definition.
    pattern=(r"Duplicate Claim|No Match Found|Unit Ineligible As Replacement|"
             r"unit's ineligible as it's replacement & was unsold")
    out["anomaly_behavior_event"]=out["ClaimItemStatusReason"].astype("string").str.contains(
        pattern,case=False,na=False,regex=True).astype("int8")
    return out

def make_parquet_safe(df):
    """Convert mixed Python object columns to one consistent text type.

    Excel columns can contain both numbers and text. Pandas then labels the
    column as object, but Parquet requires one data type per column. Converting
    only object columns to pandas StringDtype prevents pyarrow errors without
    changing date or numeric columns.
    """
    safe = df.copy()
    converted = []
    for column in safe.columns:
        if safe[column].dtype == "object":
            safe[column] = safe[column].astype("string")
            converted.append(column)
    if converted:
        print("[parquet] Converted mixed object columns to text: " + ", ".join(converted))
    return safe

def run(input_path=INPUT_FILE,output_path=OUTPUT_FILE,excel_path=EXCEL_FILE):
    out=prepare(read_file(input_path)); output_path=Path(output_path); output_path.parent.mkdir(parents=True,exist_ok=True)
    out = make_parquet_safe(out)
    out.to_parquet(output_path,index=False,engine="pyarrow")
    with pd.ExcelWriter(excel_path,engine="openpyxl") as w:
        out.to_excel(w,sheet_name="claims_base",index=False)
        out["ClaimItemStatus"].astype("string").value_counts(dropna=False).rename_axis("status").reset_index(name="rows").to_excel(w,sheet_name="population_summary",index=False)
        pd.DataFrame([{"rule":"date_cutoff","value":"2024-01-01","date_field":"ClaimSubmitionDate","rows_retained":len(out)}]).to_excel(w,sheet_name="date_filter_summary",index=False)
    print(f"[save] {output_path}\n[save] {excel_path}"); return out

def main():
    p=argparse.ArgumentParser(); p.add_argument("--input",default=INPUT_FILE); p.add_argument("--output",default=OUTPUT_FILE); p.add_argument("--excel",default=EXCEL_FILE)
    a=p.parse_args(); run(a.input,a.output,a.excel)
if __name__=="__main__": main()
