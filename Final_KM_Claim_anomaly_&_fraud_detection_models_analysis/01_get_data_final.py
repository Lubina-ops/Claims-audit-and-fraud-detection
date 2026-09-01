"""Step 1: clean Updated_KM_data_0726.xlsx and create base claim fields."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
INPUT_FILE = BASE / "Updated_KM_data_0726.xlsx"
OUTPUT_FILE = BASE / "data" / "01_claims_base.parquet"
EXCEL_OUTPUT = BASE / "data" / "01_claims_base.xlsx"

REQUIRED = ["ClaimFormID", "ClaimId", "ClaimantName", "DealerName", "ProductID",
            "ProductSoldDate", "ClaimSubmitionDate", "ItemQuantity", "SaleAmount",
            "ClaimItemStatus"]
# TEXT_COLS = ["ClaimFormID", "ClaimId", "ClaimItemId", "SalesTransactionID",
#              "ClaimantName", "DealerName", "CustomerID", "CustomerName", "ProductID",
#              "ProductName", "SerialNumber", "ClaimItemStatus", "ClaimItemStatusReason",
#              "AuditComment", "IsSerializedProduct"]
TEXT_COLS = [
    "ClaimFormID",
    "ClaimId",
    "ClaimItemId",
    "SalesTransactionID",
    "ClaimantName",
    "DealerName",
    "ClaimName",          # ADD THIS
    "ProductID",
    "ProductName",
    "SerialNumber",
    "ClaimItemStatus",
    "ClaimItemStatusReason",
    "AuditComment"
]

def path_value(value):
    p = Path(value).expanduser()
    return (p if p.is_absolute() else BASE / p).resolve()


def read_input(path):
    p = path_value(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")
    return pd.read_parquet(p) if p.suffix.lower() == ".parquet" else pd.read_excel(p, engine="openpyxl")


def prepare(df, delay_days=90):
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    missing = [c for c in REQUIRED if c not in out]
    if missing:
        raise ValueError(f"Missing required columns: {missing}\nAvailable: {list(out.columns)}")

    for c in TEXT_COLS:
        if c in out:
            out[c] = out[c].astype("string").str.strip()
    if "SerialNumber" in out:
        out["SerialNumber"] = out["SerialNumber"].str.upper()

    # Replace the old IsSerializedProduct logic with this
    if "SerialNumber" in out:
        out["is_serialized_product"] = (
            out["SerialNumber"]
            .notna()
            .astype("int8")
        )
    else:
        out["is_serialized_product"] = 0

    for c in ["ProductSoldDate", "ClaimSubmitionDate", "ClaimItemStatusDate", "InvoiceDate", "ClaimDate"]:
        if c in out:
            out[c] = pd.to_datetime(out[c], errors="coerce")

    out["ItemQuantity"] = pd.to_numeric(out["ItemQuantity"], errors="coerce")
    if "SaleQuantity" in out:
        out["SaleQuantity"] = pd.to_numeric(out["SaleQuantity"], errors="coerce")
    out["SaleAmount"] = pd.to_numeric(out["SaleAmount"], errors="coerce")

    out["claim_days_since_sale"] = (
        out["ClaimSubmitionDate"] - out["ProductSoldDate"]
    ).dt.days.astype("Int64")
    out["num_days"] = out["claim_days_since_sale"]
    out["claim_file_delay"] = out["claim_days_since_sale"].ge(delay_days).fillna(False).astype("int8")
    out["claim_days_since_sale_bucket"] = pd.cut(
        out["claim_days_since_sale"],
        [-np.inf, -1, 29, 59, 89, 179, 364, 729, np.inf],
        labels=["Negative", "0-29 days", "30-59 days", "60-89 days", "90-179 days",
                "180-364 days", "365-729 days", "730+ days"])
    status = out["ClaimItemStatus"].str.casefold()
    out["rejected_claim"] = status.eq("rejected").fillna(False).astype("int8")
    out["returned_claim"] = status.eq("returned").fillna(False).astype("int8")
    out["processed_claim"] = status.eq("processed").fillna(False).astype("int8")
    out["cancelled_claim"] = status.eq("cancelled").fillna(False).astype("int8")
    out["problem_claim_returned_or_rejected"] = status.isin(["returned", "rejected"]).astype("int8")

    
    reason_pattern = (
                r"Duplicate Claim"
                r"|No Match Found"
                r"|unit's ineligible as it's replacement & was unsold"
                r"|Rejected"
                )

    status_pattern = r"Rejected|Duplicate Claim|No Match Found"

    out["failed_validation"] = (out["ClaimItemStatusReason"].astype(str).str.contains(
    reason_pattern, case=False, na=False)
    | out["ClaimItemStatus"].astype(str).str.contains(
    status_pattern, case=False, na=False)).astype("int8")



    out["missing_product_sold_date_flag"] = out["ProductSoldDate"].isna().astype("int8")
    out["missing_claim_submission_date_flag"] = out["ClaimSubmitionDate"].isna().astype("int8")
    out["claim_month"] = out["ClaimSubmitionDate"].dt.month.astype("Int64")
    out["claim_prod_qty"] = out["ItemQuantity"]
    out["claim_prod_amt"] = out["SaleAmount"]

    #out["is_serialized_product"] = out.get("IsSerializedProduct", pd.Series(pd.NA, index=out.index)).astype("string").str.upper().eq("Y").astype("int8")

    for c in ["fraud_flag", "anamoly_flag", "anomaly_flag", "_source_row_id"]:
        if c in out:
            out = out.drop(columns=c)
    print(f"[validation] rows={len(out):,}; distinct ClaimId={out['ClaimId'].nunique():,}; repeated rows={out['ClaimId'].duplicated().sum():,}")
    return out


# def run(input_path=INPUT_FILE, output_path=OUTPUT_FILE, excel_output=EXCEL_OUTPUT, save_excel=True):
#     out = prepare(read_input(input_path))
#     op = path_value(output_path); op.parent.mkdir(parents=True, exist_ok=True)
#     out.to_parquet(op, index=False, engine="pyarrow"); print(f"[save] {op}")
#     if save_excel:
#         xp = path_value(excel_output); xp.parent.mkdir(parents=True, exist_ok=True)
#         out.to_excel(xp, index=False, engine="openpyxl"); print(f"[save] {xp}")
#     return out
def run(input_path=INPUT_FILE, output_path=OUTPUT_FILE,
        excel_output=EXCEL_OUTPUT, save_excel=True):

    out = prepare(read_input(input_path))

    op = path_value(output_path)
    op.parent.mkdir(parents=True, exist_ok=True)

    # Fix mixed object datatype columns before parquet write
    for c in out.columns:

        if out[c].dtype == "object":

            print(
                f"Converting object column: {c}"
            )

            out[c] = (
                out[c]
                .fillna("")
                .astype(str)
            )

    out.to_parquet(
        op,
        index=False,
        engine="pyarrow"
    )

    print(f"[save] {op}")

def main():
    p = argparse.ArgumentParser(); p.add_argument("--input", default=INPUT_FILE); p.add_argument("--output", default=OUTPUT_FILE)
    p.add_argument("--excel-output", default=EXCEL_OUTPUT); p.add_argument("--no-excel", action="store_true")
    a = p.parse_args(); run(a.input, a.output, a.excel_output, not a.no_excel)

if __name__ == "__main__": main()
