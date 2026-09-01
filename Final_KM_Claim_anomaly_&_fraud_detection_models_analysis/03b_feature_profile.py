"""03b_feature_profile.py - feature profiling + use_in_model contract for step 4."""
from pathlib import Path
import re
import argparse
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
INPUT_FILE = BASE / "data" / "03_claims_features.parquet"
OUTPUT_FILE = BASE / "data" / "03b_feature_profile.xlsx"

NON_FEATURE_HINTS = ["id","name","serial","invoice","comment","description","status","reason"]


def pv(v):
    p = Path(v).expanduser()
    return (p if p.is_absolute() else BASE / p).resolve()


def profile_column(series):
    n = len(series); non_null = series.dropna()
    pct_null = 100*(n-len(non_null))/n if n else 100
    if len(non_null) == 0:
        return dict(pct_null=100.0, pct_most_common_value=0.0, n_unique=0, mode_value=None,
                    min_value=None, max_value=None, is_constant=0, is_all_null=1, near_constant_flag=0)
    vc = non_null.value_counts()
    pct_common = 100*vc.iloc[0]/n
    numeric = pd.api.types.is_numeric_dtype(non_null)
    return dict(
        pct_null=round(pct_null,2), pct_most_common_value=round(pct_common,2),
        n_unique=int(non_null.nunique()), mode_value=str(vc.index[0]),
        min_value=(float(non_null.min()) if numeric else None),
        max_value=(float(non_null.max()) if numeric else None),
        is_constant=int(non_null.nunique() <= 1), is_all_null=0,
        near_constant_flag=int(pct_common >= 99))


def looks_like_id(name, series):
    n = name.lower()
    # Match ID-like tokens as whole words / boundaries, so "failed_validation"
    # (which merely contains "id") is NOT flagged as an identifier.
    id_tokens = ["id", "name", "serial", "invoice", "comment", "description"]
    tokens = re.split(r"[^a-z0-9]+", n)          # split on _ and other separators
    if any(t in tokens for t in id_tokens):
        return 1
    if n.endswith("id") or n.endswith("_id"):
        return 1
    if series.notna().sum() > 0 and series.nunique() >= 0.95*series.notna().sum() and not pd.api.types.is_numeric_dtype(series):
        return 1
    return 0


def recommended_action(row, name, series):
    if row["is_all_null"]: return "drop: entirely null"
    if row["is_constant"]: return "drop: constant"
    if pd.api.types.is_datetime64_any_dtype(series) or "date" in name.lower(): return "drop: date field"
    if looks_like_id(name, series): return "drop: identifier or free text"
    if row["near_constant_flag"]: return "review: near-constant"
    if pd.api.types.is_numeric_dtype(series): return "keep: numeric"
    return "review: categorical"


def build_profile(df):
    recs = []
    for col in df.columns:
        p = profile_column(df[col]); p["column"] = col; p["dtype"] = str(df[col].dtype)
        p["looks_like_id"] = looks_like_id(col, df[col])
        p["is_date_field"] = int(pd.api.types.is_datetime64_any_dtype(df[col]) or "date" in col.lower())
        p["recommended_action"] = recommended_action(p, col, df[col])
        # single machine-readable decision for step 4
        p["use_in_model"] = int(p["recommended_action"].startswith("keep") and pd.api.types.is_numeric_dtype(df[col]))
        recs.append(p)
    order = ["column","dtype","pct_null","pct_most_common_value","n_unique","mode_value",
             "min_value","max_value","is_constant","is_all_null","near_constant_flag",
             "looks_like_id","is_date_field","recommended_action","use_in_model"]
    return pd.DataFrame(recs)[order].sort_values("column").reset_index(drop=True)


def analyze_item_quantity(df):
    if "ItemQuantity" not in df.columns: return
    q = pd.to_numeric(df["ItemQuantity"], errors="coerce")
    print(f"\n[ItemQuantity] % == 1: {100*(q==1).sum()/q.notna().sum():.1f}%")
    if "IsSerializedProduct" in df.columns:
        ser = df["IsSerializedProduct"].astype(str).str.upper().eq("Y")
        if ser.sum():
            print(f"[ItemQuantity] serialized rows with qty==1: {100*((q==1)&ser).sum()/ser.sum():.1f}%")


def run(input_file=INPUT_FILE, output_file=OUTPUT_FILE):
    df = pd.read_parquet(pv(input_file))
    prof = build_profile(df)
    analyze_item_quantity(df)
    keep = int(prof["use_in_model"].sum())
    print(f"\n[profile] {len(df):,} rows, {len(df.columns)} cols; use_in_model=1 for {keep} features")
    op = pv(output_file); op.parent.mkdir(parents=True, exist_ok=True)
    prof.to_excel(op, index=False)
    print(f"[save] {op}")
    return prof


def main():
    p = argparse.ArgumentParser(); p.add_argument("--input", default=INPUT_FILE); p.add_argument("--output", default=OUTPUT_FILE)
    a = p.parse_args(); run(a.input, a.output)


if __name__ == "__main__":
    main()
