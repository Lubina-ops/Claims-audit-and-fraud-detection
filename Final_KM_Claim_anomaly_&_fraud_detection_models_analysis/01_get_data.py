# ============================================================================
# 01_get_data.py  -  LOAD, CLEAN, and DERIVE the base claim-level dataset
# ============================================================================
#
# Step 1 of the KM claims fraud/anomaly pipeline.
#
# Responsibilities:
#   1. Read the raw claims extract (Excel).
#   2. Standardize dates and compute filing delay (num_days + buckets).
#   3. Derive the two business-rule labels: fraud_flag and anamoly_flag.
#   4. Save the prepared dataframe to parquet for the next steps.
#
# Run directly:      python "01_get_data.py"
# Or import:         from importlib ...  get_data(input_path, output_path)
#
# See 01_get_data.md for the meaning of every claim status / reason.
# ============================================================================

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
INPUT_FILE  = 'consolidated_Vishalquery_detail_data.xlsx'
OUTPUT_FILE = 'data/01_claims_base.parquet'

DATE_COLS = ['InvoiceDate', 'ItemClaimedDate', 'ClaimItemStatusDate']

NUM_DAYS_BINS   = [-1, 0, 7, 14, 21, 30, 60, 90, 180, float('inf')]
NUM_DAYS_LABELS = ['0 days', '1-week', '2-week', '3-week', '4-week',
                   '30-60 days', '60-90 days', '90-180 days', '180+ days']

# fraud_flag: claim was rejected / duplicated / had no matching product
FRAUD_REASON_PATTERN = (r"Duplicate Claim|No Match Found|"
                        r"unit's ineligible as it's replacement & was unsold|Rejected")
FRAUD_STATUS_PATTERN = r"Rejected|Duplicate Claim|No Match Found"

# anamoly_flag (iteration 2): recalled OR filed very late (90+ days)
ANOMALY_LATE_BUCKETS = ['90-180 days', '180+ days']


# ============================================================================
# 1. LOAD
# ============================================================================

def load_raw(path=INPUT_FILE):
    """Read the raw Excel extract and print a short overview."""
    df = pd.read_excel(path)
    print(f"[load] rows={len(df):,}  cols={df.shape[1]}")
    return df


# ============================================================================
# 2. TIMING FEATURES
# ============================================================================

def add_timing_features(df):
    """Standardize dates, compute num_days (sale -> claim) and its buckets."""
    work = df.copy()

    # standardize the selected date columns to YYYY-MM-DD strings
    for col in DATE_COLS:
        if col in work.columns:
            work[col] = pd.to_datetime(work[col], errors='coerce').dt.strftime('%Y-%m-%d')

    # num_days = days between invoice (sold) and item-claimed (filed)
    invoice = pd.to_datetime(work['InvoiceDate'], errors='coerce')
    claimed = pd.to_datetime(work['ItemClaimedDate'], errors='coerce')
    work['num_days'] = (claimed - invoice).dt.days.round(0).astype('Int64')

    # banded version
    work['num_days_buckets'] = pd.cut(work['num_days'],
                                      bins=NUM_DAYS_BINS, labels=NUM_DAYS_LABELS)

    n_neg = int((work['num_days'] < 0).sum())
    if n_neg:
        print(f"[timing] WARNING: {n_neg:,} claims have negative num_days "
              "(claim date precedes invoice date - possible data-entry issue)")

    return work


# ============================================================================
# 3. BUSINESS-RULE LABELS
# ============================================================================

def add_fraud_flag(df):
    """fraud_flag = 1 if the claim was rejected / duplicated / unmatched."""
    work = df.copy()
    work['fraud_flag'] = (
        work['ClaimItemStatusReason'].astype(str)
            .str.contains(FRAUD_REASON_PATTERN, case=False, na=False)
        | work['ClaimItemStatus'].astype(str)
            .str.contains(FRAUD_STATUS_PATTERN, case=False, na=False)
    ).astype(int)
    return work


def add_anomaly_flag(df):
    """anamoly_flag = 1 if recalled OR filed 90+ days after sale.

    NOTE the column is intentionally spelled 'anamoly_flag' to match the
    source data; downstream helpers accept either spelling.
    """
    work = df.copy()
    work['anamoly_flag'] = (
        work['ClaimItemStatusReason'].astype(str)
            .str.contains("Recalled", case=False, na=False)
        | work['num_days_buckets'].isin(ANOMALY_LATE_BUCKETS)
    ).astype(int)
    return work


# ============================================================================
# 4. ORCHESTRATOR
# ============================================================================

def get_data(input_path=INPUT_FILE, output_path=OUTPUT_FILE, save=True):
    """Run all of step 1 and return the prepared claim-level dataframe."""
    df = load_raw(input_path)
    df = add_timing_features(df)
    df = add_fraud_flag(df)
    df = add_anomaly_flag(df)

    print(f"[flags] fraud_flag  = {int(df['fraud_flag'].sum()):,} "
          f"({100*df['fraud_flag'].mean():.2f}%)")
    print(f"[flags] anamoly_flag= {int(df['anamoly_flag'].sum()):,} "
          f"({100*df['anamoly_flag'].mean():.2f}%)")

    if save:
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_parquet(output_path, index=False)
        print(f"[save] -> {output_path}  ({len(df):,} rows)")

    return df


if __name__ == '__main__':
    get_data()
