# ============================================================================
# 03_build_features.py  -  BEHAVIORAL FEATURE ENGINEERING (claim level)
# ============================================================================
#
# Step 3 of the pipeline. Reads step-2 output and engineers the behavioural
# features the anomaly models consume. Produces ONE model-ready claim-level
# dataframe, saved to parquet.
#
# Two behavioural ideas are added:
#   A. Product-quantity anomalies - each claim's quantity vs its product norm
#   B. Claimant monthly activity  - each claimant's month vs their own history
#
# NAMING CONVENTION
#   product_*            product-level baseline statistics
#   quantity_*           claim-level quantity comparison to the product norm
#   claimant_*           claimant-level monthly baseline
#   monthly_*            claimant-month observations & their z-scores
#   *_flag               0/1 indicator
#
# Run directly:   python "03_build_features.py"
# ============================================================================

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
INPUT_FILE  = 'data/02_claims_velocity.parquet'
OUTPUT_FILE = 'data/03_claims_features.parquet'

QUANTITY_Z_THRESHOLD = 3     # products: large, stable baseline -> strict
MONTHLY_Z_THRESHOLD  = 2     # claimants: few months, noisy std -> looser

# Behavioural columns this module generates (dropped first, so re-runs are safe)
_GENERATED_COLS = [
    'product_claim_count', 'product_total_units', 'product_avg_units',
    'product_std_units', 'quantity_vs_product_avg',
    'quantity_ratio_to_product_avg', 'quantity_z_score',
    'high_quantity_anomaly_flag',
    'monthly_claim_count', 'monthly_units_claimed', 'monthly_unique_dealers',
    'monthly_unique_products', 'monthly_fraud_count', 'monthly_anomaly_count',
    'claimant_avg_monthly_claims', 'claimant_std_monthly_claims',
    'claimant_avg_monthly_units', 'claimant_std_monthly_units',
    'claimant_active_months', 'monthly_claim_count_z_score',
    'monthly_units_z_score', 'high_monthly_claim_activity_flag',
    'low_monthly_claim_activity_flag', 'high_monthly_units_flag',
]


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def _anomaly_col(df):
    """Return whichever anomaly-flag spelling exists."""
    for c in ('anamoly_flag', 'anomaly_flag'):
        if c in df.columns:
            return c
    raise KeyError("Need 'anamoly_flag' or 'anomaly_flag' in the dataframe.")


# ============================================================================
# 1. PREPARE
# ============================================================================

def prepare_behavioral_data(df):
    """Clean copy, re-run safe, numeric quantity, claim_month."""
    work = df.copy()
    work = work.loc[:, ~work.columns.duplicated()]
    work = work.drop(columns=[c for c in _GENERATED_COLS if c in work.columns],
                     errors='ignore')

    work['ItemQuantity'] = pd.to_numeric(work['ItemQuantity'], errors='coerce').fillna(0)
    work['ItemClaimedDate'] = pd.to_datetime(work['ItemClaimedDate'], errors='coerce')

    if 'claim_month' not in work.columns:
        work['claim_month'] = work['ItemClaimedDate'].dt.to_period('M').astype(str)
        work['claim_month'] = work['claim_month'].replace('NaT', 'Unknown')

    return work


# ============================================================================
# 2 & 3. PRODUCT-QUANTITY FEATURES
# ============================================================================

def add_product_quantity_features(df, product_col='ProductID'):
    """Compare each claim's quantity to its product's typical quantity."""
    work = df.copy()

    baseline = (
        work.groupby(product_col)
        .agg(product_claim_count=('ClaimID', 'count'),
             product_total_units=('ItemQuantity', 'sum'),
             product_avg_units=('ItemQuantity', 'mean'),
             product_std_units=('ItemQuantity', 'std'))
        .reset_index()
    )
    baseline['product_std_units'] = baseline['product_std_units'].fillna(0)

    dup = [c for c in baseline.columns if c != product_col and c in work.columns]
    work = work.drop(columns=dup, errors='ignore')
    work = work.merge(baseline, on=product_col, how='left')

    work['quantity_vs_product_avg'] = work['ItemQuantity'] - work['product_avg_units']

    work['quantity_ratio_to_product_avg'] = (
        work['ItemQuantity'] / work['product_avg_units'].replace(0, np.nan))
    work['quantity_ratio_to_product_avg'] = (
        work['quantity_ratio_to_product_avg']
        .replace([np.inf, -np.inf], np.nan).fillna(0))

    work['quantity_z_score'] = (
        (work['ItemQuantity'] - work['product_avg_units'])
        / work['product_std_units'].replace(0, np.nan))
    work['quantity_z_score'] = (
        work['quantity_z_score'].replace([np.inf, -np.inf], np.nan).fillna(0))

    work['high_quantity_anomaly_flag'] = (
        work['quantity_z_score'] >= QUANTITY_Z_THRESHOLD).astype(int)

    return work


# ============================================================================
# 4 & 5. CLAIMANT MONTHLY FEATURES
# ============================================================================

def add_claimant_monthly_features(df):
    """Compare each claimant's month to their OWN typical monthly behaviour."""
    work = df.copy()
    acol = _anomaly_col(work)

    monthly = (
        work.groupby(['ClaimantName', 'claim_month'])
        .agg(monthly_claim_count=('ClaimID', 'count'),
             monthly_units_claimed=('ItemQuantity', 'sum'),
             monthly_unique_dealers=('DealerName', 'nunique'),
             monthly_unique_products=('ProductID', 'nunique'),
             monthly_fraud_count=('fraud_flag', 'sum'),
             monthly_anomaly_count=(acol, 'sum'))
        .reset_index()
    )

    baseline = (
        monthly.groupby('ClaimantName')
        .agg(claimant_avg_monthly_claims=('monthly_claim_count', 'mean'),
             claimant_std_monthly_claims=('monthly_claim_count', 'std'),
             claimant_avg_monthly_units=('monthly_units_claimed', 'mean'),
             claimant_std_monthly_units=('monthly_units_claimed', 'std'),
             claimant_active_months=('claim_month', 'nunique'))
        .reset_index().fillna(0)
    )
    monthly = monthly.merge(baseline, on='ClaimantName', how='left')

    monthly['monthly_claim_count_z_score'] = (
        (monthly['monthly_claim_count'] - monthly['claimant_avg_monthly_claims'])
        / monthly['claimant_std_monthly_claims'].replace(0, np.nan))
    monthly['monthly_claim_count_z_score'] = (
        monthly['monthly_claim_count_z_score']
        .replace([np.inf, -np.inf], np.nan).fillna(0))

    monthly['monthly_units_z_score'] = (
        (monthly['monthly_units_claimed'] - monthly['claimant_avg_monthly_units'])
        / monthly['claimant_std_monthly_units'].replace(0, np.nan))
    monthly['monthly_units_z_score'] = (
        monthly['monthly_units_z_score']
        .replace([np.inf, -np.inf], np.nan).fillna(0))

    monthly['high_monthly_claim_activity_flag'] = (
        monthly['monthly_claim_count_z_score'] >= MONTHLY_Z_THRESHOLD).astype(int)
    monthly['low_monthly_claim_activity_flag'] = (
        monthly['monthly_claim_count_z_score'] <= -MONTHLY_Z_THRESHOLD).astype(int)
    monthly['high_monthly_units_flag'] = (
        monthly['monthly_units_z_score'] >= MONTHLY_Z_THRESHOLD).astype(int)

    # merge every monthly column back to claim level (dynamic -> re-run safe)
    monthly_cols = [c for c in monthly.columns
                    if c not in ('ClaimantName', 'claim_month')]
    work = work.drop(columns=[c for c in monthly_cols if c in work.columns],
                     errors='ignore')
    work = work.merge(monthly[['ClaimantName', 'claim_month'] + monthly_cols],
                      on=['ClaimantName', 'claim_month'], how='left')

    return work


# ============================================================================
# 6. ORCHESTRATOR
# ============================================================================

def build_features(df):
    """Return ONE claim-level dataframe with all behavioural features."""
    work = prepare_behavioral_data(df)
    work = add_product_quantity_features(work, product_col='ProductID')
    work = add_claimant_monthly_features(work)
    return work


def run(input_path=INPUT_FILE, output_path=OUTPUT_FILE, save=True):
    """Read step-2 output, build features, save, and return the dataframe."""
    df = pd.read_parquet(input_path)
    df = build_features(df)

    print(f"[features] rows = {len(df):,}")
    print(f"[features] high_quantity_anomaly_flag   = "
          f"{int(df['high_quantity_anomaly_flag'].sum()):,}")
    print(f"[features] high_monthly_claim_activity  = "
          f"{int(df['high_monthly_claim_activity_flag'].sum()):,}")

    if save:
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_parquet(output_path, index=False)
        print(f"[save] -> {output_path}  ({len(df):,} rows)")
        df.to_excel(output_path.replace('.parquet', '.xlsx'), index=False) # NEW

        print(f"[save] -> {output_path} (+ .xlsx) ({len(df):,} rows)")
    return df


if __name__ == '__main__':
    run()
