# ============================================================================
# 02_product_velocity.py  -  PRODUCT VELOCITY FEATURE
# ============================================================================
#
# Step 2 of the pipeline. Reads the base dataset from step 1, adds the
# product-velocity feature, and saves the result for step 3.
#
# WHAT IT MEASURES :
#   Existing product features measure QUANTITY PER CLAIM.
#   This measures CLAIMS PER MONTH - a surge in how OFTEN a product is
#   claimed, even when every individual quantity looks normal.
#  It is a continuous feature (ratio) and a binary spike flag (ratio >= 2x).
# It measures the SEASONALITY of a product's claims, and is a strong predictor of fraud.
#
# It is deliberately kept SEPARATE from anamoly_flag (folding it in diluted
# the flag's fraud-lift). It is exposed instead as a continuous feature for
# the model to weigh on its own merits.
#
# Run directly:   python "02_product_velocity.py"
# ============================================================================

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
INPUT_FILE  = 'data/01_claims_base.parquet'
OUTPUT_FILE = 'data/02_claims_velocity.parquet'

VELOCITY_MULTIPLIER = 2.0   # a month exceeding 2x the product's own average
VELOCITY_MIN_MONTHS = 3     # require some history before judging

VELOCITY_COLS = ['product_month_claims', 'product_avg_month_claims',
                 'product_active_months', 'product_velocity_ratio',
                 'product_velocity_spike_flag']


# ============================================================================
# FEATURE
# ============================================================================

def add_product_velocity(df,
                         multiplier=VELOCITY_MULTIPLIER,
                         min_months=VELOCITY_MIN_MONTHS):
    """Add product-velocity columns to the claim-level dataframe.

    Adds:
      product_month_claims        claims for this product in this month
      product_avg_month_claims    that product's typical claims-per-month
      product_active_months       distinct months the product appeared
      product_velocity_ratio      month_claims / avg_month_claims
      product_velocity_spike_flag ratio >= multiplier AND months >= min_months
    """
    work = df.copy()

    # claim_month is needed for the monthly grouping
    work['ItemClaimedDate'] = pd.to_datetime(work['ItemClaimedDate'], errors='coerce')
    work['claim_month'] = (work['ItemClaimedDate'].dt.to_period('M')
                           .astype(str).replace('NaT', 'Unknown'))

    # STEP 1: claims per product per month
    pm = (work.groupby(['ProductID', 'claim_month'])
              .agg(product_month_claims=('ClaimID', 'count'))
              .reset_index())

    # STEP 2: each product's own monthly baseline
    pb = (pm.groupby('ProductID')
            .agg(product_avg_month_claims=('product_month_claims', 'mean'),
                 product_active_months=('claim_month', 'nunique'))
            .reset_index())
    pm = pm.merge(pb, on='ProductID', how='left')

    # STEP 3: ratio of this month vs a normal month
    pm['product_velocity_ratio'] = (
        pm['product_month_claims'] / pm['product_avg_month_claims'].replace(0, np.nan))
    pm['product_velocity_ratio'] = (pm['product_velocity_ratio']
                                    .replace([np.inf, -np.inf], np.nan).fillna(0))

    # STEP 4: spike flag (guarded by minimum history)
    pm['product_velocity_spike_flag'] = (
        (pm['product_velocity_ratio'] >= multiplier)
        & (pm['product_active_months'] >= min_months)
    ).astype(int)

    # STEP 5: merge back to claim level (suffix-proof, re-run safe)
    work = work.drop(columns=[c for c in VELOCITY_COLS if c in work.columns],
                     errors='ignore')
    work = work.merge(pm[['ProductID', 'claim_month'] + VELOCITY_COLS],
                      on=['ProductID', 'claim_month'], how='left')
    work[VELOCITY_COLS] = work[VELOCITY_COLS].fillna(0)

    return work


def report_velocity_lift(df):
    """Quick sanity check: does the spike flag predict fraud_flag?"""
    base = df['fraud_flag'].mean()
    mask = df['product_velocity_spike_flag'] == 1
    n = int(mask.sum())
    print(f"[velocity] spikes = {n:,} claims ({100*n/len(df):.2f}%)")
    if n:
        lift = df.loc[mask, 'fraud_flag'].mean() / base
        verdict = ('strong, keep' if lift >= 2.0
                   else 'moderate, keep as model feature' if lift >= 1.5
                   else 'weak; raise multiplier or rely on the continuous ratio')
        print(f"[velocity] lift vs fraud = {lift:.2f}x  "
              f"(base {100*base:.2f}%)  -> {verdict}")


# ============================================================================
# ORCHESTRATOR
# ============================================================================

def run(input_path=INPUT_FILE, output_path=OUTPUT_FILE, save=True):
    """Read step-1 output, add velocity, save, and return the dataframe."""
    df = pd.read_parquet(input_path)
    df = add_product_velocity(df)
    report_velocity_lift(df)

    if save:
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_parquet(output_path, index=False)
        print(f"[save] -> {output_path}  ({len(df):,} rows)")

    return df


if __name__ == '__main__':
    run()
