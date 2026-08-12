# ============================================================================
# entity_features.py  -  SHARED helper: claim-level  ->  entity-level tables
# ============================================================================
#
# Both 04_riskscore_isolationforest.py and 05_riskscore_4models.py need the
# SAME claimant / dealer feature tables. Keeping the builders here (one place)
# means:
#   - no duplicated logic
#   - any algorithm can import the identical entity tables and compare fairly
#
# INPUT : the fully-featured claim-level dataframe produced by 03_build_features
#         (it already carries fraud_flag, anamoly_flag, quantity z-scores,
#          monthly z-scores/flags, and product-velocity columns).
# OUTPUT: one row per claimant / per dealer.
#
# NAMING CONVENTION (applied throughout the project)
#   product_*    -> product-level statistics
#   claim / qty  -> a single claim line
#   claimant_*   -> claimant-level aggregates
#   dealer_*     -> dealer-level aggregates
#   *_count      -> how many times a flag fired (sum)
#   *_rate       -> what fraction of claims fired  (mean)
#   *_flag       -> a 0/1 indicator
# ============================================================================

import numpy as np
import pandas as pd


def _velocity_aggs(df):
    """Velocity aggregation specs - only for columns that exist (safe if the
    velocity step has not been run yet)."""
    specs = {}
    if 'product_velocity_spike_flag' in df.columns:
        specs['velocity_spike_count'] = ('product_velocity_spike_flag', 'sum')
        specs['velocity_spike_rate']  = ('product_velocity_spike_flag', 'mean')
    if 'product_velocity_ratio' in df.columns:
        specs['max_product_velocity_ratio'] = ('product_velocity_ratio', 'max')
        specs['avg_product_velocity_ratio'] = ('product_velocity_ratio', 'mean')
    return specs


def build_claimant_features(df):
    """Aggregate claim-level behavioural flags to ONE ROW PER CLAIMANT."""
    aggs = {
        # --- volume ---
        'total_claims':  ('ClaimID', 'count'),
        'num_dealers':   ('DealerName', 'nunique'),
        'num_products':  ('ProductID', 'nunique'),

        # --- business-rule flags (count + rate) ---
        'fraud_flag_count':   ('fraud_flag', 'sum'),
        'fraud_flag_rate':    ('fraud_flag', 'mean'),
        'anomaly_flag_count': ('anamoly_flag', 'sum'),
        'anomaly_flag_rate':  ('anamoly_flag', 'mean'),

        # --- product-quantity behaviour ---
        'high_quantity_anomaly_flag_count': ('high_quantity_anomaly_flag', 'sum'),
        'high_quantity_anomaly_flag_rate':  ('high_quantity_anomaly_flag', 'mean'),

        # --- monthly-activity behaviour ---
        'high_monthly_claim_activity_flag_count': ('high_monthly_claim_activity_flag', 'sum'),
        'high_monthly_claim_activity_flag_rate':  ('high_monthly_claim_activity_flag', 'mean'),
        'low_monthly_claim_activity_flag_count':  ('low_monthly_claim_activity_flag', 'sum'),
        'low_monthly_claim_activity_flag_rate':   ('low_monthly_claim_activity_flag', 'mean'),
        'high_monthly_units_flag_count':          ('high_monthly_units_flag', 'sum'),
        'high_monthly_units_flag_rate':           ('high_monthly_units_flag', 'mean'),

        # --- z-score summaries (worst / average intensity) ---
        'max_quantity_z_score':            ('quantity_z_score', 'max'),
        'avg_quantity_z_score':            ('quantity_z_score', 'mean'),
        'max_monthly_claim_count_z_score': ('monthly_claim_count_z_score', 'max'),
        'max_monthly_units_z_score':       ('monthly_units_z_score', 'max'),
        'claimant_active_months':          ('claimant_active_months', 'max'),

        # --- timing ---
        'claims_over_90days': ('num_days', lambda x: (x >= 90).sum()),
        'avg_days_to_claim':  ('num_days', 'mean'),
    }
    aggs.update(_velocity_aggs(df))          # add product-velocity aggregates

    cf = df.groupby('ClaimantName').agg(**aggs).reset_index()

    # --- normalized ratios (reduce bias toward high-volume claimants) ---
    cf['claims_over_90days_rate'] = cf['claims_over_90days'] / cf['total_claims']
    cf['products_per_claim']      = cf['num_products'] / cf['total_claims']
    cf['dealers_per_claim']       = cf['num_dealers'] / cf['total_claims']

    return cf.fillna(0)


def build_dealer_features(df):
    """Aggregate claim-level behavioural flags to ONE ROW PER DEALER."""
    aggs = {
        'total_claims':  ('ClaimID', 'count'),
        'num_claimants': ('ClaimantName', 'nunique'),
        'num_products':  ('ProductID', 'nunique'),

        'fraud_flag_count':   ('fraud_flag', 'sum'),
        'fraud_flag_rate':    ('fraud_flag', 'mean'),
        'anomaly_flag_count': ('anamoly_flag', 'sum'),
        'anomaly_flag_rate':  ('anamoly_flag', 'mean'),

        'high_quantity_anomaly_flag_count': ('high_quantity_anomaly_flag', 'sum'),
        'high_quantity_anomaly_flag_rate':  ('high_quantity_anomaly_flag', 'mean'),

        'high_monthly_claim_activity_flag_count': ('high_monthly_claim_activity_flag', 'sum'),
        'high_monthly_claim_activity_flag_rate':  ('high_monthly_claim_activity_flag', 'mean'),
        'low_monthly_claim_activity_flag_count':  ('low_monthly_claim_activity_flag', 'sum'),
        'low_monthly_claim_activity_flag_rate':   ('low_monthly_claim_activity_flag', 'mean'),
        'high_monthly_units_flag_count':          ('high_monthly_units_flag', 'sum'),
        'high_monthly_units_flag_rate':           ('high_monthly_units_flag', 'mean'),

        'max_quantity_z_score':            ('quantity_z_score', 'max'),
        'avg_quantity_z_score':            ('quantity_z_score', 'mean'),
        'max_monthly_claim_count_z_score': ('monthly_claim_count_z_score', 'max'),
        'max_monthly_units_z_score':       ('monthly_units_z_score', 'max'),

        'claims_over_90days': ('num_days', lambda x: (x >= 90).sum()),
        'avg_days_to_claim':  ('num_days', 'mean'),
    }
    aggs.update(_velocity_aggs(df))

    dz = df.groupby('DealerName').agg(**aggs).reset_index()

    dz['claims_over_90days_rate'] = dz['claims_over_90days'] / dz['total_claims']
    dz['claimants_per_claim']     = dz['num_claimants'] / dz['total_claims']
    dz['products_per_claim']      = dz['num_products'] / dz['total_claims']

    return dz.fillna(0)
