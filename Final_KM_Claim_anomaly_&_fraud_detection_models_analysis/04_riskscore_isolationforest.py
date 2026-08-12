# ============================================================================
# 04_riskscore_isolationforest.py  -  ISOLATION FOREST RISK SCORING
# ============================================================================
#
# Step 4 (algorithm A). Reads the model-ready claim-level features from step 3,
# builds claimant/dealer tables (shared entity_features.py), scores them with
# Isolation Forest, applies business-aware risk levels, and ranks claims.
#
# Isolation Forest is a strong starting algorithm: it surfaces unusual
# claimant / dealer behaviour WITHOUT needing labelled fraud data.
#
# Outputs (parquet):
#   claimant_risk   ~1,969 rows  (one per claimant)
#   dealer_risk     ~  224 rows  (one per dealer)
#   claims_ranked   128,805 rows (one per claim, final audit queue)
#
# Run directly:   python "04_riskscore_isolationforest.py"
# ============================================================================

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest

from entity_features import build_claimant_features, build_dealer_features

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
INPUT_FILE    = 'data/03_claims_features.parquet'
OUT_CLAIMANT  = 'data/04_claimant_risk.parquet'
OUT_DEALER    = 'data/04_dealer_risk.parquet'
OUT_CLAIMS    = 'data/04_claims_ranked.parquet'


CONTAMINATION = 0.01     # conservative: fraud is rare
HIGH_RATE_THRESHOLD = 0.50
MOD_RATE_THRESHOLD  = 0.25
HIGH_PCT = 0.90          # top 10% of claims -> High
MED_PCT  = 0.80          # next 10%          -> Medium


# ============================================================================
# PART A - type cleanup (features already engineered upstream)
# ============================================================================

def clean_types(df):
    work = df.loc[:, ~df.columns.duplicated()].copy()
    work['ItemQuantity'] = pd.to_numeric(work['ItemQuantity'], errors='coerce').fillna(0)
    work['num_days'] = pd.to_numeric(work['num_days'], errors='coerce').astype(float)
    work['ItemClaimedDate'] = pd.to_datetime(work['ItemClaimedDate'], errors='coerce')
    return work


# ============================================================================
# PART C - Isolation Forest scoring
# ============================================================================

def score_anomalies(features_df, contamination=CONTAMINATION):
    """Add model_anomaly_flag, fraud_risk_score (0-100), model_risk_level."""
    result = features_df.copy()

    feature_cols = result.select_dtypes(include=np.number).columns.tolist()
    X = (result[feature_cols].replace([np.inf, -np.inf], np.nan)
         .fillna(0).astype('float64'))

    non_constant = X.columns[X.nunique() > 1].tolist() or feature_cols
    X = X[non_constant]

    X_scaled = RobustScaler().fit_transform(X)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    model = IsolationForest(n_estimators=100, contamination=contamination,
                            random_state=42)
    model.fit(X_scaled)

    result['model_anomaly_flag'] = (model.predict(X_scaled) == -1).astype(int)

    raw = -model.score_samples(X_scaled)
    result['fraud_risk_score'] = (
        MinMaxScaler(feature_range=(0, 100)).fit_transform(raw.reshape(-1, 1)).flatten())
    result['fraud_risk_score'] = result['fraud_risk_score'].fillna(0).clip(0, 100)

    result['model_risk_level'] = pd.cut(
        result['fraud_risk_score'], bins=[-np.inf, 60, 80, np.inf],
        labels=['Low', 'Medium', 'High'])

    return result


# ============================================================================
# PART D - business-aware risk levels + escalation reason
# ============================================================================

def assign_revised_risk_level(df, entity_type='claimant'):
    """Combine the model score with business rules to set the risk level."""
    result = df.copy()

    if entity_type == 'claimant':
        score_col, anomaly_col = 'fraud_risk_score_claimant', 'claimant_anomaly_flag'
        revised_col, model_col = 'claimant_risk_level', 'claimant_model_risk_level'
    elif entity_type == 'dealer':
        score_col, anomaly_col = 'fraud_risk_score_dealer', 'dealer_anomaly_flag'
        revised_col, model_col = 'dealer_risk_level', 'dealer_model_risk_level'
    else:
        raise ValueError("entity_type must be 'claimant' or 'dealer'")

    result[model_col] = pd.cut(
        result[score_col], bins=[-np.inf, 60, 80, np.inf],
        labels=['Low', 'Medium', 'High'])

    rate_cols = [c for c in [
        'fraud_flag_rate', 'anomaly_flag_rate', 'high_quantity_anomaly_flag_rate',
        'high_monthly_claim_activity_flag_rate', 'low_monthly_claim_activity_flag_rate',
        'high_monthly_units_flag_rate', 'claims_over_90days_rate']
        if c in result.columns]

    has_high_rate = (result[rate_cols].max(axis=1) >= HIGH_RATE_THRESHOLD).astype(int)
    has_mod_rate  = (result[rate_cols].max(axis=1) >= MOD_RATE_THRESHOLD).astype(int)

    high_condition = (
        (result[score_col] >= 80)                                        # model
        | (result[anomaly_col] == 1)                                     # business rule
        | (result.get('high_monthly_claim_activity_flag_count', 0) > 2)  # business rule- claimant_monthly['high_monthly_claim_activity_flag'] = (claimant_monthly['monthly_claim_count_z_score'] >= 2)
        | (result.get('high_monthly_units_flag_count', 0) > 2)           # business rule-claimant_monthly['high_monthly_units_flag'] = (claimant_monthly['monthly_units_z_score'] >= 2)
        | (result.get('high_quantity_anomaly_flag_count', 0) > 2)        # business rule-((work['quantity_z_score'] >= 3))
        | (has_high_rate == 1)                                           # business rule
    )
    medium_condition = (result[score_col] >= 60) | (has_mod_rate == 1)

    result[revised_col] = np.select(
        [high_condition, medium_condition], ['High', 'Medium'], default='Low')

    # record WHICH condition escalated each entity to High
    reason_col = f'{entity_type}_escalation_reason'

    def _why_high(r):
        reasons = []
        if r[score_col] >= 80:                                       reasons.append('score>=80')
        if r.get(anomaly_col, 0) == 1:                              reasons.append('anomaly_flag')
        if r.get('high_monthly_claim_activity_flag_count', 0) > 2:  reasons.append('monthly_activity>2')
        if r.get('high_monthly_units_flag_count', 0) > 2:           reasons.append('monthly_units>2')
        if r.get('high_quantity_anomaly_flag_count', 0) > 2:        reasons.append('quantity>2')
        if not reasons:                                             reasons.append('worst_rate>=0.50')
        return ', '.join(reasons)

    result[reason_col] = np.where(
        result[revised_col] == 'High', result.apply(_why_high, axis=1), '')

    return result


def round_numeric_output(df, decimals=2):
    result = df.copy()
    fcols = result.select_dtypes(include=['float', 'float64', 'float32']).columns
    result[fcols] = result[fcols].round(decimals)
    return result


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_isolationforest_pipeline(df, contamination=CONTAMINATION):
    """Return (claimant_risk, dealer_risk, claims_ranked)."""
    clean_df = clean_types(df)

    # STEP 2: entity tables (shared builders)
    claimant_features = build_claimant_features(clean_df)
    dealer_features   = build_dealer_features(clean_df)

    # STEP 3: Isolation Forest
    claimant_scored = score_anomalies(claimant_features, contamination)
    dealer_scored   = score_anomalies(dealer_features, contamination)

    # STEP 4: rename scoring columns
    claimant_scored = claimant_scored.rename(columns={
        'fraud_risk_score': 'fraud_risk_score_claimant',
        'model_anomaly_flag': 'claimant_anomaly_flag',
        'model_risk_level': 'claimant_model_risk_level'})
    dealer_scored = dealer_scored.rename(columns={
        'fraud_risk_score': 'fraud_risk_score_dealer',
        'model_anomaly_flag': 'dealer_anomaly_flag',
        'model_risk_level': 'dealer_model_risk_level'})

    # STEP 5: business-aware risk levels
    claimant_scored = assign_revised_risk_level(claimant_scored, 'claimant')
    dealer_scored   = assign_revised_risk_level(dealer_scored, 'dealer')

    # STEP 6/7: merge entity risk back to claim level
    clean_df = clean_df.merge(
        claimant_scored[['ClaimantName', 'fraud_risk_score_claimant',
                         'claimant_anomaly_flag', 'claimant_model_risk_level',
                         'claimant_risk_level']],
        on='ClaimantName', how='left')
    clean_df = clean_df.merge(
        dealer_scored[['DealerName', 'fraud_risk_score_dealer',
                       'dealer_anomaly_flag', 'dealer_model_risk_level',
                       'dealer_risk_level']],
        on='DealerName', how='left')

    # STEP 8: combined claim score (50/50)
    clean_df['final_score'] = (clean_df['fraud_risk_score_claimant'] * 0.50
                               + clean_df['fraud_risk_score_dealer'] * 0.50)

    # STEP 9: claim-level risk by RANK (percentile) + hard claim evidence
    _hi = clean_df['final_score'].quantile(HIGH_PCT)
    _md = clean_df['final_score'].quantile(MED_PCT)
    clean_df['risk_level'] = np.select(
        [
            (clean_df['final_score'] >= _hi)
            | (clean_df['high_quantity_anomaly_flag'] == 1)
            | (clean_df['fraud_flag'] == 1),
            (clean_df['final_score'] >= _md),
        ],
        ['High', 'Medium'], default='Low')

    clean_df = clean_df.sort_values('final_score', ascending=False)

    # STEP 10: round floats
    claimant_scored = round_numeric_output(claimant_scored)
    dealer_scored   = round_numeric_output(dealer_scored)
    clean_df        = round_numeric_output(clean_df)

    return claimant_scored, dealer_scored, clean_df


def run(input_path=INPUT_FILE, contamination=CONTAMINATION, save=True):
    """Read step-3 output, run the IF pipeline, save the three tables."""
    df = pd.read_parquet(input_path)
    claimant_risk, dealer_risk, claims_ranked = run_isolationforest_pipeline(
        df, contamination)

    print("[if] claimant_risk:", claimant_risk.shape)
    print("[if] dealer_risk  :", dealer_risk.shape)
    print("[if] claims_ranked:", claims_ranked.shape)
    print("\n[if] claim risk_level distribution:")
    print(claims_ranked['risk_level'].value_counts())

    if save:
        import os
        os.makedirs('data', exist_ok=True)
        claimant_risk.to_parquet(OUT_CLAIMANT, index=False)
        dealer_risk.to_parquet(OUT_DEALER, index=False)
        claims_ranked.to_parquet(OUT_CLAIMS, index=False)

        # --- NEW: Excel copies ---
        claimant_risk.to_excel(OUT_CLAIMANT.replace('.parquet', '.xlsx'), index=False)
        dealer_risk.to_excel(OUT_DEALER.replace('.parquet', '.xlsx'), index=False)
        claims_ranked.to_excel(OUT_CLAIMS.replace('.parquet', '.xlsx'), index=False)

        print(f"[save] parquet + xlsx -> {OUT_CLAIMANT.replace('.parquet','')} "
        f"(claimant / dealer / claims)")


        #print(f"[save] -> {OUT_CLAIMANT}, {OUT_DEALER}, {OUT_CLAIMS}")

    return claimant_risk, dealer_risk, claims_ranked


if __name__ == '__main__':
    run()
