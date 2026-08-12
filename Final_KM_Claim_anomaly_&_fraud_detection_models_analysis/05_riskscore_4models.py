# ============================================================================
# 05_riskscore_4models.py  -  MULTI-ALGORITHM ANOMALY DETECTION
# ============================================================================
#
# Step 5 (algorithm comparison). Reads the SAME model-ready features from
# step 3, builds the SAME entity tables (shared entity_features.py), and runs
# FOUR unsupervised detectors so their outputs can be compared:
#
#   IsolationForest  - globally unusual
#   LocalOutlierFactor - unusual vs nearest neighbours
#   OneClassSVM      - outside the learned normal boundary
#   KMeans distance  - far from its cluster centre  (weakest signal)
#
# WHY NORMALIZE: the four models return incompatible scales, so every score is
# rescaled to 0-100 before averaging into an ensemble_score. The key output is
# models_agreeing - an entity flagged by 2+ methods is highest confidence.
#
# NOTE: One-Class SVM is O(n^2). Fine on ~2,000 claimants / ~224 dealers;
# do NOT run it on the 128K claim rows without subsampling.
#
# Run directly:   python "05_riskscore_4models.py"
# ============================================================================

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans

from entity_features import build_claimant_features, build_dealer_features

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
INPUT_FILE      = 'data/03_claims_features.parquet'
OUT_CLAIMANT    = 'data/05_claimant_multi.parquet'
OUT_DEALER      = 'data/05_dealer_multi.parquet'
OUT_CLAIMS      = 'data/05_claims_multi.parquet'

CONTAMINATION = 0.05


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def to_100(x):
    """Rescale any score to 0-100 so models can be averaged together."""
    x = np.nan_to_num(np.asarray(x, dtype=float),
                      nan=0.0, posinf=0.0, neginf=0.0).reshape(-1, 1)
    return MinMaxScaler(feature_range=(0, 100)).fit_transform(x).flatten()


def prep_matrix(features_df):
    """Numeric -> float64 -> drop constant columns -> robust scale."""
    cols = features_df.select_dtypes(include=np.number).columns.tolist()
    X = (features_df[cols].replace([np.inf, -np.inf], np.nan)
         .fillna(0).astype('float64'))
    X = X[X.columns[X.nunique() > 1].tolist() or cols]
    Xs = RobustScaler().fit_transform(X)
    return np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)


# ============================================================================
# run four models on ONE entity table
# ============================================================================

def run_all_models(features_df, id_col, contamination=CONTAMINATION):
    """Return (dataframe with per-model flags/scores + consensus, flag_cols)."""
    X = prep_matrix(features_df)
    n = len(features_df)
    out = features_df[[id_col]].copy()

    # 1. Isolation Forest
    iso = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
    iso.fit(X)
    out['IForest_flag']  = (iso.predict(X) == -1).astype(int)
    out['IForest_score'] = to_100(-iso.score_samples(X))

    # 2. Local Outlier Factor  (neighbourhood must fit the population)
    k = min(20, max(5, n // 10))
    lof = LocalOutlierFactor(n_neighbors=k, contamination=contamination)
    out['LOF_flag']  = (lof.fit_predict(X) == -1).astype(int)
    out['LOF_score'] = to_100(-lof.negative_outlier_factor_)

    # 3. One-Class SVM
    svm = OneClassSVM(nu=contamination, kernel='rbf', gamma='scale')
    svm.fit(X)
    out['SVM_flag']  = (svm.predict(X) == -1).astype(int)
    out['SVM_score'] = to_100(-svm.decision_function(X))

    # 4. KMeans distance (hand-built flag: farthest 5% from cluster centre)
    km = KMeans(n_clusters=min(5, max(2, n // 50)), random_state=42, n_init=10)
    labels = km.fit_predict(X)
    dist = np.linalg.norm(X - km.cluster_centers_[labels], axis=1)
    out['KMeans_flag']  = (dist >= np.percentile(dist, 95)).astype(int)
    out['KMeans_score'] = to_100(dist)

    # consensus + ensemble
    flag_cols  = [c for c in out.columns if c.endswith('_flag')]
    score_cols = [c for c in out.columns if c.endswith('_score')]
    out['models_agreeing'] = out[flag_cols].sum(axis=1)
    out['ensemble_score']  = out[score_cols].mean(axis=1).round(2)

    return out.sort_values('ensemble_score', ascending=False), flag_cols


def summarize(label, res, flag_cols):
    print("\n" + "=" * 60)
    print(f"{label} - flagged by each model")
    print("=" * 60)
    for c in flag_cols:
        print(f"  {c:<16}{int(res[c].sum()):>6}  ({100*res[c].mean():5.1f}%)")
    print("  Flagged by 2+ models:",
          int((res['models_agreeing'] >= 2).sum()), " <- highest confidence")


# ============================================================================
# combine entity ensembles -> claim level
# ============================================================================

def combine_to_claims(base_df, claimant_multi, dealer_multi):
    """Blend claimant + dealer ensemble scores onto every claim (50/50)."""
    claims = base_df[['ClaimID', 'ClaimantName', 'DealerName']].copy()

    claims = claims.merge(
        claimant_multi[['ClaimantName', 'ensemble_score', 'models_agreeing']].rename(
            columns={'ensemble_score': 'claimant_ensemble',
                     'models_agreeing': 'claimant_models_agreeing'}),
        on='ClaimantName', how='left')
    claims = claims.merge(
        dealer_multi[['DealerName', 'ensemble_score', 'models_agreeing']].rename(
            columns={'ensemble_score': 'dealer_ensemble',
                     'models_agreeing': 'dealer_models_agreeing'}),
        on='DealerName', how='left')

    claims[['claimant_ensemble', 'dealer_ensemble']] = (
        claims[['claimant_ensemble', 'dealer_ensemble']].fillna(0))

    claims['combined_score'] = (claims['claimant_ensemble'] * 0.50
                                + claims['dealer_ensemble'] * 0.50).round(2)

    _hi = claims['combined_score'].quantile(0.90)
    _md = claims['combined_score'].quantile(0.80)
    claims['combined_risk_level'] = np.select(
        [claims['combined_score'] >= _hi, claims['combined_score'] >= _md],
        ['High', 'Medium'], default='Low')

    return claims.sort_values('combined_score', ascending=False)


# ============================================================================
# ORCHESTRATOR
# ============================================================================

def run_multimodel(df, contamination=CONTAMINATION):
    """Return (claimant_multi, dealer_multi, claims_multi)."""
    claimant_features = build_claimant_features(df)
    dealer_features   = build_dealer_features(df)

    claimant_multi, cl_flags = run_all_models(claimant_features, 'ClaimantName', contamination)
    summarize("CLAIMANT", claimant_multi, cl_flags)

    dealer_multi, dl_flags = run_all_models(dealer_features, 'DealerName', contamination)
    summarize("DEALER", dealer_multi, dl_flags)

    claims_multi = combine_to_claims(df, claimant_multi, dealer_multi)

    print("\n[multi] combined claim risk distribution:")
    print(claims_multi['combined_risk_level'].value_counts())

    return claimant_multi, dealer_multi, claims_multi


def run(input_path=INPUT_FILE, contamination=CONTAMINATION, save=True):
    df = pd.read_parquet(input_path)
    claimant_multi, dealer_multi, claims_multi = run_multimodel(df, contamination)

    if save:
        import os
        os.makedirs('data', exist_ok=True)
        claimant_multi.to_parquet(OUT_CLAIMANT, index=False)
        dealer_multi.to_parquet(OUT_DEALER, index=False)
        claims_multi.to_parquet(OUT_CLAIMS, index=False)

        # --- NEW: Excel copies ---
        claimant_multi.to_excel(OUT_CLAIMANT.replace('.parquet', '.xlsx'), index=False)
        dealer_multi.to_excel(OUT_DEALER.replace('.parquet', '.xlsx'), index=False)
        claims_multi.to_excel(OUT_CLAIMS.replace('.parquet', '.xlsx'), index=False)

        print(f"[save] -> {OUT_CLAIMANT}, {OUT_DEALER}, {OUT_CLAIMS} (+ .xlsx)")

        
    return claimant_multi, dealer_multi, claims_multi


if __name__ == '__main__':
    run()
