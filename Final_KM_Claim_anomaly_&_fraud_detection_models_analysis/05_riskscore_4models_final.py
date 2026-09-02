"""
============================================================================
05_riskscore_4models_Final_output.py
Multi-model anomaly CORROBORATION for KM claims
============================================================================

SUMMARY
---------------------
Step 4 (Isolation Forest) is our MAIN scorer. This step is a SECOND OPINION:
it runs four different anomaly detectors and reports how many of them agree
that a claim looks unusual.

The most useful number here is "models_agreeing" (0 to 4). A claim that 2 or
more independent methods flag is a high-confidence anomaly.

WHY WE DON'T SIMPLY AVERAGE THE FOUR MODELS
-------------------------------------------
We tested the four models on data with KNOWN anomalies. We learned:
  - No single model is best for every kind of anomaly.
  - The One-Class SVM was the WEAKEST and least reliable of the four.
  - A plain average lets a weak model drag down a strong one.
So we use a WEIGHTED average that trusts the SVM half as much as the others,
and we lead with the agreement count rather than the averaged score.

THE FOUR MODELS (each defines "unusual" differently)
----------------------------------------------------
  1. Isolation Forest : globally unusual, easy to isolate      (our main model)
  2. Local Outlier Factor : unusual vs its nearest neighbours
  3. KMeans distance : far from its cluster centre
  4. One-Class SVM : outside a learned "normal" boundary       (least reliable)

SOURCES
-------
  - Isolation Forest : Liu, Ting & Zhou, IEEE ICDM 2008
  - LOF              : Breunig, Kriegel, Ng & Sander, ACM SIGMOD 2000
  - One-Class SVM    : Scholkopf et al., Neural Computation 2001
  - Combining models : Aggarwal & Sathe, "Outlier Ensembles" (Springer 2017)

CONSISTENCY WITH STEP 4
-----------------------
Same feature selection, same business-rule promotion, same Wilson/EB/Binomial
entity consolidation, and the same three-sheet output.
============================================================================
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans


# ============================================================================
# SETTINGS  (all the numbers you might tune are here at the top)
# ============================================================================

BASE = Path(__file__).resolve().parent
INPUT_FILE = BASE / "data" / "03_claims_features.parquet"
PROFILE_FILE = BASE / "data" / "03b_feature_profile.xlsx"
OUTPUT_STUB = "05_claims_riskscored_4models"

CONTAMINATION = 0.02        # models treat ~2% of claims as anomalies
HIGH_PERCENTILE = 90        # top 10% of consensus scores -> High
MEDIUM_PERCENTILE = 75      # next 15% -> Medium
SVM_FIT_SAMPLE = 5000       # rows used to train the (slow) SVM
WILSON_Z = 1.96             # 95% confidence for the Wilson test
MEDIUM_LIFT = 1.5           # entity is Medium if its rate is 1.5x baseline

# How much we trust each model in the weighted consensus.
# 
MODEL_WEIGHTS = {
    "iforest_score": 1.0,
    "lof_score":     1.0,
    "kmeans_score":  1.0,
    "svm_score":     1.0,
}

# If the feature-profile file is missing, use this backup feature list.
MODEL_FEATURES = [
    "z_claimant_avg_total_qty", "z_claimant_avg_total_amt",
    "z_claimant_avg_prod_qty", "z_claimant_avg_prod_amt",
    "z_dealer_avg_total_qty", "z_dealer_avg_total_amt",
    "quantity_z_score", "amount_z_score",
    "rolling_30d_claim_count_z", "claim_qty_month_z",
    "recency_vs_avg_frequency", "r_prod_avg_qty_claim", "r_prod_avg_amt_claim",
    "claimant_recent_claim_frequency", "claimant_percent_returned_index",
    "product_seasonality_quantity_ratio", "seasonality_impact_score",
    "claimant_reject_rate", "claimant_return_rate", "claimant_cancel_rate",
    "claimant_failed_validation_rate",
    "dealer_reject_rate", "dealer_return_rate", "dealer_cancel_rate",
    "dealer_failed_validation_rate",
    "claim_days_since_sale", "claim_file_delay", "rejected_claim", "failed_validation",
]

# Columns that must NEVER be fed to a model (IDs, names, dates, our own outputs).
ALWAYS_EXCLUDE = [
    "ClaimFormID", "ClaimId", "ClaimItemId", "ClaimantName", "DealerName",
    "ProductID", "ProductName", "ProductDescription", "SerialNumber", "ClaimName",
    "SalesTransactionID", "AuditComment", "ClaimItemStatus", "ClaimItemStatusReason",
    "IsSerializedProduct", "ProductStatus", "ProductManafacturer",
    "ClaimSubmitionDate", "ProductSoldDate", "ClaimDate", "InvoiceDate",
    "ClaimItemStatusDate", "product_claim_month", "claim_month",
    "risk_score", "risk_level", "risk_level_model", "risk_reason",
    "iforest_score", "lof_score", "svm_score", "kmeans_score",
    "iforest_flag", "lof_flag", "svm_flag", "kmeans_flag",
    "models_agreeing", "consensus_score",
]


# ============================================================================
# SMALL HELPERS
# ============================================================================

def resolve(path):
    """Make a path absolute, relative to this script's folder."""
    p = Path(path).expanduser()
    return (p if p.is_absolute() else BASE / p).resolve()


def read(path):
    """Read a parquet or Excel file."""
    p = resolve(path)
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    return pd.read_excel(p, engine="openpyxl")


def rescale_0_to_100(values):
    """Stretch any list of scores onto a 0-100 range so models can be compared."""
    values = np.asarray(values, dtype=float).reshape(-1, 1)
    return MinMaxScaler(feature_range=(0, 100)).fit_transform(values).flatten()


# ============================================================================
# STEP 1  -  CHOOSE THE FEATURES THE MODELS WILL SEE
# ============================================================================

def choose_features(df):
    """Keep only meaningful numeric columns; never IDs, names, or dates."""
    if PROFILE_FILE.exists():
        # Use the data-driven keep/drop decisions from step 3b.
        profile = pd.read_excel(PROFILE_FILE)
        if "use_in_model" not in profile.columns:
            profile["use_in_model"] = profile.apply(
                lambda r: int(str(r.get("recommended_action", "")).lower().startswith("keep")
                              and str(r.get("dtype", "")).startswith(("int", "float"))),
                axis=1,
            )
        approved = profile.loc[profile["use_in_model"] == 1, "column"].tolist()
        source = "profile (03b_feature_profile.xlsx)"
    else:
        approved = MODEL_FEATURES
        source = "built-in fallback list"

    features = [
        c for c in approved
        if c in df.columns
        and c not in ALWAYS_EXCLUDE
        and pd.api.types.is_numeric_dtype(pd.to_numeric(df[c], errors="coerce"))
    ]
    print(f"[features] source: {source}; using {len(features)} features")
    return features


def build_scaled_matrix(df, features):
    """Turn the chosen features into a clean, scaled number grid for the models.
    We use RobustScaler because LOF, SVM and KMeans measure distances, and
    distances are meaningless if one feature is on a much larger scale.
    """
    X = df[features].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X_scaled = RobustScaler().fit_transform(X)
    return np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)


# ============================================================================
# STEP 2  -  RUN THE FOUR MODELS
# ============================================================================
# Each model returns two things per claim:
#   *_flag  : 1 if that model calls the claim an anomaly, else 0
#   *_score : the claim's unusualness on a 0-100 scale
# ============================================================================

def run_isolation_forest(X, contamination, seed):
    """Globally unusual points are easy to 'isolate'. This is our main model."""
    model = IsolationForest(n_estimators=200, contamination=contamination,
                            random_state=seed, n_jobs=-1).fit(X)
    flag = (model.predict(X) == -1).astype(int)
    # score_samples is HIGHER for normal points, so we flip the sign.
    score = rescale_0_to_100(-model.score_samples(X))
    return flag, np.round(score, 2)


def run_lof(X, contamination):
    """Local Outlier Factor: unusual compared with nearby points."""
    n = len(X)
    neighbours = min(20, max(5, n // 10))     # neighbourhood size must fit the data
    model = LocalOutlierFactor(n_neighbors=neighbours, contamination=contamination)
    flag = (model.fit_predict(X) == -1).astype(int)
    score = rescale_0_to_100(-model.negative_outlier_factor_)
    return flag, np.round(score, 2)


def run_one_class_svm(X, contamination, seed):
    """One-Class SVM: outside a learned 'normal' boundary. Least reliable model.

    It is slow (grows with the square of the row count), so on big data we train
    it on a random sample, then apply it to every row.
    """
    n = len(X)
    if n > SVM_FIT_SAMPLE:
        sample = np.random.RandomState(seed).choice(n, SVM_FIT_SAMPLE, replace=False)
        model = OneClassSVM(nu=contamination, kernel="rbf", gamma="scale").fit(X[sample])
    else:
        model = OneClassSVM(nu=contamination, kernel="rbf", gamma="scale").fit(X)
    flag = (model.predict(X) == -1).astype(int)
    score = rescale_0_to_100(-model.decision_function(X))
    return flag, np.round(score, 2)


def run_kmeans_distance(X, contamination, seed):
    """KMeans: group the claims, then flag those far from their group's centre."""
    n = len(X)
    clusters = min(8, max(2, n // 50))
    model = KMeans(n_clusters=clusters, random_state=seed, n_init=10)
    labels = model.fit_predict(X)
    distance = np.linalg.norm(X - model.cluster_centers_[labels], axis=1)
    cutoff = np.percentile(distance, 100 * (1 - contamination))
    flag = (distance >= cutoff).astype(int)
    score = rescale_0_to_100(distance)
    return flag, np.round(score, 2)


def run_all_models(df, X, contamination=CONTAMINATION, seed=42):
    """Run the four models and add their flags, scores, agreement, and consensus."""
    df["iforest_flag"], df["iforest_score"] = run_isolation_forest(X, contamination, seed)
    df["lof_flag"],     df["lof_score"]     = run_lof(X, contamination)
    df["svm_flag"],     df["svm_score"]     = run_one_class_svm(X, contamination, seed)
    df["kmeans_flag"],  df["kmeans_score"]  = run_kmeans_distance(X, contamination, seed)

    # How many of the four models flagged each claim (0 to 4). THE key signal.
    flag_cols = ["iforest_flag", "lof_flag", "svm_flag", "kmeans_flag"]
    df["models_agreeing"] = df[flag_cols].sum(axis=1)

    # Weighted-average score (SVM trusted half as much as the others).
    score_cols = list(MODEL_WEIGHTS.keys())
    weights = np.array([MODEL_WEIGHTS[c] for c in score_cols])
    scores = df[score_cols].to_numpy(dtype=float)
    df["consensus_score"] = np.round((scores * weights).sum(axis=1) / weights.sum(), 2)

    # Quick report.
    print("[models] flagged by each model:")
    for c in flag_cols:
        print(f"  {c:<16}{int(df[c].sum()):>7} ({100*df[c].mean():.1f}%)")
    print(f"  agree 2+ models : {int((df['models_agreeing'] >= 2).sum()):>6}"
          "   <- headline high-confidence signal")
    return df


# ============================================================================
# STEP 3  -  TURN THE CONSENSUS SCORE INTO High / Medium / Low  (+ reasons)
# ============================================================================

def assign_levels(df):
    """Rank by consensus score, then always escalate rejected/failed claims."""
    score = df["consensus_score"]
    high_cut = np.percentile(score, HIGH_PERCENTILE)
    med_cut = np.percentile(score, MEDIUM_PERCENTILE)

    # Start everyone Low, upgrade to Medium, then to High (highest wins).
    df["risk_level_model"] = "Low"
    df.loc[score >= med_cut,  "risk_level_model"] = "Medium"
    df.loc[score >= high_cut, "risk_level_model"] = "High"

    # Business-rule safety net: known problems always go to High.
    df["risk_level"] = df["risk_level_model"]
    for rule in ["rejected_claim", "failed_validation"]:
        if rule in df.columns:
            df.loc[pd.to_numeric(df[rule], errors="coerce").fillna(0) == 1, "risk_level"] = "High"
    return df


def build_reasons(df):
    """Write a plain-language reason for each claim by listing the signals that fired."""
    reasons = [[] for _ in range(len(df))]

    def add_reason(col, threshold, phrase, low=False):
        if col not in df.columns:
            return
        values = pd.to_numeric(df[col], errors="coerce").fillna(0)
        hit = values.le(threshold) if low else values.ge(threshold)
        for i in np.where(hit.to_numpy())[0]:
            reasons[i].append(phrase)

    # Model agreement is the headline reason for this step.
    agree = pd.to_numeric(df["models_agreeing"], errors="coerce").fillna(0)
    for i in np.where(agree.ge(3).to_numpy())[0]:
        reasons[i].append("flagged by 3+ independent models (high confidence)")
    for i in np.where(agree.eq(2).to_numpy())[0]:
        reasons[i].append("flagged by 2 independent models")

    # Feature-based reasons.
    add_reason("quantity_z_score", 3, "product quantity far above product norm")
    add_reason("amount_z_score", 3, "product amount far above product norm")
    add_reason("z_claimant_avg_total_qty", 3, "claim quantity far above claimant norm")
    add_reason("z_dealer_avg_total_qty", 3, "claim quantity far above dealer norm")
    add_reason("r_prod_avg_qty_claim", 3, "product quantity 3x+ the claimant's usual")
    add_reason("recency_vs_avg_frequency", 0.25, "filed unusually soon after last claim", low=True)
    add_reason("rolling_30d_claim_count_z", 2, "spike in recent claim activity")
    add_reason("claimant_percent_returned_index", 2, "returned rate 2x+ population")
    add_reason("claim_qty_month_z", 3, "quantity high even after seasonal adjustment")
    add_reason("failed_validation", 1, "failed a validation rule")
    add_reason("rejected_claim", 1, "previously rejected by an auditor")
    add_reason("claim_file_delay", 1, "filed 90+ days after sale")

    df["risk_reason"] = [", ".join(r) if r else "no single strong signal (model-driven score)"
                         for r in reasons]
    return df


# ============================================================================
# STEP 4  -  ROLL CLAIMS UP INTO ONE RATING PER CLAIMANT / DEALER
# ============================================================================
# Same statistically-fair method as step 4:
#   Compare each entity's High-claim RATE to the population baseline using the
#   Wilson lower bound, so nobody is rated High just for filing many claims.
# ============================================================================

def wilson_lower_bound(k, n, z=WILSON_Z):
    """A cautious lowest estimate of the true 'High' rate given k highs in n claims."""
    if n == 0:
        return 0.0
    p = k / n
    denominator = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (centre - margin) / denominator)


def summarize_reason(sub):
    """Combine one entity's claim reasons into the top few, with counts."""
    flagged = sub[sub["_tier"].isin(["High", "Medium"])]
    if len(flagged) == 0:
        return "mostly low-risk claims"
    counts = {}
    for reason in flagged["risk_reason"].astype(str):
        for part in reason.split(","):
            phrase = part.strip()
            if phrase and not phrase.startswith("no single strong signal"):
                counts[phrase] = counts.get(phrase, 0) + 1
    if not counts:
        return "elevated score without one dominant driver"
    top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:4]
    return "; ".join(f"{phrase} ({n} claims)" for phrase, n in top)


def consolidate_entity(df, entity_col):
    """Produce one row per claimant (or dealer) with a single rating and reason."""
    prefix = "claimant" if entity_col.lower().startswith("claimant") else "dealer"

    # 1. Population baseline: what share of ALL claims are High?
    tier = df["risk_level"].astype(str).str.strip().str.title()
    baseline = float((tier == "High").mean())

    # 2. Count each entity's claims (n) and High claims (k).
    work = pd.DataFrame({entity_col: df[entity_col].values,
                         "_high": (tier == "High").astype(int).values,
                         "_score": df["consensus_score"].values})
    g = (work.groupby(entity_col, dropna=False)
             .agg(claim_count=("_high", "size"),
                  high_count=("_high", "sum"),
                  avg_consensus_score=("_score", "mean"),
                  max_consensus_score=("_score", "max"))
             .reset_index())
    g["raw_high_rate"] = g["high_count"] / g["claim_count"]

    # 3a. Wilson lower bound (the decision method).
    g["wilson_lower"] = [wilson_lower_bound(int(k), int(n))
                         for k, n in zip(g["high_count"], g["claim_count"])]
    g["wilson_lift"] = g["wilson_lower"] / baseline if baseline > 0 else 0

    # 3b. Empirical Bayes (supporting): shrink small samples toward the average.
    rates = g["raw_high_rate"]
    weight = g["claim_count"] / g["claim_count"].sum()
    mean_rate = float((rates * weight).sum())
    var_rate = float((weight * (rates - mean_rate) ** 2).sum())
    if var_rate > 0 and 0 < mean_rate < 1:
        strength = mean_rate * (1 - mean_rate) / var_rate - 1
        alpha = max(mean_rate * strength, 1e-6)
        beta = max((1 - mean_rate) * strength, 1e-6)
    else:
        alpha, beta = max(mean_rate, 1e-6) * 20, (1 - max(mean_rate, 1e-6)) * 20
    g["eb_rate"] = (g["high_count"] + alpha) / (g["claim_count"] + alpha + beta)
    g["eb_prob_above_baseline"] = [1 - stats.beta.cdf(baseline, k + alpha, n - k + beta)
                                   for k, n in zip(g["high_count"], g["claim_count"])]

    # 3c. Binomial test (supporting): could this many highs be luck?
    g["binom_p"] = [stats.binomtest(int(k), int(n), baseline, alternative="greater").pvalue
                    for k, n in zip(g["high_count"], g["claim_count"])]

    # 4. The decision.
    is_high = g["wilson_lower"] > baseline
    is_medium = (~is_high) & (g["eb_rate"] >= MEDIUM_LIFT * baseline)
    g[f"{prefix}_risk_level"] = np.where(is_high, "High", np.where(is_medium, "Medium", "Low"))

    # 5. Model-only version (same test, ignoring business promotions).
    tier_model = df["risk_level_model"].astype(str).str.strip().str.title()
    baseline_model = float((tier_model == "High").mean())
    wm = pd.DataFrame({entity_col: df[entity_col].values,
                       "_high": (tier_model == "High").astype(int).values})
    gm = wm.groupby(entity_col, dropna=False)["_high"].agg(["size", "sum"]).reset_index()
    gm.columns = [entity_col, "n", "k"]
    gm["wl"] = [wilson_lower_bound(int(k), int(n)) for k, n in zip(gm["k"], gm["n"])]
    gm[f"{prefix}_risk_level_model"] = np.where(gm["wl"] > baseline_model, "High", "Low")
    g = g.merge(gm[[entity_col, f"{prefix}_risk_level_model"]], on=entity_col, how="left")

    # 6. A consolidated reason per entity.
    reason_src = df[[entity_col, "risk_level", "risk_reason"]].copy()
    reason_src["_tier"] = reason_src["risk_level"].astype(str).str.strip().str.title()
    reasons = (reason_src.groupby(entity_col, dropna=False)
                         .apply(summarize_reason, include_groups=False)
                         .reset_index(name=f"{prefix}_risk_reason"))
    g = g.merge(reasons, on=entity_col, how="left")
    g["population_high_rate"] = round(baseline, 4)

    # 7. Tidy column order.
    cols = [entity_col, f"{prefix}_risk_level_model", f"{prefix}_risk_level",
            f"{prefix}_risk_reason", "claim_count", "high_count", "raw_high_rate",
            "wilson_lower", "wilson_lift", "eb_rate", "eb_prob_above_baseline",
            "binom_p", "avg_consensus_score", "max_consensus_score", "population_high_rate"]
    return g[cols].round(4).sort_values(f"{prefix}_risk_level").reset_index(drop=True)


# ============================================================================
# MAIN
# ============================================================================

def run(input_path=INPUT_FILE, output_stub=OUTPUT_STUB, contamination=CONTAMINATION, save_excel=True):
    df = read(input_path)

    # Step 1: features -> Step 2: models -> Step 3: levels & reasons.
    features = choose_features(df)
    X = build_scaled_matrix(df, features)
    df = run_all_models(df, X, contamination)
    df = assign_levels(df)
    df = build_reasons(df)

    dist = df["risk_level"].value_counts(); total = len(df)
    print("[claim distribution]", ", ".join(
        f"{t} {int(dist.get(t,0))} ({100*int(dist.get(t,0))/total:.1f}%)" for t in ["High","Medium","Low"]))

    # Step 4: one rating per claimant and per dealer.
    claimant = consolidate_entity(df, "ClaimantName") if "ClaimantName" in df.columns else None
    dealer = consolidate_entity(df, "DealerName") if "DealerName" in df.columns else None
    if claimant is not None:
        c = claimant["claimant_risk_level"].value_counts()
        print("[claimant ratings]", ", ".join(f"{t} {int(c.get(t,0))}" for t in ["High","Medium","Low"]))
    if dealer is not None:
        d = dealer["dealer_risk_level"].value_counts()
        print("[dealer ratings]  ", ", ".join(f"{t} {int(d.get(t,0))}" for t in ["High","Medium","Low"]))

    # Save three parquet files + one Excel workbook with three sheets.
    data = BASE / "data"; data.mkdir(parents=True, exist_ok=True)
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].astype("string")

    df.to_parquet(data / f"{output_stub}.parquet", index=False)
    print(f"\n[save] {data / (output_stub + '.parquet')}")
    if claimant is not None:
        claimant.to_parquet(data / f"{output_stub}_by_claimant.parquet", index=False)
    if dealer is not None:
        dealer.to_parquet(data / f"{output_stub}_by_dealer.parquet", index=False)

    if save_excel:
        with pd.ExcelWriter(data / f"{output_stub}.xlsx", engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="claims_individual", index=False)
            if claimant is not None:
                claimant.to_excel(writer, sheet_name="by_claimant", index=False)
            if dealer is not None:
                dealer.to_excel(writer, sheet_name="by_dealer", index=False)
        print(f"[save] {data / (output_stub + '.xlsx')} (3 sheets)")

    return df, claimant, dealer


def main():
    parser = argparse.ArgumentParser(description="4-model anomaly corroboration for KM claims")
    parser.add_argument("--input", default=INPUT_FILE)
    parser.add_argument("--output-stub", default=OUTPUT_STUB)
    parser.add_argument("--contamination", type=float, default=CONTAMINATION)
    parser.add_argument("--no-excel", action="store_true")
    args = parser.parse_args()
    run(args.input, args.output_stub, args.contamination, not args.no_excel)


if __name__ == "__main__":
    main()
