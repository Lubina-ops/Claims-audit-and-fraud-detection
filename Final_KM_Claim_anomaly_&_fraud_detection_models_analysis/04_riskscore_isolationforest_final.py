"""
04_riskscore_isolationforest.py
==============================================================================
Isolation Forest scoring, ranking, explanations and entity ratings for KM claims.

WHAT THIS STEP PRODUCES
-----------------------
For EVERY claim (claim-item grain), in ONE output row:
  * ClaimantName FIRST, then the key identifiers
  * every remaining source column and every engineered feature from Step 3
  * risk_score          0-100  how unusual the claim is
  * risk_score_1000     1-1000 the same risk on a finer scale (stakeholder request)
  * risk_percentile     0-100  99.4 = riskier than 99.4% of all claims
  * risk_rank           1      = the single riskiest claim in the file
  * audit_band          Top 1% / Top 5% / Top 10% / Top 20% / Top 25% / Bottom 75%
  * in_top_1pct ... in_top_25pct   instant review-queue filters
  * risk_level_model    the model's own High / Medium / Low
  * risk_level          the FINAL High / Medium / Low after business rules
  * risk_reason         short reason text (kept for backward compatibility)
  * Why this rating     a full plain-English sentence
  * Driver 1/2/3 (field = value)  the FIELD NAME and value behind the reason,
                        so any explanation can be verified on the same row
  * features_used_for_rating      the exact field names that fired for this claim

PLUS one consolidated, statistically fair rating per claimant and per dealer.

COLUMNS DELIBERATELY OMITTED (redundant or duplicated in the source extract):
  ClaimItemId, ProductName, ClaimName, ClaimantName.1, ProductName.1,
  ItemDescription, ProductManafacturer, AuditComment

SerialNumber is retained but moved towards the end of the sheet.

KEY DESIGN POINTS
-----------------
* Only meaningful numeric signals reach the model. Names, IDs, dates and outcome
  flags never do (outcome flags drive business rules only - this avoids leakage).
* Business rules override the model: a rejected or failed-validation claim is
  always High, whatever its score.
* Entity ratings use the Wilson score, so nobody is rated High simply for filing
  a lot of claims.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from openpyxl.styles import Font, PatternFill, Alignment


# ============================================================================
# SETTINGS
# ============================================================================

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
INPUT_FILE = DATA / "03_claims_features.parquet"
PROFILE_FILE = DATA / "03b_feature_profile.xlsx"
OUTPUT_STUB = "04_claims_riskscored"

CONTAMINATION = 0.02        # model treats ~2% of claims as anomalies
HIGH_PERCENTILE = 90        # top 10% of scores -> High
MEDIUM_PERCENTILE = 75      # next 15% -> Medium
WILSON_Z = 1.96             # 95% confidence for the entity rating
MEDIUM_LIFT = 1.5           # entity is Medium if its rate is 1.5x baseline
TOP_BANDS = [1, 5, 10, 20, 25]      # review-queue sizes a user can filter to
SCORE_1000_MIN = 1                  # stakeholder asked for a 1-1000 scale
SCORE_1000_MAX = 1000

# Redundant / duplicated source columns to leave out of the output entirely.
DROP_COLUMNS = [
    "ClaimItemId", "ProductName", "ClaimName", "ClaimantName.1",
    "ProductName.1", "ItemDescription", "ProductManafacturer", "AuditComment",
]

# Fallback feature list, used only if the 03b profile file is missing.
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
    "claim_days_since_sale", "claim_file_delay",
]

# Never model inputs: identifiers, names, dates, outcome flags and our outputs.
ALWAYS_EXCLUDE = [
    "ClaimFormID", "ClaimId", "ClaimItemId", "ClaimantName", "DealerName",
    "ProductID", "ProductName", "ProductDescription", "SerialNumber", "ClaimName",
    "SalesTransactionID", "AuditComment", "ClaimItemStatus", "ClaimItemStatusReason",
    "IsSerializedProduct", "ProductStatus", "ProductManafacturer",
    "ClaimSubmitionDate", "ProductSoldDate", "ClaimDate", "InvoiceDate",
    "ClaimItemStatusDate", "product_claim_month", "claim_month",
    "risk_score", "risk_level", "risk_level_model", "risk_reason",
    "risk_percentile", "risk_score_1000", "risk_rank", "audit_band",
    # outcome flags: business rules only, never model inputs (avoids leakage)
    "rejected_claim", "returned_claim", "cancelled_claim", "failed_validation",
    "processed_claim", "problem_claim_returned_or_rejected",
]

# Explanation rules: field, the test that fires it, and how to say it plainly.
RULES = [
    ("failed_validation",                 lambda v: v == 1,   lambda v: "failed a validation check (rejected / duplicate / no match)"),
    ("rejected_claim",                    lambda v: v == 1,   lambda v: "was previously rejected by an auditor"),
    ("r_prod_avg_qty_claim",              lambda v: v >= 2,   lambda v: f"quantity was {v:.1f}x this claimant's usual for the product"),
    ("quantity_z_score",                  lambda v: v >= 3,   lambda v: f"quantity far above the product norm ({v:.1f} std above)"),
    ("z_claimant_avg_total_qty",          lambda v: v >= 3,   lambda v: f"quantity far above this claimant's own norm ({v:.1f} std above)"),
    ("z_dealer_avg_total_qty",            lambda v: v >= 3,   lambda v: f"quantity far above this dealer's norm ({v:.1f} std above)"),
    ("amount_z_score",                    lambda v: v >= 3,   lambda v: f"dollar amount far above the product norm ({v:.1f} std above)"),
    ("r_prod_avg_amt_claim",              lambda v: v >= 2,   lambda v: f"amount was {v:.1f}x this claimant's usual"),
    ("rolling_30d_claim_count_z",         lambda v: v >= 2,   lambda v: f"unusual burst of claims in the last 30 days ({v:.1f} std above normal)"),
    ("claimant_recent_claim_frequency",   lambda v: v >= 2,   lambda v: f"filing {v:.1f}x more often than usual recently"),
    ("recency_vs_avg_frequency",          lambda v: v <= 0.5, lambda v: f"filed much sooner after the last claim than usual ({v:.2f}x their normal gap)"),
    ("claim_qty_month_z",                 lambda v: v >= 3,   lambda v: f"quantity still unusual after allowing for seasonality ({v:.1f} std above)"),
    ("claimant_percent_returned_index",   lambda v: v >= 2,   lambda v: f"claimant returned rate {v:.1f}x the population average"),
    ("claim_file_delay",                  lambda v: v == 1,   lambda v: "filed 90+ days after the sale"),
    ("insufficient_claimant_history_flag", lambda v: v == 1,  lambda v: "claimant has very little history, so estimates are less certain"),
]

# What "normal" looks like - used to explain why a Low claim is Low.
NORMAL_CHECKS = [
    ("quantity_z_score",          lambda v: abs(v) < 1.5,    "quantity in line with the product norm"),
    ("r_prod_avg_qty_claim",      lambda v: 0.5 <= v <= 1.5, "quantity about the same as the claimant's usual"),
    ("rolling_30d_claim_count_z", lambda v: abs(v) < 1.5,    "normal recent filing activity"),
    ("claim_file_delay",          lambda v: v == 0,          "filed within the normal window"),
]

# Column order: ClaimantName first, then identifiers, then the risk outputs.
FRONT_COLUMNS = [
    "ClaimantName",
    "ClaimFormID", "ClaimId", "DealerName", "ProductID", "ClaimItemStatus",
    "risk_level", "risk_level_model",
    "risk_score", "risk_score_1000", "risk_percentile", "risk_rank", "audit_band",
    "Why this rating",
    "Driver 1 (field = value)", "Driver 2 (field = value)", "Driver 3 (field = value)",
    "features_used_for_rating", "risk_reason",
]

# Columns pushed towards the end of the sheet.
BACK_COLUMNS = ["SerialNumber"]


# ============================================================================
# HELPERS
# ============================================================================

def resolve(path):
    p = Path(path).expanduser()
    return (p if p.is_absolute() else BASE / p).resolve()


def read(path):
    p = resolve(path)
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    return pd.read_excel(p, engine="openpyxl")


def num(row, col):
    """Read one numeric value from a row, or None if missing/non-numeric."""
    if col not in row.index:
        return None
    v = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
    return None if pd.isna(v) else float(v)


# ============================================================================
# STEP 1 - CHOOSE THE FEATURES THE MODEL MAY SEE
# ============================================================================

def choose_features(df):
    """Use the data-driven keep/drop list from step 3b; fall back to a safe list."""
    if PROFILE_FILE.exists():
        profile = pd.read_excel(PROFILE_FILE)
        if "use_in_model" not in profile.columns:
            profile["use_in_model"] = profile.apply(
                lambda r: int(str(r.get("recommended_action", "")).lower().startswith("keep")
                              and str(r.get("dtype", "")).startswith(("int", "float"))), axis=1)
        approved = profile.loc[profile["use_in_model"] == 1, "column"].tolist()
        source = "step 3b profile"
    else:
        approved = MODEL_FEATURES
        source = "built-in fallback list"

    features = [c for c in approved
                if c in df.columns and c not in ALWAYS_EXCLUDE
                and pd.api.types.is_numeric_dtype(pd.to_numeric(df[c], errors="coerce"))]
    print(f"[features] source: {source}; using {len(features)} features for the model")
    return features


# ============================================================================
# STEP 2 - SCORE EVERY CLAIM (0-100)
# ============================================================================

def score_claims(df, features, contamination=CONTAMINATION, seed=42):
    """Isolation Forest: claims that are quick to 'isolate' are unusual."""
    X = df[features].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    model = IsolationForest(n_estimators=200, contamination=contamination,
                            random_state=seed, n_jobs=-1).fit(X)
    raw = -model.score_samples(X)              # flip: higher = more unusual
    lo, hi = raw.min(), raw.max()
    score = 100 * (raw - lo) / (hi - lo) if hi > lo else np.zeros_like(raw)
    return np.round(score, 2)


# ============================================================================
# STEP 3 - TURN THE SCORE INTO A RATING (+ business-rule safety net)
# ============================================================================

def assign_levels(df):
    score = df["risk_score"]
    high_cut = np.percentile(score, HIGH_PERCENTILE)
    med_cut = np.percentile(score, MEDIUM_PERCENTILE)

    df["risk_level_model"] = "Low"
    df.loc[score >= med_cut,  "risk_level_model"] = "Medium"
    df.loc[score >= high_cut, "risk_level_model"] = "High"

    # Known problems always escalate, whatever the score.
    df["risk_level"] = df["risk_level_model"]
    for rule in ["rejected_claim", "failed_validation"]:
        if rule in df.columns:
            df.loc[pd.to_numeric(df[rule], errors="coerce").fillna(0) == 1, "risk_level"] = "High"

    print(f"[thresholds] Medium score >= {med_cut:.2f}; High score >= {high_cut:.2f}")
    return df


# ============================================================================
# STEP 4 - RANKING (percentile, 1-1000 score, rank, audit band)
# ============================================================================

def add_ranking(df):
    """Give reviewers a flexible way to size their own audit queue."""
    score = pd.to_numeric(df["risk_score"], errors="coerce")

    # How this claim ranks against ALL claims. 99.4 = riskier than 99.4% of them.
    df["risk_percentile"] = (score.rank(pct=True, method="average") * 100).round(2)

    # The same risk on a 1-1000 scale (stakeholder request).
    lo, hi = score.min(), score.max()
    if hi > lo:
        scaled = SCORE_1000_MIN + (score - lo) / (hi - lo) * (SCORE_1000_MAX - SCORE_1000_MIN)
    else:
        scaled = pd.Series(SCORE_1000_MIN, index=df.index)
    df["risk_score_1000"] = scaled.round(0).astype("Int64")

    # 1 = the single riskiest claim in the file.
    df["risk_rank"] = score.rank(ascending=False, method="min").astype("Int64")

    # Ready-made filters for an instant review queue.
    for b in TOP_BANDS:
        df[f"in_top_{b}pct"] = (df["risk_percentile"] >= (100 - b)).astype("int8")

    def band(p):
        for b in TOP_BANDS:
            if p >= 100 - b:
                return f"Top {b}%"
        return f"Bottom {100 - TOP_BANDS[-1]}%"
    df["audit_band"] = df["risk_percentile"].apply(band)
    return df


# ============================================================================
# STEP 5 - EXPLAIN EVERY RATING (field names included, so it is verifiable)
# ============================================================================

def explain_row(row):
    """Return (why, driver1, driver2, driver3, fields_used)."""
    fired = []
    for col, test, phrase in RULES:
        v = num(row, col)
        if v is not None and test(v):
            fired.append((col, f"{col} = {v:g} -> {phrase(v)}", phrase(v)))

    level = str(row.get("risk_level", "")).title()

    if fired:
        fields = [f[0] for f in fired]
        drivers = [f[1] for f in fired[:3]]
        words = [f[2] for f in fired[:3]]
        lead = ("Rated HIGH because " if level == "High"
                else "Rated MEDIUM because " if level == "Medium"
                else "Rated LOW; minor signals noted: ")
        why = lead + "; ".join(words) + "."
    else:
        normals = []
        for col, test, phrase in NORMAL_CHECKS:
            v = num(row, col)
            if v is not None and test(v):
                normals.append((col, f"{col} = {v:g} -> {phrase}", phrase))
        fields = [n[0] for n in normals]
        drivers = [n[1] for n in normals[:3]]
        words = [n[2] for n in normals[:3]]
        if level == "Low":
            why = ("Rated LOW because nothing stood out: "
                   + ("; ".join(words) if words else "all signals near normal") + ".")
        else:
            why = (f"Rated {level} on the model's overall pattern; "
                   "no single signal crossed a review threshold.")

    drivers = (drivers + ["", "", ""])[:3]
    return why, drivers[0], drivers[1], drivers[2], ", ".join(fields)


def add_explanations(df):
    exp = df.apply(explain_row, axis=1, result_type="expand")
    exp.columns = ["Why this rating", "Driver 1 (field = value)",
                   "Driver 2 (field = value)", "Driver 3 (field = value)",
                   "features_used_for_rating"]
    out = pd.concat([df, exp], axis=1)

    # Short reason text, kept so earlier reports still work.
    out["risk_reason"] = out["Why this rating"].str.split("because", n=1).str[-1] \
                                               .str.split("noted:", n=1).str[-1] \
                                               .str.strip().str.rstrip(".")
    return out


# ============================================================================
# STEP 6 - ONE FAIR RATING PER CLAIMANT / DEALER (Wilson score)
# ============================================================================

def wilson_lower_bound(k, n, z=WILSON_Z):
    """Cautious lowest estimate of the true 'High' rate given k highs in n claims."""
    if n == 0:
        return 0.0
    p = k / n
    denominator = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (centre - margin) / denominator)


def summarize_entity_reason(sub):
    """Most common drivers across the entity's High and Medium claims."""
    flagged = sub[sub["_tier"].isin(["High", "Medium"])]
    if len(flagged) == 0:
        return "mostly low-risk claims"
    counts = {}
    for reason in flagged["risk_reason"].astype(str):
        for part in reason.split(";"):
            phrase = part.strip().rstrip(".")
            if phrase and not phrase.startswith("nothing stood out"):
                counts[phrase] = counts.get(phrase, 0) + 1
    if not counts:
        return "elevated score without one dominant driver"
    top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:4]
    return "; ".join(f"{p} ({n} claims)" for p, n in top)


def consolidate_entity(df, entity_col):
    """One row per entity: rating decided by rate-vs-baseline, not raw counts."""
    prefix = "claimant" if entity_col.lower().startswith("claimant") else "dealer"

    tier = df["risk_level"].astype(str).str.strip().str.title()
    baseline = float((tier == "High").mean())

    work = pd.DataFrame({entity_col: df[entity_col].values,
                         "_high": (tier == "High").astype(int).values,
                         "_score": df["risk_score"].values,
                         "_s1000": pd.to_numeric(df["risk_score_1000"], errors="coerce").values})
    g = (work.groupby(entity_col, dropna=False)
             .agg(claim_count=("_high", "size"), high_count=("_high", "sum"),
                  avg_risk_score=("_score", "mean"), max_risk_score=("_score", "max"),
                  avg_risk_score_1000=("_s1000", "mean"), max_risk_score_1000=("_s1000", "max"))
             .reset_index())
    g["raw_high_rate"] = g["high_count"] / g["claim_count"]

    # The deciding number.
    g["wilson_lower"] = [wilson_lower_bound(int(k), int(n))
                         for k, n in zip(g["high_count"], g["claim_count"])]
    g["wilson_lift"] = g["wilson_lower"] / baseline if baseline > 0 else 0

    # Supporting cross-checks.
    rates = g["raw_high_rate"]; weight = g["claim_count"] / g["claim_count"].sum()
    mean_rate = float((rates * weight).sum())
    var_rate = float((weight * (rates - mean_rate) ** 2).sum())
    if var_rate > 0 and 0 < mean_rate < 1:
        strength = mean_rate * (1 - mean_rate) / var_rate - 1
        alpha, beta = max(mean_rate * strength, 1e-6), max((1 - mean_rate) * strength, 1e-6)
    else:
        alpha, beta = max(mean_rate, 1e-6) * 20, (1 - max(mean_rate, 1e-6)) * 20
    g["eb_rate"] = (g["high_count"] + alpha) / (g["claim_count"] + alpha + beta)
    g["eb_prob_above_baseline"] = [1 - stats.beta.cdf(baseline, k + alpha, n - k + beta)
                                   for k, n in zip(g["high_count"], g["claim_count"])]
    g["binom_p"] = [stats.binomtest(int(k), int(n), baseline, alternative="greater").pvalue
                    for k, n in zip(g["high_count"], g["claim_count"])]

    # The decision.
    is_high = g["wilson_lower"] > baseline
    is_medium = (~is_high) & (g["eb_rate"] >= MEDIUM_LIFT * baseline)
    g[f"{prefix}_risk_level"] = np.where(is_high, "High", np.where(is_medium, "Medium", "Low"))

    # Model-only view (same test, ignoring business promotions).
    tier_model = df["risk_level_model"].astype(str).str.strip().str.title()
    baseline_model = float((tier_model == "High").mean())
    wm = pd.DataFrame({entity_col: df[entity_col].values,
                       "_high": (tier_model == "High").astype(int).values})
    gm = wm.groupby(entity_col, dropna=False)["_high"].agg(["size", "sum"]).reset_index()
    gm.columns = [entity_col, "n", "k"]
    gm["wl"] = [wilson_lower_bound(int(k), int(n)) for k, n in zip(gm["k"], gm["n"])]
    gm[f"{prefix}_risk_level_model"] = np.where(gm["wl"] > baseline_model, "High", "Low")
    g = g.merge(gm[[entity_col, f"{prefix}_risk_level_model"]], on=entity_col, how="left")

    # A readable reason per entity.
    src = df[[entity_col, "risk_level", "risk_reason"]].copy()
    src["_tier"] = src["risk_level"].astype(str).str.strip().str.title()
    reasons = (src.groupby(entity_col, dropna=False)
                  .apply(summarize_entity_reason, include_groups=False)
                  .reset_index(name=f"{prefix}_risk_reason"))
    g = g.merge(reasons, on=entity_col, how="left")
    g["population_high_rate"] = round(baseline, 4)
    g["model_population_high_rate"] = round(baseline_model, 4)
    g["counting_unit"] = "distinct claim rows"

    cols = [entity_col, f"{prefix}_risk_level_model", f"{prefix}_risk_level",
            f"{prefix}_risk_reason", "claim_count", "high_count", "raw_high_rate",
            "wilson_lower", "wilson_lift", "eb_rate", "eb_prob_above_baseline",
            "binom_p", "avg_risk_score", "max_risk_score",
            "avg_risk_score_1000", "max_risk_score_1000",
            "population_high_rate", "model_population_high_rate", "counting_unit"]
    return g[cols].round(4).sort_values(f"{prefix}_risk_level").reset_index(drop=True)


# ============================================================================
# STEP 7 - SAVE
# ============================================================================

def order_columns(df):
    """ClaimantName first, then identifiers and risk outputs, then features.
    SerialNumber is pushed towards the end."""
    front = [c for c in FRONT_COLUMNS if c in df.columns]
    back = [c for c in BACK_COLUMNS if c in df.columns]
    middle = [c for c in df.columns if c not in front and c not in back]
    return df[front + middle + back]


def style_sheet(ws, wide=("Why this rating",), driver_prefix="Driver"):
    hdr = Font(bold=True, color="FFFFFF", name="Arial", size=10)
    fill = PatternFill("solid", fgColor="1F4E78")
    colours = {"High": "F4CCCC", "Medium": "FFF2CC", "Low": "D9EAD3"}
    for c in ws[1]:
        c.font = hdr; c.fill = fill
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    headers = [c.value for c in ws[1]]
    lvl_cols = [i + 1 for i, h in enumerate(headers)
                if h in ("risk_level", "claimant_risk_level", "dealer_risk_level")]
    for r in range(2, ws.max_row + 1):
        for i in lvl_cols:
            v = ws.cell(r, i).value
            if v in colours:
                ws.cell(r, i).fill = PatternFill("solid", fgColor=colours[v])
                ws.cell(r, i).alignment = Alignment(horizontal="center")
        for i, h in enumerate(headers, 1):
            if h in wide or str(h).startswith(driver_prefix) or h == "features_used_for_rating":
                ws.cell(r, i).alignment = Alignment(wrap_text=True, vertical="top")
    for i, h in enumerate(headers, 1):
        letter = ws.cell(1, i).column_letter
        ws.column_dimensions[letter].width = (
            58 if h in wide else 44 if str(h).startswith(driver_prefix)
            else 34 if h == "features_used_for_rating"
            else 22 if h == "ClaimantName" else 16)
    ws.freeze_panes = "B2"


def save_outputs(df, claimant, dealer, stub):
    DATA.mkdir(parents=True, exist_ok=True)

    out = order_columns(df.copy())
    for c in out.columns:
        if out[c].dtype == "object":
            out[c] = out[c].astype("string")

    out.to_parquet(DATA / f"{stub}.parquet", index=False)
    if claimant is not None:
        claimant.to_parquet(DATA / f"{stub}_by_claimant.parquet", index=False)
    if dealer is not None:
        dealer.to_parquet(DATA / f"{stub}_by_dealer.parquet", index=False)

    order = {"High": 0, "Medium": 1, "Low": 2}
    if "ClaimantName" in out.columns:
        claimant_view = out.copy()
        claimant_view["_o"] = claimant_view["risk_level"].map(order).fillna(9)
        claimant_view = (claimant_view.sort_values(["ClaimantName", "_o", "risk_score"],
                                                   ascending=[True, True, False])
                         .drop(columns="_o"))
    else:
        claimant_view = out

    readme = pd.DataFrame({"How to use this output": [
        "SHEET 'Why each claim' - ONE ROW PER CLAIM. ClaimantName is the first column,",
        "   followed by the key identifiers, the risk outputs, and then every remaining",
        "   source field and engineered feature. SerialNumber sits towards the end.",
        "",
        "   risk_level          the FINAL rating: High / Medium / Low.",
        "   risk_level_model    the model's own rating, before business rules.",
        "   risk_score          0-100, how unusual the claim is.",
        "   risk_score_1000     the same risk on a 1-1000 scale.",
        "   risk_percentile     0-100. 99.4 = riskier than 99.4% of all claims.",
        "   risk_rank           1 = the single riskiest claim in the file.",
        "   audit_band          Top 1% / Top 5% / Top 10% / Top 20% / Top 25%.",
        "   Why this rating     a full sentence - no interpretation needed.",
        "   Driver 1/2/3        the FIELD NAME and value behind the reason, so you",
        "                       can verify it directly on the same row.",
        "   features_used_for_rating   the exact fields that fired for this claim.",
        "",
        "SHEET 'Claimant view' - the same claims grouped by claimant, High first, so you",
        "                        can compare a claimant's High and Low claims directly.",
        "SHEET 'By claimant' / 'By dealer' - one consolidated rating per entity.",
        "",
        "TO SIZE YOUR OWN QUEUE: filter in_top_1pct / in_top_5pct / in_top_10pct /",
        "in_top_20pct / in_top_25pct to 1, or use the 'audit_band' column.",
        "",
        "HOW A CLAIM RATING IS DECIDED",
        "  The model scores each claim 0-100. The top 10% become High, the next 15%",
        "  Medium, the rest Low. Business rules override: a claim that failed validation",
        "  or was rejected is always High, whatever its score.",
        "",
        "HOW AN ENTITY RATING IS DECIDED",
        "  An entity is High only if its RATE of High claims is convincingly above the",
        "  population baseline (Wilson score), so nobody is rated High simply for filing",
        "  a lot of claims.",
        "",
        "WHY TWO CLAIMS FROM THE SAME PERSON CAN DIFFER",
        "  The model reacts to the SIZE of each signal. One claim may be 4x the claimant's",
        "  usual quantity while another is about average - so they score differently.",
    ]})

    xl = DATA / f"{stub}.xlsx"
    with pd.ExcelWriter(xl, engine="openpyxl") as xw:
        readme.to_excel(xw, sheet_name="Read me", index=False)
        out.to_excel(xw, sheet_name="Why each claim", index=False)
        claimant_view.to_excel(xw, sheet_name="Claimant view", index=False)
        if claimant is not None:
            claimant.to_excel(xw, sheet_name="By claimant", index=False)
        if dealer is not None:
            dealer.to_excel(xw, sheet_name="By dealer", index=False)
        for s in ["Why each claim", "Claimant view", "By claimant", "By dealer"]:
            if s in xw.sheets:
                style_sheet(xw.sheets[s])
        xw.sheets["Read me"].column_dimensions["A"].width = 92

    print(f"\n[save] {DATA / (stub + '.parquet')}   ({len(out):,} rows x {len(out.columns)} columns)")
    print(f"[save] {xl}")
    print("        sheets: Read me | Why each claim | Claimant view | By claimant | By dealer")
    return out


# ============================================================================
# MAIN
# ============================================================================

def run(input_path=INPUT_FILE, output_stub=OUTPUT_STUB, contamination=CONTAMINATION):
    df = read(input_path)
    print(f"[input] {len(df):,} rows x {len(df.columns)} columns")

    # Remove the redundant / duplicated source columns.
    dropped = [c for c in DROP_COLUMNS if c in df.columns]
    if dropped:
        df = df.drop(columns=dropped)
        print(f"[omit]  removed {len(dropped)} redundant columns: {dropped}")

    features = choose_features(df)
    df["risk_score"] = score_claims(df, features, contamination)
    df = assign_levels(df)
    df = add_ranking(df)
    df = add_explanations(df)

    dist = df["risk_level"].value_counts(); total = len(df)
    print("[claims] " + ", ".join(
        f"{t} {int(dist.get(t,0)):,} ({100*int(dist.get(t,0))/total:.1f}%)"
        for t in ["High", "Medium", "Low"]))
    print("[score]  risk_score_1000 range: "
          f"{int(df['risk_score_1000'].min())} to {int(df['risk_score_1000'].max())}")
    print("[queues] " + ", ".join(
        f"top {b}% = {int(df[f'in_top_{b}pct'].sum()):,}" for b in TOP_BANDS))

    claimant = consolidate_entity(df, "ClaimantName") if "ClaimantName" in df.columns else None
    dealer = consolidate_entity(df, "DealerName") if "DealerName" in df.columns else None
    if claimant is not None:
        c = claimant["claimant_risk_level"].value_counts()
        print("[claimants] " + ", ".join(f"{t} {int(c.get(t,0)):,}" for t in ["High", "Medium", "Low"]))
    if dealer is not None:
        d = dealer["dealer_risk_level"].value_counts()
        print("[dealers]   " + ", ".join(f"{t} {int(d.get(t,0)):,}" for t in ["High", "Medium", "Low"]))

    out = save_outputs(df, claimant, dealer, output_stub)
    return out, claimant, dealer


def main():
    p = argparse.ArgumentParser(description="Isolation Forest scoring, ranking and explanations")
    p.add_argument("--input", default=str(INPUT_FILE))
    p.add_argument("--output-stub", default=OUTPUT_STUB)
    p.add_argument("--contamination", type=float, default=CONTAMINATION)
    a = p.parse_args()
    run(a.input, a.output_stub, a.contamination)


if __name__ == "__main__":
    main()
