"""
03_build_features_final.py - KM anomaly feature builder (trimmed)
================================================================

Goal
----
Keep only the features that are actually used for anomaly detection:
  - Kelly's requested features
  - The additional features we evolved and agreed to keep

We deliberately DROP:
  - Intermediate _std and _count columns (used internally, not exported)
  - The 12 month_multiplier_1..12 columns (replaced by one seasonality ratio)
  - Redundant _avg columns that duplicate a rate

Grain
-----
ClaimFormID = parent claim form
ClaimId     = claim item

Repeated ClaimId rows (different SalesTransactionID) are preserved in the final
output. Historical stats use one logical record per ClaimId. All historical
windows exclude the current claim (closed="left").
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
INPUT_FILE = BASE / "data" / "02_claims_velocity.parquet"
OUTPUT_FILE = BASE / "data" / "03_claims_features.parquet"
EXCEL_OUTPUT = BASE / "data" / "03_claims_features.xlsx"
HISTORY_DAYS = 1095   # 3 years


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def pv(value):
    """Resolve a path relative to this script."""
    p = Path(value).expanduser()
    return (p if p.is_absolute() else BASE / p).resolve()


def read(path):
    """Read Parquet or Excel."""
    p = pv(path)
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    return pd.read_excel(p, engine="openpyxl")


def ratio(numerator, denominator):
    """Divide safely. Returns NaN when the denominator is 0 or missing."""
    a = pd.to_numeric(numerator, errors="coerce")
    b = pd.to_numeric(denominator, errors="coerce")
    return (a / b.where(b.ne(0))).replace([np.inf, -np.inf], np.nan)


def zscore(value, mean, std):
    """How many standard deviations above/below the historical average."""
    v = pd.to_numeric(value, errors="coerce")
    m = pd.to_numeric(mean, errors="coerce")
    s = pd.to_numeric(std, errors="coerce")
    out = pd.Series(0.0, index=v.index)
    ok = v.notna() & m.notna() & s.gt(0)
    out.loc[ok] = (v.loc[ok] - m.loc[ok]) / s.loc[ok]
    return out.replace([np.inf, -np.inf], 0).fillna(0)


def prior_mean_std(df, group, value, days=HISTORY_DAYS):
    """Return prior-window mean and std for one value column.

    The current claim is excluded (closed='left'). We only return mean and std
    because those are all the z-scores need.
    """
    mean = pd.Series(np.nan, index=df.index)
    std = pd.Series(np.nan, index=df.index)

    for _, g in df.groupby(group, sort=False, dropna=False):
        g = g[g["ClaimSubmitionDate"].notna()].sort_values(
            ["ClaimSubmitionDate", "ClaimFormID"], kind="stable"
        )
        if g.empty:
            continue
        series = pd.Series(
            pd.to_numeric(g[value], errors="coerce").to_numpy(),
            index=pd.DatetimeIndex(g["ClaimSubmitionDate"]),
        )
        mean.loc[g.index] = series.rolling(f"{days}D", closed="left", min_periods=1).mean().to_numpy()
        std.loc[g.index] = series.rolling(f"{days}D", closed="left", min_periods=2).std().to_numpy()

    return mean, std


def prior_rate_count(df, group, flag, days=HISTORY_DAYS):
    """Return prior-window rate (mean of a 0/1 flag) and prior claim count."""
    rate = pd.Series(np.nan, index=df.index)
    count = pd.Series(0, index=df.index, dtype="int64")

    for _, g in df.groupby(group, sort=False, dropna=False):
        g = g[g["ClaimSubmitionDate"].notna()].sort_values(
            ["ClaimSubmitionDate", "ClaimFormID"], kind="stable"
        )
        if g.empty:
            continue
        series = pd.Series(
            pd.to_numeric(g[flag], errors="coerce").to_numpy(),
            index=pd.DatetimeIndex(g["ClaimSubmitionDate"]),
        )
        w = series.rolling(f"{days}D", closed="left", min_periods=1)
        rate.loc[g.index] = w.mean().to_numpy()
        count.loc[g.index] = w.count().fillna(0).astype(int).to_numpy()

    return rate, count


# ---------------------------------------------------------------------------
# Step A: one logical row per ClaimId
# ---------------------------------------------------------------------------

def logical_claim_items(source):
    """Collapse repeated SalesTransactionID rows into one row per ClaimId."""
    order = source.sort_values(["ClaimSubmitionDate", "ClaimId"], kind="stable")
    item = order.drop_duplicates("ClaimId", keep="first").copy()

    # Quantity is counted once per claim item.
    item["claim_prod_qty"] = pd.to_numeric(item["ItemQuantity"], errors="coerce")

    # Amount is summed once per distinct ClaimId + SalesTransactionID.
    if "SalesTransactionID" in source.columns:
        tx = (source[["ClaimId", "SalesTransactionID", "SaleAmount"]]
              .drop_duplicates(["ClaimId", "SalesTransactionID"]))
        tx["SaleAmount"] = pd.to_numeric(tx["SaleAmount"], errors="coerce")
        amt = tx.groupby("ClaimId")["SaleAmount"].sum(min_count=1)
        item["claim_prod_amt"] = item["ClaimId"].map(amt)
    else:
        item["claim_prod_amt"] = pd.to_numeric(item["SaleAmount"], errors="coerce")

    return item


# ---------------------------------------------------------------------------
# Step B: claim-form and claim-product tables
# ---------------------------------------------------------------------------

def claim_tables(items):
    """Build one row per claim form and one row per claim-product."""
    forms = (items.groupby("ClaimFormID", sort=False).agg(
        ClaimSubmitionDate=("ClaimSubmitionDate", "min"),
        ClaimantName=("ClaimantName", "first"),
        DealerName=("DealerName", "first"),
        claim_total_qty=("claim_prod_qty", lambda x: x.sum(min_count=1)),
        claim_total_amt=("claim_prod_amt", lambda x: x.sum(min_count=1)),
        rejected_claim=("rejected_claim", "max"),
        returned_claim=("returned_claim", "max"),
        cancelled_claim=("cancelled_claim", "max"),
    ).reset_index())

    products = (items.groupby(["ClaimFormID", "ProductID"], sort=False).agg(
        ClaimSubmitionDate=("ClaimSubmitionDate", "min"),
        ClaimantName=("ClaimantName", "first"),
        DealerName=("DealerName", "first"),
        claim_prod_qty=("claim_prod_qty", lambda x: x.sum(min_count=1)),
        claim_prod_amt=("claim_prod_amt", lambda x: x.sum(min_count=1)),
    ).reset_index())

    return forms, products


# ---------------------------------------------------------------------------
# Step C: claimant and dealer history (claim-form level)
# ---------------------------------------------------------------------------

def form_history(forms):
    """Add claimant/dealer averages, z-scores, recency, frequency, and rates."""
    out = forms.copy()

    # Recency: days since this claimant/dealer's previous claim.
    out = out.sort_values(["ClaimantName", "ClaimSubmitionDate", "ClaimFormID"], kind="stable")
    out["claim_claimant_recency"] = out.groupby("ClaimantName")["ClaimSubmitionDate"].diff().dt.days

    out = out.sort_values(["DealerName", "ClaimSubmitionDate", "ClaimFormID"], kind="stable")
    out["claim_dealer_recency"] = out.groupby("DealerName")["ClaimSubmitionDate"].diff().dt.days

    # Claimant average number of days between claims + prior claim count.
    claimant_freq_mean, _ = prior_mean_std(out, "ClaimantName", "claim_claimant_recency")
    out["claimant_avg_frequency"] = claimant_freq_mean
    _, claimant_count = prior_rate_count(out, "ClaimantName", "rejected_claim")
    out["claimant_claim_count"] = claimant_count

    # Claimant quantity + amount z-scores.
    for value, feat in [("claim_total_qty", "total_qty"), ("claim_total_amt", "total_amt")]:
        mean, std = prior_mean_std(out, "ClaimantName", value)
        out[f"claimant_avg_{feat}"] = mean
        out[f"z_claimant_avg_{feat}"] = zscore(out[value], mean, std)

    # Dealer quantity + amount z-scores.
    for value, feat in [("claim_total_qty", "total_qty"), ("claim_total_amt", "total_amt")]:
        mean, std = prior_mean_std(out, "DealerName", value)
        out[f"dealer_avg_{feat}"] = mean
        out[f"z_dealer_avg_{feat}"] = zscore(out[value], mean, std)

    # Recency ratio: how fast the claim arrived vs the claimant's normal pace.
    out["recency_vs_avg_frequency"] = ratio(out["claim_claimant_recency"], out["claimant_avg_frequency"])

    # Prior outcome RATES only (no _std / _count clutter).
    for entity, prefix in [("ClaimantName", "claimant"), ("DealerName", "dealer")]:
        for flag, name in [("rejected_claim", "reject"), ("returned_claim", "return"), ("cancelled_claim", "cancel")]:
            rate, _ = prior_rate_count(out, entity, flag)
            out[f"{prefix}_{name}_rate"] = rate

    return out


# ---------------------------------------------------------------------------
# Step D: product history (claim-product level)
# ---------------------------------------------------------------------------

def product_history(products):
    """Add product, claimant-product, and dealer-product features."""
    out = products.copy()

    # Product recency + average days between product claims.
    out = out.sort_values(["ProductID", "ClaimSubmitionDate", "ClaimFormID"], kind="stable")
    out["claim_product_recency"] = out.groupby("ProductID")["ClaimSubmitionDate"].diff().dt.days
    prod_freq_mean, _ = prior_mean_std(out, "ProductID", "claim_product_recency")
    out["prod_avg_frequency"] = prod_freq_mean

    # Product quantity + amount averages and z-scores.
    qmean, qstd = prior_mean_std(out, "ProductID", "claim_prod_qty")
    out["prod_avg_qty_claim"] = qmean
    out["quantity_z_score"] = zscore(out["claim_prod_qty"], qmean, qstd)
    # Keep product qty std internally for the seasonal z-score below.
    out["_prod_qty_std"] = qstd

    amean, astd = prior_mean_std(out, "ProductID", "claim_prod_amt")
    out["prod_avg_amt_claim"] = amean
    out["amount_z_score"] = zscore(out["claim_prod_amt"], amean, astd)

    # Claimant-product averages, z-scores, and ratios.
    cq_mean, cq_std = prior_mean_std(out, ["ClaimantName", "ProductID"], "claim_prod_qty")
    out["claimant_avg_prod_qty"] = cq_mean
    out["z_claimant_avg_prod_qty"] = zscore(out["claim_prod_qty"], cq_mean, cq_std)
    out["r_prod_avg_qty_claim"] = ratio(out["claim_prod_qty"], cq_mean)

    ca_mean, ca_std = prior_mean_std(out, ["ClaimantName", "ProductID"], "claim_prod_amt")
    out["claimant_avg_prod_amt"] = ca_mean
    out["z_claimant_avg_prod_amt"] = zscore(out["claim_prod_amt"], ca_mean, ca_std)
    out["r_prod_avg_amt_claim"] = ratio(out["claim_prod_amt"], ca_mean)

    return out


# ---------------------------------------------------------------------------
# Step E: quantity seasonality (ONE ratio, not 12 multipliers)
# ---------------------------------------------------------------------------

def quantity_seasonality(products):
    """Adjust product quantity for its own month-of-year pattern.

    Instead of 12 month_multiplier columns, we keep a single ratio plus the
    seasonally adjusted quantity and its z-score.
    """
    out = products.copy()
    out["claim_month"] = out["ClaimSubmitionDate"].dt.month

    same_month = pd.Series(np.nan, index=out.index)
    all_month = pd.Series(np.nan, index=out.index)

    for _, g in out.groupby("ProductID", sort=False, dropna=False):
        g = g[g["ClaimSubmitionDate"].notna()].sort_values(
            ["ClaimSubmitionDate", "ClaimFormID"], kind="stable"
        )
        for idx, row in g.iterrows():
            window_start = row["ClaimSubmitionDate"] - pd.Timedelta(days=HISTORY_DAYS)
            prior = g[(g["ClaimSubmitionDate"] < row["ClaimSubmitionDate"]) &
                      (g["ClaimSubmitionDate"] >= window_start)]
            all_month.loc[idx] = pd.to_numeric(prior["claim_prod_qty"], errors="coerce").mean()
            same = prior[prior["ClaimSubmitionDate"].dt.month == row["claim_month"]]
            same_month.loc[idx] = pd.to_numeric(same["claim_prod_qty"], errors="coerce").mean()

    # One seasonality ratio for the claim month.
    out["product_seasonality_quantity_ratio"] = ratio(same_month, all_month)

    # Single impact score: how far the month is from normal.
    out["seasonality_impact_score"] = (out["product_seasonality_quantity_ratio"] - 1).abs()

    # Seasonally adjusted quantity and its z-score.
    out["claim_qty_month_comp"] = ratio(out["claim_prod_qty"], out["product_seasonality_quantity_ratio"])
    out["claim_qty_month_z"] = zscore(
        out["claim_qty_month_comp"], out["prod_avg_qty_claim"], out["_prod_qty_std"]
    )

    return out


# ---------------------------------------------------------------------------
# Step F: recent claimant activity (claim-item level)
# ---------------------------------------------------------------------------

def recent_activity(items):
    """Add recent 30-day claim frequency and its z-score."""
    out = items.copy()
    recent = pd.Series(np.nan, index=out.index)
    hist_avg = pd.Series(np.nan, index=out.index)
    hist_std = pd.Series(np.nan, index=out.index)
    hist_cnt = pd.Series(0, index=out.index, dtype="int64")

    for _, g in out.groupby("ClaimantName", sort=False, dropna=False):
        g = g[g["ClaimSubmitionDate"].notna()].sort_values(
            ["ClaimSubmitionDate", "ClaimId"], kind="stable"
        )
        if g.empty:
            continue
        dates = pd.DatetimeIndex(g["ClaimSubmitionDate"])
        ones = pd.Series(1.0, index=dates)
        rolling30 = ones.rolling("30D", closed="left").sum().fillna(0)
        recent.loc[g.index] = rolling30.to_numpy()
        hist = pd.Series(rolling30.to_numpy(), index=dates)
        w = hist.rolling(f"{HISTORY_DAYS}D", closed="left", min_periods=1)
        hist_avg.loc[g.index] = w.mean().to_numpy()
        hist_cnt.loc[g.index] = w.count().fillna(0).astype(int).to_numpy()
        hist_std.loc[g.index] = hist.rolling(f"{HISTORY_DAYS}D", closed="left", min_periods=2).std().to_numpy()

    out["prior_30d_claim_count"] = recent.fillna(0).astype("int64")
    out["claimant_recent_claim_frequency"] = ratio(recent, hist_avg)
    out["rolling_30d_claim_count_z"] = zscore(recent, hist_avg, hist_std)
    out["insufficient_claimant_history_flag"] = (hist_cnt < 2).astype("int8")
    return out


# ---------------------------------------------------------------------------
# Step G: Kelly returned index (claim-item level, prior 12 months)
# ---------------------------------------------------------------------------

def returned_index(items):
    """Claimant returned rate vs population returned rate (prior 12 months)."""
    out = items.copy()
    valid = out[out["ClaimSubmitionDate"].notna()].sort_values(["ClaimSubmitionDate", "ClaimId"])

    # Overall prior-12-month returned and processed counts.
    overall = valid.set_index("ClaimSubmitionDate")
    all_ret = pd.Series(np.nan, index=out.index)
    all_proc = pd.Series(np.nan, index=out.index)
    all_ret.loc[valid.index] = overall["returned_claim"].rolling("365D", closed="left").sum().fillna(0).to_numpy()
    all_proc.loc[valid.index] = overall["processed_claim"].rolling("365D", closed="left").sum().fillna(0).to_numpy()

    # Claimant prior-12-month returned and processed counts.
    cl_ret = pd.Series(np.nan, index=out.index)
    cl_proc = pd.Series(np.nan, index=out.index)
    for _, g in valid.groupby("ClaimantName", dropna=False):
        g = g.sort_values(["ClaimSubmitionDate", "ClaimId"])
        ts = g.set_index("ClaimSubmitionDate")
        cl_ret.loc[g.index] = ts["returned_claim"].rolling("365D", closed="left").sum().fillna(0).to_numpy()
        cl_proc.loc[g.index] = ts["processed_claim"].rolling("365D", closed="left").sum().fillna(0).to_numpy()

    claimant_rate = ratio(cl_ret, cl_proc)
    population_rate = ratio(all_ret, all_proc)
    out["claimant_percent_returned_index"] = ratio(claimant_rate, population_rate)
    return out


# ---------------------------------------------------------------------------
# Assemble the final output
# ---------------------------------------------------------------------------

def build_features(df):
    source = df.copy()
    source["ClaimSubmitionDate"] = pd.to_datetime(source["ClaimSubmitionDate"], errors="coerce")

    items = logical_claim_items(source)
    forms, products = claim_tables(items)

    forms = form_history(forms)
    products = product_history(products)
    products = quantity_seasonality(products)
    items = recent_activity(items)
    items = returned_index(items)

    # Remove internal helper columns before merging.
    products = products.drop(columns=["_prod_qty_std"], errors="ignore")

    # Merge claim-form features onto every source row.
    out = source.copy()
    base = set(out.columns)
    fc = [c for c in forms.columns if c == "ClaimFormID" or c not in base]
    out = out.merge(forms[fc], on="ClaimFormID", how="left", validate="many_to_one")

    # Merge claim-product features.
    base = set(out.columns)
    pc = [c for c in products.columns if c in ["ClaimFormID", "ProductID"] or c not in base]
    out = out.merge(products[pc], on=["ClaimFormID", "ProductID"], how="left", validate="many_to_one")

    # Merge claim-item features.
    base = set(out.columns)
    ic = [c for c in items.columns if c == "ClaimId" or c not in base]
    out = out.merge(items[ic], on="ClaimId", how="left", validate="many_to_one")

    return out.reset_index(drop=True)


def run(input_path=INPUT_FILE, output_path=OUTPUT_FILE, excel_output=EXCEL_OUTPUT, save_excel=True):
    out = build_features(read(input_path))

    # Guard against mixed object columns before Parquet.
    for c in out.columns:
        if out[c].dtype == "object":
            out[c] = out[c].astype("string")

    op = pv(output_path)
    op.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(op, index=False, engine="pyarrow")
    print(f"[save] {op}")

    if save_excel:
        xp = pv(excel_output)
        xp.parent.mkdir(parents=True, exist_ok=True)
        out.to_excel(xp, index=False, engine="openpyxl")
        print(f"[save] {xp}")

    print(f"[validation] rows={len(out):,}; columns={len(out.columns):,}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=INPUT_FILE)
    p.add_argument("--output", default=OUTPUT_FILE)
    p.add_argument("--excel-output", default=EXCEL_OUTPUT)
    p.add_argument("--no-excel", action="store_true")
    a = p.parse_args()
    run(a.input, a.output, a.excel_output, not a.no_excel)


if __name__ == "__main__":
    main()
