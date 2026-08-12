# 03 — Build Features

**Step 3 of the pipeline**

Engineers the behavioural features the anomaly models consume. Reads
`data/02_claims_velocity.parquet`, writes the model-ready
`data/03_claims_features.parquet` (one row per claim, all features attached).

---

## The two behavioural ideas

### A. Product-quantity anomalies (`quantity_*`)
Compare each claim's quantity to its **product's** typical quantity, via a
z-score. Flagged at **z ≥ 3** — products have thousands of claims, so the
baseline is stable and a strict cutoff is safe.

### B. Claimant monthly activity (`monthly_*`)
Compare each claimant's month to **their own** monthly history, via a z-score.
Flagged at **z ≥ 2** — each claimant has only a handful of months, so the std
is noisy and a looser cutoff is needed. Captures the "usually 3 claims,
suddenly 9" pattern.

> Why z-scores: they make "unusual" mean the same thing across 150 products and
> 400 claimants with one rule, instead of 550 hand-tuned thresholds. A z-score
> self-calibrates to each product's / person's own spread.

---

## Naming convention

| Prefix | Level | Example |
|---|---|---|
| `product_*` | Product baseline statistics | `product_avg_units`, `product_std_units` |
| `quantity_*` | A single claim's quantity vs its product norm | `quantity_z_score` |
| `claimant_*` | A claimant's own monthly baseline | `claimant_avg_monthly_claims` |
| `monthly_*` | A claimant-month observation & its z-score | `monthly_claim_count_z_score` |
| `*_flag` | 0/1 indicator | `high_quantity_anomaly_flag` |

This convention is carried through the whole project so any dataframe is
self-describing — you can tell what level a feature belongs to from its name.

---

## Columns produced

**Product baseline**
`product_claim_count`, `product_total_units`, `product_avg_units`,
`product_std_units`

**Claim-level quantity comparison**
`quantity_vs_product_avg`, `quantity_ratio_to_product_avg`,
`quantity_z_score`, `high_quantity_anomaly_flag`

**Claimant monthly baseline & observations**
`monthly_claim_count`, `monthly_units_claimed`, `monthly_unique_dealers`,
`monthly_unique_products`, `monthly_fraud_count`, `monthly_anomaly_count`,
`claimant_avg_monthly_claims`, `claimant_std_monthly_claims`,
`claimant_avg_monthly_units`, `claimant_std_monthly_units`,
`claimant_active_months`, `monthly_claim_count_z_score`,
`monthly_units_z_score`, `high_monthly_claim_activity_flag`,
`low_monthly_claim_activity_flag`, `high_monthly_units_flag`

---

## Re-run safety

`prepare_behavioral_data` drops any previously generated feature columns and
duplicate column names before rebuilding. The monthly merge derives its column
list **dynamically** from the data, so commenting an aggregation in or out
never breaks the merge.

---

## Functions

| Function | Purpose |
|---|---|
| `prepare_behavioral_data(df)` | Clean copy, numeric quantity, `claim_month` |
| `add_product_quantity_features(df)` | Product baseline + quantity z-scores/flag |
| `add_claimant_monthly_features(df)` | Claimant monthly baseline + z-scores/flags |
| `build_features(df)` | Run all three → one claim-level dataframe |
| `run(input, output)` | Read step-2 output, build, save |

---

## The result — a reusable dataframe

`data/03_claims_features.parquet` is the **hand-off point**. It carries every
feature the models need, with clear names and preserved dtypes. Any algorithm
(steps 4, 5, or a new one you write) can pick it up and run — that is the whole
point of stopping here and saving.

---

## Run it

```bash
python "03_build_features.py"
```

**Input:** `data/02_claims_velocity.parquet`
**Output:** `data/03_claims_features.parquet`
