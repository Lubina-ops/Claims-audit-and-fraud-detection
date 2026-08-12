# 05 — Risk Score: 4-Model Comparison

**Step 5 (algorithm comparison)**

Runs **four** unsupervised anomaly detectors on the same entity tables so their
outputs can be compared. Reads `data/03_claims_features.parquet`; writes three
`*_multi.parquet` tables.

This step exists to answer: *"which claimants/dealers do multiple independent
methods agree are unusual?"* — a stronger signal than any single model.

---

## The four algorithms

| Algorithm | Detects | Notes |
|---|---|---|
| **Isolation Forest** | Globally unusual | Random partitioning |
| **Local Outlier Factor** | Unusual vs nearest neighbours | Local density |
| **One-Class SVM** | Outside the normal boundary | Kernel boundary |
| **KMeans distance** | Far from its cluster centre | Weakest signal |

---

## Why normalize to 0–100

The four models return incompatible scales (LOF ~1–3, One-Class SVM ~ −1 to 1,
KMeans a raw distance). Averaging is only meaningful after rescaling each score
to a common **0–100** range (`to_100`). Only then is `ensemble_score` = the
average of the four.

---

## Outputs

| File | Contents |
|---|---|
| `data/05_claimant_multi.parquet` | Per-model flags/scores, `models_agreeing`, `ensemble_score` |
| `data/05_dealer_multi.parquet` | Same, at dealer level |
| `data/05_claims_multi.parquet` | Claim-level 50/50 blend + `combined_risk_level` |

---

## The column that matters: `models_agreeing`

Counts how many of the four models flagged the entity (0–4). An entity flagged
by **2+ models** is your highest-confidence anomaly — because each method
detects a *different* kind of unusualness, agreement across them is far stronger
evidence than a single flag.

---

## Flags ≠ fraud

| Column | Type | Says |
|---|---|---|
| `fraud_flag` | Business rule | A rule flagged this claim |
| `IForest/LOF/SVM/KMeans_flag` | Model output | This entity behaves **unusually** |
| `anamoly_flag` | Business rule | Routine-but-notable states |

The model flags say *"statistically different from the rest"* — never
*"fraudulent."* The models are unsupervised and were never shown a fraud example.

---

## Two cautions

- **One-Class SVM is O(n²).** Fine on ~2,000 claimants / ~224 dealers; do **not**
  run it on the 128K claim rows without subsampling. (It also tends to
  over-flag on the small dealer table.)
- **KMeans is not a true anomaly detector.** "Far from centroid" correlates with
  unusualness but wasn't designed for it — treat it as the weakest vote.

---

## Functions

| Function | Purpose |
|---|---|
| `to_100(x)` | Rescale any score to 0–100 |
| `prep_matrix(features)` | Numeric → float64 → drop constants → robust scale |
| `run_all_models(features, id_col, contamination)` | Run the 4 models on one table |
| `combine_to_claims(base, claimant_multi, dealer_multi)` | Blend to claim level |
| `run_multimodel(df, contamination)` | Full comparison → 3 tables |
| `run(input, contamination)` | Read step-3 output, run, save |

---

## Run it

```bash
python "05_riskscore_4models.py"
```

**Input:** `data/03_claims_features.parquet`
**Outputs:** `data/05_claimant_multi.parquet`, `data/05_dealer_multi.parquet`,
`data/05_claims_multi.parquet`

---

## Extending the comparison

Because this step reads the **same** `03_claims_features.parquet` and the
**same** `entity_features.py` builders as step 4, adding a new algorithm is
easy: write a `06_riskscore_yourmodel.py` that reads step 3, builds the entity
tables, scores them, and saves. All models then compare on identical inputs.
