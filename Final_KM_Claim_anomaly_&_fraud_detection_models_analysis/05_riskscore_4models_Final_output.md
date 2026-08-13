# 05 — Risk Score: 4-Model Comparison

**Step 5 (algorithm comparison)**

Runs **four** unsupervised anomaly detectors on the same entity tables so their
outputs can be compared, produces a claim-level queue with a plain-English
reason for every claim, and rolls everything up to a **final claimant-level
summary**. Reads `data/03_claims_features.parquet`; writes four tables (parquet
+ Excel).

This step answers: *"which claimants/dealers do multiple independent methods
agree are unusual?"* — a stronger signal than any single model.

---

## The four algorithms

| Algorithm | Detects | Notes |
|---|---|---|
| **Isolation Forest** | Globally unusual | Random partitioning |
| **Local Outlier Factor** | Unusual vs nearest neighbours | Local density |
| **One-Class SVM** | Outside the normal boundary | Kernel boundary |
| **KMeans distance** | Far from its cluster centre | Weakest signal |

### Why normalize to 0–100

The four models return incompatible scales (LOF ~1–3, One-Class SVM ~ −1 to 1,
KMeans a raw distance). Averaging is only meaningful after rescaling each score
to 0–100 (`to_100`). Only then is `ensemble_score` = the average of the four.

### The column that matters: `models_agreeing`

Counts how many of the four models flagged the entity (0–4). An entity flagged
by **2+ models** is your highest-confidence anomaly — independent methods
corroborating one another is far stronger evidence than a single flag.

---

## Outputs

| File | Grain | Contents |
|---|---|---|
| `data/05_claimant_multi.{parquet,xlsx}` | 1 per claimant | Per-model flags/scores, `models_agreeing`, `ensemble_score` |
| `data/05_dealer_multi.{parquet,xlsx}` | 1 per dealer | Same, at dealer level |
| `data/05_claims_multi.{parquet,xlsx}` | 1 per claim | Tier + **`risk_reason`** |
| `data/05_claimant_risk_summary.{parquet,xlsx}` | 1 per claimant | **Final rollup** with summed reasons |

Set `SAVE_EXCEL = False` to skip the Excel copies.

---

## Claim-level tiering (consistent with step 4)

Claims are ranked by a 50/50 blend of the claimant and dealer ensemble scores
(`combined_score`), then tiered by **percentile plus hard evidence** — the same
logic as `04_riskscore_isolationforest`:

| Tier | Rule |
|---|---|
| **High** | top 10% by `combined_score` **OR** `fraud_flag = 1` **OR** `high_quantity_anomaly_flag = 1` |
| **Medium** | next 10% by `combined_score` (≥ p80) |
| **Low** | everything else |

The two OR conditions guarantee **100% capture** of fraud-flagged and
quantity-outlier claims.

> **Anomaly rule is CONTEXT, not a trigger.** `anamoly_flag` is deliberately
> **not** a High trigger (it covers ~6% of claims and using it as a trigger
> inflated the queue). It is surfaced in the reason wherever it applies, so a
> reviewer still sees it — without pushing the claim to High. Keeps the queue at
> the calibrated size (~11–13% High).

---

## `risk_reason` — per-claim explanation, tagged by source

| Tag | Drives the tier? | Meaning |
|---|:---:|---|
| `[Model]` | Yes | The anomaly score placed it (top-10% / moderate / low) |
| `[Business rule - Fraud]` | Yes | KM status = Rejected / Duplicate / No Match |
| `[Behavioural]` | Yes | Quantity 3+ std above the product norm |
| `[Context - Anomaly rule]` | No | Recalled or 90+ days late — informational only |

The first three **escalate**; the fourth only **explains**. Model corroboration
(`corroborated by N of 4 models`) is appended when 2+ models agreed.

---

## Final rollup — `05_claimant_risk_summary`

One row per claimant, with the reasons **summed** into a single readable line.
This is the "final claim risk at the claimant level" file.

### Columns

| Column | Meaning |
|---|---|
| `total_claims` | Claims filed by the claimant |
| `high_claims` / `medium_claims` / `low_claims` | How many fell in each tier |
| `avg_combined_score` | Average claim-level score |
| `avg_claimant_ensemble` / `avg_dealer_ensemble` | Average ensemble scores |
| `fraud_rule_hits` | Number of the claimant's claims that hit the fraud rule |
| `anomaly_rule_hits` | Number that hit the anomaly rule |
| `high_quantity_hits` | Number flagged as quantity outliers |
| `max_models_agreeing` | Most models that agreed on any of their claims |
| `claimant_risk_level` | Worst tier the claimant has any claim in |
| `risk_reason_summary` | Plain-English summed explanation |

### How `claimant_risk_level` is set

It is the **worst tier** the claimant appears in — if any of their claims is
High, the claimant is High; else Medium if any Medium; else Low. This is
deliberately conservative for triage (one High claim is enough to warrant a
look at the claimant).

### Example `risk_reason_summary`

```
18 of 72 claims High; fraud rule applied 6x; anomaly rule applied 7x;
high-quantity outlier 6x; corroborated by up to 4 of 4 models
```

A reviewer reads this in one line: how much of the claimant's book is High, how
often each business rule fired, and how strongly the models agreed.

---

## Two cautions

- **One-Class SVM is O(n²).** Fine on ~2,000 claimants / ~224 dealers; do **not**
  run it on the 128K claim rows without subsampling.
- **KMeans is not a true anomaly detector.** Treat "far from centroid" as the
  weakest vote.

---

## Flags ≠ fraud

| Column | Type | Says |
|---|---|---|
| `fraud_flag` | Business rule | A rule flagged this claim |
| `IForest/LOF/SVM/KMeans_flag` | Model output | This entity behaves **unusually** |
| `anamoly_flag` | Business rule | Routine-but-notable states |

The model flags say *"statistically different"* — never *"fraudulent."*

---

## Functions

| Function | Purpose |
|---|---|
| `to_100(x)` | Rescale any score to 0–100 |
| `prep_matrix(features)` | Numeric → float64 → drop constants → robust scale |
| `run_all_models(features, id_col, contamination)` | Run the 4 models on one table |
| `combine_to_claims(base, claimant_multi, dealer_multi)` | Blend to claim level + `risk_reason` |
| `build_claimant_summary(claims)` | **Final rollup** — one row per claimant, summed reasons |
| `run_multimodel(df, contamination)` | Full comparison → 4 tables |
| `run(input, contamination)` | Read step-3 output, run, save (parquet + Excel) |

---

## Run it

```bash
python "05_riskscore_4models.py"
```

**Input:** `data/03_claims_features.parquet`
**Outputs:** `05_claimant_multi`, `05_dealer_multi`, `05_claims_multi`,
`05_claimant_risk_summary` — each as `.parquet` and `.xlsx`.

---

## Extending the comparison

Because this step reads the **same** `03_claims_features.parquet` and the
**same** `entity_features.py` builders as step 4, adding a new algorithm is
easy: write a `06_riskscore_yourmodel.py` that reads step 3, builds the entity
tables, scores them, and saves. All models then compare on identical inputs.
