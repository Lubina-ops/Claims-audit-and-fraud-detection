# 05 — Risk Score: 4-Model Comparison

**Step 5 (algorithm comparison)**

Runs **four** unsupervised anomaly detectors on the same entity tables so their
outputs can be compared, then produces a claim-level queue with a
plain-English reason for every claim. Reads `data/03_claims_features.parquet`;
writes three `*_multi.parquet` (and `.xlsx`) tables.

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
to a common **0–100** range (`to_100`). Only then is `ensemble_score` = the
average of the four.

### The column that matters: `models_agreeing`

Counts how many of the four models flagged the entity (0–4). An entity flagged
by **2+ models** is your highest-confidence anomaly — independent methods
corroborating one another is far stronger evidence than a single flag.

---

## Outputs

| File | Contents |
|---|---|
| `data/05_claimant_multi.parquet` / `.xlsx` | Per-model flags/scores, `models_agreeing`, `ensemble_score` |
| `data/05_dealer_multi.parquet` / `.xlsx` | Same, at dealer level |
| `data/05_claims_multi.parquet` / `.xlsx` | Claim-level tier + **`risk_reason`** |

Both parquet and Excel are written (set `SAVE_EXCEL = False` to skip Excel).

---

## Claim-level tiering (consistent with step 4)

Claims are ranked by a 50/50 blend of the claimant and dealer ensemble scores
(`combined_score`), then tiered by **percentile plus hard evidence** — the same
logic as `04_riskscore_isolationforest`, so the two models' claim tiers match:

| Tier | Rule |
|---|---|
| **High** | top 10% by `combined_score` **OR** `fraud_flag = 1` **OR** `high_quantity_anomaly_flag = 1` |
| **Medium** | next 10% by `combined_score` (≥ p80) |
| **Low** | everything else |

The two OR conditions guarantee **100% capture** of fraud-flagged and
quantity-outlier claims, even when their score falls just short of the top 10%.

> **Anomaly rule is CONTEXT, not a trigger.** `anamoly_flag` is deliberately
> **not** a High trigger. It covers ~6% of claims, and using it as a trigger
> inflated the queue. Instead it is surfaced in the reason wherever it applies,
> so a reviewer still sees it — without it pushing the claim to High. This keeps
> the queue at the calibrated size (~11–13% High).

---

## The `risk_reason` column — built for a layman

Every claim carries a plain-English explanation that names **why** it was tiered
and **whether the cause was the model or a business rule**. Each cause is tagged
by its source:

| Tag | Drives the tier? | Meaning |
|---|:---:|---|
| `[Model]` | ✅ Yes | The anomaly score placed it (top-10% / moderate / low) |
| `[Business rule - Fraud]` | ✅ Yes | KM claim status = Rejected / Duplicate / No Match |
| `[Behavioural]` | ✅ Yes | Claimed quantity 3+ standard deviations above the product norm |
| `[Context - Anomaly rule]` | ❌ No | Recalled or filed 90+ days late — informational only |

The first three **escalate** a claim; the fourth only **explains**. That
distinction is what makes the output readable — a reviewer sees at a glance what
put the claim in the queue versus what is merely additional context. Model
corroboration (`corroborated by N of 4 models`) is appended when 2+ models
agreed on the claimant or dealer.

### Example reasons

| Tier | `risk_reason` |
|---|---|
| High | `[Business rule - Fraud] Claim was Rejected, a Duplicate, or had No Matching product \| corroborated by 4 of 4 models at claimant level` |
| High | `[Model] Among the top 10% most unusual claims by anomaly score (score 94.2 >= 63.3)` |
| High | `[Behavioural] Claimed quantity far above this product's norm (3+ standard deviations)` |
| Medium | `[Model] Moderate anomaly score (63.3, in the 80-90% band >= 56.7)` |
| Low | `[Model] Low anomaly score (20.5 < 79.5) \| [Context - Anomaly rule] Claim was Recalled or filed 90+ days after the sale` |

The last row is the intended behaviour: a Low-scored claim still **explains**
that it hit the anomaly rule, without being escalated for it.

---

## Two cautions

- **One-Class SVM is O(n²).** Fine on ~2,000 claimants / ~224 dealers; do **not**
  run it on the 128K claim rows without subsampling. (It also tends to over-flag
  on the small dealer table.)
- **KMeans is not a true anomaly detector.** "Far from centroid" correlates with
  unusualness but wasn't designed for it — treat it as the weakest vote.

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

## Functions

| Function | Purpose |
|---|---|
| `to_100(x)` | Rescale any score to 0–100 |
| `prep_matrix(features)` | Numeric → float64 → drop constants → robust scale |
| `run_all_models(features, id_col, contamination)` | Run the 4 models on one table |
| `combine_to_claims(base, claimant_multi, dealer_multi)` | Blend to claim level + `risk_reason` |
| `run_multimodel(df, contamination)` | Full comparison → 3 tables |
| `run(input, contamination)` | Read step-3 output, run, save (parquet + Excel) |

---

## Run it

```bash
python "05_riskscore_4models.py"
```

**Input:** `data/03_claims_features.parquet`
**Outputs:** `data/05_claimant_multi.{parquet,xlsx}`,
`data/05_dealer_multi.{parquet,xlsx}`, `data/05_claims_multi.{parquet,xlsx}`

---

## Extending the comparison

Because this step reads the **same** `03_claims_features.parquet` and the
**same** `entity_features.py` builders as step 4, adding a new algorithm is
easy: write a `06_riskscore_yourmodel.py` that reads step 3, builds the entity
tables, scores them, and saves. All models then compare on identical inputs.
