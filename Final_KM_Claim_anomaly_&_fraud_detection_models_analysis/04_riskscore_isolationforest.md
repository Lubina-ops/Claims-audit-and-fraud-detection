# 04 — Risk Score: Isolation Forest

**Step 4 (algorithm A)**

Reads the model-ready features from step 3, builds claimant/dealer tables,
scores them with **Isolation Forest**, applies business-aware risk levels, and
ranks every claim into a final audit queue.

Reads `data/03_claims_features.parquet`; writes three tables.

---

## Why Isolation Forest

It is a strong **starting** algorithm: it surfaces unusual claimant/dealer
behaviour **without labelled fraud data**. It isolates points that are "few and
different" — exactly how rare anomalies behave.

---

## Outputs

| File | Rows | Contents |
|---|---|---|
| `data/04_claimant_risk.parquet` | ~1 per claimant | scores + risk level + escalation reason |
| `data/04_dealer_risk.parquet` | ~1 per dealer | scores + risk level + escalation reason |
| `data/04_claims_ranked.parquet` | 1 per claim | final audit queue, sorted by score |

---

## The pipeline (10 steps)

| Step | Action |
|---|---|
| A | Type cleanup (features already engineered) |
| 2 | Build claimant & dealer tables (shared `entity_features.py`) |
| 3 | Isolation Forest → `fraud_risk_score` (0–100), `model_anomaly_flag` |
| 4 | Rename scoring columns (claimant vs dealer) |
| 5 | Apply business-aware risk levels |
| 6–7 | Merge entity risk back to claim level |
| 8 | Combined claim score = 0.5·claimant + 0.5·dealer |
| 9 | Claim `risk_level` by **percentile** + hard evidence |
| 10 | Round floats |

---

## Two risk levels per entity

| Column | Based on | Question |
|---|---|---|
| `*_model_risk_level` | Score band only | What did the model alone say? |
| `*_risk_level` | Score **+ business rules** | Final call after business knowledge |

A **High** row can have a **Medium** model level — that means a business rule
escalated it. The `*_escalation_reason` column records exactly which rule fired
(e.g. `anomaly_flag`, `monthly_activity>2`, `worst_rate>=0.50`).

**Business rules** (human-set thresholds, on top of the model score):

- `high_monthly_claim_activity_flag_count > 2`
- `high_monthly_units_flag_count > 2`
- `high_quantity_anomaly_flag_count > 2`
- worst behavioural rate ≥ 0.50 → High, ≥ 0.25 → Medium

---

## Step 9 — claim ranking by percentile (the key design choice)

Entity flags cascade by volume: one large dealer (e.g. 26,584 claims, ~20% of
the book) flagged High would drag all its claims to High if claims *inherited*
dealer status. That produced ~74% High.

Instead, each claim is ranked on its **own** `final_score`:

```python
HIGH_PCT = 0.90            # top 10% by score -> High
_hi = clean_df['final_score'].quantile(HIGH_PCT)

risk_level = High if (final_score >= _hi)
                  or (high_quantity_anomaly_flag == 1)   # hard evidence
                  or (fraud_flag == 1)                   # hard evidence
```

- The **percentile** guarantees a fixed, workable queue (~10% High) regardless
  of score distribution.
- The two **OR** conditions guarantee 100% capture of fraud-flagged and
  quantity-outlier claims, even if their score falls just short.

---

## Contamination

Set to **0.01** (conservative — fraud is rare). Contamination controls
**queue size**, not model quality; a higher percentage flagged is not "better".

---

## Functions

| Function | Purpose |
|---|---|
| `clean_types(df)` | Type cleanup passthrough |
| `score_anomalies(features, contamination)` | Isolation Forest scoring |
| `assign_revised_risk_level(df, entity_type)` | Business-aware level + reason |
| `run_isolationforest_pipeline(df, contamination)` | Full pipeline → 3 tables |
| `run(input, contamination)` | Read step-3 output, run, save |

---

## Run it

```bash
python "04_riskscore_isolationforest.py"
```

**Input:** `data/03_claims_features.parquet`
**Outputs:** `data/04_claimant_risk.parquet`, `data/04_dealer_risk.parquet`,
`data/04_claims_ranked.parquet`

---

## Important limitation

`fraud_flag` is a **rule-based proxy** (from claim status), not
investigator-confirmed fraud. Risk scores are an audit-**triage** aid — they
prioritise where to look; they do not determine fraud.
