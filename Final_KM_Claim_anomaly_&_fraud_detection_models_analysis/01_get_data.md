# 01 — Get Data

**Step 1 of the KM claims fraud/anomaly pipeline**

Loads the raw claims extract, standardizes dates, derives the filing-delay
feature, and produces the two business-rule labels. Saves the result to
`data/01_claims_base.parquet` for step 2.

---

## What this step produces

| Column | Type | Meaning |
|---|---|---|
| `num_days` | Int64 | Days between the invoice (sale) date and the item-claimed (filed) date |
| `num_days_buckets` | category | Banded version of `num_days` |
| `fraud_flag` | 0/1 | Claim was rejected, duplicated, or had no matching product |
| `anamoly_flag` | 0/1 | Claim was recalled **or** filed 90+ days after sale |

*(All original claim columns are retained.)*

---

## Functions

| Function | Purpose |
|---|---|
| `load_raw(path)` | Read the raw Excel extract |
| `add_timing_features(df)` | Standardize dates; compute `num_days` and its buckets |
| `add_fraud_flag(df)` | Derive `fraud_flag` |
| `add_anomaly_flag(df)` | Derive `anamoly_flag` |
| `get_data(input, output)` | Run all of the above and save |

---

## The claim lifecycle — what the statuses mean

Once a claim passes **System Validation**, it enters the **Audit Process**:

- **Approved → Completed** status
- **Edit / fix problems → Returned**
- **Rejected**

### Status-reason meanings

| Status / Reason | Meaning |
|---|---|
| **Rejected due to KMAP Sale** | KM-specific sale type; the claim was not eligible for sales, so it was rejected |
| **Completed** | Claim returned; the system confirmed the Product ID and Product Unique ID match and that someone else has claimed it |
| **Pending** | Submitted and awaiting approval after all required documents are in and initial validation has passed |
| **Draft** | May be cancelled by the submitter/claimant. Pending / Returned claims may be recalled to Draft to update information |

> **Validation note:** if a product has **no Product Unique ID** (serial number,
> VIN) to validate against, the claim **passes system validation automatically**.

---

## Label definitions (the rules encoded in code)

**`fraud_flag = 1`** when `ClaimItemStatusReason` matches
`Duplicate Claim | No Match Found | unit's ineligible... | Rejected`
**or** `ClaimItemStatus` matches `Rejected | Duplicate Claim | No Match Found`.

**`anamoly_flag = 1`** when `ClaimItemStatusReason` contains `Recalled`
**or** `num_days_buckets` is in `['90-180 days', '180+ days']`.

> The `Cancelled | Returned | Draft | Pending` status condition was tested and
> **removed** — it pushed the anomaly rate to ~20% (near the population average)
> and diluted the flag's fraud-lift. The tighter definition above scores ~6%
> with ~2.5x lift.

---

## Timing feature — `num_days` buckets

| Bucket | Range (days) |
|---|---|
| 0 days | exactly 0 |
| 1-week | 1 – 7 |
| 2-week | 8 – 14 |
| 3-week | 15 – 21 |
| 4-week | 22 – 30 |
| 30-60 days | 31 – 60 |
| 60-90 days | 61 – 90 |
| 90-180 days | 91 – 180 |
| 180+ days | over 180 |

> **Data-quality note:** `num_days` can be **negative** if a claim date
> precedes its invoice date — usually a data-entry issue. The step prints a
> warning with the count.

---

## Run it

```bash
python "01_get_data.py"
```

Or from another script:

```python
# see run_pipeline.py for the importlib pattern (numbered filenames)
get_data.get_data(input_path='my_extract.xlsx',
                  output_path='data/01_claims_base.parquet')
```

**Output:** `data/01_claims_base.parquet`
