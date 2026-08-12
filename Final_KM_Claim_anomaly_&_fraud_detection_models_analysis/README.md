# Claims Audit & Fraud Detection
## Approach, Results & Technical Reference

**Konica Minolta (KM)  |  Prepared by Data Science  |  Confidential – Internal Use Only**

> **In one sentence:** We taught the system what "normal" looks like — for each
> product and each person — so it can surface what does not fit, turning
> 128,805 claims into a prioritised audit queue rather than a random sample.

### Contents

- **Part 1** · For Everyone: What We Did and Why
- **Part 2** · Results & Algorithm Comparison
- **Part 3** · How the Risk Levels Are Decided
- **Part 4** · Audit Selection & Claim-Status Findings
- **Part 5** · The Pipeline — Technical Reference
- **Part 6** · Limitations & Next Steps

---
---

# Part 1 — For Everyone: What We Did and Why

*A plain-language guide, no technical background needed.*

## The Problem We Are Solving

We hold roughly 128,000 warranty claims. No team can review them all, and
picking claims at random wastes effort. We also cannot simply ask the computer
"is this claim fraudulent?" — nobody has labelled which past claims were
genuinely fraudulent, so there is nothing for it to learn from.

So we ask a different question — one the data can actually answer:

> **The question we ask:** Is this claim unusual compared with what normally
> happens? Unusual does not mean wrong — but unusual is where an auditor should
> look first.

## The Two Questions We Ask About Every Claim

### Question 1 — Is the quantity unusual for this product?

Every product has a normal claim size. Some items are almost always claimed one
at a time; others are ordered in batches. A quantity that looks alarming for one
product may be perfectly routine for another, so each product is judged against
its own history.

- Work out the typical claim quantity for each product.
- Compare every individual claim against that product's norm.
- Flag only the claims that are dramatically larger than usual.

| Product | Typical claim | This claim | Verdict |
|---|---|---|---|
| Printer A | 1 unit | 1 unit | Normal |
| Printer A | 1 unit | 15 units | Unusual — flag it |

*Why we are strict here: we have thousands of claims for each product, so we
understand its pattern very well. That lets us set a high bar and flag only
genuinely extreme cases — roughly the top one in a thousand — keeping the review
list short and worthwhile.*

### Question 2 — Is this claimant unusually busy this month?

Here we compare each person against their own history, not against anyone else.
A rep who files 50 claims a month is not suspicious if that is simply their
normal workload. What matters is a change in their own pattern.

- Build each claimant's normal monthly rhythm from their own past activity.
- Compare every month against that person's own baseline.
- Flag months that sit well above their usual level.

> **Example:** A sales rep normally files three claims a month. In March they
> file nine. That is out of character for them — worth a look.

*Why we are more lenient here: we may only have six or twelve months of history
for a person. With so few data points we cannot be as confident, so applying the
same strict bar as the product check would mean almost nobody is ever flagged.*

## Why Two Different Standards

This is the one design choice worth understanding, because it looks inconsistent
until the reason is clear.

| Check | Compared against | Standard | Reason |
|---|---|---|---|
| **Product quantity** | Thousands of claims for that product | Strict | Plenty of data — confident and highly selective |
| **Monthly activity** | One person's handful of months | Looser | Little data — a strict rule would catch almost nothing |

**An analogy.** Think of it like speeding tolerance. On a busy motorway — huge
traffic, well-understood patterns — you only ticket the truly reckless, or you
would write millions of tickets. But on one person's daily commute, if they
suddenly drive very differently one day, that change alone is worth noticing.

## What Comes Out of It

The result is a table in which every claim carries context it did not have
before:

- How its quantity compares with the norm for that product.
- How the claimant's activity compares with their own history.
- Simple yes / no markers highlighting the unusual cases.

That context is what turns 128,000 undifferentiated rows into a ranked list an
audit team can actually work through.

> **Important — "unusual" is not "fraudulent":** These markers say "look here
> first" — they do not say "this is fraud." A flagged claim may be entirely
> legitimate: a large customer ordering in bulk, a quarter-end surge, or a new
> territory ramping up. Every flagged case needs a person to review it before
> any conclusion is drawn. The analysis narrows the search; the judgement stays
> with the reviewer.

---
---

# Part 2 — Results & Algorithm Comparison

*What the models produced and how they compare.*

## The Dataset

| Dimension | Scale |
|---|---:|
| Claim records | **128,805** |
| Unique claimants | ~1,969 |
| Unique dealers | ~224 |
| Distinct products | ~735 |

The catalogue is entirely KM office / print equipment (Bizhub MFPs,
AccurioPress, scanners, finishers) — **0% tire-related**. Two rule-based labels
are derived first:

| Label | Rule | Rate |
|---|---|---:|
| `fraud_flag` | Status = Rejected / Duplicate Claim / No Match Found | 2.14% |
| `anamoly_flag` | Recalled **or** filed 90+ days after sale | 5.96% |

> **Read this before the numbers:** `fraud_flag` is a **rule-based proxy** from
> claim status — not investigator-confirmed fraud. Every "lift" figure below
> measures agreement with these business rules, not true fraud detection.

## Algorithm 1 — Isolation Forest (the primary model)

Scores claimants and dealers 0–100, then ranks claims by a combined score with a
percentile cutoff.

### Final claim risk distribution

| Tier | Claims | % of total |
|---|---:|---:|
| **High** | 14,889 | **11.6%** |
| Medium | 12,354 | 9.6% |
| Low | 101,562 | 78.8% |

This is the result **after calibration**. An early un-tuned version flagged
98.85% High. The fix was to (a) tighten the business-rule thresholds and (b)
rank claims by score **percentile** rather than letting them inherit their
parent entity's status — a few large dealers otherwise swept ~74% of claims into
High by volume alone.

### Validation — does the High tier contain the signal?

Hard claim-level evidence is fully captured:

| Signal | In High | Med/Low | Capture |
|---|---:|---:|---:|
| `fraud_flag` | 2,762 | 0 | **100%** |
| `high_quantity_anomaly_flag` | 54 | 0 | **100%** |

Enrichment (lift) vs base rate:

| Signal | % of High | Base rate | Lift |
|---|---:|---:|---:|
| `claimant_anomaly` | 26.9% | 7.9% | **3.41×** |
| `dealer_anomaly` | 92.5% | 28.9% | **3.20×** |
| `anamoly_flag` | 17.5% | 6.0% | **2.93×** |

All signals decline monotonically High → Medium → Low, confirming the tiering
concentrates risk rather than sorting arbitrarily. A High-tier claim is ~3×
more likely to carry an anomaly signal than a random claim.

### Contamination — a queue-size dial, not a quality dial

| contamination | claimants flagged | % flagged |
|---:|---:|---:|
| 0.01 | 20 | 1.02% |
| 0.03 | 60 | 3.05% |
| 0.05 | 99 | 5.03% |
| 0.10 | 197 | 10.01% |

*Chosen: 0.01 — conservative, appropriate for a rare target. The parameter
simply sets how many are flagged; a higher percentage is not "better."*

## Algorithm 2 — Four-Model Comparison

Four unsupervised detectors run on the **same** entity tables so their outputs
can be compared. Scores are rescaled to 0–100 before averaging.

| Algorithm | Detects | Strength |
|---|---|---|
| **Isolation Forest** | Globally unusual | Fast, scales well |
| **Local Outlier Factor** | Unusual vs nearest neighbours | Local density |
| **One-Class SVM** | Outside the learned boundary | Flexible boundary |
| **KMeans distance** | Far from cluster centre | Weakest — not a true detector |

Consensus at the claimant level (contamination 0.05) — each model flags ~5% by
design, but the overlap is small:

| Models agreeing | Claimants | % |
|---:|---:|---:|
| 0 | 1,759 | 89.3% |
| 1 | 104 | 5.3% |
| 2 | 49 | 2.5% |
| 3 | 33 | 1.7% |
| 4 | 24 | 1.2% |

> **Key insight:** Only ~5.4% of claimants are flagged by two or more methods —
> and that intersection (106 claimants) is the most defensible audit target.
> Independent methods corroborating one another is far stronger evidence than
> any single flag. Use `models_agreeing` as the priority signal, not any
> individual score.

*Caveats: One-Class SVM is O(n²) — safe on ~2,000 claimants / ~224 dealers, but
not on the 128K claim rows without subsampling. KMeans is not a true anomaly
detector; treat "far from centroid" as the weakest vote.*

## Algorithm 3 — Supervised Benchmark (exploratory)

A Random Forest was tested against the rule-based `fraud_flag` to gauge how
predictable the labels are.

| Metric | Value | Baseline | Reading |
|---|---:|---:|---|
| ROC-AUC | 0.892 | 0.50 | Inflated by class imbalance |
| **PR-AUC** | **0.331** | **0.021** | ~15× better than random |
| Precision @ 0.5 | 0.09 | 0.021 | 4.2× lift |
| Recall @ 0.5 | 0.77 | — | — |

PR-AUC is the honest metric on a 2.14% base rate: 0.33 vs a 0.021 baseline is a
respectable first model — the low precision is a consequence of rarity, not a
broken model. Not adopted (`fraud_flag` is a proxy), but useful as a benchmark.
Feature-importance analysis flagged that **product features dominated (~80%)** —
worth investigating whether the model is learning product eligibility rules
rather than fraud behaviour.

## Which Algorithm to Use

| Approach | Needs labels? | Output | Best use |
|---|---|---|---|
| **Isolation Forest** | No | 0–100 score + tiers | Primary — the production queue |
| **4-model ensemble** | No | Consensus count (0–4) | Highest-confidence corroboration |
| **Random Forest** | Yes (proxy) | Fraud probability | Benchmark; future path once outcomes exist |

> **Recommendation:** Isolation Forest drives the operational queue; the 4-model
> `models_agreeing` count is a strong secondary filter for the very top of the
> queue. The supervised route becomes viable once confirmed audit outcomes are
> captured.

## Why the Anomaly Flag Is Defined the Way It Is

The `anamoly_flag` definition was tuned empirically. Broadening it destroyed its
signal — **lift, not coverage, is the correct measure:**

| Definition | Rate | Lift | Verdict |
|---|---:|---:|---|
| **Recall + 90/180+ days** | **5.96%** | **2.56×** | Adopted |
| Volume rules, mult 2.0 | 31.6% | 1.26× | Rejected |
| Volume rules, mult 2.5 | 24.2% | 1.26× | Rejected |
| Volume rules, mult 3.0 | 17.7% | 1.03× | ≈ random |

The tight definition (5.96% coverage, 2.56× lift) beat every broad alternative.
This is why the product-velocity feature is kept **separate** from the anomaly
flag rather than folded into it.

---
---

# Part 3 — How the Risk Levels Are Decided

*The two risk levels, the three cuts, and the "worst behavioural rate."*

## Two Risk Levels, Two Questions

| Column | Based on | Question it answers |
|---|---|---|
| `*_model_risk_level` | Score only | What did the model alone say? |
| `*_risk_level` (revised) | Score + business rules | Our final call after adding business knowledge |

The revised level is built in three "cuts." They look similar but do different
jobs.

### Cut 1 — the model-only level

```python
result[model_col] = pd.cut(result[score_col],
    bins=[-np.inf, 60, 80, np.inf],
    labels=['Low', 'Medium', 'High'])
```

Slices the 0–100 score into bands: 0–60 Low, 60–80 Medium, 80–100 High. This is
the model's opinion **before** any business rules — kept so we can see how much
the rules later changed things.

### Cut 2 — two helper switches (not a level yet)

```python
has_high_rate = (result[behavior_rate_cols].max(axis=1) >= 0.50)
has_mod_rate  = (result[behavior_rate_cols].max(axis=1) >= 0.25)
```

Two 0/1 switches: `has_high_rate = 1` when the entity's worst behavioural rate is
≥ 50%; `has_mod_rate = 1` when it is ≥ 25%. They feed into Cut 3.

### Cut 3 — the revised level (the real output)

```python
high_condition = (
    (score >= 80) | (anomaly_flag == 1)            # model
    | (high_monthly_claim_activity_count > 2)      # business rule
    | (high_monthly_units_count > 2)               # business rule
    | (high_quantity_anomaly_count > 2)            # business rule
    | (has_high_rate == 1))                         # business rule

medium_condition = (score >= 60) | (has_mod_rate == 1)

risk_level = np.select([high_condition, medium_condition],
                       ['High', 'Medium'], default='Low')
```

The `|` means OR — any single condition being true sets the level. `np.select`
checks High first, then Medium, else Low. The "business rules" are human-set
thresholds layered on top of the model score, so a claimant the model rated
Medium can still be escalated to High.

## What Is the "Worst Behavioural Rate"?

From `result[behavior_rate_cols].max(axis=1)` — the single highest rate among all
of an entity's behavioural-rate columns. `axis=1` reads **across** a row and
keeps the largest.

| Rate column (Claimant A) | Value |
|---|---|
| `fraud_flag_rate` | 0.05 |
| `anomaly_flag_rate` | 0.30 |
| `high_quantity_anomaly_flag_rate` | 0.00 |
| `high_monthly_claim_activity_flag_rate` | **0.55 ← worst** |
| `low_monthly_claim_activity_flag_rate` | 0.10 |
| `high_monthly_units_flag_rate` | 0.20 |
| `claims_over_90days_rate` | 0.40 |

`max = 0.55`: 55% of this claimant's claims fell in spike months — their single
most extreme behaviour. That 0.55 trips both thresholds.

> **Why maximum, not sum or average:** Summing proportions is meaningless (1.60
> "of what"?). Averaging dilutes a single alarm — one rate of 0.90 among six
> zeros averages to just 0.13 and slips past every threshold. Maximum preserves
> the strongest signal, so one serious red flag always surfaces.

**Optional — name the culprit.** Maximum tells you HOW extreme, not WHICH
behaviour. `idxmax` adds the reason for the auditor:

```python
result['worst_rate']   = result[behavior_rate_cols].max(axis=1)
result['worst_reason'] = result[behavior_rate_cols].idxmax(axis=1)
```

---
---

# Part 4 — Audit Selection & Claim-Status Findings

*Source: KM consolidated claims dataset (128,805 records).*

## Audit Selection Outcomes

- Of the 128,805 records, **1,532 claims (1.19%)** carried the reason "Select
  For Audit." All 1,532 were subsequently **Processed** — audit selection did
  not result in any rejections in this dataset.
- Within the audited population, **1,490 claims (97.3%)** related to serialized
  products (`IsSerializedProduct = Y`), and all were Processed.
- The remaining **38 audited claims (2.5%)** related to non-serialized products,
  and were likewise all Processed.

> **Observation:** Every claim selected for audit — serialized or not —
> completed processing. This suggests the audit process functioned as a
> validation checkpoint rather than a rejection mechanism during the period
> covered.

---
---

# Part 5 — The Pipeline: Technical Reference

*A sequential, file-per-step pipeline — reuse any intermediate dataframe.*

Each numbered script reads the previous step's saved dataframe, does one job,
and saves its own output — so you can pick up any intermediate dataframe and try
a different algorithm on it without re-running everything.

## Files

| File | Role |
|---|---|
| `01_get_data.py` / `.md` | Load raw extract; timing features; `fraud_flag`, `anamoly_flag` |
| `02_product_velocity.py` / `.md` | Product claims-per-month velocity feature |
| `03_build_features.py` / `.md` | Behavioural features (quantity + monthly z-scores) |
| `04_riskscore_isolationforest.py` / `.md` | Isolation Forest scoring + risk levels |
| `05_riskscore_4models.py` / `.md` | 4-algorithm comparison (IF, LOF, SVM, KMeans) |
| `entity_features.py` | **Shared** claimant/dealer table builders (used by 04 & 05) |
| `run_pipeline.py` | Orchestrator — runs every step in order |

## Data Flow

```
raw .xlsx
   |  01_get_data
   v
data/01_claims_base.parquet
   |  02_product_velocity
   v
data/02_claims_velocity.parquet
   |  03_build_features
   v
data/03_claims_features.parquet   <- model-ready hand-off point
   |__ 04_riskscore_isolationforest -> 04_claimant / dealer / claims_ranked
   |__ 05_riskscore_4models         -> 05_claimant / dealer / claims_multi
```

Both modeling steps read the **same** `03_claims_features.parquet` and the
**same** `entity_features.py`, so every algorithm is compared on identical
inputs.

## Quick Start

```bash
python run_pipeline.py                 # steps 1-4 (Isolation Forest)
python run_pipeline.py --multimodel    # also run step 5 (4-model compare)
python run_pipeline.py --input my.xlsx --contamination 0.03

python "03_build_features.py"           # or run any single step
```

## Naming Convention (project-wide)

| Prefix | Level |
|---|---|
| `product_*` | Product baseline statistics |
| `quantity_*` | A single claim's quantity vs its product norm |
| `claimant_*` | Claimant-level aggregate / baseline |
| `dealer_*` | Dealer-level aggregate |
| `monthly_*` | Claimant-month observation |
| `*_count` / `*_rate` | Sum of a flag / mean of a flag |
| `*_flag` | 0/1 indicator |

Names are self-describing — you can tell a feature's level from its prefix.

## Reusing Data & Extending

Reuse an intermediate dataframe — every hand-off is parquet with preserved
dtypes:

```python
import pandas as pd
feats = pd.read_parquet('data/03_claims_features.parquet')
# feats is model-ready - try any algorithm on it
```

**Add a new algorithm:** copy `05_riskscore_4models.py` to
`06_riskscore_yourmodel.py`, read `data/03_claims_features.parquet`, build entity
tables with `entity_features.build_claimant_features / build_dealer_features`,
score, and save. It compares fairly against steps 4 and 5.

> **Note on numbered filenames:** Python can't `import` a module whose name
> starts with a digit, so `run_pipeline.py` loads the numbered files via
> `importlib` and calls their functions. The shared `entity_features.py` (no
> digit) imports normally.

---
---

# Part 6 — Limitations & Next Steps

*What to keep in mind, and where this goes next.*

## Key Limitation

> **Triage aid, not a verdict:** `fraud_flag` is a rule-based proxy from claim
> status — not investigator-confirmed fraud. All risk scores prioritise where to
> look; they do not determine fraud. Every High-risk item requires human review
> before any action.

## Recommended Next Steps

| # | Action | Expected benefit |
|---|---|---|
| 1 | Capture confirmed fraud outcomes from completed audits | The critical enabler — turns every metric from "agreement with rules" into "agreement with truth" and unlocks supervised modelling |
| 2 | Establish predictor importance; prune redundant / low-signal features | Sharper performance and clearer explanations (early analysis already found duplicate flags and product-feature dominance) |
| 3 | Continue refining the anomaly process on the shared feature set | Threshold tuning and challenger models, compared fairly |
| 4 | Advance toward an agentic AI framework | Specialised agents for behavioural, network, temporal and triage roles, coordinated into one explainable queue |

> **Bottom line:** The pipeline converts 128,805 claims into a workable ~11.6%
> audit queue that captures 100% of rule-flagged claims with ~3× signal
> enrichment — a defensible, reusable foundation that improves as confirmed
> outcomes are captured.
