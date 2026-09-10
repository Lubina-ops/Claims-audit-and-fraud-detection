# KM Claims Fraud & Anomaly Detection

Turning a very large pile of warranty claims into a **short, ranked, explainable
review list** — so auditors spend their limited time on the claims most worth a
closer look.

> **Important:** this system highlights claims that are **unusual**, not claims that
> are proven fraudulent. Every flagged claim still needs a human to review it.
> "Unusual" is where an auditor should look first, not a verdict.

---

## Table of Contents

- [Why we built this](#why-we-built-this)
- [The pipeline at a glance](#the-pipeline-at-a-glance)
- [Step 01 — Get and clean the data](#step-01--get-and-clean-the-data)
- [Step 02 — Product timing](#step-02--product-timing)
- [Step 03 — Build the comparison signals](#step-03--build-the-comparison-signals)
- [Step 03b — Feature health-check](#step-03b--feature-health-check)
- [Step 04 — Score every claim](#step-04--score-every-claim)
- [Step 05 — Four-model second opinion](#step-05--four-model-second-opinion)
- [Comparing Step 04 and Step 05](#comparing-step-04-and-step-05)
- [Understanding the output](#understanding-the-output)
- [The statistics, in plain language](#the-statistics-in-plain-language)
- [How to run](#how-to-run)
- [Output files](#output-files)
- [Guiding principles](#guiding-principles)
- [Honest limitations](#honest-limitations)

---

## Why we built this

Konica Minolta receives roughly **130,000 warranty and incentive claims**. Two
realities drove this project:

- **Volume.** Reviewing every claim by hand is impossible, so most claims are never
  examined closely.
- **Efficiency.** Choosing claims at random means auditors spend as long on ordinary
  claims as on suspicious ones.

We also could not simply ask a computer *"is this claim fraudulent?"*, because there
is no reliable, labelled history of which past claims were genuinely fraudulent —
the machine has nothing to copy. So we ask the question we **can** answer:

> **"Is this claim unusual compared with what normally happens — for this claimant,
> this dealer, and this product?"**

---

## The pipeline at a glance

Each stage does one job and hands a clean result to the next.

```
01  Get & clean the data      →  tidy dates and text, create business flags
02  Product timing            →  how often each product is normally claimed
03  Build comparison signals  →  compare each claim to 3 years of history
03b Feature health-check      →  keep only features that carry real information
04  Score every claim         →  risk score, rating, ranking, explanation  ← PRIMARY
05  Four-model second opinion →  corroboration cross-check
    Compare 04 vs 05          →  measure agreement with proper statistics
```

**One rule runs through everything:** no claim is ever allowed to influence its own
baseline. When we work out what is "normal" for a claimant, we use only their
*earlier* claims. A claim can never make itself look normal.

---

## Step 01 — Get and clean the data

**File:** `01_get_data.py`

### What it does

- Reads the raw claims extract.
- Standardises dates and tidies inconsistent text (so the same dealer isn't counted
  as two different dealers).
- Understands the claim structure confirmed with the business:
  - **`ClaimFormID`** = the parent claim form (the invoice)
  - **`ClaimId`** = the individual claim item / product line ← *this is the level we analyse*
- Calculates the filing delay: **days between the product sale and the claim submission**.
- Creates clearly named business flags.

### Key fields created

| Field | Meaning |
|---|---|
| `claim_days_since_sale` | Days between the sale and the claim |
| `claim_file_delay` | 1 if the claim was filed **90 or more days** after the sale |
| `failed_validation` | 1 if the claim was **rejected, a duplicate, or had no match** |
| `rejected_claim`, `returned_claim`, `processed_claim`, `cancelled_claim` | Claim status flags |

### Why the naming matters

Two deliberate renames, requested by the business:

- **`failed_validation`**, *not* "fraud flag" — a rejected or duplicate claim is a
  **validation failure, not proven fraud**. The old name overstated what we know.
- **`claim_file_delay`** — the word *"anomaly"* was removed from feature names so
  each name describes exactly what it measures.

### Effect

This step produces trustworthy data. Without it, every later calculation would be
built on sand. It is also where the plain-English business flags are born, so the
rest of the system can reuse them.

---

## Step 02 — Product timing

**File:** `02_product_velocity.py`

### What it does

For each product, it measures **how often that product is normally claimed each
month**, then compares the current month against that product's own normal level.

- Counts **distinct claims** per product per month (so repeated rows for the same
  claim don't inflate the count).
- Builds the product's normal monthly baseline from **prior months only**.
- Flags a surge only when activity is at least **2× normal** *and* the product has
  at least **3 months** of history.

### Why it matters

Sometimes a problem shows up not as one strange claim, but as a **sudden surge in
how frequently a product is claimed** — even when each individual claim looks
ordinary. A single-claim view would miss this entirely.

### Effect

Adds an early-warning signal at the product level, with a minimum-history guard so
we don't judge a product on thin data.

---

## Step 03 — Build the comparison signals

**File:** `03_build_features_final.py`

This is the analytical heart. For every claim, it looks back over **three years of
history** and compares the claim against several baselines — always excluding the
current claim.

### The five comparison lenses

| Lens | The question it answers |
|---|---|
| **Claimant** | Is this unusual for *this person's* own history? |
| **Dealer** | Is this unusual for *this dealer*? |
| **Product** | Is this unusual for *this product* across everyone? |
| **Recency** | Is the claimant filing faster than their normal pace? |
| **Return behaviour** | Are this claimant's claims returned or rejected more often than average? |

### The two kinds of signal it produces

**Z-score — "how many steps from normal?"**
Measures how many standard steps above or below the historical average a value sits.
Near 0 is normal; 3 or more is clearly unusual. It accounts for how much a value
naturally varies, so it is fair across products with different volatility.

**Ratio — "how many times the usual?"**
A ratio of 1.0 means exactly average; 3.0 means three times the usual.

> **Worked example:** a claimant normally claims 4 units of a product. This claim is
> for 12. The ratio is `12 ÷ 4 = 3.0` — three times their usual.

### Seasonal adjustment

Some products are naturally claimed more in certain months. Without adjusting for
this, a claim in a busy month could look unusual purely because of timing. The
seasonal adjustment re-expresses quantity at an **"average month" level**, so a
normal December claim isn't flagged just because December is busy.

### The critical safeguard — no leakage

Every historical calculation excludes the current claim. In plain terms: **a claim
can never make itself look normal.**

### Effect

Raw data becomes meaningful. By always comparing a claim to what is normal *for that
person, dealer, and product*, a genuinely unusual claim stands out fairly — whether
the claimant is large or small, new or long-established.

---

## Step 03b — Feature health-check

**File:** `03b_feature_profile.py`

### Why this step exists

Step 03 produces many signals, but not every signal is useful. Some columns are
completely empty; some always contain the same value; some are just labels like
names or IDs that must never enter a model. Without a check, weak or inappropriate
columns could quietly influence the results with no explanation.

**This step was added in direct response to a request** for a feature summary showing
the percentage of repeated values, whether a column is entirely null, the number of
unique values, the most common value, and the range — so features could be dropped
with evidence rather than intuition.

### What it reports for every column

- **% blank** (100% = entirely empty)
- **% that repeat the same value**
- **Number of unique values**
- **Most common value (mode)** and **range**
- A recommended action: `keep` / `drop-null` / `drop-constant` / `drop-id` / `drop-date`
- A simple `use_in_model` flag (1 = feed to the model, 0 = don't)

### How Step 04 uses it

Step 04 **reads this report** and keeps only the columns marked `use_in_model = 1`.
So the model's inputs are chosen from **evidence**, not a hand-typed list, and
identifiers, names and dates can never accidentally reach the model.

### What it caught — a real finding

Serialised products are recorded **one serial number at a time**, so their quantity
is almost always exactly 1 — even when several units were sold. This means
*"quantity"* carries very little information for those products. Surfacing this let
us handle it properly instead of being quietly misled by it.

### Effect

Three concrete benefits:

- **Auditability** — every feature kept or dropped has a documented reason.
- **Quality** — empty and constant columns are removed before they weaken the model.
- **Safety** — identifiers, names and dates are *provably* excluded.

---

## Step 04 — Score every claim

**File:** `04_riskscore_isolationforest.py` — **this is the primary scorer**

This step has three parts: score each claim, rank it, and explain it. Then it rolls
claims up into one fair rating per claimant and per dealer.

### Part A — Score each claim

**The model: Isolation Forest.** In plain terms: imagine repeatedly splitting the
claims into smaller and smaller random groups. An **ordinary** claim sits in the
crowd and takes many splits to separate. An **unusual** claim is off on its own and
gets isolated in just a few splits. Claims that are easy to isolate are the unusual
ones. The model does this thousands of times and turns the result into a **0–100
risk score**.

**Why this model:** it needs no examples of past fraud (we have none), handles many
signals at once, and is built to find rare, unusual cases — exactly what an audit
queue needs.

**Turning the score into a rating:**
- Top 10% of scores → **High**, next 15% → **Medium**, the rest → **Low**.
- **Business-rule safety net:** any claim that was **rejected** or **failed
  validation** is always promoted to **High**, regardless of its score.

> **Note on the 10%:** "High" is the top 10% of *all* claims, not the top 10% of each
> claimant's claims. This is deliberate — it means a claimant with only ordinary
> claims can legitimately have zero High claims, and it is what makes the entity-level
> comparison below meaningful.

### Part B — Rank each claim

Added at stakeholder request, so reviewers can size their own audit queue:

| Column | What it gives you |
|---|---|
| `risk_score` | 0–100, how unusual the claim is |
| `risk_score_1000` | The same risk on a **1–1000 scale** |
| `risk_percentile` | 99.4 = riskier than 99.4% of all claims |
| `risk_rank` | 1 = the single riskiest claim in the file |
| `audit_band` | Top 1% / Top 5% / Top 10% / Top 20% / Top 25% |
| `in_top_1pct` … `in_top_25pct` | Filter to `1` for an instant review queue |

If a team can review 200 claims this month, they simply take `risk_rank` 1–200.

### Part C — Explain each claim

Every claim gets a plain-English explanation, plus the **field name and value** behind
it so any reason can be verified on the same row.

```
Why this rating : Rated HIGH because quantity was 3.6x this claimant's usual for the
                  product; quantity far above the product norm (4.4 std above).
Driver 1        : r_prod_avg_qty_claim = 3.65 -> quantity was 3.6x this claimant's usual
Driver 2        : quantity_z_score = 4.38 -> quantity far above the product norm
Features used   : r_prod_avg_qty_claim, quantity_z_score, z_claimant_avg_total_qty, ...
```

This directly answers the question *"why did two claims from the same person score
differently?"* — the model reacts to the **size** of each signal, so one claim may be
4× the claimant's usual quantity while another is about average.

### Part D — One fair rating per claimant and dealer

A person can have many claims. To give them **one** overall rating, we must be
careful: because "High" is the top 10% by design, about 10% of *everyone's* claims
are High. So simply **counting** High claims would unfairly flag anyone who files a
lot of claims.

**The fix — the Wilson score.** We judge each entity by their **rate** of High
claims, compared to the population baseline, using a cautious estimate that accounts
for how many claims they filed.

> **Worked example.** Two claimants each have **11 High claims**:
>
> | Claimant | Total claims | High rate | Rating |
> |---|---:|---:|---|
> | Clinton Parker | 116 | 9.5% (≈ normal) | **Low** ✓ |
> | Genuinely elevated | 40 | 27.5% (clearly above) | **High** ✓ |
>
> Same raw count, correct opposite ratings — because we judge the **rate**, not the
> volume.

### Effect

This is the ranked review list auditors actually use. The score focuses attention,
the business rules ensure known problems always surface, the ranking lets reviewers
size their own queue, and the written reason makes every flag defensible.

---

## Step 05 — Four-model second opinion

**File:** `05_riskscore_4models_Final_output.py`

### What it does

Runs **four independent anomaly detectors** on the same claims and reports **how many
of them agree** a claim is unusual.

| Method | How it spots something unusual |
|---|---|
| **Isolation Forest** | Easy to separate from the crowd (our primary model) |
| **Local Outlier Factor** | Doesn't fit its nearest neighbours |
| **KMeans distance** | Far from its group's centre |
| **One-Class SVM** | Falls outside a learned "normal" boundary |

*Everyday analogy:* like asking four independent doctors to examine the same patient.
If several agree on a concern, you trust it far more.

### An honest finding

We tested this on data with **deliberately planted, known anomalies** and found that
simply **averaging** the four scores can sometimes **dilute** a strong signal that one
good model catches — if the weaker models don't corroborate it. In one test the
average caught only 9 of 20 planted anomalies while Isolation Forest alone caught all
20.

**So we changed the design:**
- The headline output is **`models_agreeing`** (0–4), not the averaged score. A claim
  flagged by 2 or more independent methods is high-confidence.
- Isolation Forest (Step 04) remains the **primary** rating.
- Step 05 is framed as a **corroboration cross-check**, not a replacement.

### Effect

Adds a confidence dimension: where several independent methods agree, you can act
with more certainty.

---

## Comparing Step 04 and Step 05

**Files:** `compare_outputs.py`, `export_comparison_to_excel.py`

Both steps use the identical fair consolidation (Wilson); only the scoring engine
differs. We compared their final ratings using three established statistics.

### Results on the real data (1,990 claimants)

| Measure | Value | Meaning |
|---|---|---|
| Exact agreement | 1,864 of 1,990 = **93.7%** | Both methods gave the same rating |
| **Cohen's kappa** | **0.72** | **Substantial** agreement (beyond chance) |
| McNemar's test | **p = 0.75** | Neither method is systematically stricter |
| Disagreements | **126 claimants** | The priority review list |

### Recommendation

- Keep **Isolation Forest (Step 04)** as the **primary** rating.
- Use **Step 05** as corroboration; lead with "how many models agree".
- **Review the 126 disagreement cases first** — agreement means confidence,
  disagreement means "look closer".

---

## Understanding the output

The main output has one row per claim, with **ClaimantName as the first column**,
followed by the key identifiers, the risk results, and then every remaining source
field and engineered feature. Nothing is hidden — a rating can be traced back to the
raw values on the same row.

### The six risk columns

| Column | Answers | Example |
|---|---|---|
| `risk_level` | The **final** rating an auditor acts on | High |
| `risk_level_model` | What the model said **before** business rules | High |
| `risk_score` | How unusual? (0–100) | 76.84 |
| `risk_score_1000` | The same, on a finer 1–1000 scale | 769 |
| `risk_percentile` | Riskier than what share of claims? | 92.47% |
| `risk_rank` | What position in the review queue? | 227th |

**When `risk_level` differs from `risk_level_model`,** a business rule fired — the
claim was rejected or failed validation, so it was escalated to High.

### The explanation columns

| Column | What it gives you |
|---|---|
| `Why this rating` | A full plain-English sentence |
| `Driver 1 / 2 / 3 (field = value)` | The field name and value behind the reason |
| `features_used_for_rating` | Every field that fired for this claim |

### Excel sheets

| Sheet | Purpose |
|---|---|
| **Read me** | How to use every column |
| **Why each claim** | One row per claim, full detail |
| **Claimant view** | Grouped by claimant, High first — compare their claims directly |
| **By claimant** | One consolidated rating per person |
| **By dealer** | One consolidated rating per dealer |

---

## The statistics, in plain language

### `raw_high_rate` = High claims ÷ total claims
The plain share of an entity's claims that were rated High (e.g. 6 of 116 = 5.2%).
The honest starting number — but it can mislead on small samples (2 of 3 = 67% looks
alarming but is only 3 claims). Shown for transparency; the **decision** uses Wilson.

### Wilson score lower bound — the rating rule
A cautious, sample-size-aware estimate of an entity's *true* High rate. With few
claims it stays low (we're not sure yet); with many claims it tightens around the
observed rate. An entity is rated High only if even this cautious estimate is above
the population baseline.

> *Analogy:* a batting average — one hit in two at-bats doesn't make a great hitter;
> you wait for enough at-bats.
> *(Wilson 1927; recommended over simpler methods by Brown, Cai & DasGupta 2001.)*

### Cohen's kappa — agreement beyond chance
Plain agreement percentages can look high purely by luck, so kappa corrects for
chance. 0 = chance-level; 0.41–0.60 = moderate; 0.61–0.80 = **substantial**;
0.81–1.00 = almost perfect. Our claimant kappa was **0.72**.
*(Cohen 1960; scale from Landis & Koch 1977.)*

### McNemar's test — is one method stricter?
Checks whether one method is consistently harsher about calling entities High, or
whether they simply differ in both directions by chance. A p-value above 0.05 means
no systematic difference. Ours was **p = 0.75**.
*(McNemar 1947.)*

### Supporting cross-checks
- **Empirical Bayes** — gently pulls small-sample rates toward the population average
  until there is enough evidence, and reports the probability the true rate exceeds
  the baseline. *(Efron & Morris 1975.)*
- **Binomial test** — asks whether an entity's number of High claims could have
  happened by chance. *(Clopper & Pearson 1934.)*

### The algorithms, one line each
- **Isolation Forest** — finds points that are quick to isolate *(Liu, Ting & Zhou, 2008)*
- **Local Outlier Factor** — finds points in low-density neighbourhoods *(Breunig et al., 2000)*
- **One-Class SVM** — learns a boundary around "normal" *(Schölkopf et al., 2001)*
- **KMeans distance** — flags points far from their cluster centre (a corroborator)

---

## How to run

```bash
python 01_get_data.py
python 02_product_velocity.py
python 03_build_features_final.py
python 03b_feature_profile.py
python 04_riskscore_isolationforest.py        # primary
python 05_riskscore_4models_Final_output.py   # corroboration

# comparison and interactive workbook
python compare_outputs.py
python export_comparison_to_excel.py
```

### Requirements

```
pandas, numpy, scipy, scikit-learn, pyarrow, openpyxl
```

---

## Output files

**Step 04 (primary)**
- `04_claims_riskscored.parquet` / `.xlsx` — one row per claim, full detail
- `04_claims_riskscored_by_claimant.parquet` — one rating per claimant
- `04_claims_riskscored_by_dealer.parquet` — one rating per dealer

**Step 05 (corroboration)**
- `05_claims_riskscored_4models.parquet` / `.xlsx` — per claim, plus each model's
  score and the agreement count
- `05_claims_riskscored_4models_by_claimant.parquet`
- `05_claims_riskscored_4models_by_dealer.parquet`

**Comparison**
- `KM_Step4_vs_Step5_Comparison_Interactive.xlsx` — interactive workbook with live
  agreement statistics and a "Statistics Explained" tab

---

## Guiding principles

- **No leakage** — every historical calculation excludes the current claim.
- **No identifiers in the model** — names, IDs, serial numbers and dates are never
  used to score. Outcome flags (rejected, returned, cancelled, failed validation)
  drive business rules only, never the model.
- **Business rules on top of the model** — known failures always escalate.
- **Statistically fair consolidation** — entities are judged by *rate* against a
  baseline, not by raw counts.
- **Explainable at every level** — a plain-language reason accompanies every rating,
  with the field name and value so it can be verified.
- **Not a verdict** — ratings point auditors where to look first; humans decide.

---

## Honest limitations

- **Claim quantity definition.** Quantity is a shared denominator across several
  z-scores and ratios, and its definition involves judgment calls (serialised products
  record quantity as 1; claim-item vs claim-form level). This is with the business for
  confirmation. A sensitivity test is available to measure how much the choice
  actually changes outcomes.
- **Method comparison used synthetic data.** To compare methods fairly we used data
  with deliberately planted anomalies, because no confirmed fraud labels exist. The
  *rankings and findings* are robust; exact percentages will differ on real data. The
  agreement figures quoted above (1,990 claimants, 93.7%, kappa 0.72) are from the
  **actual** output.
- **`risk_rank` and `risk_percentile` are file-relative.** They describe position
  *within this batch*. `risk_score` and `risk_score_1000` are more stable for
  comparing across runs.
- **A high z-score, ratio or reject rate is a reason to look closer** — not proof of
  wrongdoing. Large legitimate orders, bulk purchases and promotions can all produce
  high values innocently.
