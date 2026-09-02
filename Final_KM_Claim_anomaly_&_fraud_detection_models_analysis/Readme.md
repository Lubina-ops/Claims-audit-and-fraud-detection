# KM Claims Fraud & Anomaly Detection — Pipeline README

A plain-language guide to the whole system, from raw claims to final risk ratings.
Written so both technical and non-technical readers can follow it.

---

## The Big Picture

Konica Minolta receives a very large number of warranty and incentive claims — far
too many to review by hand. This system automatically highlights the claims and the
claimants/dealers that look **unusual**, so auditors can focus their limited time on
the cases most worth a closer look.

> **Important:** the system flags what is *unusual*, not what is *proven fraud*.
> Every flagged case still needs a human to review it. "Unusual" is where an auditor
> should look first, not a final verdict.

The pipeline runs as a sequence of steps. Each step does one job and passes a clean
result to the next:

```
01_get_data           →  clean the raw claims, create basic business flags
02_product_velocity   →  learn how often each product is normally claimed
03_build_features     →  compare each claim to history (the anomaly signals)
03b_feature_profile   →  health-check every feature; decide what to keep
04_riskscore_isoforest→  score each claim (Isolation Forest) + one rating per entity
05_riskscore_4models  →  second opinion: four models + agreement, same statistics
compare (04 vs 05)    →  measure how much the two methods agree, with statistics
```

---

## Step 01 — Get & Clean the Data  (`01_get_data.py`)

**What it does**
- Reads the raw claims extract.
- Standardises dates and fixes inconsistent text.
- Uses the correct claim structure: **ClaimFormID** = the parent claim form (invoice),
  **ClaimId** = the individual claim item (product line).
- Calculates basic facts and creates clearly-named business flags.

**Key fields created**
| Field | Meaning |
|---|---|
| `claim_days_since_sale` | Days between the product sale and the claim submission |
| `claim_file_delay` | Flag = 1 when a claim is filed 90+ days after sale |
| `failed_validation` | Flag = 1 when a claim was rejected, duplicated, or had no match |
| `rejected_claim`, `returned_claim`, `processed_claim`, `cancelled_claim` | Status flags |

**Why it matters**
Raw data arrives messy. If dates or text are inconsistent, every later step is built
on sand. This step makes the foundation trustworthy and is where the plain-English
business flags are born (renamed for clarity — e.g. `failed_validation` instead of
"fraud flag", because a rejected claim is a validation failure, not proven fraud).

**Algorithm / logic:** simple rule-based cleaning and date arithmetic. No model yet.

---

## Step 02 — Product Timing  (`02_product_velocity.py`)

**What it does**
- For each product, measures how often it is **normally** claimed each month.
- Compares the current month against that product's own normal level.

**Why it matters**
Sometimes a problem shows up not as one strange claim, but as a **surge in how often**
a product is claimed — even when each individual claim looks ordinary. This step gives
an early-warning signal that a single-claim view would miss.

**Algorithm / logic:** counting and averaging (claims per product per month), with the
current month excluded from its own baseline so nothing "cheats."

---

## Step 03 — Build the Comparison Signals  (`03_build_features.py`)

This is the analytical heart. For every claim, it looks back over the prior 3 years and
compares the claim against several baselines — always excluding the current claim so it
never influences its own comparison.

**The five comparison "lenses"**
| Lens | Question it answers |
|---|---|
| Claimant | Is this unusual for *this person's* own history? |
| Dealer | Is this unusual for *this dealer*? |
| Product | Is this unusual for *this product* across everyone? |
| Recency | Is the claimant filing faster than their normal pace? |
| Return behaviour | Are this claimant's claims returned more than the population? |

**Two kinds of signal it produces**
- **Z-score** — how many "standard steps" above or below normal a value is.
  Near 0 = normal; 3+ = clearly unusual. It accounts for how much a value naturally varies.
- **Ratio** — how many times bigger than normal. A ratio of 3.0 means three times the usual.

It also produces **seasonal adjustment** features, so a claim in a naturally busy month
is not flagged just because of the time of year.

**Why it matters**
A raw number means nothing without context. Comparing each claim to what is normal *for
that person, dealer, and product* is what makes a genuinely unusual claim stand out —
fairly, whether the claimant is large or small, new or long-established.

**Algorithm / logic:** rolling historical statistics (means, standard deviations),
z-scores, ratios, and seasonal indices. Still no machine-learning model — these are the
*inputs* the model will use.

---

## Step 03b — Feature Health-Check  (`03b_feature_profile.py`)

**What it does**
Before any modelling, it produces a one-page health report of every feature:
- how often it is blank (and whether it is 100% empty),
- how often it repeats the same value,
- how many different values it has,
- its most common value and range,
- a recommended action: **keep / drop-constant / drop-null / drop-id / drop-date**.

**Why it matters**
This gives an evidence trail for every feature we keep or remove, instead of relying on
judgement. It also caught a real issue: **serialised products almost always show a
quantity of 1** (because they are recorded one serial number at a time), so quantity
carries little information for them — something we can now handle correctly.

**Algorithm / logic:** descriptive statistics per column + simple keep/drop rules.

---

## Step 04 — Score Each Claim (Isolation Forest)  (`04_riskscore_isolationforest.py`)

This step has **two parts**.

### Part A — Score each claim

**The model: Isolation Forest.** Imagine repeatedly splitting the claims into smaller
and smaller random groups. An ordinary claim sits in the crowd and takes many splits to
separate; an unusual claim is "off on its own" and gets isolated in just a few splits.
Claims that are easy to isolate are the unusual ones. The model does this thousands of
times and averages the result into a **risk score from 0 to 100**.

**Why this model:** it needs no examples of past fraud (we have none), handles many
signals at once, and is designed to find rare, unusual cases — exactly what an audit
queue needs.

**Turning the score into a label**
- Top 10% of scores → **High**, next 15% → **Medium**, the rest → **Low**.
- **Business-rule safety net:** any claim that was rejected or failed validation is
  always promoted to **High**, regardless of its score.

**Explainability:** every claim gets a plain-language `risk_reason`, e.g.
*"product quantity far above product norm; filed 90+ days after sale."*

### Part B — One rating per claimant and per dealer

A single claimant can have many claims. To give **one** overall rating fairly, we cannot
just count High claims — because "High" is the top 10% by design, ~10% of *everyone's*
claims are High, so counting would just flag high-volume claimants.

Instead we use the **Wilson score lower bound** (see Statistics below): we rate an entity
High only if its *rate* of High claims is convincingly above the ~10% baseline, after
allowing for how many claims they filed.

**Outputs (three views):**
- `claims_individual` — one row per claim (score, level, reason)
- `by_claimant` — one consolidated rating per claimant
- `by_dealer` — one consolidated rating per dealer

---

## Step 05 — Four-Model Second Opinion  (`05_riskscore_4models_Final_output.py`)

**What it does**
Runs **four** independent anomaly detectors on the same claims and reports **how many of
them agree** a claim is unusual. It then applies the **same** business rules and the
**same** Wilson consolidation as Step 4.

| Model | How it spots "unusual" |
|---|---|
| Isolation Forest | Globally unusual (easy to isolate) — the primary model |
| Local Outlier Factor | Unusual vs its nearest neighbours |
| KMeans distance | Far from its cluster's centre |
| One-Class SVM | Outside a learned "normal" boundary (least reliable here) |

**Why four models**
No single method is perfect. Like asking four independent doctors to look at the same
patient — if several agree on a concern, you trust it far more. The most reliable output
of this step is **`models_agreeing`** (0–4): a claim flagged by 2+ methods is high-confidence.

**An honest finding (from controlled testing):** simply averaging the four scores can
sometimes *dilute* a strong signal that one good model catches. So this step is a
**corroboration cross-check**, not a replacement for Step 4, and we lead with the
agreement count rather than the averaged score. The least reliable model (One-Class SVM)
is trusted less in the blend.

**Outputs:** same three-view structure as Step 4 (`_by_claimant`, `_by_dealer`, and the
per-claim file with each model's score and the agreement count).

---

## Comparing Step 04 and Step 05

**What we did**
Both steps use the identical fair consolidation (Wilson); only the scoring engine differs
(one model vs four). We compared the final claimant and dealer ratings using three
established statistics.

**Results on the real data (1,990 claimants)**
| Measure | Value | Meaning |
|---|---|---|
| Exact agreement | 1,864 of 1,990 = 93.7% | Both methods gave the same rating |
| Cohen's kappa | 0.72 | **Substantial** agreement (beyond chance) |
| McNemar's test | p = 0.75 | Neither method is systematically stricter |
| Disagreements | 126 claimants | The priority review list |

**Example — Clinton Parker:** 116 claims, rated **Low by both** methods (correct — his
High rate is around the normal baseline, so he is not flagged just for high volume).

**Recommendation**
- Keep **Isolation Forest (Step 4)** as the primary rating.
- Use **Step 5** as corroboration; lead with "how many models agree."
- Review the **126 disagreement cases first** — agreement = confidence, disagreement = look closer.

---

## The Statistics, In Plain Language

### raw_high_rate = high_count ÷ claim_count
The plain share of an entity's claims that were rated High (e.g. 6 of 116 = 5.2%). The
honest starting number — but it can mislead on small samples (2 of 3 = 67% looks alarming
but is only 3 claims). Shown for transparency; the decision uses Wilson.

### Wilson score lower bound — the rating rule
A cautious, sample-size-aware estimate of an entity's *true* High rate. With few claims it
stays low (we're not sure yet); with many claims it tightens around the observed rate. We
rate an entity High only if even this cautious estimate is above the ~10% baseline.
*Analogy: a batting average — one hit in two at-bats doesn't make a great hitter; you wait
for enough at-bats.*  (Wilson 1927; Brown, Cai & DasGupta 2001.)

### Cohen's kappa — agreement beyond chance
Plain agreement % can look high by luck, so kappa corrects for chance. 0 = chance-level;
0.41–0.60 = moderate; 0.61–0.80 = substantial; 0.81–1.00 = almost perfect. Our claimant
kappa was **0.72 — substantial**.  (Cohen 1960; scale from Landis & Koch 1977.)

### McNemar's test — is one method stricter?
Checks whether one method is consistently harsher about calling entities High, or whether
differences are just chance. p above 0.05 = no systematic difference. Ours was **p = 0.75**.
*Analogy: two graders marking the same essays — does one consistently give lower marks?*
(McNemar 1947.)

### Supporting cross-checks
- **Empirical Bayes** — gently shrinks small-sample rates toward the population average
  until there is enough evidence, and reports the probability the true rate exceeds the baseline.
- **Binomial test** — asks whether an entity's number of High claims could have happened by chance.

---

## The Algorithms, In One Line Each
- **Isolation Forest** — finds points that are quick to isolate (Liu, Ting & Zhou, 2008).
- **Local Outlier Factor** — finds points in low-density neighbourhoods (Breunig et al., 2000).
- **One-Class SVM** — learns a boundary around "normal" (Schölkopf et al., 2001).
- **KMeans distance** — flags points far from their cluster centre (a corroborator).

---

## How to Run the Pipeline

```bash
python 01_get_data.py
python 02_product_velocity.py
python 03_build_features_final.py
python 03b_feature_profile.py
python 04_riskscore_isolationforest.py
python 05_riskscore_4models_Final_output.py

# comparison + interactive workbook
python compare_outputs.py
python export_comparison_to_excel.py
```

---

## Output Files

**Step 4 (Isolation Forest)**
- `04_claims_riskscored.parquet / .xlsx` — one row per claim (score, level, reason)
- `04_claims_riskscored_by_claimant.parquet` — one rating per claimant
- `04_claims_riskscored_by_dealer.parquet` — one rating per dealer

**Step 5 (four models)**
- `05_claims_riskscored_4models.parquet / .xlsx` — per claim + each model's score + agreement count
- `05_claims_riskscored_4models_by_claimant.parquet` — one rating per claimant
- `05_claims_riskscored_4models_by_dealer.parquet` — one rating per dealer

**Comparison**
- `KM_Step4_vs_Step5_Comparison_Interactive.xlsx` — interactive workbook
  (Ratings, Summary with live Cohen's kappa, Disagreements, and a Statistics Explained tab)

---

## Guiding Principles
- **No leakage** — every historical calculation excludes the current claim.
- **No identifiers in the model** — only behavioural signals (never names, IDs, or dates).
- **Business rules on top of the model** — hard failures always escalate.
- **Statistically fair consolidation** — rate-vs-baseline, not raw counts.
- **Explainable at every level** — a reason accompanies every rating.
- **Not a verdict** — ratings point auditors where to look first; humans decide.

---

## Honest Caveats
- The method comparison used data with **known, planted anomalies** to establish how the
  approaches behave (the correct way to compare methods when no confirmed fraud labels
  exist). The real-data figures (1,990 claimants, 93.7%, kappa 0.72) are from the actual output.
- A high z-score, ratio, or reject rate is a **reason to look closer**, not proof of wrongdoing.
