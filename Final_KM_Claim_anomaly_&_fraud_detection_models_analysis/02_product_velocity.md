# 02 — Product Velocity

**Step 2 of the pipeline**

Adds the product-velocity feature. Reads `data/01_claims_base.parquet`,
writes `data/02_claims_velocity.parquet`.

---

## What it measures

Existing product features (added later in step 3) measure **quantity per
claim**. Product velocity measures something genuinely different —
**claims per month**: a surge in how *often* a product is claimed, even when
every individual quantity looks normal.

> Example: a product normally sees ~50 claims/month across all dealers. This
> month it sees 180 — with every quantity a perfectly normal 1 unit. No
> single claim looks odd; the **aggregate surge** is the signal.

---

## Columns produced

| Column | Meaning |
|---|---|
| `product_month_claims` | Claims for this product in this month |
| `product_avg_month_claims` | That product's typical claims-per-month |
| `product_active_months` | Distinct months the product appeared in claims |
| `product_velocity_ratio` | `month_claims / avg_month_claims` (continuous) |
| `product_velocity_spike_flag` | `1` if ratio ≥ 2.0 **and** active_months ≥ 3 |

---

## How it works

1. Count **claims per product per month**.
2. For each product, compute its **own** monthly average and active-month count.
3. `product_velocity_ratio` = this month ÷ a normal month.
4. Flag a spike only when the ratio is high **and** the product has ≥ 3 months
   of history (so brand-new products don't produce meaningless ratios).

`VELOCITY_MIN_MONTHS = 3` is a **noise guard**, not a tuning dial — it only
excludes products with too little history. `VELOCITY_MULTIPLIER = 2.0` is the
dial that controls sensitivity.

---

## Kept separate from `anamoly_flag` — on purpose

Folding velocity into `anamoly_flag` was tested and **rejected**: it pushed the
flag to ~31% coverage and collapsed its fraud-lift from 2.56x to ~1.26x.

Instead, velocity is exposed as a **continuous feature** (`product_velocity_ratio`)
plus a spike flag, so the anomaly model can weigh it on its own merits. The
continuous ratio matters most — Isolation Forest uses gradation better than a
binary cut.

> The step prints a lift check. If the spike flag shows **< 1.5x** lift on your
> data, rely on the continuous ratio and consider dropping the binary flag.

---

## Functions

| Function | Purpose |
|---|---|
| `add_product_velocity(df, multiplier, min_months)` | Add the five velocity columns |
| `report_velocity_lift(df)` | Print the spike-vs-fraud lift check |
| `run(input, output)` | Read step-1 output, add velocity, save |

---

## Run it

```bash
python "02_product_velocity.py"
```

**Input:** `data/01_claims_base.parquet`
**Output:** `data/02_claims_velocity.parquet`
