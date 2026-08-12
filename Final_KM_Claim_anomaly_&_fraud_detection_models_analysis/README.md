# KM Claims Audit & Fraud Detection — Modular Pipeline

A sequential, file-per-step pipeline. Each numbered script reads the previous
step's saved dataframe, does one job, and saves its own output — so you can
**pick up any intermediate dataframe and reuse it** (or try a different
algorithm on it) without re-running everything.

---

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

---

## Data flow

```
raw .xlsx
   │  01_get_data
   ▼
data/01_claims_base.parquet
   │  02_product_velocity
   ▼
data/02_claims_velocity.parquet
   │  03_build_features
   ▼
data/03_claims_features.parquet   ← model-ready hand-off point
   ├── 04_riskscore_isolationforest ─► data/04_claimant_risk / dealer_risk / claims_ranked
   └── 05_riskscore_4models         ─► data/05_claimant_multi / dealer_multi / claims_multi
```

Both modeling steps read the **same** `03_claims_features.parquet` and the
**same** `entity_features.py`, so every algorithm is compared on identical inputs.

---

## Quick start

```bash
# put your Excel next to the scripts, or pass --input
python run_pipeline.py                 # steps 1-4 (Isolation Forest)
python run_pipeline.py --multimodel    # also run step 5 (4-model compare)
python run_pipeline.py --input my.xlsx --contamination 0.03
```

Run any step on its own:

```bash
python "03_build_features.py"          # rebuild features only
python "04_riskscore_isolationforest.py"
```

---

## Naming convention (project-wide)

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

---

## Reusing an intermediate dataframe

Every hand-off is a parquet file with preserved dtypes:

```python
import pandas as pd
feats = pd.read_parquet('data/03_claims_features.parquet')
# feats is model-ready — try any algorithm on it
```

---

## Adding a new algorithm

Copy `05_riskscore_4models.py` to `06_riskscore_yourmodel.py`, read
`data/03_claims_features.parquet`, build entity tables with
`entity_features.build_claimant_features / build_dealer_features`, score, save.
It will compare fairly against steps 4 and 5.

---

## Note on the numbered filenames

Python can't `import` a module whose name starts with a digit, so
`run_pipeline.py` loads the numbered files via `importlib` and calls their
functions. The shared `entity_features.py` (no digit) imports normally.

---

## Key limitation

`fraud_flag` is a **rule-based proxy** derived from claim status — not
investigator-confirmed fraud. All risk scores are an audit-**triage** aid: they
prioritise where to look, they do not determine fraud. Capturing confirmed
audit outcomes is the highest-value next step (enables supervised modelling).
