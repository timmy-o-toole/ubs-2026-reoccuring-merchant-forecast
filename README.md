# Next Recurring Merchant Forecast (UBS, Swiss AI Week 2026)

**Task:** for each bank client, predict which subscription family recurs next
in the 90 days after `2026-01-01`:
`cloud, gym, insurance, mobile, music, software, streaming` or `none`.
Metric: macro-F1 over the 8 labels.

**Our idea:** find the right features, keep the model lean, make every
forecast explainable. We call the model **SparseLevels**: one sparse
logistic regression per level (L1 or elastic net). Here the levels are the 8
labels and we use L1 (lasso).

## Method at a glance

```
                     ┌──────────────────────────────────────┐
                     │           Raw transactions           │
                     │  amount · date · MCC · description   │
                     └──────────────────┬───────────────────┘
                                        │
  ══════════════════════ 1 · FEATURE CREATION ══════════════════════
                                        │
                     ┌──────────────────▼───────────────────┐
                     │      Tag subscription payments       │
                     │  (cloud, gym, insurance, mobile …)   │
                     └──────────────────┬───────────────────┘
                     ┌──────────────────▼───────────────────┐
                     │       Detect recurring streams       │
                     │    (same amount, roughly monthly)    │
                     └──────────────────┬───────────────────┘
                                        │
      ┌─────────────┬─────────────┬─────┴───────┬─────────────┐
      ▼             ▼             ▼             ▼             ▼
 ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
 │ Status  │   │ Habits  │   │ History │   │Portfolio│   │ Account │
 │& timing │   │ recent  │   │  age,   │   │ changes │   │behaviour│
 │live? due│   │payments │   │ charges │   │& refunds│   │& travel │
 └────┬────┘   └────┬────┘   └────┬────┘   └────┬────┘   └────┬────┘
      └─────────────┴─────────────┼─────────────┴─────────────┘
                                  ▼
                       103 features per client
                                  │
  ═════ 2 · SPARSELEVELS: ONE SPARSE LOGISTIC REGRESSION PER LABEL ══════
                                  │
   ┌───────┬───────┬───────┬──────┴┬───────┬───────┬───────┐
   ▼       ▼       ▼       ▼       ▼       ▼       ▼       ▼
┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐
│cloud│ │ gym │ │insur│ │mobil│ │music│ │softw│ │strea│ │none │
│  ?  │ │  ?  │ │  ?  │ │  ?  │ │  ?  │ │  ?  │ │  ?  │ │  ?  │
└──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘
 p=.12   p=.61   p=.08   p=.05   p=.03   p=.04   p=.02   p=.21
   └───────┴───────┴───────┴───┬───┴───────┴───────┴───────┘
                               ▼
  ══════════════════════ 3 · DECISION ══════════════════════
                               ▼
                Highest probability wins → "gym"
```

### 1 · Feature creation
Raw transactions are turned into 103 readable features per client, for example
*"recent gym payments"*, *"is the insurance still running"*, *"which
subscription is due first"*, *"does the client travel"*.

| Feature block | What it captures | Importance* |
|---|---|---|
| Subscription status & timing | still live? overdue? due first? short trial? | 22.5 |
| Habits | recent payments per family (last 90 days) | 13.5 |
| Subscription history | age, number of charges, amount per family | 4.4 |
| Portfolio changes & refunds | families gained or dropped, refunds | 3.6 |
| Account behaviour | payment mix, top-ups, merchant variety, travel | 2.9 |

\*macro-F1 points lost when the block is removed.

### 2 · SparseLevels: one sparse logistic regression per label
Each of the 8 labels gets its own yes/no model ("is it gym?"). A sparse
penalty sets unhelpful coefficients to zero; we use L1 (lasso), elastic net is
the optional alternative. So each label keeps its own
feature list (between 23 features for music and 80 for `none`, out of
103). Each label picks its own penalty strength by
cross-validation.

### 3 · Decision
The label with the highest probability is the forecast. Because every label
has a short list of coefficients, each prediction can be explained, e.g.
*gym is likely because of recent gym payments (odds ×2.5) and a long-running
gym subscription (×1.8)*.

## Result

**Macro-F1 0.504** on the validation set (1,000 clients the model has never seen).
Training data: only the 2,000 labelled train clients.

### Reference: other model types on the same data

Same 103 features, trained on the 2,000 train clients, macro-F1 on valid.
*Per label* = one yes/no model per label, highest probability wins (like
SparseLevels); *global* = one multiclass model. Fit time = training only, on a
16-core laptop.

| Setup | Model | Macro-F1 (valid) | Fit time |
|---|---|---|---|
| per label | **SparseLevels, L1 logistic regression (ours)** | **0.504** | 6.5 s |
| per label | Random forest | 0.491 | 5.4 s |
| per label | Gradient boosting (HGB) | 0.491 | 26.2 s |
| per label | XGBoost | 0.484 | 4.9 s |
| global | Logistic regression | 0.489 | 0.3 s |
| global | Random forest | 0.480 | 0.7 s |
| global | Gradient boosting (HGB) | 0.482 | 24.6 s |
| global | XGBoost | 0.476 | 4.2 s |

SparseLevels is the most accurate here and fast enough; the tree models are
not tuned and differences below ~0.01-0.02 are within noise. The SparseLevels
time includes its own penalty search per label. Reproduce with
`py -3.10 extra_info/benchmark_models.py` (needs `xgboost`).

## Repository: three independent parts

```
features/   raw transactions (a table) -> one feature row per client
model/      the model: SparseLevels, one sparse logistic regression per level (L1 or elastic net), generic
pipeline/   our task: data paths, labels, training set, evaluation, submission
data/       challenge data (dataset.zip)
extra_info/ experiment log, model benchmark, verified interpretations, coefficients per label
```

`features/` and `model/` never read files and know nothing about our data
paths; only `pipeline/` does. So you can reuse them for a similar task.

## Use it on similar data

**Transactions -> forecast.** Your transactions need the columns `client_id,
timestamp, amount, direction, type, mcc, description` (history up to your
cutoff); `y` is the label per client.

```python
import pandas as pd
from features import build_features, select_features
from model import fit_sparse_levels

tx = pd.read_json("my_transactions.jsonl", lines=True, dtype={"mcc": str})
F = build_features(tx, cutoff_date="2026-01-01")          # one row per client
X = select_features(F.drop(columns="client_id"), "lean")  # the 103 model features
model = fit_sparse_levels(X, y)                           # y: label per client (same order as F)
model.predict(X)
```

**Any feature table -> any levels.** The model alone works on any table and
any levels (class labels, cluster ids, customer segments):

```python
from model import fit_sparse_levels

model = fit_sparse_levels(X, y)            # X: feature table, y: level per row
model.predict(X_new)                       # predicted level per row
model.predict_proba(X_new)                 # probability per level (each row sums to 1)
model.coefficients(top=5)                  # the features each level uses (odds ratios)
model.explain(X_new.iloc[[0]])             # why this row got its prediction
```

Options: `Cs=(...)` candidate penalty strengths (one value = fixed penalty),
`levels=[...]` fixed order of the levels, `penalty="elasticnet"` with `l1_ratio`
as an optional alternative to L1 (not used in our final model).
`py model/sparse_levels.py` runs a small demo. The file is self-contained
(numpy, pandas, scikit-learn), so you can also copy it on its own.

## Run our pipeline

```bash
pip install numpy pandas "scikit-learn>=1.0"   # Python >= 3.10
unzip data/dataset.zip -d data/
py -3.10 -m pipeline.evaluate "my run"         # train, score on validation, log
py -3.10 -m pipeline.make_submission           # -> data/submission_final.csv
```

Challenge data and task: [Swiss-ai-Weeks/ubs-2026](https://github.com/Swiss-ai-Weeks/ubs-2026) (Apache 2.0).
