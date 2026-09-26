# Next Recurring Merchant Forecast (UBS, Swiss AI Week 2026)

**Task:** for each bank client, predict which subscription family recurs next
in the 90 days after `2026-01-01`:
`cloud, gym, insurance, mobile, music, software, streaming` or `none`.
Metric: macro-F1 over the 8 labels.

**Our idea:** find the right features, keep the model lean, make every
forecast explainable.

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
  ════════════ 2 · ONE L1 LOGISTIC REGRESSION PER LABEL ════════════
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

### 2 · One L1 logistic regression per label
Each of the 8 labels gets its own yes/no model ("is it gym?"). The L1 (lasso)
penalty sets unhelpful coefficients to zero, so each label keeps its own
feature list (in the final model between 59 features for cloud and 96 for
`none`, out of 103). Each label picks its own penalty strength by
cross-validation.

### 3 · Decision
The label with the highest probability is the forecast. Because every label
has a short list of coefficients, each prediction can be explained, e.g.
*gym is likely because of recent gym payments (odds ×2.5) and a long-running
gym subscription (×1.8)*.

## Result

**Macro-F1 0.513** on the validation set (1,000 clients the model has never seen).
Training data: 2,000 labelled clients + 3,152 pseudo-labelled extra clients.

## Use the model on your own data

The method lives in one standalone file, [`sparse_levels.py`](sparse_levels.py)
(only numpy, pandas and scikit-learn). It works for any feature table and any
"levels": class labels, cluster ids or customer segments.

```python
from sparse_levels import fit_sparse_levels

model = fit_sparse_levels(X, y)            # X: feature table, y: level per row
model.predict(X_new)                       # predicted level per row
model.predict_proba(X_new)                 # probability per level
model.coefficients(top=5)                  # the features each level uses (odds ratios)
model.explain(X_new.iloc[[0]])             # why this row got its prediction
```

Options: `penalty="l1"` (lasso, sparsest, default) or `penalty="elasticnet"`
with `l1_ratio` (more stable with strongly correlated features); `Cs=(...)`
for the candidate penalty strengths (one value = fixed penalty); `levels=[...]`
to fix the order of the levels. Run `py sparse_levels.py` for a small demo.

## Run our pipeline

```bash
pip install -e .                      # Python >= 3.10
unzip data/dataset.zip -d data/
py -3.10 -m src.evaluate "my run"     # train, score on validation, log
py -3.10 -m src.make_submission       # -> data/submission_final.csv
```

## Repository

```
sparse_levels.py  the method: one sparse logistic regression per level (standalone)
src/          category_map (tagging) · recurrence (streams) · features · model · evaluate · make_submission
data/         challenge data (dataset.zip) + pseudo-labels
extra_info/   experiment log, verified interpretations, coefficients per label
```

Challenge data and task: [Swiss-ai-Weeks/ubs-2026](https://github.com/Swiss-ai-Weeks/ubs-2026) (Apache 2.0).
