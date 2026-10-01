# Client risk monitor

> **Status: prototype.** The risk method has not been independently reviewed; treat the numbers as a first draft.

## Goal

Give a bank advisor a simple view per client: **is this client likely to miss or be late with an
upcoming recurring payment, and why?**

- One number per client, tracked over time: the **risk index** (0-100).
- A ranked list of all clients, so the riskiest and the fastest-rising stand out.
- For each change in the score, the behaviour behind it (e.g. timing, payment history, subscriptions).

**[Open the live dashboard](https://timmy-o-toole.github.io/ubs-2026-recurring-merchant-forecast/dashboard_sample/)**
(the same `index.html`, served by GitHub Pages). It shows the 1,000 validation clients; none of
them is used to fit any model.

## How it relates to the main model

The risk model reuses the main model's features and explanation style for a different question.

1. **Main model** asks: **what** does the client pay next? (e.g. *insurance*)
2. **Risk model** asks: **will** the next payments come on time? (e.g. *risk 60: likely late or missed*)

| | Main model | Risk model |
|---|---|---|
| Features | 103 lean features | the same 103 features (L1 keeps 91) |
| Model | 8 yes/no L1 logistic regressions, one per label ("is it insurance?") | one L1 logistic regression with two linked yes/no steps (see below) |
| Explanation | value × beta per feature | the same value × beta, summed into 6 groups |
| Target | next category (challenge labels) | on time / late / missed (read from the transactions) |

## The model in short

1. **Snapshots.** Every two weeks we rebuild each client's features using only the data up to that day
   (the main model does this once, at the cutoff).
2. **New target.** We take the recurring payments (subscriptions) that are due in the next 30 days
   (or up to 5 days overdue) and check what happened: **on time**, **late** (up to one billing cycle
   late) or **missed** (did not come).
3. **Model.** One sparse L1 logistic regression (a continuation-ratio logit, fitted on stacked rows)
   for two linked yes/no questions: *will a payment slip?* and *if it slips, is it missed?* Both
   share one set of betas. Unlike the main model: features are clipped to the 0.5-99.5 percentile
   range before standardising, there are no class weights, and one penalty strength is chosen by
   cross-validated log-loss (folds grouped by client).
4. **Risk index** = 50 × P(late) + 100 × P(missed). So 0 = surely on time, 50 = surely late,
   100 = surely missed.
5. **Why it moved.** As in the main model, each feature adds *feature value × beta* to the score.
   We sum these into six groups (timing, history, amounts, subscriptions, recent change, account
   activity) and show how each group changed between two snapshots.

The risk model is fitted on the transactions of the clients in the challenge's train and test
splits. It needs no challenge labels: its on-time/late/missed outcomes are read from the
transactions themselves, so the hidden test labels are never used. The validation clients shown
in the dashboard are never used for fitting. The *next payment* column is the main model itself
(`pipeline/evaluate.py`, trained on the train clients only).

## Rebuild

    python -m dashboard_sample.build

Run from the repo root after unzipping the data (see the main README). It uses `sparselevels/` and
`pipeline/` unchanged. The first run takes ~30-40 min; later runs take ~3 min.

`risk.py` builds the two-weekly snapshots and their on-time/late/missed outcomes (cached in `data/processed/`).
`risk_model.py` is the risk model (`python -m dashboard_sample.risk_model` only prints its evaluation).
`build.py` fills `template.html` with the data and writes `index.html`.
