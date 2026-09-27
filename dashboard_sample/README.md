# Client risk monitor

> **Status: prototype, vibe coded.** The method has not been reviewed or checked yet.
> Treat the numbers as a first draft, not as results.

## Goal

Give a bank advisor a simple view per client: **is this client likely to miss or be late with an
upcoming recurring payment, and why?**

- One number per client, tracked over time: the **risk index** (0-100).
- A ranked list of all clients, so the riskiest and the fastest-rising stand out.
- For each change in the score, the behaviour behind it (e.g. timing, payment history, subscriptions).

Open `index.html` in a browser to see it (download it first: GitHub only shows the source).
It shows the 1,000 validation clients; none of them is used to fit any model.

## How it relates to the main model

The risk model is **the main model's recipe with a different question**. Nothing else is new.

1. **Main model** asks: **what** does the client pay next? (e.g. *insurance*)
2. **Risk model** asks: **will** the next payments come on time? (e.g. *risk 60: likely late or missed*)

```text
                                                 +----------------+
                                                 | Transactions   |
                                                 +----------------+
                                                          |
                                                          v
                                             +------------------------+
                                             | Client features        |
                                             | 103 lean features      |
                                             | (shared)               |
                                             +------------------------+
                                                          |
                                 +------------------------+------------------------+
                                 |                                                 |
                  PATH 1: MAIN MODEL (challenge)                    PATH 2: RISK MODEL (dashboard)
                                 |  features                                       |  features
                                 v                                                 v
+----------------+   +-----------------------+    +----------------+   +-----------------------+
| TARGET         |   | MODEL                 |    | TARGET         |   | MODEL                 |
| next category  |-->| 8 x L1 logistic       |    | on time /      |-->| 1 x L1 logistic       |
| (8 challenge   |   | regressions, one      |    | late / missed  |   | regression:           |
|  labels)       |   | per category:         |    | (from the      |   | "will a payment       |
|                |   | "is it insurance?"    |    |  transactions) |   |  slip? if so,         |
+----------------+   |                       |    +----------------+   |  is it missed?"       |
                     +-----------------------+                         +-----------------------+
                                 |                                                 |
                                 v                                                 v
                     +-----------------------+                         +-----------------------+
                     | OUTPUT                |                         | OUTPUT                |
                     | next category         |                         | risk index 0-100      |
                     | e.g. insurance        |                         | e.g. 60               |
                     +-----------------------+                         +-----------------------+
                                 |                                                 |
                                 +------------------------+------------------------+
                                                          v
                                           +----------------------------+
                                           | DASHBOARD                  |
                                           | risk over time + drivers   |
                                           | (value x beta)             |
                                           +----------------------------+
```

| | Main model | Risk model |
|---|---|---|
| Features | 103 lean features | the same 103 features |
| Model | yes/no L1 logistic regressions ("is it insurance?") | the same kind: "will a payment slip?" and "if it slips, is it missed?" |
| Explanation | value × beta per feature | the same value × beta, summed into 6 groups |
| Target | next category (challenge labels) | on time / late / missed (read from the transactions) |

So the only real change is the **target**. Because the betas work the same way, anyone who can read
the main model's coefficients can read the risk drivers.

## The model in short

1. **Snapshots.** Every two weeks we rebuild each client's features using only the data up to that day
   (the main model does this once, at the cutoff).
2. **New target.** We take the recurring payments (subscriptions) that should come in the next 30 days
   and check what happened: **on time**, **late** (up to one billing cycle late) or **missed** (did not come).
3. **Model.** The same sparse L1 logistic regression as the main model, used for two linked yes/no
   questions: *will a payment slip?* and *if it slips, is it missed?* Both share one set of betas.
4. **Risk index** = 50 × P(late) + 100 × P(missed). So 0 = surely on time, 50 = surely late,
   100 = surely missed.
5. **Why it moved.** As in the main model, each feature adds *feature value × beta* to the score.
   We sum these into six groups (timing, history, amounts, subscriptions, recent change, account
   activity) and show how each group changed between two snapshots.

The risk model is trained on the train and test clients (its outcomes come from the transactions,
so no labels are needed). The *next payment* column is the main model itself (`pipeline/evaluate.py`,
trained on the train clients only).

## Rebuild

    python -m dashboard_sample.build

Run from the repo root after unzipping the data (see the main README). It uses `features/` and
`pipeline/` unchanged. The first run takes ~30-40 min; later runs take ~3 min.
