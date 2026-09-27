# SparseLevels: Individual Coefficients

Final model: **SparseLevels with L1 (lasso), one sparse logistic regression per label**, `lean` feature set (103 features), trained on the 2,000 labelled train clients, valid macro-F1 **0.504**.

How to read: *odds ×k per SD* = one standard deviation more of the feature multiplies the odds of that label (vs all other labels) by k, holding the other features fixed. **Stable** = selected in ≥ 80% of 20 bootstrap refits with the same sign in ≥ 95%. Each table shows the strongest stable coefficients per label: up to 5 that raise the odds (↑) and up to 4 that lower them (↓). Coefficients of twin features (`n_occurrences_*`, `active_*`, `over_days_*`, `late_cycles_*`) are left out because they move together with `tenure_days_*` / `is_live_*` and get offsetting signs. Full table: regenerate with `model.coefficients()` (code/model.py).

## cloud  (55 of 103 features used, 23 stable)

| Direction | Feature | Odds ×/SD | Note |
|---|---|---|---|
| ↑ | `recent_txns_cloud` | 2.63 |  |
| ↑ | `tenure_days_cloud` | 1.76 |  |
| ↑ | `first_due_cloud` | 1.66 |  |
| ↑ | `is_live_cloud` | 1.40 |  |
| ↑ | `frac_card_payment` | 1.23 |  |
| ↓ | `refund_n_90d` | 0.71 | ⚠️ refund effect not robust |
| ↓ | `port_families_ever` | 0.78 |  |
| ↓ | `refund_mobile_90d` | 0.78 |  |
| ↓ | `n_distinct_mcc` | 0.84 |  |

## gym  (58 of 103 features used, 21 stable)

| Direction | Feature | Odds ×/SD | Note |
|---|---|---|---|
| ↑ | `recent_txns_gym` | 2.46 |  |
| ↑ | `tenure_days_gym` | 1.78 |  |
| ↑ | `is_live_gym` | 1.41 |  |
| ↑ | `n_distinct_mcc` | 1.40 |  |
| ↑ | `n_topups` | 1.24 |  |
| ↓ | `recent_txns_insurance` | 0.60 |  |
| ↓ | `tenure_days_insurance` | 0.69 |  |
| ↓ | `port_families_ever` | 0.76 |  |
| ↓ | `recent_txns_mobile` | 0.78 |  |

## insurance  (53 of 103 features used, 23 stable)

| Direction | Feature | Odds ×/SD | Note |
|---|---|---|---|
| ↑ | `recent_txns_insurance` | 2.33 |  |
| ↑ | `tenure_days_insurance` | 1.77 |  |
| ↑ | `is_live_insurance` | 1.65 |  |
| ↑ | `n_distinct_mcc` | 1.35 |  |
| ↑ | `refund_insurance_90d` | 1.35 |  |
| ↓ | `refund_n_90d` | 0.68 | ⚠️ refund effect not robust |
| ↓ | `short_live_stream` | 0.76 |  |
| ↓ | `recent_txns_mobile` | 0.76 |  |
| ↓ | `recent_txns_software` | 0.77 |  |

## mobile  (55 of 103 features used, 21 stable)

| Direction | Feature | Odds ×/SD | Note |
|---|---|---|---|
| ↑ | `recent_txns_mobile` | 2.82 |  |
| ↑ | `tenure_days_mobile` | 1.79 |  |
| ↑ | `is_live_mobile` | 1.47 |  |
| ↑ | `first_due_mobile` | 1.38 |  |
| ↑ | `n_distinct_mcc` | 1.28 |  |
| ↓ | `recent_txns_software` | 0.69 |  |
| ↓ | `recent_txns_streaming` | 0.79 |  |
| ↓ | `recent_txns_insurance` | 0.80 |  |
| ↓ | `recent_txns_cloud` | 0.82 |  |

## music  (23 of 103 features used, 6 stable)

| Direction | Feature | Odds ×/SD | Note |
|---|---|---|---|
| ↑ | `recent_txns_music` | 1.88 |  |
| ↑ | `first_due_music` | 1.39 |  |
| ↓ | `port_families_last90d` | 0.84 |  |
| ↓ | `short_live_stream` | 0.84 |  |
| ↓ | `bill_day_music` | 0.93 |  |

## software  (60 of 103 features used, 24 stable)

| Direction | Feature | Odds ×/SD | Note |
|---|---|---|---|
| ↑ | `recent_txns_software` | 2.35 |  |
| ↑ | `tenure_days_software` | 2.10 |  |
| ↑ | `bill_day_streaming` | 1.52 |  |
| ↑ | `first_due_software` | 1.40 |  |
| ↑ | `is_live_software` | 1.34 |  |
| ↓ | `recent_txns_streaming` | 0.60 |  |
| ↓ | `recent_txns_gym` | 0.79 |  |
| ↓ | `first_due_music` | 0.81 |  |
| ↓ | `recent_txns_insurance` | 0.82 |  |

## streaming  (56 of 103 features used, 26 stable)

| Direction | Feature | Odds ×/SD | Note |
|---|---|---|---|
| ↑ | `recent_txns_streaming` | 2.29 |  |
| ↑ | `is_live_streaming` | 1.68 |  |
| ↑ | `first_due_streaming` | 1.41 |  |
| ↑ | `mean_amount_software` | 1.20 |  |
| ↑ | `bill_day_cloud` | 1.15 |  |
| ↓ | `n_distinct_mcc` | 0.67 |  |
| ↓ | `port_families_last90d` | 0.74 |  |
| ↓ | `refund_music_90d` | 0.78 |  |
| ↓ | `refund_gym_90d` | 0.84 |  |

## none  (80 of 103 features used, 50 stable)

| Direction | Feature | Odds ×/SD | Note |
|---|---|---|---|
| ↑ | `port_families_ever` | 1.63 | ⚠️ twin of port_families_last90d (r = 0.78); raw none rate falls with it; read together as 'shrinking portfolio' |
| ↑ | `rule_says_none` | 1.46 | = 1 when no subscription is still running, or the running one with the most charges has only 3-4 of them (looks like a trial) |
| ↑ | `refund_n_90d` | 1.34 | ⚠️ refund effect not robust (it comes from recent purchases) |
| ↑ | `max_live_n_occurrences` | 1.28 | ⚠️ raw relation is U-shaped (44% / 14% / 15% / 22%); only meaningful next to short_live_stream and rule_says_none |
| ↑ | `refund_gym_90d` | 1.22 |  |
| ↓ | `tenure_days_software` | 0.41 |  |
| ↓ | `is_live_insurance` | 0.43 |  |
| ↓ | `frac_card_payment` | 0.54 |  |
| ↓ | `is_live_software` | 0.56 |  |
