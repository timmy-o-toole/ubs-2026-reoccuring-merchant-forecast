"""Client risk monitor: payment risk index over time per client, with grouped x*beta drivers.

    python -m dashboard_sample.build

Fits the risk model (dashboard_sample/risk_model.py: sparse L1 continuation-ratio logit, trained on
train + test clients) and shows the 1,000 held-out validation clients. Also fits the final
category model (SparseLevels from pipeline.evaluate, train clients only; valid not used) for the
"next payment" column. Writes a self-contained dashboard_sample/index.html.
"""

import json
import re

import numpy as np
import pandas as pd

from code import TARGET_CATEGORIES
from pipeline.data import labelled_features, training_set
from pipeline.evaluate import build_model
from dashboard_sample.risk import SNAPSHOTS as ALL_SNAPSHOTS, build_snapshots
from dashboard_sample.risk_model import EVAL_SPLIT, evaluate, risk_index

# The 14-day grid ends on 31 Dec, one day before "today" (1 Jan): show only the latter.
SNAPSHOTS = [s for s in ALL_SNAPSHOTS if s != "2025-12-31"]
TEMPLATE_PATH = "dashboard_sample/template.html"
OUT_PATH = "dashboard_sample/index.html"

# Colours: categorical slots 1-6 in fixed order, validated colour-blind safe for adjacent stack
# segments on the card surface (no red: red is reserved for risk in the dashboard).
GROUPS = [
    ("timing", "Timing regularity", "#2A78D6"),
    ("history", "History & frequency", "#EB6834"),
    ("amount", "Amounts & funding", "#1BAF7A"),
    ("portfolio", "Subscription portfolio", "#EDA100"),
    ("change", "Recent change", "#E87BA4"),
    ("activity", "Account activity", "#008300"),
]
GROUP_PREFIXES = {
    "timing": ("over_days_", "is_live_", "first_due_", "late_cycles_", "bill_day_"),
    "history": ("n_occurrences_", "tenure_days_", "active_", "recent_txns_"),
    "amount": ("mean_amount_",),
    "change": ("refund_",),
}
GROUP_EXACT = {
    "timing": {"late_min_cycles", "min_over_days", "earliest_bill_day", "rule_says_none", "short_live_stream"},
    "history": {"max_live_n_occurrences"},
    "amount": {"n_topups", "mean_topup_amount", "topup_gap_mean_days", "topup_gap_cv"},
    "portfolio": {"n_live_streams", "port_families_ever", "port_families_last90d", "n_distinct_mcc"},
    "change": {"dormancy_score", "cooling_families", "port_dropped_families"},
}

# Plain-English name of each feature. "{c}" = the subscription family.
FEATURE_TEXT = [
    (r"n_occurrences_(\w+)", "Payments to {c} (all time)"),
    (r"tenure_days_(\w+)", "Age of the {c} subscription (days)"),
    (r"mean_amount_(\w+)", "Average {c} payment"),
    (r"active_(\w+)", "{c} has a recurring stream"),
    (r"recent_txns_(\w+)", "Recent {c} payments (last 90 d)"),
    (r"over_days_(\w+)", "Days the next {c} charge is overdue"),
    (r"is_live_(\w+)", "{c} subscription still running"),
    (r"first_due_(\w+)", "{c} is the next subscription due"),
    (r"bill_day_(\w+)", "Usual {c} billing day of month"),
    (r"late_cycles_(\w+)", "{c} lateness (cycles overdue)"),
    (r"refund_(\w+)_90d", "Recent {c} refund (90 d)"),
]
FIXED_TEXT = {
    "tenure_days": "Client age (days since first transaction)",
    "recency_days": "Days since last transaction",
    "frac_card_payment": "Share of card payments",
    "frac_p2p": "Share of P2P transfers",
    "frac_atm": "Share of ATM withdrawals",
    "frac_fee": "Share of transactions with a fee",
    "n_topups": "Number of top-ups (income)",
    "mean_topup_amount": "Average top-up amount",
    "topup_gap_mean_days": "Average days between top-ups",
    "topup_gap_cv": "Irregularity of top-ups",
    "n_distinct_mcc": "Number of different merchant types",
    "n_live_streams": "Number of running subscriptions",
    "max_live_n_occurrences": "Longest running subscription (payments)",
    "min_over_days": "Overdue days of the most on-time subscription",
    "short_live_stream": "Only a short (3-4 payment) stream is live",
    "rule_says_none": "No subscription clearly due next",
    "earliest_bill_day": "Earliest billing day of month",
    "late_min_cycles": "Lateness of the most punctual subscription",
    "port_families_ever": "Subscription families ever",
    "port_families_last90d": "Subscription families in the last 90 d",
    "port_dropped_families": "Families dropped in the last 90 d",
    "refund_n_90d": "Families with a recent refund",
    "dormancy_score": "Share of subscriptions gone quiet",
    "cooling_families": "Habits cooling down (6-12 m ago vs last 90 d)",
    "traveler_intensity": "Travel intensity",
    "days_since_trip": "Days since last trip",
}


def describe(col: str) -> str:
    if col in FIXED_TEXT:
        return FIXED_TEXT[col]
    for pattern, text in FEATURE_TEXT:
        m = re.fullmatch(pattern, col)
        if m and m.group(1) in TARGET_CATEGORIES:
            text = text.format(c=m.group(1))
            return text[0].upper() + text[1:]
    return col


def group_of(col: str) -> int:
    keys = [g[0] for g in GROUPS]
    for key, cols in GROUP_EXACT.items():
        if col in cols:
            return keys.index(key)
    for key, prefixes in GROUP_PREFIXES.items():
        if col.startswith(prefixes):
            return keys.index(key)
    return keys.index("activity")


def _r(v, digits=4):
    return None if v is None or not np.isfinite(v) else float(f"{v:.{digits}g}")


def next_category(valid_now: pd.DataFrame) -> pd.DataFrame:
    """Final category model (SparseLevels, train clients only; valid unseen) applied to the valid clients today."""
    X, y = training_set(labelled_features("train"))
    P = build_model().fit(X, y).predict_proba(valid_now[X.columns])
    return pd.DataFrame({"client_id": valid_now["client_id"].to_numpy(),
                         "next": P.idxmax(axis=1).to_numpy(), "next_p": P.max(axis=1).to_numpy()})


def method_html(ev: dict, n_features: int) -> str:
    rows = ev["rows"]
    tbl = "".join(
        f"<tr><td>{r.model}</td><td>{r.log_loss:.3f}</td><td>{r.auc_slip:.3f}</td><td>{r.auc_missed:.3f}</td></tr>"
        for r in rows.itertuples())
    cal = "".join(f"<tr><td>{i + 1}</td><td>{r.mean_index:.1f}</td><td>{r.observed_index:.1f}</td>"
                  f"<td>{100 * r.late:.0f}%</td><td>{100 * r.missed:.0f}%</td></tr>"
                  for i, r in enumerate(ev["calibration"].itertuples()))
    info = ev["info"]
    return f"""
<p><b>Question.</b> At each snapshot (every two weeks), will a recurring payment that is due in the next 30 days
arrive on time, late, or not at all? Only data before the snapshot is used.</p>
<ul>
<li><b>Due payment</b>: a detected recurring stream (same amount, roughly monthly) whose next charge is expected within 30 days.</li>
<li><b>On time</b>: arrives at most 5 days after the expected date. <b>Late</b>: later, but within one billing cycle.
<b>Missed</b>: no matching payment within one cycle. A snapshot takes the worst outcome of its due payments.</li>
<li><b>Model</b>: continuation-ratio logit (Agresti, <i>Categorical Data Analysis</i> 8.3). Step 1: will a payment slip? Step 2: if it slips,
is it missed? Both steps share one set of betas, so one risk score &eta; = &Sigma; (standardised feature &times; beta) explains everything.
Sparse L1 penalty: {n_features} of 103 features are used. No class weights, so probabilities stay calibrated.</li>
<li><b>Robustness</b>: every feature is clipped to the 0.5-99.5 percentile range of the training data before standardising,
so one unusual charge cannot push the score to an extreme.</li>
<li><b>Risk index</b> = 100 &times; expected severity on the 0 / 1 / 2 scale = 50 &times; P(late) + 100 &times; P(missed).</li>
<li><b>Drivers</b>: each feature adds (standardised value &times; beta) to &eta;; features are summed into six groups. The change in
risk points between two snapshots is split over the groups in proportion to their change in &eta;, so the groups add up exactly.</li>
<li><b>Honest numbers</b>: trained on {info['n_fit']:,} labelled snapshots of {info['n_fit_clients']:,} train and test clients (outcomes come
from the transactions, no labels needed); everything below and every client shown is from the {info['n_eval_clients']:,} held-out
validation clients ({info['n_eval']:,} labelled snapshots).</li>
</ul>
<table class="evaltbl"><thead><tr><th>Model (held-out clients)</th><th>Log-loss (lower = better)</th><th>AUC late or missed</th><th>AUC missed</th></tr></thead>
<tbody>{tbl}</tbody></table>
<p>"Separate stage betas" relaxes the shared-beta assumption; if it is not clearly better, one shared score is justified.</p>
<table class="evaltbl"><thead><tr><th>Risk decile</th><th>Mean index</th><th>Observed index</th><th>Late</th><th>Missed</th></tr></thead>
<tbody>{cal}</tbody></table>
<p>Observed index = 50 &times; average outcome (0/1/2) in that decile; close to the mean index = calibrated.</p>"""


def main() -> None:
    ev = evaluate(verbose=True)
    model, cols = ev["models"]["l1"], ev["cols"]
    keep = [j for j, b in enumerate(model.beta_) if b != 0]
    kcols = [cols[j] for j in keep]

    snap, due = build_snapshots(EVAL_SPLIT)
    snap = snap.set_index(["client_id", "snapshot"]).sort_index()
    ids = sorted(snap.index.get_level_values(0).unique())
    full = snap.reindex(pd.MultiIndex.from_product([ids, SNAPSHOTS], names=["client_id", "snapshot"]))
    X = full[cols].reset_index(drop=True)
    idx = risk_index(model.proba(X)).reshape(len(ids), len(SNAPSHOTS))
    T = len(SNAPSHOTS)
    last = pd.Timestamp(SNAPSHOTS[-1])
    i4 = int(np.argmin([abs((pd.Timestamp(s) - (last - pd.Timedelta(days=28))).days) for s in SNAPSHOTS]))

    now = full.xs(SNAPSHOTS[-1], level="snapshot").reset_index()
    nxt = next_category(now).set_index("client_id")

    due = due.sort_values("expected")
    due_by = {k: g for k, g in due.groupby(["client_id", "snapshot"])}
    clients, portfolio = {}, []
    pct = pd.Series(idx[:, -1]).rank(pct=True).to_numpy()
    for i, cid in enumerate(ids):
        rows = full.loc[cid]
        x = [[_r(v) for v in rows.loc[s, kcols].to_numpy(float)] for s in SNAPSHOTS]
        outcome = [None if pd.isna(o) else int(o) for o in rows["outcome"]]
        n_due = [0 if pd.isna(n) else int(n) for n in rows["n_due"]]
        dues = []
        for s in SNAPSHOTS:
            g = due_by.get((cid, s))
            dues.append([] if g is None else [
                [r.category, round(float(r.amount), 2), r.expected.strftime("%Y-%m-%d"),
                 None if pd.isna(r.arrival) else r.arrival.strftime("%Y-%m-%d"),
                 None if pd.isna(r.outcome) else int(r.outcome)] for r in g.itertuples()])
        clients[cid] = {"x": x, "outcome": outcome, "n_due": n_due, "due": dues}
        portfolio.append({"id": cid, "now": round(float(idx[i, -1]), 2), "delta": round(float(idx[i, -1] - idx[i, i4]), 2),
                          "pct": int(round(100 * pct[i])), "next": nxt.loc[cid, "next"], "next_p": round(float(nxt.loc[cid, "next_p"]), 3)})

    known_bad = [c for c in ids if any(o and o >= 1 for o in clients[c]["outcome"])]
    default = max(known_bad, key=lambda c: next(p["delta"] for p in portfolio if p["id"] == c))

    data = {
        "dates": SNAPSHOTS, "i4": i4,
        "groups": [{"key": k, "name": n, "color": c} for k, n, c in GROUPS],
        "features": [{"col": c, "name": describe(c), "group": group_of(c),
                      "lo": float(model.lo_[j]), "hi": float(model.hi_[j]),
                      "median": float(model.impute_.statistics_[j]), "mean": float(model.scale_.mean_[j]),
                      "scale": float(model.scale_.scale_[j]),
                      "binary": bool(set(X[c].dropna().unique()) <= {0, 1})}
                     for j, c in zip(keep, kcols)],
        "beta": [float(model.beta_[j]) for j in keep], "a1": float(model.a1_), "a2": float(model.a2_),
        "avg_index": [round(float(v), 2) for v in idx.mean(0)],
        "levels": [float(np.percentile(idx[:, -1], 33)), float(np.percentile(idx[:, -1], 67))],
        "portfolio": portfolio, "clients": clients, "default_client": default,
        "method": method_html(ev, len(keep)),
    }
    # Sanity check: JS math (standardise, x*beta, CR-logit) must reproduce the model's index.
    xk = np.clip(X[kcols].to_numpy(float), model.lo_[keep], model.hi_[keep])
    xk = np.where(np.isnan(xk), model.impute_.statistics_[keep], xk)
    eta = ((xk - model.scale_.mean_[keep]) / model.scale_.scale_[keep]) @ model.beta_[keep]
    ps = 1 / (1 + np.exp(-(model.a1_ + eta)))
    pm = ps / (1 + np.exp(-(model.a2_ + eta)))
    chk = 50 * (ps - pm) + 100 * pm
    print(f"index recompute check: max |diff| = {np.abs(chk - idx.ravel()).max():.2e}")
    assert np.allclose(chk, idx.ravel(), atol=1e-8)

    with open(TEMPLATE_PATH, encoding="utf-8") as f:
        html = f.read()
    html = html.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {OUT_PATH} ({len(html) / 1e6:.1f} MB): {len(ids)} clients x {T} snapshots, "
          f"{len(keep)} features, default client {default}")


if __name__ == "__main__":
    main()
