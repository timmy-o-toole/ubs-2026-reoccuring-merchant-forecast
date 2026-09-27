"""Payment risk index: will a client's next recurring payments arrive on time?

Unit = client x snapshot (every 14 days). At snapshot t we only use data before t:
- features = the 103 lean model features, rebuilt with cutoff t;
- due payments = recurring streams still live at t (detect_streams on data < t)
  whose next charge is expected in [t - 5 d, t + 30 d].

Outcome (ordinal, from the transactions themselves, no labels needed), per due payment:
  on time  arrives <= 5 days after the expected date (same tolerance as "live")
  late     arrives later, but within one full billing cycle
  missed   no matching payment within one full cycle (stopped or skipped)
The snapshot's outcome is the worst of its due payments: 0 on time, 1 late, 2 missed.
Snapshots without a due payment, or whose outcome window runs past the data, stay unlabelled.

A matching payment = same client, outgoing subscription-type payment (same pool as
detect_streams), amount within max(0.3, 5%) of the stream's mean amount, same
category (or none) if the stream has one, later than half a cycle after the last payment.
"""

from __future__ import annotations

import glob
import hashlib
import os

import numpy as np
import pandas as pd

import code
from code import build_features, detect_streams, select_features
from code.features_category_map import NON_SUBSCRIPTION_PHRASES, NON_SUBSCRIPTION_TYPES
from pipeline.data import DATA_DIR, read_transactions

SNAPSHOTS = [d.strftime("%Y-%m-%d") for d in pd.date_range("2025-01-15", "2025-12-31", freq="14D")] + ["2026-01-01"]
DATA_END = pd.Timestamp("2025-12-31 23:59:59", tz="UTC")
DUE_BEFORE_DAYS, DUE_AHEAD_DAYS = 5, 30
ON_TIME_DAYS = 5
AMOUNT_ABS_TOL, AMOUNT_REL_TOL = 0.3, 0.05
OUTCOMES = ["on time", "late", "missed"]
SPLITS = ("train", "valid", "test")


def _pool(df: pd.DataFrame) -> pd.DataFrame:
    """Outgoing subscription-type payments (same filter as detect_streams)."""
    out = df[df["direction"] == "out"]
    return out[~out["type"].isin(NON_SUBSCRIPTION_TYPES)
               & ~out["description"].str.contains("|".join(NON_SUBSCRIPTION_PHRASES))]


def _outcomes(streams: pd.DataFrame, by_client: dict, t: pd.Timestamp) -> pd.DataFrame:
    """One row per due payment at snapshot t with its outcome (0/1/2), NaN if the window runs past the data."""
    s = streams[streams["is_recurring"]].copy()
    s["expected"] = s["last_date"] + pd.to_timedelta(s["mean_gap_days"], unit="D")
    s = s[(s["expected"] >= t - pd.Timedelta(days=DUE_BEFORE_DAYS))
          & (s["expected"] <= t + pd.Timedelta(days=DUE_AHEAD_DAYS))]
    rows = []
    for r in s.itertuples():
        gap = pd.Timedelta(days=r.mean_gap_days)
        arrival = pd.NaT
        cand = by_client.get(r.client_id)
        if cand is not None:
            m = (((cand["amount"] - r.mean_amount).abs() <= max(AMOUNT_ABS_TOL, AMOUNT_REL_TOL * r.mean_amount))
                 & (cand["timestamp"] > r.last_date + gap / 2))
            if r.category is not None:   # rotating descriptions often carry no category: accept those
                m &= (cand["category"] == r.category) | cand["category"].isna()
            if m.any():
                arrival = cand.loc[m, "timestamp"].min()
        delay = (arrival - r.expected) / pd.Timedelta(days=1) if pd.notna(arrival) else np.inf
        if r.expected + gap > DATA_END:     # full window not observed: no label (avoids end-of-data bias)
            outcome = np.nan
        else:
            outcome = 0 if delay <= ON_TIME_DAYS else 1 if delay <= r.mean_gap_days else 2
        rows.append({"client_id": r.client_id, "category": r.category, "amount": r.mean_amount,
                     "expected": r.expected, "arrival": arrival, "outcome": outcome})
    return pd.DataFrame(rows, columns=["client_id", "category", "amount", "expected", "arrival", "outcome"])


def build_snapshots(split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(client x snapshot features + outcome, due-payment detail). Cached by code hash."""
    cache = f"{DATA_DIR}/processed/risk_{split}_{_code_hash()}.pkl"
    if os.path.exists(cache):
        return pd.read_pickle(cache)
    df = read_transactions(split)
    by_client = dict(tuple(_pool(df).groupby("client_id")))
    frames, dues = [], []
    for snap in SNAPSHOTS:
        t = pd.Timestamp(snap, tz="UTC")
        before = df[df["timestamp"] < t]
        feats = build_features(before, snap)
        due = _outcomes(detect_streams(before), by_client, t)
        due["snapshot"] = snap
        worst = due.groupby("client_id")["outcome"].agg(lambda o: np.nan if o.isna().any() else o.max())
        feats["snapshot"] = snap
        feats["n_due"] = feats["client_id"].map(due.groupby("client_id").size()).fillna(0).astype(int)
        feats["outcome"] = feats["client_id"].map(worst)
        frames.append(feats)
        dues.append(due)
        print(f"  {split} {snap}: {len(feats)} clients, {int(feats['outcome'].notna().sum())} labelled")
    out = (pd.concat(frames, ignore_index=True), pd.concat(dues, ignore_index=True))
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    pd.to_pickle(out, cache)
    return out


def _code_hash() -> str:
    """Hash of the feature code and this file: the cache is rebuilt when either changes."""
    h = hashlib.sha1()
    for path in sorted(glob.glob(os.path.join(os.path.dirname(code.__file__), "features_*.py"))) + [__file__]:
        with open(path, "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:10]


def lean_columns(snap: pd.DataFrame) -> list[str]:
    X = snap.drop(columns=["client_id", "snapshot", "n_due", "outcome"])
    return list(select_features(X, "lean").columns)


if __name__ == "__main__":
    import sys
    for split in sys.argv[1:] or SPLITS:
        snap, due = build_snapshots(split)
        lab = snap["outcome"].dropna()
        print(split, "labelled snapshots:", len(lab), lab.value_counts().sort_index().to_dict(),
              "| due payments:", due["outcome"].value_counts(dropna=False).to_dict())
