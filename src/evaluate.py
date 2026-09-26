"""Score the model on the validation set and log the result.

    py -3.10 -m src.evaluate "short description of the change"

Rebuilds train/valid features from the raw jsonl (so changes in
category_map / recurrence / features are picked up), trains the model on the
train clients + pseudo-labelled pretrain clients (never on valid), prints
macro-F1, per-label F1 and the confusion matrix on valid, and appends one row
per run to extra_info/experiments.csv.

Also prints stream-detection diagnostics on valid, independent of the model:
  c_detect      share of non-'none' clients whose target family is an active stream
  c_nextdue     share of non-'none' clients where the active stream due soonest is the target
  c_none_active share of 'none' clients with >= 1 active stream
  c_fams        mean number of active families per client
"""

from __future__ import annotations

import datetime as dt
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

from src.features import build_features, select_features
from src.model import ALL_LABELS, LABEL_COL, build_model
from src.pseudo import pseudo_labelled_features
from src.recurrence import detect_streams, load_transactions

CUTOFF = "2026-01-01"
LOG_PATH = "extra_info/experiments.csv"
FEATURE_SET = "lean"


def labelled_features(split: str) -> pd.DataFrame:
    feats = build_features(f"data/{split}_transactions.jsonl", CUTOFF)
    labels = pd.read_csv(f"data/{split}_labels.csv")[["client_id", LABEL_COL]]
    return labels.merge(feats, on="client_id", how="left")


def training_set(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Model features of the train clients + pseudo-labelled pretrain clients."""
    X = select_features(train.drop(columns=["client_id", LABEL_COL]), FEATURE_SET)
    pseudo = pseudo_labelled_features()
    X = pd.concat([X, pseudo[X.columns]], ignore_index=True)
    y = pd.concat([train[LABEL_COL], pseudo[LABEL_COL]], ignore_index=True)
    return X, y


def macro_f1(y_true, y_pred) -> float:
    return f1_score(y_true, y_pred, average="macro", labels=ALL_LABELS, zero_division=0)


def stream_diagnostics(split: str) -> dict[str, float]:
    cutoff = pd.Timestamp(CUTOFF, tz="UTC")
    labels = pd.read_csv(f"data/{split}_labels.csv").set_index("client_id")[LABEL_COL]
    streams = detect_streams(load_transactions(f"data/{split}_transactions.jsonl"))
    active = streams[streams["is_recurring"] & streams["category"].notna()].copy()

    families = active.groupby("client_id")["category"].apply(set)
    active["due"] = active["mean_gap_days"] - (cutoff - active["last_date"]).dt.days
    live = active[active["due"] >= -active["mean_gap_days"]]
    first_due = live.sort_values("due").groupby("client_id")["category"].first()

    targets = labels[labels != "none"]
    nones = labels[labels == "none"]
    return {
        "c_detect": np.mean([t in families.get(c, set()) for c, t in targets.items()]),
        "c_nextdue": np.mean([first_due.get(c) == t for c, t in targets.items()]),
        "c_none_active": np.mean([len(families.get(c, set())) > 0 for c in nones.index]),
        "c_fams": np.mean([len(families.get(c, set())) for c in labels.index]),
    }


def main() -> None:
    note = sys.argv[1] if len(sys.argv) > 1 else ""

    X_train, y_train = training_set(labelled_features("train"))
    valid = labelled_features("valid")
    y_valid = valid[LABEL_COL]
    pred = build_model().fit(X_train, y_train).predict(valid[X_train.columns])
    score = macro_f1(y_valid, pred)
    diag = stream_diagnostics("valid")

    print(f"\n{note}")
    print(f"  model macro-F1 (valid) = {score:.4f}   ({X_train.shape[1]} features, {len(X_train)} training rows)")
    print("stream diagnostics (valid):")
    for k, v in diag.items():
        print(f"  {k:14s} {v:.3f}")
    print("\nper-label F1:")
    for lab, s in zip(ALL_LABELS, f1_score(y_valid, pred, average=None, labels=ALL_LABELS, zero_division=0)):
        print(f"  {lab:10s} {s:.3f}")
    print("\nconfusion (rows=true, cols=pred):")
    print(pd.DataFrame(confusion_matrix(y_valid, pred, labels=ALL_LABELS), index=ALL_LABELS, columns=ALL_LABELS))

    row = {"time": dt.datetime.now().isoformat(timespec="seconds"), "note": note,
           "final_sparse_lean_pseudo": f"{score:.4f}", **{k: f"{v:.3f}" for k, v in diag.items()}}
    old = pd.read_csv(LOG_PATH, dtype=str) if os.path.exists(LOG_PATH) else pd.DataFrame(columns=list(row))
    cols = list(old.columns) + [c for c in row if c not in old.columns]
    pd.concat([old, pd.DataFrame([row])], ignore_index=True)[cols].to_csv(LOG_PATH, index=False)


if __name__ == "__main__":
    main()
