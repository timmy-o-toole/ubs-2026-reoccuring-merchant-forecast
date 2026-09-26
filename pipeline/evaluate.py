"""Train the model, score it on the validation set and log the result.

    py -3.10 -m pipeline.evaluate "short description of the change"

The model is trained only on the labelled train clients (never on valid).
Prints macro-F1, per-label F1 and the confusion matrix on valid, and appends
one row per run to extra_info/experiments.csv.
"""

from __future__ import annotations

import datetime as dt
import os
import sys

import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

from model import SparseLevels
from pipeline.data import LABEL_COL, LABELS, labelled_features, training_set

LOG_PATH = "extra_info/experiments.csv"


def build_model() -> SparseLevels:
    """SparseLevels for our task: one sparse logistic regression per label, L1 (lasso) penalty."""
    return SparseLevels(levels=LABELS)


def macro_f1(y_true, y_pred) -> float:
    return f1_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0)


def main() -> None:
    note = sys.argv[1] if len(sys.argv) > 1 else ""

    X_train, y_train = training_set(labelled_features("train"))
    valid = labelled_features("valid")
    y_valid = valid[LABEL_COL]
    pred = build_model().fit(X_train, y_train).predict(valid[X_train.columns])
    score = macro_f1(y_valid, pred)

    print(f"\n{note}")
    print(f"  model macro-F1 (valid) = {score:.4f}   ({X_train.shape[1]} features, {len(X_train)} training rows)")
    print("\nper-label F1:")
    for lab, s in zip(LABELS, f1_score(y_valid, pred, average=None, labels=LABELS, zero_division=0)):
        print(f"  {lab:10s} {s:.3f}")
    print("\nconfusion (rows=true, cols=pred):")
    print(pd.DataFrame(confusion_matrix(y_valid, pred, labels=LABELS), index=LABELS, columns=LABELS))

    row = {"time": dt.datetime.now().isoformat(timespec="seconds"), "note": note, "sparse_logreg_lean": f"{score:.4f}"}
    old = pd.read_csv(LOG_PATH, dtype=str) if os.path.exists(LOG_PATH) else pd.DataFrame(columns=list(row))
    cols = list(old.columns) + [c for c in row if c not in old.columns]
    pd.concat([old, pd.DataFrame([row])], ignore_index=True)[cols].to_csv(LOG_PATH, index=False)


if __name__ == "__main__":
    main()
