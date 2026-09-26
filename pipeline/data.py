"""Our task's data: file paths, labels, cutoff and the training set.

This is the only place that knows where files live. features/ turns a table
of transactions into features, model/ fits on a feature table; neither reads
files by itself.

Training uses only the labelled train clients; the valid clients are only
used for scoring.
"""

from __future__ import annotations

import pandas as pd

from features import TARGET_CATEGORIES, build_features, load_transactions, select_features

DATA_DIR = "data"
CUTOFF = "2026-01-01"
LABEL_COL = "target_next_recurring_merchant"
LABELS = TARGET_CATEGORIES + ["none"]
FEATURE_SET = "lean"  # 103 features (see features.select_features)


def read_transactions(split: str) -> pd.DataFrame:
    """Raw transactions of a split: train, valid or test."""
    return load_transactions(f"{DATA_DIR}/{split}_transactions.jsonl")


def labelled_features(split: str) -> pd.DataFrame:
    """client_id, label and all features for train or valid."""
    feats = build_features(read_transactions(split), CUTOFF)
    labels = pd.read_csv(f"{DATA_DIR}/{split}_labels.csv")[["client_id", LABEL_COL]]
    return labels.merge(feats, on="client_id", how="left")


def training_set(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Model features (lean set) and labels of the labelled train clients."""
    X = select_features(train.drop(columns=["client_id", LABEL_COL]), FEATURE_SET)
    return X, train[LABEL_COL]
