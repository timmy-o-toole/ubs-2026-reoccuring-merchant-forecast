"""Our task's data: file paths, labels, cutoff and the training set.

This is the only place that knows where files live. features/ turns a table
of transactions into features, model/ fits on a feature table; neither reads
files by itself.
"""

from __future__ import annotations

import glob
import hashlib
import os

import pandas as pd

from features import TARGET_CATEGORIES, build_features, load_transactions, select_features

DATA_DIR = "data"
CUTOFF = "2026-01-01"
LABEL_COL = "target_next_recurring_merchant"
LABELS = TARGET_CATEGORIES + ["none"]
FEATURE_SET = "lean"  # 103 features (see features.select_features)

# Pseudo-labels for the 10,000 unlabeled pretrain clients (from Salim's branch,
# origin/Salim 34ae845): kept only where all three label sources agree
# (unsupervised, LLM, model trained on train only), 3,152 clients.
PSEUDO_LABELS_PATH = f"{DATA_DIR}/pretrain_labels_merged.csv"
MIN_SOURCES_AGREE = 3


def read_transactions(split: str) -> pd.DataFrame:
    """Raw transactions of a split: train, valid, test or unlabeled_pretrain."""
    return load_transactions(f"{DATA_DIR}/{split}_transactions.jsonl")


def labelled_features(split: str) -> pd.DataFrame:
    """client_id, label and all features for train or valid."""
    feats = build_features(read_transactions(split), CUTOFF)
    labels = pd.read_csv(f"{DATA_DIR}/{split}_labels.csv")[["client_id", LABEL_COL]]
    return labels.merge(feats, on="client_id", how="left")


def _feature_code_hash() -> str:
    h = hashlib.sha1()
    for path in sorted(glob.glob("features/*.py")):
        with open(path, "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:10]


def pretrain_features() -> pd.DataFrame:
    """Features of the pretrain clients, cached in data/processed (rebuilt when feature code changes)."""
    cache = f"{DATA_DIR}/processed/pretrain_features_{_feature_code_hash()}.pkl"
    if os.path.exists(cache):
        return pd.read_pickle(cache)
    feats = build_features(read_transactions("unlabeled_pretrain"), CUTOFF)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    feats.to_pickle(cache)
    return feats


def pseudo_labelled_features() -> pd.DataFrame:
    """client_id, label and features of the pretrain clients whose three label sources agree."""
    labels = pd.read_csv(PSEUDO_LABELS_PATH)
    labels = labels.loc[labels["n_sources_agree"] >= MIN_SOURCES_AGREE, ["client_id", LABEL_COL]]
    return labels.merge(pretrain_features(), on="client_id", how="inner")


def training_set(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Model features (lean set) and labels: train clients + pseudo-labelled pretrain clients."""
    X = select_features(train.drop(columns=["client_id", LABEL_COL]), FEATURE_SET)
    pseudo = pseudo_labelled_features()
    X = pd.concat([X, pseudo[X.columns]], ignore_index=True)
    y = pd.concat([train[LABEL_COL], pseudo[LABEL_COL]], ignore_index=True)
    return X, y
