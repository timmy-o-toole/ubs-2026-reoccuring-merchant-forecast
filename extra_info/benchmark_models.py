"""Reference benchmark: SparseLevels vs other model types, same data, same features.

    python extra_info/benchmark_models.py      (from the repo root)

Needs, on top of the pipeline: interpret-core (EBM) and tabpfn (TabPFN v2
weights, downloaded once, no login; runs on CPU).

All models: 103 lean features, trained on the 2,000 labelled train clients,
macro-F1 on the 1,000 valid clients. "Per label" = one yes/no model per label,
highest probability wins (like SparseLevels); "global" = one multiclass model.
Class imbalance: balanced class / sample weights everywhere (TabPFN:
balance_probabilities). Other models use default settings (no tuning);
SparseLevels includes its own penalty search per label. Time = training +
prediction on valid (features built once before), because TabPFN does its
work at prediction time.
"""

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

warnings.filterwarnings("ignore")
sys.path.insert(0, os.getcwd())
from pipeline.data import LABEL_COL, LABELS, labelled_features, training_set  # noqa: E402
from pipeline.evaluate import build_model, macro_f1  # noqa: E402


def binary_model(kind):
    return {
        "HGB": lambda: HistGradientBoostingClassifier(random_state=0),
        "EBM": lambda: ExplainableBoostingClassifier(random_state=0),
    }[kind]()


def fit_per_label(kind, X, y):
    models = {}
    for lab in LABELS:
        yb = (y == lab).astype(int).values
        w = compute_sample_weight("balanced", yb)
        m = binary_model(kind)
        m.fit(X, yb, sample_weight=w)
        models[lab] = m
    return models


def predict_per_label(models, X):
    P = np.column_stack([models[lab].predict_proba(X)[:, 1] for lab in LABELS])
    return np.array(LABELS)[P.argmax(axis=1)]


def global_model(kind):
    return {
        "LogReg": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                        LogisticRegression(max_iter=2000, class_weight="balanced")),
        "HGB": lambda: HistGradientBoostingClassifier(random_state=0),
        "EBM": lambda: ExplainableBoostingClassifier(random_state=0),
        "TabPFN": lambda: TabPFNClassifier.create_default_for_version(
            ModelVersion.V2, device="cpu", balance_probabilities=True, random_state=0),
    }[kind]()


def main():
    X, y = training_set(labelled_features("train"))
    valid = labelled_features("valid")
    Xv, yv = valid[X.columns], valid[LABEL_COL]
    rows = []

    def log(setup, kind, pred, t):
        rows.append((setup, kind, macro_f1(yv, pred), time.time() - t))
        print(rows[-1], flush=True)

    t = time.time()
    log("per label", "SparseLevels, L1 LogReg (ours)", build_model().fit(X, y).predict(Xv), t)
    for kind in ("HGB", "EBM"):
        t = time.time()
        log("per label", kind, predict_per_label(fit_per_label(kind, X, y), Xv), t)

    codes = {lab: i for i, lab in enumerate(LABELS)}
    yi = y.map(codes).values
    w = compute_sample_weight("balanced", y)
    for kind in ("LogReg", "HGB", "EBM", "TabPFN"):
        t = time.time()
        m = global_model(kind)
        if kind in ("HGB", "EBM"):
            m.fit(X, yi, sample_weight=w)
        else:
            m.fit(X, yi)
        log("global", kind, np.array(LABELS)[m.predict(Xv).astype(int)], t)

    out = pd.DataFrame(rows, columns=["setup", "model", "valid_macro_f1", "seconds"]).round({"valid_macro_f1": 4, "seconds": 1})
    print(out.to_string(index=False))
    out.to_csv("extra_info/benchmark_models.csv", index=False)


if __name__ == "__main__":
    main()
