"""Reference benchmark: SparseLevels vs other model types, same data, same features.

    py -3.10 extra_info/benchmark_models.py      (from the repo root)

All models: 103 lean features, trained on the 2,000 labelled train clients,
macro-F1 on the 1,000 valid clients. "Per label" = one yes/no model per label,
highest probability wins (like SparseLevels); "global" = one multiclass model.
Class imbalance: balanced class / sample weights everywhere. Tree models use
fixed default-style settings (no tuning); SparseLevels includes its own
penalty search per label. fit time = training only (features built once before).
"""

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
sys.path.insert(0, os.getcwd())
from pipeline.data import LABEL_COL, LABELS, labelled_features, training_set  # noqa: E402
from pipeline.evaluate import build_model, macro_f1  # noqa: E402


def binary_model(kind):
    return {
        "HGB": lambda: HistGradientBoostingClassifier(random_state=0),
        "XGBoost": lambda: XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1, n_jobs=-1,
                                         random_state=0, eval_metric="logloss"),
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
        "LogReg": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                LogisticRegression(max_iter=2000, class_weight="balanced")),
        "HGB": HistGradientBoostingClassifier(random_state=0),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1, n_jobs=-1, random_state=0,
                                 eval_metric="mlogloss"),
    }[kind]


def main():
    X, y = training_set(labelled_features("train"))
    valid = labelled_features("valid")
    Xv, yv = valid[X.columns], valid[LABEL_COL]
    rows = []

    t = time.time(); sl = build_model().fit(X, y); ft = time.time() - t
    rows.append(("per label", "SparseLevels, L1 LogReg (ours)", macro_f1(yv, sl.predict(Xv)), ft))
    for kind in ("HGB", "XGBoost"):
        t = time.time(); models = fit_per_label(kind, X, y); ft = time.time() - t
        rows.append(("per label", kind, macro_f1(yv, predict_per_label(models, Xv)), ft))

    codes = {lab: i for i, lab in enumerate(LABELS)}
    yi = y.map(codes).values
    w = compute_sample_weight("balanced", y)
    for kind in ("LogReg", "HGB", "XGBoost"):
        m = global_model(kind)
        t = time.time()
        if kind in ("HGB", "XGBoost"):
            m.fit(X, yi, sample_weight=w)
        else:
            m.fit(X, yi)
        ft = time.time() - t
        rows.append(("global", kind, macro_f1(yv, np.array(LABELS)[m.predict(Xv).astype(int)]), ft))

    out = pd.DataFrame(rows, columns=["setup", "model", "valid_macro_f1", "fit_seconds"]).round({"valid_macro_f1": 4, "fit_seconds": 1})
    print(out.to_string(index=False))
    out.to_csv("extra_info/benchmark_models.csv", index=False)


if __name__ == "__main__":
    main()
