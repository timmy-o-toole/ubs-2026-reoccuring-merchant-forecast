"""Models for target_next_recurring_merchant (macro-F1 on valid).

  - sparse   - FINAL model: one sparse L1 logistic regression per label,
               from the standalone file sparse_levels.py (build_sparse_logreg).
  - logreg   - one global multinomial L2 logistic regression (benchmark).
  - rule     - no learning: the live subscription due first, else 'none'.
  - majority - always the most common label (floor).
  - hgb      - gradient boosting (reference only, not used).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sparse_levels import SparseLevelModel
from src.category_map import TARGET_CATEGORIES

LABEL_COL = "target_next_recurring_merchant"
ALL_LABELS = TARGET_CATEGORIES + ["none"]


def load_xy(path: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    y = df[LABEL_COL]
    X = df.drop(columns=["client_id", LABEL_COL])
    return X, y


def majority_predict(y_train: pd.Series, n: int) -> np.ndarray:
    majority = y_train.value_counts().idxmax()
    return np.full(n, majority)


def rule_predict(X: pd.DataFrame) -> np.ndarray:
    """No learning: use the engineered features directly.

    'none' if no stream is live (all overdue = stopped) or the longest live
    stream has only 3-4 charges (a trial that ends); else the live family
    due soonest (largest over_days = closest to its next charge).
    """
    preds = []
    for _, row in X.iterrows():
        live = [
            (cat, row[f"over_days_{cat}"])
            for cat in TARGET_CATEGORIES
            if row[f"is_live_{cat}"] == 1
        ]
        if not live or row["short_live_stream"] == 1:
            preds.append("none")
            continue
        live.sort(key=lambda t: -t[1])
        preds.append(live[0][0])
    return np.array(preds)


def build_logreg() -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(max_iter=2000, class_weight="balanced"),
            ),
        ]
    )


def build_sparse_logreg() -> SparseLevelModel:
    """Final model: one sparse L1 logistic regression per label (see sparse_levels.py).

    Imputation and scaling happen inside the model. Interpret a fitted model
    with .coefficients(), .summary() and .explain().
    """
    return SparseLevelModel(levels=ALL_LABELS)


def build_hgb() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(class_weight="balanced", random_state=0)


def evaluate(name: str, y_true: pd.Series, y_pred: np.ndarray) -> float:
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=ALL_LABELS, zero_division=0)
    print(f"=== {name} ===")
    print(f"macro-F1: {macro_f1:.4f}")
    print(classification_report(y_true, y_pred, labels=ALL_LABELS, zero_division=0))
    print()
    return macro_f1


if __name__ == "__main__":
    X_train, y_train = load_xy("data/processed/train_features.csv")
    X_valid, y_valid = load_xy("data/processed/valid_features.csv")

    results = {}

    pred_majority = majority_predict(y_train, len(y_valid))
    results["majority"] = evaluate("Majority baseline", y_valid, pred_majority)

    pred_rule = rule_predict(X_valid)
    results["rule"] = evaluate("Rule-based (no learning)", y_valid, pred_rule)

    logreg = build_logreg()
    logreg.fit(X_train, y_train)
    pred_logreg = logreg.predict(X_valid)
    results["logreg"] = evaluate("Logistic Regression", y_valid, pred_logreg)

    hgb = build_hgb()
    hgb.fit(X_train, y_train)
    pred_hgb = hgb.predict(X_valid)
    results["hgb"] = evaluate("HistGradientBoosting", y_valid, pred_hgb)

    print("=== Summary (macro-F1) ===")
    for name, score in sorted(results.items(), key=lambda t: -t[1]):
        print(f"  {name:10s} {score:.4f}")

    print()
    print("=== Confusion matrix: best model ===")
    best_name = max(results, key=results.get)
    best_pred = {"majority": pred_majority, "rule": pred_rule, "logreg": pred_logreg, "hgb": pred_hgb}[
        best_name
    ]
    cm = confusion_matrix(y_valid, best_pred, labels=ALL_LABELS)
    cm_df = pd.DataFrame(cm, index=ALL_LABELS, columns=ALL_LABELS)
    print(cm_df)
