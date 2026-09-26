"""SparseLevels: one sparse logistic regression per level (L1 or elastic net).

    from model import fit_sparse_levels      # or copy this file and: from sparse_levels import ...

    model = fit_sparse_levels(X, y)          # X: feature table, y: a level per row
    model.predict(X_new)                     # predicted level per row
    model.predict_proba(X_new)               # probability per level, each row sums to 1
    model.coefficients(top=5)                # the few features each level uses
    model.summary()                          # penalty and number of features per level
    model.explain(X_new.iloc[[0]])           # why this row got its prediction

How it works:
  1. Features are median-imputed and standardized, so every coefficient
     means "effect of +1 standard deviation".
  2. Each level ("is it gym?", "is it cluster 3?", ...) gets its own yes/no
     logistic regression with an L1 (lasso) or elastic-net penalty. The
     penalty sets unhelpful coefficients to zero, so each level keeps only
     its own short list of features. Each level picks its own penalty
     strength C by cross-validation.
  3. The level with the highest probability is the prediction.

"Levels" can be anything categorical: class labels, cluster ids, customer
segments. Only numpy, pandas and scikit-learn are needed; the file does not
depend on the rest of this repository.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

__all__ = ["SparseLevels", "fit_sparse_levels"]


class SparseLevels(BaseEstimator, ClassifierMixin):
    """One sparse (L1 or elastic-net) yes/no logistic regression per level.

    Parameters
    ----------
    levels : list or None
        Levels to model, in the order of the probability columns. None means
        the sorted unique values of y.
    penalty : "l1" or "elasticnet"
        Shrinkage form. "l1" (lasso) gives the sparsest models; "elasticnet"
        mixes in L2, which is more stable when features are strongly correlated.
    l1_ratio : float
        Only for "elasticnet": share of L1 in the penalty (1.0 = pure lasso).
    Cs : sequence of float
        Candidate penalty strengths (smaller = sparser). Each level picks its
        own C by cross-validation; a single value means a fixed penalty.
    cv : int
        Folds of the inner cross-validation that picks C per level.
    class_weight : "balanced" or None
        "balanced" up-weights the positive rows of rare levels.
    random_state : int
        Seed for the CV split and the solver.
    """

    def __init__(self, levels=None, penalty="l1", l1_ratio=0.5, Cs=(0.03, 0.1, 0.3, 1.0), cv=3,
                 class_weight="balanced", random_state=0):
        self.levels = levels
        self.penalty = penalty
        self.l1_ratio = l1_ratio
        self.Cs = Cs
        self.cv = cv
        self.class_weight = class_weight
        self.random_state = random_state

    # ---------------------------------------------------------------- fitting
    def _logreg(self, C):
        if self.penalty == "l1":
            return LogisticRegression(penalty="l1", solver="liblinear", C=C, class_weight=self.class_weight,
                                      max_iter=2000, random_state=self.random_state)
        if self.penalty == "elasticnet":
            return LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=self.l1_ratio, C=C,
                                      class_weight=self.class_weight, max_iter=5000, random_state=self.random_state)
        raise ValueError(f"penalty must be 'l1' or 'elasticnet', got {self.penalty!r}")

    def _prepare(self, X, fit=False):
        if fit:
            self.feature_names_in_ = (list(map(str, X.columns)) if isinstance(X, pd.DataFrame)
                                      else [f"x{i}" for i in range(np.asarray(X).shape[1])])
            self.imputer_ = SimpleImputer(strategy="median").fit(np.asarray(X, dtype=float))
            # columns that are completely empty are dropped by the imputer
            self.features_ = [f for f, s in zip(self.feature_names_in_, self.imputer_.statistics_) if not np.isnan(s)]
            self.scaler_ = StandardScaler().fit(self.imputer_.transform(np.asarray(X, dtype=float)))
        return self.scaler_.transform(self.imputer_.transform(np.asarray(X, dtype=float)))

    def fit(self, X, y, sample_weight=None):
        """Fit one sparse logistic regression per level. sample_weight: optional weight per row."""
        y = np.asarray(y)
        self.classes_ = np.array(sorted(pd.unique(y)) if self.levels is None else list(self.levels))
        unknown = set(pd.unique(y)) - set(self.classes_)
        if unknown:
            raise ValueError(f"y contains levels not in `levels`: {sorted(map(str, unknown))}")
        Z = self._prepare(X, fit=True)
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        self.models_, self.C_ = {}, {}
        for level in self.classes_:
            yb = (y == level).astype(int)
            if yb.sum() == 0:
                raise ValueError(f"level {level!r} has no rows in y")
            if len(self.Cs) == 1:
                best_C = self.Cs[0]
            else:
                folds = StratifiedKFold(self.cv, shuffle=True, random_state=self.random_state)
                score = {
                    C: np.mean([f1_score(yb[te], self._logreg(C).fit(Z[tr], yb[tr], sample_weight=w[tr]).predict(Z[te]),
                                         sample_weight=w[te], zero_division=0)
                                for tr, te in folds.split(Z, yb)])
                    for C in self.Cs
                }
                best_C = max(score, key=score.get)
            self.C_[level] = best_C
            self.models_[level] = self._logreg(best_C).fit(Z, yb, sample_weight=w)
        return self

    # ------------------------------------------------------------- predicting
    def predict_proba(self, X):
        """Probability of each level (one column per level, rows as in X); each row sums to 1.

        The per-level yes/no probabilities are rescaled to a distribution, which
        does not change which level has the highest probability.
        """
        Z = self._prepare(X)
        P = np.column_stack([self.models_[lv].predict_proba(Z)[:, 1] for lv in self.classes_])
        P = P / P.sum(axis=1, keepdims=True)
        index = X.index if isinstance(X, pd.DataFrame) else None
        return pd.DataFrame(P, columns=self.classes_, index=index)

    def predict(self, X):
        """Level with the highest probability for each row."""
        return self.classes_[np.argmax(self.predict_proba(X).to_numpy(), axis=1)]

    # ---------------------------------------------------------- interpreting
    def coefficients(self, top=None):
        """Non-zero coefficients per level, strongest first.

        coef = change in log-odds of the level per +1 SD of the feature;
        odds_ratio = exp(coef), e.g. 2.5 means "odds x2.5 per SD".
        """
        rows = []
        for level in self.classes_:
            coef = pd.Series(self.models_[level].coef_[0], index=self.features_)
            coef = coef[coef != 0]
            coef = coef.reindex(coef.abs().sort_values(ascending=False).index)
            for feature, c in (coef if top is None else coef[:top]).items():
                rows.append({"level": level, "feature": feature, "coef": round(float(c), 4),
                             "odds_ratio": round(float(np.exp(c)), 3)})
        return pd.DataFrame(rows, columns=["level", "feature", "coef", "odds_ratio"])

    def summary(self):
        """Per level: chosen penalty C and number of features kept."""
        return pd.DataFrame([{"level": lv, "C": self.C_[lv],
                              "n_features": int(np.sum(self.models_[lv].coef_[0] != 0)),
                              "n_features_total": len(self.features_)} for lv in self.classes_])

    def explain(self, X, top=5):
        """For each row: predicted level and the features that pushed it most.

        contribution = coef x standardized feature value (log-odds units).
        """
        Z = pd.DataFrame(self._prepare(X), columns=self.features_,
                         index=X.index if isinstance(X, pd.DataFrame) else None)
        proba = self.predict_proba(X)
        rows = []
        for idx, z in Z.iterrows():
            level = proba.loc[idx].idxmax()
            contrib = pd.Series(self.models_[level].coef_[0], index=self.features_) * z
            contrib = contrib[contrib != 0]
            for feature, c in contrib.reindex(contrib.abs().sort_values(ascending=False).index)[:top].items():
                rows.append({"row": idx, "predicted": level, "probability": round(float(proba.loc[idx, level]), 3),
                             "feature": feature, "contribution": round(float(c), 3)})
        return pd.DataFrame(rows)


def fit_sparse_levels(X, y, sample_weight=None, **kwargs) -> SparseLevels:
    """Fit a SparseLevels in one call; kwargs go to SparseLevels (levels, penalty, Cs, ...)."""
    return SparseLevels(**kwargs).fit(X, y, sample_weight=sample_weight)


if __name__ == "__main__":
    # Tiny demo on synthetic data: 3 "clusters", 12 features of which few matter.
    from sklearn.datasets import make_classification

    Xa, ya = make_classification(n_samples=600, n_features=12, n_informative=4, n_redundant=2,
                                 n_classes=3, n_clusters_per_class=1, random_state=0)
    X = pd.DataFrame(Xa, columns=[f"feature_{i}" for i in range(Xa.shape[1])])
    y = np.array(["cluster_A", "cluster_B", "cluster_C"])[ya]

    model = fit_sparse_levels(X.iloc[:500], y[:500])
    print("accuracy on held-out rows:", round(float(np.mean(model.predict(X.iloc[500:]) == y[500:])), 3))
    print("\nsummary:\n", model.summary().to_string(index=False))
    print("\ntop coefficients:\n", model.coefficients(top=3).to_string(index=False))
    print("\nwhy row 500:\n", model.explain(X.iloc[[500]], top=3).to_string(index=False))
