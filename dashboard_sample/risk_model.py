"""Payment risk index model: continuation-ratio (sequential) logit on the risk snapshots.

    python -m dashboard_sample.risk_model      # fit on train- and test-split clients, report on valid clients

Outcome per client x snapshot (dashboard_sample/risk.py): 0 on time, 1 late, 2 missed.
Continuation-ratio logit (Agresti, Categorical Data Analysis, sec. 8.3), shared betas:
    stage 1  P(slip)            = sigmoid(a1 + eta)      slip = late or missed
    stage 2  P(missed | slip)   = sigmoid(a2 + eta)
    eta      = sum_j z_j * beta_j      z = winsorised (train 0.5-99.5 pct), imputed, standardised feature
It is fitted exactly as ONE logistic regression on stacked rows (every snapshot gives a
stage-1 row; slipped ones also a stage-2 row, with a stage dummy), so the sparse L1 and the
global L2 benchmark are both plain sklearn fits. No class weights: probabilities stay calibrated.

Risk index (0-100) = 100 x expected severity on the equally spaced 0/1/2 scale
                    = 50 * P(late) + 100 * P(missed),   monotone in eta.
Outcomes are read from the transactions, not from the challenge labels, so the test-split
clients' transactions can be used for fitting too (the hidden test labels are never needed);
valid clients are held out for every number reported here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from dashboard_sample.risk import build_snapshots, lean_columns

STAGE_SCALE = 10.0     # stage dummy value: big, so the penalty barely shrinks the stage offset
FIT_SPLITS, EVAL_SPLIT = ("train", "test"), "valid"
CS = (0.003, 0.01, 0.03, 0.1, 0.3)
CLIP_PCT = (0.5, 99.5)  # winsorise: one odd charge (e.g. a 570 "streaming" payment) must not extrapolate


class _Prep:
    """Winsorise at the fit data's percentiles, impute the median, standardise."""

    def fit_prep(self, X: pd.DataFrame):
        self.lo_, self.hi_ = (np.nanpercentile(X.to_numpy(float), q, axis=0) for q in CLIP_PCT)
        self.impute_ = SimpleImputer(strategy="median").fit(self._clip(X))
        self.scale_ = StandardScaler().fit(self.impute_.transform(self._clip(X)))
        return self

    def _clip(self, X) -> np.ndarray:
        return np.clip(np.asarray(X, float), self.lo_, self.hi_)   # NaN stays NaN

    def transform(self, X) -> np.ndarray:
        return self.scale_.transform(self.impute_.transform(self._clip(X)))


class ContinuationRatioLogit(_Prep):
    """Shared-beta continuation-ratio logit on 3 ordered outcomes (0 < 1 < 2)."""

    def __init__(self, penalty: str = "l1", C: float = 0.1):
        self.penalty, self.C = penalty, C

    def _clf(self):
        if self.penalty == "l1":
            return LogisticRegression(penalty="l1", solver="liblinear", C=self.C, max_iter=5000,
                                      intercept_scaling=STAGE_SCALE, random_state=0)
        return LogisticRegression(penalty="l2", C=self.C, max_iter=5000)

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        Z = self.fit_prep(X).transform(X)
        slip = y >= 1
        Zs = np.vstack([np.c_[Z, np.zeros(len(Z))], np.c_[Z[slip], np.full(slip.sum(), STAGE_SCALE)]])
        ys = np.r_[slip.astype(int), (y[slip] == 2).astype(int)]
        clf = self._clf().fit(Zs, ys)
        self.beta_ = clf.coef_[0][:-1]
        self.a1_ = clf.intercept_[0]
        self.a2_ = clf.intercept_[0] + clf.coef_[0][-1] * STAGE_SCALE
        return self

    def eta(self, X) -> np.ndarray:
        return self.transform(X) @ self.beta_

    def proba(self, X) -> np.ndarray:
        """Columns: P(on time), P(late), P(missed)."""
        return outcome_proba(self.eta(X), self.a1_, self.a2_)


def outcome_proba(eta, a1, a2) -> np.ndarray:
    p_slip = 1 / (1 + np.exp(-(a1 + eta)))
    p_miss = p_slip / (1 + np.exp(-(a2 + eta)))
    return np.column_stack([1 - p_slip, p_slip - p_miss, p_miss])


def risk_index(P: np.ndarray) -> np.ndarray:
    return 50 * P[:, 1] + 100 * P[:, 2]


class SeparateStages(_Prep):
    """Proportionality check: the same two stages, but each with its own betas."""

    def __init__(self, penalty="l1", C=0.1):
        self.penalty, self.C = penalty, C

    def fit(self, X, y):
        Z, slip = self.fit_prep(X).transform(X), y >= 1
        mk = lambda: (LogisticRegression(penalty="l1", solver="liblinear", C=self.C, max_iter=5000, random_state=0)
                      if self.penalty == "l1" else LogisticRegression(C=self.C, max_iter=5000))
        self.s1_ = mk().fit(Z, slip.astype(int))
        self.s2_ = mk().fit(Z[slip], (y[slip] == 2).astype(int))
        return self

    def proba(self, X):
        Z = self.transform(X)
        p_slip, p_m = self.s1_.predict_proba(Z)[:, 1], self.s2_.predict_proba(Z)[:, 1]
        return np.column_stack([1 - p_slip, p_slip * (1 - p_m), p_slip * p_m])


def labelled(splits) -> tuple[pd.DataFrame, list[str]]:
    snap = pd.concat([build_snapshots(s)[0].assign(split=s) for s in splits], ignore_index=True)
    return snap, lean_columns(snap.drop(columns="split"))


def past_rate_baseline(snap: pd.DataFrame, prior: np.ndarray, weight: float = 2.0,
                       lag_days: int = 84) -> np.ndarray:
    """Naive benchmark: the client's own past outcome mix (snapshots whose window had closed), smoothed."""
    s = snap[["client_id", "snapshot", "outcome"]].copy()
    s["t"] = pd.to_datetime(s["snapshot"])
    out = np.tile(prior, (len(s), 1)).astype(float)
    for cid, g in s.groupby("client_id"):
        g = g.sort_values("t")
        for i, row in g.iterrows():
            past = g[(g["t"] <= row["t"] - pd.Timedelta(days=lag_days)) & g["outcome"].notna()]["outcome"]
            counts = np.bincount(past.astype(int), minlength=3)
            out[s.index.get_loc(i)] = (counts + weight * prior) / (counts.sum() + weight)
    return out


def score(name, y, P) -> dict:
    P = np.clip(P, 1e-9, 1)
    P = P / P.sum(1, keepdims=True)
    idx = risk_index(P)
    return {"model": name, "log_loss": log_loss(y, P, labels=[0, 1, 2]),
            "auc_slip": roc_auc_score(y >= 1, idx), "auc_missed": roc_auc_score(y == 2, idx),
            "mean_index": idx.mean()}


def choose_C(X, y, groups, penalty) -> float:
    cv = {}
    for C in CS:
        losses = []
        for tr, te in GroupKFold(3).split(X, y, groups):
            m = ContinuationRatioLogit(penalty, C).fit(X.iloc[tr], y[tr])
            losses.append(log_loss(y[te], np.clip(m.proba(X.iloc[te]), 1e-9, 1), labels=[0, 1, 2]))
        cv[C] = np.mean(losses)
    print(f"  {penalty} CV log-loss by C:", {c: round(v, 4) for c, v in cv.items()})
    return min(cv, key=cv.get)


def evaluate(verbose: bool = True) -> dict:
    """Fit on FIT_SPLITS, score on the held-out EVAL_SPLIT clients. Returns tables and fitted models."""
    fit, cols = labelled(FIT_SPLITS)
    ev, _ = labelled([EVAL_SPLIT])
    fit_l, ev_l = fit[fit["outcome"].notna()], ev[ev["outcome"].notna()]
    X, y = fit_l[cols], fit_l["outcome"].astype(int).to_numpy()
    Xv, yv = ev_l[cols], ev_l["outcome"].astype(int).to_numpy()
    info = {"n_fit": len(fit_l), "n_fit_clients": fit_l["client_id"].nunique(), "fit_mix": np.bincount(y) / len(y),
            "n_eval": len(ev_l), "n_eval_clients": ev_l["client_id"].nunique(), "eval_mix": np.bincount(yv) / len(yv)}
    if verbose:
        print(f"fit: {info['n_fit']} labelled snapshots of {info['n_fit_clients']} clients, mix {info['fit_mix'].round(3)}")
        print(f"eval ({EVAL_SPLIT}): {info['n_eval']} labelled snapshots, mix {info['eval_mix'].round(3)}")

    prior = np.bincount(y, minlength=3) / len(y)
    rows = [score("constant (training mix)", yv, np.tile(prior, (len(yv), 1))),
            score("client's own past rate", yv, past_rate_baseline(ev, prior)[ev["outcome"].notna().to_numpy()])]
    models = {}
    for penalty in ("l1", "l2"):
        C = choose_C(X, y, fit_l["client_id"], penalty)
        m = ContinuationRatioLogit(penalty, C).fit(X, y)
        models[penalty] = m
        name = "sparse L1 CR-logit" if penalty == "l1" else "global L2 CR-logit"
        rows.append({**score(f"{name} (C={C})", yv, m.proba(Xv)), "n_features": int((m.beta_ != 0).sum())})
        rows.append(score("  same, separate stage betas", yv, SeparateStages(penalty, C).fit(X, y).proba(Xv)))
    rows = pd.DataFrame(rows)

    ev_l = ev_l.assign(index=risk_index(models["l1"].proba(Xv)))
    ev_l["decile"] = pd.qcut(ev_l["index"], 10, labels=False, duplicates="drop")
    cal = ev_l.groupby("decile").agg(mean_index=("index", "mean"),
                                     observed_index=("outcome", lambda o: 50 * o.mean()),
                                     late=("outcome", lambda o: (o == 1).mean()),
                                     missed=("outcome", lambda o: (o == 2).mean()), n=("outcome", "size"))
    if verbose:
        print(rows.round(4).to_string(index=False))
        print("\ncalibration on valid (observed index = 50 x mean outcome):")
        print(cal.round(3).to_string())
        coef = pd.Series(models["l1"].beta_, index=cols)
        print("\ntop betas (sparse L1):")
        print(coef[coef != 0].sort_values(key=abs, ascending=False).head(15).round(3).to_string())
    return {"rows": rows, "calibration": cal, "models": models, "cols": cols, "info": info}


if __name__ == "__main__":
    evaluate()
