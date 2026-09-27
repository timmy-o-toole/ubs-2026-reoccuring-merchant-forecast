"""Features and model: raw transactions table -> one feature row per client -> SparseLevels.

Input: one row per transaction with the columns
    client_id, timestamp (ISO string or datetime), amount, direction ("in"/"out"),
    type (card_payment, refund, topup, atm, fee, p2p_transfer, ...), mcc (string), description.
Other columns (fee, currency, ...) are ignored. A 'category' column is added
by tag_transactions if missing. Pass each client's history up to the cutoff.

Usage:
    import pandas as pd
    from code import build_features, select_features, fit_sparse_levels
    X = build_features(pd.read_json("data/valid_transactions.jsonl", lines=True, dtype={"mcc": str}), "2026-01-01")
    X_lean = select_features(X.drop(columns="client_id"), "lean")
    model = fit_sparse_levels(X_lean, y)

Modules: features_category_map (transaction -> family tag), features_recurrence
(recurring streams), features_build (client feature table), model (SparseLevels,
one sparse logistic regression per level). None of them reads files.
"""

from code.features_build import FEATURE_SETS, build_features, select_features
from code.features_category_map import TARGET_CATEGORIES
from code.features_recurrence import detect_streams, load_transactions, tag_transactions
from code.model import SparseLevels, fit_sparse_levels

__all__ = [
    "build_features",
    "select_features",
    "FEATURE_SETS",
    "load_transactions",
    "tag_transactions",
    "detect_streams",
    "TARGET_CATEGORIES",
    "SparseLevels",
    "fit_sparse_levels",
]
