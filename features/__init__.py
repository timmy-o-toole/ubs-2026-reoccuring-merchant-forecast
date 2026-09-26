"""Feature generation: raw transactions table -> one feature row per client.

Input: one row per transaction with the columns
    client_id, timestamp (ISO string or datetime), amount, direction ("in"/"out"),
    type (card_payment, refund, topup, atm, fee, p2p_transfer, ...), mcc (string), description.
Other columns (fee, currency, ...) are ignored. A 'category' column is added
by tag_transactions if missing. Pass each client's history up to the cutoff.

Usage:
    import pandas as pd
    from features import build_features, select_features
    X = build_features(pd.read_json("data/valid_transactions.jsonl", lines=True, dtype={"mcc": str}), "2026-01-01")
    X_lean = select_features(X.drop(columns="client_id"), "lean")

Modules: category_map (transaction -> family tag), recurrence (recurring
streams), build (client feature table).
"""

from features.build import FEATURE_SETS, build_features, select_features
from features.category_map import TARGET_CATEGORIES
from features.recurrence import detect_streams, load_transactions, tag_transactions

__all__ = [
    "build_features",
    "select_features",
    "FEATURE_SETS",
    "load_transactions",
    "tag_transactions",
    "detect_streams",
    "TARGET_CATEGORIES",
]
