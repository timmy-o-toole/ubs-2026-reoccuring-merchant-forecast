"""The model: one sparse L1 logistic regression per label.

The method itself lives in the standalone file sparse_levels.py; this module
only fixes it to our task (the 8 labels, in a fixed order). Interpret a fitted
model with .coefficients(), .summary() and .explain().
"""

from __future__ import annotations

from sparse_levels import SparseLevelModel
from src.category_map import TARGET_CATEGORIES

LABEL_COL = "target_next_recurring_merchant"
ALL_LABELS = TARGET_CATEGORIES + ["none"]


def build_model() -> SparseLevelModel:
    """One sparse (L1) yes/no logistic regression per label; highest probability wins."""
    return SparseLevelModel(levels=ALL_LABELS)
