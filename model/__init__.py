"""The model: sparse per-label logistic regression (L1 / lasso). See sparse_levels.py."""

from model.sparse_levels import SparseLevelModel, fit_sparse_levels

__all__ = ["SparseLevelModel", "fit_sparse_levels"]
