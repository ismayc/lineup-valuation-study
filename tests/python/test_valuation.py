"""Unit tests for lineup-valuation-study/python/02_analysis.py model core."""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest


def test_ridge_zero_penalty_equals_wls(valuation):
    rng = np.random.default_rng(7)
    X = np.column_stack([np.ones(50), rng.normal(size=(50, 3))])
    beta_true = np.array([1.0, 2.0, -1.0, 0.5])
    w = rng.uniform(0.5, 2.0, 50)
    y = X @ beta_true + rng.normal(scale=0.01, size=50)
    beta = valuation.ridge_fit(X, y, w, lam=0.0)
    Xw = X * w[:, None]
    expected = np.linalg.solve(X.T @ Xw, Xw.T @ y)
    assert np.allclose(beta, expected, atol=1e-10)


def test_ridge_large_penalty_shrinks_players_not_intercept(valuation):
    rng = np.random.default_rng(8)
    X = np.column_stack([np.ones(100), rng.integers(0, 2, size=(100, 2)).astype(float)])
    y = 5.0 + rng.normal(size=100)
    w = np.ones(100)
    beta = valuation.ridge_fit(X, y, w, lam=1e9)
    assert abs(beta[1]) < 1e-3 and abs(beta[2]) < 1e-3   # players -> 0
    assert beta[0] == pytest.approx(y.mean(), abs=0.05)  # intercept free


def test_build_design_replacement_pooling(valuation):
    # Player "1" and "2" clear the possession threshold; "9" does not and
    # must land in the replacement column, not get its own.
    df = pl.DataFrame({
        "player_ids": [["1", "2", "3", "4", "5"], ["1", "2", "3", "4", "9"]],
        "net_100": [10.0, -5.0],
        # "9" appears only in the second lineup, one possession short of the
        # threshold; everyone else clears it via the first lineup.
        "POSS": [float(valuation.MIN_POSS), float(valuation.MIN_POSS - 1)],
    })
    X, y, w, kept = valuation.build_design(df)
    assert "9" not in kept
    assert set(["1", "2", "3", "4", "5"]) <= set(kept)
    assert X.shape == (2, len(kept) + 2)
    assert X[:, 0].tolist() == [1.0, 1.0]         # intercept
    assert X[0, -1] == 0.0 and X[1, -1] == 1.0    # replacement count
    assert X[0, 1:-1].sum() == 5.0                # five kept players in row 0


def test_cv_folds_deterministic(valuation):
    rng = np.random.default_rng(9)
    X = np.column_stack([np.ones(40), rng.integers(0, 2, size=(40, 3)).astype(float)])
    y = rng.normal(size=40)
    w = np.ones(40)
    best1, curve1 = valuation.cv_lambda(X, y, w)
    best2, curve2 = valuation.cv_lambda(X, y, w)
    assert best1 == best2
    assert curve1.equals(curve2)
