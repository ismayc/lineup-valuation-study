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

# ---- bootstrap CIs -----------------------------------------------------------
# The study's headline is the width of these intervals ("treat any lineup-value
# claim without an error bar as marketing"), so they get the same scrutiny as
# the point estimates rather than being taken on trust.


def _toy_design(seed=11, n=240, p=6):
    """A small lineup-shaped problem: intercept, p player indicators, each row
    a 'lineup' of three players drawn from the pool."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n, p + 1))
    X[:, 0] = 1.0
    for i in range(n):
        for j in rng.choice(p, size=3, replace=False):
            X[i, 1 + j] = 1.0
    beta = np.concatenate([[0.0], np.linspace(-3, 3, p)])
    w = rng.uniform(50, 400, n)
    y = X @ beta + rng.normal(scale=4.0, size=n)
    return X, y, w


def test_bootstrap_ci_is_reproducible_from_the_seed(valuation):
    # The published CIs are only meaningful if they regenerate, so the seed
    # has to fully determine them.
    X, y, w = _toy_design()
    a = valuation.bootstrap_ci(X, y, w, lam=100.0, reps=40, seed=3)
    b = valuation.bootstrap_ci(X, y, w, lam=100.0, reps=40, seed=3)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    c = valuation.bootstrap_ci(X, y, w, lam=100.0, reps=40, seed=4)
    assert not np.allclose(a[0], c[0])


def test_bootstrap_ci_is_ordered_and_shaped_like_the_design(valuation):
    X, y, w = _toy_design()
    lo, hi = valuation.bootstrap_ci(X, y, w, lam=100.0, reps=60, seed=5)
    # One entry per design column, intercept included: the pipeline slices
    # [1:1+len(kept)] off both these and beta, so a shape mismatch would
    # silently attach every player to the wrong interval.
    assert lo.shape == hi.shape == (X.shape[1],)
    assert np.all(lo < hi)


def test_bootstrap_ci_brackets_the_point_estimate(valuation):
    # A percentile interval that does not contain the full-sample fit would
    # mean the resampling is not centered on the estimator it describes.
    X, y, w = _toy_design()
    lam = 100.0
    beta = valuation.ridge_fit(X, y, w, lam)
    lo, hi = valuation.bootstrap_ci(X, y, w, lam, reps=200, seed=6)
    assert np.all(lo <= beta) and np.all(beta <= hi)


def test_bootstrap_ci_reports_the_requested_quantiles(valuation):
    # Pin the 2.5/97.5 levels: a percentile bootstrap silently reporting a
    # different width is exactly the failure the study's headline can't take.
    X, y, w = _toy_design()
    lam, reps, seed = 100.0, 120, 7
    lo, hi = valuation.bootstrap_ci(X, y, w, lam, reps=reps, seed=seed)
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    boots = np.array([
        valuation.ridge_fit(*(lambda i: (X[i], y[i], w[i]))(
            rng.integers(0, n, n)), lam) for _ in range(reps)])
    want_lo, want_hi = np.quantile(boots, [0.025, 0.975], axis=0)
    assert np.allclose(lo, want_lo) and np.allclose(hi, want_hi)


def test_bootstrap_ci_narrows_as_evidence_grows(valuation):
    # The study's whole claim about error bars is that they track evidence.
    # Ten times the lineups should tighten the intervals, not just move them.
    lam = 100.0
    Xs, ys, ws = _toy_design(seed=12, n=120)
    Xl, yl, wl = _toy_design(seed=12, n=1200)
    small = np.subtract(*reversed(valuation.bootstrap_ci(
        Xs, ys, ws, lam, reps=80, seed=8)))
    large = np.subtract(*reversed(valuation.bootstrap_ci(
        Xl, yl, wl, lam, reps=80, seed=8)))
    assert np.median(large) < np.median(small)


def test_bootstrap_resamples_lineups_not_observations(valuation):
    # Every replicate must draw n rows with replacement from the n available.
    # Sampling without replacement would refit the same data 500 times and
    # collapse the intervals to zero width.
    X, y, w = _toy_design()
    lo, hi = valuation.bootstrap_ci(X, y, w, lam=100.0, reps=50, seed=9)
    assert np.all(hi - lo > 0)
