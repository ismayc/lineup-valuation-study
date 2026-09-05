# Pure functions for the lineup valuation model, extracted so both the
# pipeline script (02_analysis.R) and the test suite (../../tests/R) can
# source them.

#' Closed-form possession-weighted ridge; column 1 (intercept) unpenalized.
#' Storage-agnostic: X may be a base matrix or a Matrix-package sparse matrix
#' (the design is ~7 nonzeros per row, so sparse crossprod is what makes the
#' 500-rep bootstrap finish in seconds rather than an hour).
ridge_fit <- function(X, y, w, lam) {
  Xw <- X * w
  A <- as.matrix(crossprod(X, Xw))
  pen <- rep(lam, ncol(X)); pen[1] <- 0
  diag(A) <- diag(A) + pen
  as.vector(solve(A, as.matrix(crossprod(Xw, y))))
}

#' Deterministic 5-fold CV over a lambda grid; fold of 0-based row i is
#' i %% n_folds. Returns list(best = lambda, curve = tibble).
cv_lambda <- function(X, y, w, lambdas, n_folds = 5) {
  folds <- (seq_along(y) - 1) %% n_folds
  curve <- purrr::map_dfr(lambdas, function(lam) {
    sse <- 0; wsum <- 0
    for (k in 0:(n_folds - 1)) {
      tr <- folds != k; te <- folds == k
      beta <- ridge_fit(X[tr, , drop = FALSE], y[tr], w[tr], lam)
      resid <- y[te] - X[te, , drop = FALSE] %*% beta
      sse <- sse + sum(w[te] * resid^2)
      wsum <- wsum + sum(w[te])
    }
    tibble::tibble(lam = lam, cv_wmse = sse / wsum)
  })
  list(best = curve$lam[which.min(curve$cv_wmse)], curve = curve)
}

#' Percentile bootstrap over lineups at a fixed penalty.
#'
#' Resampling rows treats the lineup-season as the sampling unit, so the
#' interval reflects how much a player's estimate depends on which lineups
#' happened to exist rather than on within-lineup sampling. Returns a 2 x
#' ncol(X) matrix: row 1 the 2.5th percentile, row 2 the 97.5th, one column
#' per design column (column 1 is the intercept, as in ridge_fit).
#'
#' Seeded per call, which is what makes the published CIs reproducible. The
#' Python twin draws from a different generator on purpose; the two are
#' reconciled by interval overlap, not by equality.
bootstrap_ci <- function(X, y, w, lam, reps, seed = 2026) {
  set.seed(seed)
  n <- nrow(X)
  boots <- matrix(NA_real_, reps, ncol(X))
  for (b in seq_len(reps)) {
    idx <- sample.int(n, n, replace = TRUE)
    boots[b, ] <- ridge_fit(X[idx, , drop = FALSE], y[idx], w[idx], lam)
  }
  apply(boots, 2, quantile, probs = c(0.025, 0.975))
}
