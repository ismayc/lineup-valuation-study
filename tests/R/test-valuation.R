# Unit tests for lineup-valuation-study/R/functions.R

source(file.path(REPO, "R", "functions.R"))

test_that("ridge with zero penalty equals weighted least squares", {
  set.seed(7)
  X <- cbind(1, matrix(rnorm(150), 50, 3))
  beta_true <- c(1, 2, -1, 0.5)
  w <- runif(50, 0.5, 2)
  y <- X %*% beta_true + rnorm(50, sd = 0.01)
  beta <- ridge_fit(X, y, w, lam = 0)
  expected <- solve(crossprod(X, X * w), crossprod(X * w, y))[, 1]
  expect_equal(beta, expected, tolerance = 1e-10)
})

test_that("large penalty shrinks players but not the intercept", {
  set.seed(8)
  X <- cbind(1, matrix(rbinom(200, 1, 0.5), 100, 2))
  y <- 5 + rnorm(100)
  beta <- ridge_fit(X, y, rep(1, 100), lam = 1e9)
  expect_lt(abs(beta[2]), 1e-3)
  expect_lt(abs(beta[3]), 1e-3)
  expect_equal(beta[1], mean(y), tolerance = 0.05)
})

test_that("cv_lambda is deterministic and returns a grid value", {
  set.seed(9)
  X <- cbind(1, matrix(rbinom(120, 1, 0.5), 40, 3))
  y <- rnorm(40)
  w <- rep(1, 40)
  grid <- c(1, 10, 100)
  a <- cv_lambda(X, y, w, grid)
  b <- cv_lambda(X, y, w, grid)
  expect_equal(a$best, b$best)
  expect_equal(a$curve, b$curve)
  expect_true(a$best %in% grid)
})

# ---- bootstrap CIs -----------------------------------------------------------
# The study's headline is the width of these intervals, so they get the same
# scrutiny as the point estimates rather than being taken on trust.

toy_design <- function(seed = 11, n = 240, p = 6) {
  set.seed(seed)
  X <- matrix(0, n, p + 1)
  X[, 1] <- 1
  for (i in seq_len(n)) X[i, 1 + sample.int(p, 3)] <- 1
  beta <- c(0, seq(-3, 3, length.out = p))
  w <- runif(n, 50, 400)
  y <- as.vector(X %*% beta) + rnorm(n, sd = 4)
  list(X = X, y = y, w = w)
}

test_that("bootstrap CIs are reproducible from the seed", {
  d <- toy_design()
  a <- bootstrap_ci(d$X, d$y, d$w, lam = 100, reps = 40, seed = 3)
  b <- bootstrap_ci(d$X, d$y, d$w, lam = 100, reps = 40, seed = 3)
  expect_identical(a, b)
  c2 <- bootstrap_ci(d$X, d$y, d$w, lam = 100, reps = 40, seed = 4)
  expect_false(isTRUE(all.equal(a, c2)))
})

test_that("bootstrap CIs are ordered and shaped like the design", {
  # The pipeline slices columns 2:(1 + p) off this and beta together, so a
  # shape mismatch would attach every player to the wrong interval.
  d <- toy_design()
  ci <- bootstrap_ci(d$X, d$y, d$w, lam = 100, reps = 60, seed = 5)
  expect_equal(dim(ci), c(2L, ncol(d$X)))
  expect_true(all(ci[1, ] < ci[2, ]))
})

test_that("bootstrap CIs bracket the point estimate", {
  # A percentile interval that misses the full-sample fit would mean the
  # resampling is not centered on the estimator it describes.
  d <- toy_design()
  beta <- ridge_fit(d$X, d$y, d$w, 100)
  ci <- bootstrap_ci(d$X, d$y, d$w, lam = 100, reps = 200, seed = 6)
  expect_true(all(ci[1, ] <= beta))
  expect_true(all(beta <= ci[2, ]))
})

test_that("bootstrap CIs narrow as evidence grows", {
  # The claim about error bars is that they track evidence, so ten times the
  # lineups must tighten the intervals rather than merely move them.
  small <- toy_design(seed = 12, n = 120)
  large <- toy_design(seed = 12, n = 1200)
  ws <- diff(bootstrap_ci(small$X, small$y, small$w, 100, reps = 80, seed = 8))
  wl <- diff(bootstrap_ci(large$X, large$y, large$w, 100, reps = 80, seed = 8))
  expect_lt(median(wl), median(ws))
})

test_that("bootstrap resamples lineups with replacement", {
  # Sampling without replacement would refit identical data every rep and
  # collapse the intervals to zero width.
  d <- toy_design()
  ci <- bootstrap_ci(d$X, d$y, d$w, lam = 100, reps = 50, seed = 9)
  expect_true(all(ci[2, ] - ci[1, ] > 0))
})
