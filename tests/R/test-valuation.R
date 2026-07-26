# Unit tests for lineup-valuation-study/R/functions.R

source(file.path(REPO, "lineup-valuation-study", "R", "functions.R"))

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
