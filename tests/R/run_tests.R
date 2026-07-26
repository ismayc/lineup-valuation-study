# Run the R unit-test suite: Rscript tests/R/run_tests.R
# Exits non-zero on any failure so it can gate a commit alongside pytest.

suppressPackageStartupMessages(library(testthat))

args <- commandArgs()
this_file <- sub("^--file=", "", grep("^--file=", args, value = TRUE)[1])
REPO <<- normalizePath(file.path(dirname(this_file), "..", ".."))

results <- test_dir(dirname(this_file), env = globalenv(), stop_on_failure = TRUE)
