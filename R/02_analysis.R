# Player value from lineup data --- R / tidyverse implementation
#
# Independent implementation of python/02_analysis.py: same model (possession-
# weighted ridge with an unpenalized intercept and a pooled replacement column),
# same deterministic 5-fold CV, written from the definitions so that
# python/03_reconcile.py is a real check.
#
# Reconciliation notes, the two places where R and Python silently diverge:
#   * Row and player ordering use C-locale byte order (method = "radix"),
#     matching polars' string sort. R's default locale-aware sort does NOT.
#   * The CV fold of row i (0-based) is i %% 5 on the sorted rows.
#
# Outputs per season: output/<season>/player_values_r.csv, cv_curve_r.csv,
#                     lineup_table_r.csv, validation_r.csv, bootstrap CIs
#                     figures/<season>/fig1_player_values_r.png
#
# Run:  Rscript R/02_analysis.R

suppressPackageStartupMessages({
  library(nanoparquet)
  library(dplyr)
  library(tidyr)
  library(stringr)
  library(purrr)
  library(readr)
  library(ggplot2)
})

ROOT <- normalizePath(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1])), ".."))
if (is.na(ROOT) || !dir.exists(ROOT)) ROOT <- normalizePath(".")

MIN_POSS <- 300
LAMBDAS <- c(100, 200, 400, 800, 1600, 3200, 6400)
N_FOLDS <- 5
BOOT_REPS <- 500

PAL <- c(blue = "#2a78d6", orange = "#eb6834")
INK <- "#0b0b0b"; INK2 <- "#52514e"; MUTED <- "#898781"
GRID <- "#e1e0d9"; AXIS <- "#c3c2b7"; SURFACE <- "#fcfcfb"

# ridge_fit() and cv_lambda() live in R/functions.R and are unit-tested.
source(file.path(ROOT, "R", "functions.R"))

run_season <- function(season) {
  raw <- file.path(ROOT, "data", "raw", season)
  out_dir <- file.path(ROOT, "output", season)
  fig_dir <- file.path(ROOT, "figures", season)
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)

  message("=== ", season, " ===")

  read_all <- function(pattern) {
    files <- sort(list.files(raw, pattern = pattern, full.names = TRUE))
    map_dfr(files, read_parquet)
  }
  base <- read_all("^lineups_base_.*\\.parquet$")
  adv <- read_all("^lineups_advanced_.*\\.parquet$")

  df_full <- base %>%
    select(GROUP_ID, TEAM_ID, TEAM_ABBREVIATION, GROUP_NAME, MIN, PLUS_MINUS) %>%
    inner_join(adv %>% select(GROUP_ID, TEAM_ID, POSS, NET_RATING),
               by = c("GROUP_ID", "TEAM_ID"))
  # The model drops zero-possession lineups (net_100 undefined: defensive-only
  # micro-stints), but the coverage validation keeps them - their plus-minus
  # points are real and team totals must reconstruct exactly.
  df <- df_full %>%
    filter(POSS > 0) %>%
    mutate(net_100 = 100 * PLUS_MINUS / POSS)

  # C-locale byte-order sort to match polars exactly.
  df <- df[order(df$GROUP_ID, df$TEAM_ID, method = "radix"), ]

  id_lists <- str_split(str_remove_all(df$GROUP_ID, "^-|-$"), "-")
  stopifnot(all(lengths(id_lists) == 5))

  message(sprintf("%s lineups, %d teams, %s possessions",
                  format(nrow(df), big.mark = ","), n_distinct(df$TEAM_ID),
                  format(round(sum(df$POSS)), big.mark = ",")))

  # ---- design matrix -------------------------------------------------------
  poss_by_player <- tapply(rep(df$POSS, each = 5), unlist(id_lists), sum)
  kept <- sort(names(poss_by_player)[poss_by_player >= MIN_POSS], method = "radix")
  col_of <- setNames(seq_along(kept), kept)

  n <- nrow(df); p <- length(kept)
  X <- matrix(0, n, p + 2)   # [intercept | players | replacement]
  X[, 1] <- 1
  for (i in seq_len(n)) {
    for (pid in id_lists[[i]]) {
      j <- col_of[pid]
      if (is.na(j)) X[i, p + 2] <- X[i, p + 2] + 1 else X[i, 1 + j] <- 1
    }
  }
  y <- df$net_100
  w <- df$POSS

  # ---- CV over the lambda grid (deterministic folds) -----------------------
  cv <- cv_lambda(X, y, w, LAMBDAS, N_FOLDS)
  curve <- cv$curve
  write_csv(curve, file.path(out_dir, "cv_curve_r.csv"))
  best_lam <- cv$best
  message("lambda by 5-fold CV: ", best_lam)

  beta <- ridge_fit(X, y, w, best_lam)

  # ---- bootstrap over lineups (R RNG; reconciled by CI overlap) ------------
  set.seed(2026)
  boots <- matrix(NA_real_, BOOT_REPS, ncol(X))
  for (b in seq_len(BOOT_REPS)) {
    idx <- sample.int(n, n, replace = TRUE)
    boots[b, ] <- ridge_fit(X[idx, , drop = FALSE], y[idx], w[idx], best_lam)
  }
  ci <- apply(boots, 2, quantile, probs = c(0.025, 0.975))

  poss_kept <- poss_by_player[kept]

  names_tbl <- read_parquet(file.path(raw, "player_stats.parquet")) %>%
    transmute(player_id = as.character(PLAYER_ID), player = PLAYER_NAME,
              team = TEAM_ABBREVIATION, season_min = MIN, season_pm = PLUS_MINUS)

  values <- tibble(
    player_id = kept,
    rapm_100 = beta[2:(1 + p)],
    ci_lo = ci[1, 2:(1 + p)],
    ci_hi = ci[2, 2:(1 + p)],
    poss = as.numeric(poss_kept),
  ) %>%
    left_join(names_tbl, by = "player_id") %>%
    arrange(desc(rapm_100)) %>%
    mutate(rank = row_number(), .before = 1)
  write_csv(values, file.path(out_dir, "player_values_r.csv"))

  df %>%
    filter(POSS >= 1000) %>%
    select(TEAM_ABBREVIATION, GROUP_NAME, MIN, POSS, PLUS_MINUS, net_100, NET_RATING) %>%
    arrange(desc(net_100)) %>%
    write_csv(file.path(out_dir, "lineup_table_r.csv"))

  # ---- external validation (same checks as Python) -------------------------
  teams <- read_parquet(file.path(raw, "team_stats.parquet"))
  cov <- df_full %>%
    group_by(TEAM_ID) %>%
    summarise(lineup_min = sum(MIN), lineup_pm = sum(PLUS_MINUS), .groups = "drop") %>%
    inner_join(teams %>% select(TEAM_ID, team_min = MIN, team_pm = PLUS_MINUS),
               by = "TEAM_ID")
  min_ratio_worst <- max(abs(cov$lineup_min / cov$team_min - 1))
  pm_err_max <- max(abs(cov$lineup_pm - cov$team_pm))

  big <- df %>% filter(POSS >= 200)
  r_net <- cor(big$net_100, big$NET_RATING)

  vv <- values %>% filter(!is.na(season_pm)) %>%
    mutate(pm_per_min = season_pm / season_min)
  rho_pm <- cor(vv$rapm_100, vv$pm_per_min, method = "spearman")

  validation <- tibble(
    check = c("lineup minutes reconstruct team minutes",
              "lineup plus-minus reconstructs team plus-minus (max abs pts)",
              "net_100 vs NBA NET_RATING correlation (POSS>=200)",
              "Spearman(RAPM, raw plus-minus per minute)"),
    value = c(min_ratio_worst, pm_err_max, r_net, rho_pm),
    threshold = c(0.005, 1.0, 0.97, 0.5),
    pass = c(min_ratio_worst <= 0.005, pm_err_max <= 1.0, r_net >= 0.97, rho_pm >= 0.5)
  )
  write_csv(validation, file.path(out_dir, "validation_r.csv"))
  print(as.data.frame(validation))
  stopifnot(all(validation$pass))

  # ---- figure --------------------------------------------------------------
  show <- bind_rows(head(values, 15), tail(values, 15)) %>%
    mutate(player = factor(player, levels = player[order(rapm_100)]))
  p1 <- ggplot(show, aes(rapm_100, player)) +
    geom_vline(xintercept = 0, colour = AXIS, linewidth = 0.3) +
    geom_errorbarh(aes(xmin = ci_lo, xmax = ci_hi), height = 0,
                   colour = MUTED, linewidth = 0.4) +
    geom_point(aes(colour = rapm_100 > 0), size = 2.2, show.legend = FALSE) +
    scale_colour_manual(values = c(`TRUE` = PAL[["blue"]], `FALSE` = PAL[["orange"]])) +
    labs(
      title = paste0("Regularized player value from ", season, " lineup data"),
      subtitle = paste0("Points per 100 possessions vs league average · ridge, lambda=",
                        best_lam, " by CV · 95% bootstrap CI"),
      x = "Net points per 100 possessions (vs average)", y = NULL
    ) +
    theme_minimal(base_size = 11) +
    theme(
      plot.background = element_rect(fill = SURFACE, colour = NA),
      panel.background = element_rect(fill = SURFACE, colour = NA),
      panel.grid.major = element_line(colour = GRID, linewidth = 0.3),
      panel.grid.minor = element_blank(),
      axis.text = element_text(colour = MUTED, size = 8.5),
      axis.title = element_text(colour = INK2, size = 9),
      plot.title = element_text(colour = INK, face = "bold", size = 12),
      plot.subtitle = element_text(colour = INK2, size = 9)
    )
  ggsave(file.path(fig_dir, "fig1_player_values_r.png"), p1,
         width = 7.5, height = 6.3, dpi = 200)

  message("Top 5: ", paste(head(values$player, 5), collapse = ", "))
}

seasons <- list.dirs(file.path(ROOT, "data", "raw"), recursive = FALSE, full.names = FALSE)
for (s in sort(seasons)) run_season(s)
message("R analysis complete.")
