# Lineup valuation study — regularized player value from five-man lineup data

Work sample addressing `../skills_matrix.md` requirements **#4/#20** (*build and
validate models evaluating players, teams, lineups*), **#3** (*statistical
modeling and ML*), and **#5** (*lineup/personnel data*). This was the gap the
other three studies could not close: none of them valued a player or a lineup.

**The question.** Which players most improve their team's net scoring when they
are on the floor, once you stop giving them credit for who they share the floor
with — and which five-man lineups actually worked?

**Two seasons:** 2023-24 (matching `../playbyplay-study`) and **2025-26, the
just-completed season** — the lineup endpoints are live, so the model runs on
current basketball (see `../docs/public-data-availability.md`).

---

## 1. Where this sits in the literature

Nothing here is invented in a vacuum; the model is the entry point of a
well-developed public lineage, and knowing that lineage is part of the point:

- **Adjusted plus-minus (APM)** — Rosenbaum (2004), building on Winston &
  Sagarin's WINVAL: regress stint score margins on player indicators, so a
  player's rating is *adjusted* for teammates and opponents rather than raw
  on-court margin.
- **Regularized APM (RAPM)** — Sill (2010, MIT Sloan Sports Analytics
  Conference): APM's coefficients explode because lineups are collinear; ridge
  regression (Tikhonov regularization) shrinks players toward league average
  in proportion to evidence and dramatically improves out-of-sample accuracy.
  This is the backbone of most public player metrics since.
- **Modern descendants** — ESPN RPM, FiveThirtyEight RAPTOR, Dunks & Threes
  EPM, DARKO DPM: all blend a RAPM-style on/off core with box-score and
  tracking priors to stabilize small samples.

This study implements the **ridge core of that lineage on season-aggregated
five-man lineup data**. The honest difference from full RAPM: season lineup
aggregates do not record *opponents*, so opponent strength is not adjusted
(limitation 1). The public route to full stint-level RAPM — play-by-play with
players-on-court filled in — is documented in
`../docs/public-data-availability.md` and is the natural next step.

## 2. Data

`stats.nba.com` via `nba_api`, per team (the league-wide lineup call silently
caps at 2,000 rows — per-team calls stay under it):

| Endpoint | Content |
|---|---|
| `LeagueDashLineups` (Base, Totals) | every five-man lineup's minutes and raw plus-minus |
| `LeagueDashLineups` (Advanced, Totals) | possessions and the NBA's own NET_RATING per lineup |
| `LeagueDashPlayerStats` | per-player totals — external reference for validation |
| `LeagueDashTeamStats` | team totals — the coverage check target |

Scale: ~15,900 lineup rows (2023-24) and ~19,500 (2025-26) covering ~255k
possessions per season.

## 3. Model

Each lineup's net points per 100 possessions is the sum of five player effects
plus an intercept, fitted by **possession-weighted ridge regression**:

- **Weights:** possessions — a 900-possession lineup carries 900× the evidence
  of a 1-possession lineup.
- **Penalty:** shrinks every player toward league average; chosen by **5-fold
  cross-validation** on a log-spaced grid. Folds are deterministic (row index
  mod 5 on byte-order-sorted rows) so the R implementation reproduces the
  choice exactly.
- **Replacement pool:** players under 300 possessions become one shared
  "replacement" column — their individual effects are unidentifiable, and
  pooling is more honest than reporting shrunken noise.
- **Uncertainty:** 500 bootstrap resamples of lineups, 95% percentile CIs on
  every player.
- **Zero-possession lineups** (defensive-only micro-stints, e.g. on the floor
  solely for opponent free throws) are excluded from the model — net-per-100
  is undefined for them — but **retained in the coverage validation**, where
  their points matter.

## 4. Pipeline

```
python/01_harvest_lineups.py   stats.nba.com -> data/raw/<season>/*.parquet (resumable)
python/02_analysis.py          -> output/<season>/*.csv, figures/<season>/*
R/02_analysis.R                independent R/tidyverse implementation -> *_r.csv
python/03_reconcile.py         R vs Python row-by-row, non-zero exit on mismatch
python/04_findings.py          writes the Findings section below
```

Same discipline as the other studies: two implementations written from the
definitions, reconciled to numeric tolerance (bootstrap CIs, which use each
language's RNG, are compared for per-player overlap instead).

## 5. Validation

Four external checks, all of which must pass before findings are written:

1. **Lineup minutes reconstruct team minutes** (max relative error < 0.5%).
2. **Lineup plus-minus reconstructs team plus-minus exactly** — the full
   lineup table sums to each team's season plus-minus to the point.
3. **Independent rate agrees with the NBA's own NET_RATING** (r ≥ 0.97 on
   200+ possession lineups; not identical by construction, since the league
   computes off/def ratings on separate possession denominators).
4. **Sanity against raw plus-minus** (Spearman ≥ 0.5) — high enough to show
   the model is anchored in reality, and deliberately *not* 1: the daylight
   between raw plus-minus and the regularized estimate is the teammate
   adjustment doing its work.

## 6. Limitations

1. **No opponent adjustment.** Season lineup aggregates do not say who the
   lineup played against. A player whose minutes come against opposing bench
   units gets bench-unit-inflated numbers. Full stint-level RAPM fixes this
   and is the documented next step.
2. **One season at a time.** Single-season RAPM-family estimates are noisy;
   the bootstrap CIs say so honestly (±2-3 points per 100). Public production
   metrics stabilize with multi-season priors and box/tracking blends.
3. **Offense and defense are not separated.** Net effects only. The same
   machinery runs on OFF_RATING/DEF_RATING per lineup; not done here.
4. **Low-minute players are pooled**, not estimated. The 300-possession
   threshold is a judgment call, stated rather than hidden.
5. **Trades blur team labels.** Players are valued across all their lineups
   regardless of team; the team column in the output is their season-end
   listing.

---

## Findings

*(generated by `python/04_findings.py` — run the pipeline first)*
