# Lineup valuation study — regularized player value from five-man lineup data

Player and lineup value from public data, in the [basketball-data-science](https://github.com/ismayc/basketball-data-science)
family. The sibling studies describe events; this one values the people on
the floor.

**The question.** Which players most improve their team's net scoring when they
are on the floor, once you stop giving them credit for who they share the floor
with — and which five-man lineups actually worked?

**Two seasons:** 2023-24 (matching [playbyplay-study](https://github.com/ismayc/playbyplay-study)) and **2025-26, the
just-completed season** — the lineup endpoints are live, so the model runs on
current basketball (see [public-data-availability](https://github.com/ismayc/basketball-data-science/blob/main/docs/public-data-availability.md)).

---

<!-- terms -->
> **Terms used in this analysis.** Dotted-underlined terms anywhere below repeat these definitions on hover ([full glossary](https://github.com/ismayc/basketball-data-science/blob/main/docs/glossary.md)).
>
> - **RAPM** — Regularized adjusted plus-minus: a ridge regression crediting each player with net points per 100 possessions while adjusting for the other nine players on the floor.
> - **stint** — A stretch of game time with no substitution at either end - the unit of observation for the stint-level RAPM model.
> - **net_100** — Net points per 100 possessions: 100 x plus-minus / possessions; sits on the same scale as the NBA's published net rating.
> - **plus-minus** — Points scored minus points allowed while a player or lineup is on the floor.
> - **ridge** — Ridge regression: least squares with an L2 penalty that keeps many-parameter models stable at the cost of a little bias.
> - **CI** — Confidence interval.
> - **Spearman** — Rank correlation: correlation of the orderings rather than the raw values.
<!-- /terms -->

## 1. Where this sits in the literature

Nothing here is invented in a vacuum; the model is the entry point of a
well-developed public lineage, and knowing that lineage is part of the point:

- **Adjusted <abbr title="Points scored minus points allowed while a player or lineup is on the floor.">plus-minus</abbr> (APM)** — Rosenbaum (2004), building on Winston &
  Sagarin's WINVAL: regress <abbr title="A stretch of game time with no substitution at either end - the unit of observation for the stint-level RAPM model.">stint</abbr> score margins on player indicators, so a
  player's rating is *adjusted* for teammates and opponents rather than raw
  on-court margin.
- **Regularized APM (<abbr title="Regularized adjusted plus-minus: a ridge regression crediting each player with net points per 100 possessions while adjusting for the other nine players on the floor.">RAPM</abbr>)** — Sill (2010, MIT Sloan Sports Analytics
  Conference): APM's coefficients explode because lineups are collinear; <abbr title="Ridge regression: least squares with an L2 penalty that keeps many-parameter models stable at the cost of a little bias.">ridge</abbr>
  regression (Tikhonov regularization) shrinks players toward league average
  in proportion to evidence and dramatically improves out-of-sample accuracy.
  This is the backbone of most public player metrics since.
- **Modern descendants** — ESPN RPM, FiveThirtyEight RAPTOR, Dunks & Threes
  EPM, DARKO DPM: all blend a <abbr title="Regularized adjusted plus-minus: a ridge regression crediting each player with net points per 100 possessions while adjusting for the other nine players on the floor.">RAPM</abbr>-style on/off core with box-score and
  tracking priors to stabilize small samples.

This study implements the **<abbr title="Ridge regression: least squares with an L2 penalty that keeps many-parameter models stable at the cost of a little bias.">ridge</abbr> core of that lineage on season-aggregated
five-man lineup data**. The honest difference from full <abbr title="Regularized adjusted plus-minus: a ridge regression crediting each player with net points per 100 possessions while adjusting for the other nine players on the floor.">RAPM</abbr>: season lineup
aggregates do not record *opponents*, so opponent strength is not adjusted
(limitation 1). The public route to full <abbr title="A stretch of game time with no substitution at either end - the unit of observation for the stint-level RAPM model.">stint</abbr>-level <abbr title="Regularized adjusted plus-minus: a ridge regression crediting each player with net points per 100 possessions while adjusting for the other nine players on the floor.">RAPM</abbr> — play-by-play with
players-on-court filled in — is documented in
[public-data-availability](https://github.com/ismayc/basketball-data-science/blob/main/docs/public-data-availability.md) and is the natural next step.

## 2. Data

`stats.nba.com` via `nba_api`, per team (the league-wide lineup call silently
caps at 2,000 rows — per-team calls stay under it):

| Endpoint | Content |
|---|---|
| `LeagueDashLineups` (Base, Totals) | every five-man lineup's minutes and raw <abbr title="Points scored minus points allowed while a player or lineup is on the floor.">plus-minus</abbr> |
| `LeagueDashLineups` (Advanced, Totals) | possessions and the NBA's own <abbr title="Net points per 100 possessions: 100 x plus-minus / possessions; sits on the same scale as the NBA's published net rating.">NET_RATING</abbr> per lineup |
| `LeagueDashPlayerStats` | per-player totals — external reference for validation |
| `LeagueDashTeamStats` | team totals — the coverage check target |

Scale: ~15,900 lineup rows (2023-24) and ~19,500 (2025-26) covering ~255k
possessions per season. `POSS` is the lineup's **offensive-possession count**
(per-team sums land at 8,200-8,800, matching a season's offensive possession
totals), so `100 * PLUS_MINUS / POSS` sits on the same scale as the league's
<abbr title="Net points per 100 possessions: 100 x plus-minus / possessions; sits on the same scale as the NBA's published net rating.">NET_RATING</abbr> - verified: regressing one on the other gives slope 1.001,
intercept +0.07.

## 3. Model

Each lineup's net points per 100 possessions is the sum of five player effects
plus an intercept, fitted by **possession-weighted <abbr title="Ridge regression: least squares with an L2 penalty that keeps many-parameter models stable at the cost of a little bias.">ridge</abbr> regression**:

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
- **Zero-possession lineups** (defensive-only micro-<abbr title="A stretch of game time with no substitution at either end - the unit of observation for the stint-level RAPM model.">stints</abbr>, e.g. on the floor
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
python/05_build_stints.py      bulk pbp + nba-on-court -> data/stints_2023.parquet
python/06_stint_rapm.py        stint-level RAPM WITH opponent adjustment (2023-24)
```

The last two are the wired-in upgrade path (see
[public-data-availability](https://github.com/ismayc/basketball-data-science/blob/main/docs/public-data-availability.md)): <abbr title="A stretch of game time with no substitution at either end - the unit of observation for the stint-level RAPM model.">stints</abbr> carry both five-man lineups, so
the +1/−1 design adjusts each player for teammates *and opponents* — the thing
the season-aggregate model above cannot do. The <abbr title="A stretch of game time with no substitution at either end - the unit of observation for the stint-level RAPM model.">stint</abbr> model's intercept doubles
as a home-court-advantage estimate, and its own validation gates (final-margin
reconstruction, on-floor minutes vs official minutes) run before comparison.

Same discipline as the other studies: two implementations written from the
definitions, reconciled to numeric tolerance (bootstrap CIs, which use each
language's RNG, are compared for per-player overlap instead).

## 5. Validation

Four external checks, all of which must pass before findings are written:

1. **Lineup minutes reconstruct team minutes** (max relative error < 0.5%).
2. **Lineup <abbr title="Points scored minus points allowed while a player or lineup is on the floor.">plus-minus</abbr> reconstructs team <abbr title="Points scored minus points allowed while a player or lineup is on the floor.">plus-minus</abbr> exactly** — the full
   lineup table sums to each team's season <abbr title="Points scored minus points allowed while a player or lineup is on the floor.">plus-minus</abbr> to the point.
3. **Independent rate agrees with the NBA's own <abbr title="Net points per 100 possessions: 100 x plus-minus / possessions; sits on the same scale as the NBA's published net rating.">NET_RATING</abbr>** (r ≥ 0.97 on
   200+ possession lineups; not identical by construction, since the league
   computes off/def ratings on separate possession denominators).
4. **Sanity against raw <abbr title="Points scored minus points allowed while a player or lineup is on the floor.">plus-minus</abbr>** (<abbr title="Rank correlation: correlation of the orderings rather than the raw values.">Spearman</abbr> ≥ 0.5) — high enough to show
   the model is anchored in reality, and deliberately *not* 1: the daylight
   between raw <abbr title="Points scored minus points allowed while a player or lineup is on the floor.">plus-minus</abbr> and the regularized estimate is the teammate
   adjustment doing its work.

## 6. Limitations

1. **No opponent adjustment in the aggregate model** — now addressed: the
   wired-in <abbr title="A stretch of game time with no substitution at either end - the unit of observation for the stint-level RAPM model.">stint</abbr>-level <abbr title="Regularized adjusted plus-minus: a ridge regression crediting each player with net points per 100 possessions while adjusting for the other nine players on the floor.">RAPM</abbr> (Findings, last section) adjusts for opponents
   and lands within <abbr title="Rank correlation: correlation of the orderings rather than the raw values.">Spearman</abbr> 0.95 of the aggregate model, which quantifies
   how much this limitation actually cost (less than the standard caveat
   implies, over a full season).
2. **One season at a time.** Single-season <abbr title="Regularized adjusted plus-minus: a ridge regression crediting each player with net points per 100 possessions while adjusting for the other nine players on the floor.">RAPM</abbr>-family estimates are noisy;
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

Every number below is generated by `python/04_findings.py` from `output/`, not
typed by hand. Both implementations (R and Python) reconcile before this is
written.

### 2023-24

Penalty chosen by CV: **λ = 3200**. 454 players over the possession
threshold; median 95% <abbr title="Confidence interval.">CI</abbr> width **±2.5 points per 100** — the
honest single-season noise floor for this model family.

| Rank | Player | Effect / 100 poss | 95% <abbr title="Confidence interval.">CI</abbr> | Poss |
|---|---|---|---|---|
| 1 | Nikola Jokić | **+5.53** | [+3.16, +7.62] | 5,796 |
| 2 | OG Anunoby | **+5.27** | [+2.65, +8.18] | 3,596 |
| 3 | Jalen Brunson | **+4.94** | [+2.25, +7.26] | 5,679 |
| 4 | Paul George | **+4.84** | [+1.80, +7.47] | 5,222 |
| 5 | Shai Gilgeous-Alexander | **+4.80** | [+2.17, +7.27] | 5,647 |
| 6 | Derrick White | **+4.20** | [+1.83, +6.43] | 5,058 |
| 7 | Jusuf Nurkić | **+4.07** | [+1.70, +6.22] | 4,460 |
| 8 | Dante Exum | **+3.77** | [+0.87, +6.79] | 2,369 |
| 9 | Luka Dončić | **+3.77** | [+1.14, +6.24] | 5,741 |
| 10 | Sam Hauser | **+3.76** | [+1.15, +6.29] | 3,655 |

**Best high-usage lineup** (1,000+ possessions):
B. Lopez - D. Lillard - K. Middleton - G. Antetokounmpo - M. Beasley (MIL) —
**+15.3 per 100** over 1,302 possessions.
**Worst:** T. Jones - K. Kuzma - D. Gafford - J. Poole - D. Avdija (WAS),
-3.4 per 100 over 1,249.

Validation: lineup minutes and <abbr title="Points scored minus points allowed while a player or lineup is on the floor.">plus-minus</abbr> reconstruct team totals
(exact on points); independent net-rate agrees with the NBA's <abbr title="Net points per 100 possessions: 100 x plus-minus / possessions; sits on the same scale as the NBA's published net rating.">NET_RATING</abbr> at
r = 0.989; <abbr title="Rank correlation: correlation of the orderings rather than the raw values.">Spearman</abbr> vs raw <abbr title="Points scored minus points allowed while a player or lineup is on the floor.">plus-minus</abbr> = 0.81 — anchored in reality,
and the gap from 1.0 is the teammate adjustment working.

### 2025-26

Penalty chosen by CV: **λ = 3200**. 490 players over the possession
threshold; median 95% <abbr title="Confidence interval.">CI</abbr> width **±2.6 points per 100** — the
honest single-season noise floor for this model family.

| Rank | Player | Effect / 100 poss | 95% <abbr title="Confidence interval.">CI</abbr> | Poss |
|---|---|---|---|---|
| 1 | Victor Wembanyama | **+6.50** | [+3.59, +9.24] | 4,090 |
| 2 | Shai Gilgeous-Alexander | **+6.22** | [+3.24, +8.95] | 4,962 |
| 3 | Kawhi Leonard | **+5.69** | [+2.95, +8.29] | 4,356 |
| 4 | Chet Holmgren | **+5.18** | [+2.58, +7.77] | 4,314 |
| 5 | Nikola Jokić | **+4.94** | [+2.13, +7.07] | 4,996 |
| 6 | Derrick White | **+4.54** | [+1.95, +7.44] | 5,467 |
| 7 | Bam Adebayo | **+4.19** | [+1.81, +7.07] | 5,275 |
| 8 | Ajay Mitchell | **+4.02** | [+1.11, +6.92] | 3,229 |
| 9 | Cade Cunningham | **+3.90** | [+1.82, +6.60] | 4,734 |
| 10 | Neemias Queta | **+3.86** | [+1.30, +6.45] | 3,932 |

**Best high-usage lineup** (1,000+ possessions):
M. Bridges - L. Ball - M. Diabaté - B. Miller - K. Knueppel (CHA) —
**+26.6 per 100** over 1,054 possessions.
**Worst:** K. Towns - O. Anunoby - J. Hart - M. Bridges - J. Brunson (NYK),
+1.8 per 100 over 1,131.

Validation: lineup minutes and <abbr title="Points scored minus points allowed while a player or lineup is on the floor.">plus-minus</abbr> reconstruct team totals
(exact on points); independent net-rate agrees with the NBA's <abbr title="Net points per 100 possessions: 100 x plus-minus / possessions; sits on the same scale as the NBA's published net rating.">NET_RATING</abbr> at
r = 0.992; <abbr title="Rank correlation: correlation of the orderings rather than the raw values.">Spearman</abbr> vs raw <abbr title="Points scored minus points allowed while a player or lineup is on the floor.">plus-minus</abbr> = 0.83 — anchored in reality,
and the gap from 1.0 is the teammate adjustment working.

### Stint-level RAPM: adding the opponent adjustment (2023-24)

The wired-in upgrade (`python/05_build_stints.py` + `python/06_stint_rapm.py`):
69,767 <abbr title="A stretch of game time with no substitution at either end - the unit of observation for the stint-level RAPM model.">stints</abbr> from all 1,230 games, built from bulk play-by-play
with the ten on-court players
filled offline, micro-<abbr title="A stretch of game time with no substitution at either end - the unit of observation for the stint-level RAPM model.">stints</abbr> from free-throw substitutions absorbed into their
neighbours so no points leave the model, and a +1/−1 design that adjusts every
player for teammates *and opponents*. Validation gates all pass — <abbr title="A stretch of game time with no substitution at either end - the unit of observation for the stint-level RAPM model.">stint</abbr> points
reconstruct the official league point total, per-player on-floor minutes
reconstruct official minutes (r = 0.999), and the unpenalized intercept
estimates **home-court advantage at +2.2 points per 100** without being
asked to.

Top 5 opponent-adjusted: Shai Gilgeous-Alexander, OG Anunoby, Nikola Jokić, Jalen Brunson, Paul George.

**The instructive result is how little changes: <abbr title="Rank correlation: correlation of the orderings rather than the raw values.">Spearman</abbr> 0.96 against the
lineup-aggregate model.** Over an 82-game season, opponent strength largely
averages out, so the cheaper model was a better approximation than the usual
caveat implies — a claim now measured on this data instead of argued.
The players the adjustment moves up most (Luguentz Dort (+1.5), Dillon Brooks (+1.5), Dean Wade (+1.5)) are where schedule and
matchup context mattered. (An earlier build covered 1,200 of 1,230 games —
the boxscore-range endpoint nba-on-court needs was hanging on nba_api's
stale User-Agent; the builder's retry pass completed the season once that
was diagnosed and patched.)

### Reading the two seasons together

- The model finds the consensus stars without being told who they are — the
  top of each table is recognisable to anyone who watched that season. That
  is a *face-validity check*, not the product. The product is the ordering of
  everyone else, where intuition runs out.
- <abbr title="Regularized adjusted plus-minus: a ridge regression crediting each player with net points per 100 possessions while adjusting for the other nine players on the floor.">RAPM</abbr>-family models famously promote elite role players alongside stars.
  That is not a bug: it is the model reporting that scoring volume and
  lineup impact are different quantities.
- Single-season CIs of ±2-3 per 100 are why serious deployments blend
  multiple seasons and add box/tracking priors (see "Where this sits in the
  literature"). A model whose error bars you can defend is worth more than a
  sharper-looking one whose error bars you cannot.
- **Stability across seasons — the honest backtest.** For the 317 players fitted in both 2023-24 and 2025-26 (two seasons apart), estimates correlate at r = 0.31 (r = 0.38 for the 181 with 2,000+ possessions in both). That is the well-documented reliability ceiling of single-season <abbr title="Regularized adjusted plus-minus: a ridge regression crediting each player with net points per 100 possessions while adjusting for the other nine players on the floor.">RAPM</abbr>-family estimates, quantified on this data rather than assumed — and it is the empirical argument for the multi-season priors that production metrics add.
