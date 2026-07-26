# One finding, three audiences — player & lineup value

Same model, three registers. Numbers from `output/` (README Findings holds the
generated originals).

---

## For the front office (one minute)

**Who actually moves the scoreboard when they're on the floor — after you stop
crediting them for their teammates?**

We modelled every five-man lineup's scoring margin across a full season
(~15,000–19,000 lineups per season, covering every minute of every game) and
split the credit among the five players on the floor, shrinking toward average
wherever the evidence is thin.

- The model finds the stars on its own: Jokić, Gilgeous-Alexander, Brunson,
  Dončić at the top of 2023-24; Wembanyama, Gilgeous-Alexander, Leonard,
  Holmgren in 2025-26. **Nobody told it who they are** — that's the sanity
  check, not the product.
- The product is everyone else: it surfaces high-impact "connector" players
  whose box scores undersell them (the Derrick White / Sam Hauser tier), and
  it prices lineups directly — which five-man groups earned their minutes and
  which bled points.
- **Honest error bars:** one season of lineup data pins a player's impact down
  to about ±2–3 points per 100 possessions. That's why we'd blend seasons and
  add play-level data before using this to drive a decision — and the write-up
  says exactly what it would take.

## For analytics peers

Possession-weighted ridge on five-man lineup aggregates (`LeagueDashLineups`,
harvested per team to dodge the silent 2,000-row cap), net points per 100 on
player indicators, unpenalized intercept, sub-300-possession players pooled to
a replacement column. λ by deterministic 5-fold CV; 500-rep lineup bootstrap
for CIs. Sits in the Rosenbaum→Sill RAPM lineage. The opponent-adjustment upgrade is
now wired in: 69,767 stints across all 1,230 games (bulk pbp +
players-on-court filled offline), +1/−1 design with an unpenalized intercept
that estimates home-court advantage at +2.2/100, free-throw micro-stints
absorbed so no points leave the model.
Spearman 0.96 vs the aggregate model — over 82 games opponent strength mostly
averages out, now measured rather than argued. Validation gates: lineup rows
reconstruct team minutes and plus-minus exactly (zero-possession
defensive-only stints retained for the audit, excluded from the model);
independent net-rate vs NBA NET_RATING r ≈ 0.99; Spearman vs raw plus-minus
≈ 0.81 — anchored but not identical, which is the point. Dual R/Python
implementations reconcile to 1e-6 (C-locale sort discipline matters: R's
default collation silently breaks fold assignment).

## For the executive summary (three bullets)

- A season-scale player- and lineup-value model in the standard RAPM family,
  run on both 2023-24 and the just-finished 2025-26 season, with validation
  gates that reconstruct official team totals exactly.
- It identifies star and connector value beyond box scores, with honest
  uncertainty (±2–3 pts/100 per season of evidence) and a stated upgrade path.
- Methodology is literature-grounded, cross-implemented in R and Python, and
  unit-tested.
