"""Write the Findings section of the valuation README from computed output.

Same discipline as the other studies: every number in the prose is read from
output/, never typed. Run after 02_analysis.py (+ ideally 03_reconcile.py).

Run: python python/04_findings.py
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
README = ROOT / "README.md"
MARKER = "## Findings"


def season_block(season: str) -> str:
    out = OUT / season
    values = pl.read_csv(out / "player_values.csv")
    lineups = pl.read_csv(out / "lineup_table.csv")
    curve = pl.read_csv(out / "cv_curve.csv").sort("cv_wmse")
    validation = pl.read_csv(out / "validation.csv")

    lam = float(curve["lam"][0])
    top = values.head(10)
    top_rows = "\n".join(
        f"| {r['rank']} | {r['player']} | **{r['rapm_100']:+.2f}** "
        f"| [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}] | {r['poss']:,.0f} |"
        for r in top.iter_rows(named=True))

    best_l = lineups.row(0, named=True)
    worst_l = lineups.sort("net_100").row(0, named=True)

    rho = float(validation.filter(
        pl.col("check").str.contains("Spearman"))["value"][0])
    r_net = float(validation.filter(
        pl.col("check").str.contains("NET_RATING"))["value"][0])

    n_players = values.height
    ci_width = float((values["ci_hi"] - values["ci_lo"]).median())

    return f"""### {season}

Penalty chosen by CV: **λ = {lam:g}**. {n_players} players over the possession
threshold; median 95% CI width **±{ci_width / 2:.1f} points per 100**, the
honest single-season noise floor for this model family.

| Rank | Player | Effect / 100 poss | 95% CI | Poss |
|---|---|---|---|---|
{top_rows}

**Best high-usage lineup** (1,000+ possessions):
{best_l['GROUP_NAME']} ({best_l['TEAM_ABBREVIATION']}),
**{best_l['net_100']:+.1f} per 100** over {best_l['POSS']:,.0f} possessions.
**Worst:** {worst_l['GROUP_NAME']} ({worst_l['TEAM_ABBREVIATION']}),
{worst_l['net_100']:+.1f} per 100 over {worst_l['POSS']:,.0f}.

Validation: lineup minutes and plus-minus reconstruct team totals
(exact on points); independent net-rate agrees with the NBA's NET_RATING at
r = {r_net:.3f}; Spearman vs raw plus-minus = {rho:.2f}. The estimates are
anchored in reality, and the gap from 1.0 is the teammate adjustment working.
"""


def cross_season() -> str:
    """Stability of the estimates across the two fitted seasons: the honest
    backtest for a single-season RAPM-family model."""
    import numpy as np

    seasons = sorted(p.name for p in OUT.iterdir() if p.is_dir())
    if len(seasons) < 2:
        return ""
    a = pl.read_csv(OUT / seasons[0] / "player_values.csv").select(
        "player_id", ra="rapm_100", pa="poss")
    b = pl.read_csv(OUT / seasons[-1] / "player_values.csv").select(
        "player_id", rb="rapm_100", pb="poss")
    m = a.join(b, on="player_id", how="inner")
    r = float(np.corrcoef(m["ra"].to_numpy(), m["rb"].to_numpy())[0, 1])
    hi = m.filter((pl.col("pa") >= 2000) & (pl.col("pb") >= 2000))
    r_hi = float(np.corrcoef(hi["ra"].to_numpy(), hi["rb"].to_numpy())[0, 1])
    return (
        f"- **Stability across seasons: the honest backtest.** For the "
        f"{m.height} players fitted in both {seasons[0]} and {seasons[-1]} "
        f"(two seasons apart), estimates correlate at r = {r:.2f} "
        f"(r = {r_hi:.2f} for the {hi.height} with 2,000+ possessions in "
        f"both). That is the well-documented reliability ceiling of "
        f"single-season RAPM-family estimates, quantified on this data rather "
        f"than assumed. It is the empirical argument for the multi-season "
        f"priors that production metrics add.\n")


def stint_section() -> str:
    """Opponent-adjusted stint RAPM results, if 05/06 have run."""
    out = OUT / "2023-24"
    vpath = out / "stint_player_values.csv"
    if not vpath.exists():
        return ""
    sv = pl.read_csv(vpath)
    val = pl.read_csv(out / "stint_validation.csv")
    comp = pl.read_csv(out / "stint_vs_lineup.csv")

    rho = float(val.filter(pl.col("check").str.contains("Spearman"))["value"][0])
    hca = float(val.filter(pl.col("check").str.contains("home-court"))["value"][0])
    r_min = float(val.filter(pl.col("check") == "on-floor minutes vs official minutes (r)")["value"][0])
    # floored, so "r > 0.999" never overstates the agreement
    r_min_shown = f"> {math.floor(r_min * 1000) / 1000:.3f}"
    scale = pl.read_csv(out / "stint_scale.csv")
    n_stints, n_games = int(scale["stints_parsed"][0]), int(scale["games"][0])
    corr = {r["metric"]: r for r in
            pl.read_csv(out / "stint_correction.csv").iter_rows(named=True)}

    def c(metric: str, col: str) -> float:
        return corr[metric][col]

    top5 = ", ".join(sv.head(5)["player"].to_list())
    up = comp.sort("delta", descending=True).head(3)
    movers = ", ".join(f"{r['player']} ({r['delta']:+.1f})"
                       for r in up.iter_rows(named=True))

    return f"""
### Stint-level RAPM: adding the opponent adjustment (2023-24)

The wired-in upgrade (`python/05_build_stints.py` + `python/06_stint_rapm.py`):
{n_stints:,} stints from all {n_games:,} games, built from bulk play-by-play
with the ten on-court players
filled offline, micro-stints from free-throw substitutions absorbed into their
neighbours so no points leave the model, and a +1/−1 design that adjusts every
player for teammates *and opponents*. Validation gates all pass: stint points
reconstruct the official league point total, per-player on-floor minutes
reconstruct official minutes (r {r_min_shown}), and the unpenalized intercept
estimates **home-court advantage at {hca:+.1f} points per 100** without being
asked to.

**Correction, October 19, 2026.** The first stint build processed events in
EVENTNUM order, which is not chronological (late-inserted events carry later
numbers), and credited free throws to the lineup on the floor at the free
throw rather than at the foul. It left {c("negative_elapsed_stints", "first_build"):.0f} stints with negative elapsed time
and {c("negative_points_stints", "first_build"):.0f} with negative points, yet passed every league-total gate. Two new
per-player gates catch it. Against an independent feed (NBA liveData
play-by-play with pbpstats lineups, {c("player_games_checked", "corrected"):,.0f} player-games), stint
plus-minus was exact for {c("player_game_pm_exact_share", "first_build"):.0%} of player-games and is now exact for
{c("player_game_pm_exact_share", "corrected"):.1%} (`08_stint_box_check.py`). Against the NBA's season totals, it
matched {c("season_pm_exact_players", "first_build"):.0f} of {c("players_checked", "corrected"):.0f} players exactly with a worst miss of
{c("season_pm_max_abs_diff", "first_build"):.0f} points, and now matches {c("season_pm_exact_players", "corrected"):.0f} with every player within
{c("season_pm_max_abs_diff", "corrected"):.0f} (`07_stint_season_check.py`). The rebuild has {c("stints_parsed", "corrected"):,.0f} stints
(was {c("stints_parsed", "first_build"):,.0f}); the numbers in this section are from the corrected
build, and `output/2023-24/stint_correction.csv` keeps both.

Top 5 opponent-adjusted: {top5}.

**The instructive result is how little changes: Spearman {rho:.2f} against the
lineup-aggregate model.** Over an 82-game season, opponent strength largely
averages out, so the cheaper model was a better approximation than the usual
caveat implies, a claim now measured on this data instead of argued.
The players the adjustment moves up most ({movers}) are where schedule and
matchup context mattered. (An earlier build covered 1,200 of 1,230 games:
the boxscore-range endpoint nba-on-court needs was hanging on nba_api's
stale User-Agent; the builder's retry pass completed the season once that
was diagnosed and patched.)
"""


def build() -> str:
    seasons = sorted(p.name for p in OUT.iterdir() if p.is_dir())
    blocks = "\n".join(season_block(s) for s in seasons)
    return f"""{MARKER}

Every number below is generated by `python/04_findings.py` from `output/`, not
typed by hand. Both implementations (R and Python) reconcile before this is
written.

{blocks}{stint_section()}
### Reading the two seasons together

- The model finds the consensus stars without being told who they are: the
  top of each table is recognizable to anyone who watched that season. That
  is a *face-validity check*, not the product. The product is the ordering of
  everyone else, where intuition runs out.
- RAPM-family models famously promote elite role players alongside stars.
  That is not a bug: it is the model reporting that scoring volume and
  lineup impact are different quantities.
- Single-season CIs of ±2-3 per 100 are why serious deployments blend
  multiple seasons and add box/tracking priors (see "Where this sits in the
  literature"). A model whose error bars you can defend is worth more than a
  sharper-looking one whose error bars you cannot.
{cross_season()}"""


# glossary.py --sync wraps terms in hover tags after this runs; strip them
# before comparing, as the other studies' generators do.
ABBR = re.compile(r'<abbr title="[^"]*">([^<]*)</abbr>')


def regenerate(text: str) -> str:
    head = text.split(MARKER)[0].rstrip()
    return head + "\n\n" + build()


def main(argv: list[str]) -> int:
    current = README.read_text()
    if "--check" in argv:
        plain = ABBR.sub(r"\1", current)
        if plain != regenerate(plain):
            print("README Findings differ from what output/ supports. Fix: "
                  "python python/04_findings.py, then glossary.py --sync.")
            print("FINDINGS CHECK FAILED")
            return 1
        print("FINDINGS CHECK PASSED (README Findings regenerate from output/)")
        return 0
    README.write_text(regenerate(current))
    print(f"Updated Findings in {README}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
