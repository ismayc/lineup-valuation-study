"""Player value from lineup data: possession-weighted ridge regression.

The model
---------
Every five-man lineup's net scoring rate (points scored minus allowed, per 100
possessions) is modeled as the sum of five player effects plus an intercept:

    net_100_i  ~  intercept + sum_{p in lineup i} beta_p        (weight: POSS_i)

Ordinary least squares on this design is hopeless: lineups are massively
collinear (stars share the floor with the same teammates all season), and
low-possession lineups have wild observed rates. Ridge regularization shrinks
every player toward zero (league average) in proportion to how little evidence
there is - the standard solution in the public literature (see README,
"Where this sits in the literature"). The penalty is chosen by 5-fold
cross-validation with deterministic folds so the R implementation can
reproduce the choice exactly.

Players below a possession threshold are pooled into a single "replacement"
column: their individual effects are unidentifiable, and pooling them is more
honest than reporting noise shrunk toward zero.

What this is NOT: stint-level RAPM with opponent adjustment. Season lineup
aggregates do not record who the opponents were, so strength of opposition is
not controlled. The README's limitations section says so plainly.

Inputs : data/raw/*.parquet          (from 01_harvest_lineups.py)
Outputs: output/player_values.csv    per-player estimate + bootstrap CI
         output/lineup_table.csv     high-possession lineup ratings
         output/cv_curve.csv         cross-validation curve
         output/validation.csv      external checks (see 04_findings.py)
         figures/*.png + *.html

Run: python python/02_analysis.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "data" / "raw"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"

MIN_POSS = 300          # below this a player is pooled into "replacement"
LAMBDAS = [100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0, 6400.0]
N_FOLDS = 5
BOOT_REPS = 500
SEED = 2026


def base_layout(title: str, subtitle: str, x_title: str, y_title: str) -> go.Layout:
    return go.Layout(
        title=dict(text=f"<b>{title}</b><br><span style='font-size:12px;color:{INK2}'>"
                        f"{subtitle}</span>",
                   font=dict(size=17, color=INK), x=0, xanchor="left"),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif",
                  size=12, color=INK2),
        xaxis=dict(title=dict(text=x_title, font=dict(color=INK2, size=12)),
                   showgrid=True, gridcolor=GRID, gridwidth=0.5, zeroline=False,
                   linecolor=AXIS, tickfont=dict(color=MUTED, size=11)),
        yaxis=dict(title=dict(text=y_title, font=dict(color=INK2, size=12)),
                   showgrid=True, gridcolor=GRID, gridwidth=0.5, zeroline=False,
                   linecolor=AXIS, tickfont=dict(color=MUTED, size=11)),
        showlegend=False, margin=dict(l=80, r=120, t=85, b=55),
        width=900, height=520,
    )


def save(fig: go.Figure, stem: str) -> None:
    fig.write_image(FIG / f"{stem}.png", scale=2)
    fig.write_html(FIG / f"{stem}.html", include_plotlyjs="cdn")


# ------------------------------------------------------------------ loading --
def load_lineups(raw: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Returns (model_frame, full_frame).

    The model frame drops zero-possession lineups (net_100 is undefined for
    them - they are defensive-only micro-stints, e.g. on the floor solely for
    opponent free throws). The FULL frame keeps them, because the external
    coverage validation must reconstruct team totals exactly, and the dropped
    stints carry real plus-minus points.
    """
    base = pl.concat([pl.read_parquet(f) for f in sorted(raw.glob("lineups_base_*.parquet"))])
    adv = pl.concat([pl.read_parquet(f) for f in sorted(raw.glob("lineups_advanced_*.parquet"))])

    df = (base.select("GROUP_ID", "TEAM_ID", "TEAM_ABBREVIATION", "GROUP_NAME",
                      "MIN", "PLUS_MINUS")
          .join(adv.select("GROUP_ID", "TEAM_ID", "POSS", "NET_RATING"),
                on=["GROUP_ID", "TEAM_ID"], how="inner"))

    full = df
    zero = df.filter(pl.col("POSS") <= 0)
    if zero.height:
        print(f"  model excludes {zero.height} zero-possession lineups "
              f"(defensive-only micro-stints, {zero['PLUS_MINUS'].sum():+.0f} pts "
              f"kept in the coverage validation)")
    df = (df.filter(pl.col("POSS") > 0)
          .with_columns(
              net_100=100.0 * pl.col("PLUS_MINUS") / pl.col("POSS"),
              player_ids=pl.col("GROUP_ID").str.strip_chars("-").str.split("-"),
          )
          # Deterministic row order: the CV fold assignment and the bootstrap
          # both index rows, and R must be able to reproduce them.
          .sort(["GROUP_ID", "TEAM_ID"]))
    assert (df["player_ids"].list.len() == 5).all()
    return df, full


def player_names(raw: Path) -> pl.DataFrame:
    ps = pl.read_parquet(raw / "player_stats.parquet")
    return ps.select(
        player_id=pl.col("PLAYER_ID").cast(pl.Utf8),
        player=pl.col("PLAYER_NAME"),
        team=pl.col("TEAM_ABBREVIATION"),
        season_min=pl.col("MIN"),
        season_pm=pl.col("PLUS_MINUS"),
    )


# ------------------------------------------------------------- design matrix --
def build_design(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Rows: lineups. Columns: intercept, kept players (sorted id), replacement."""
    poss_by_player: dict[str, float] = {}
    for ids, poss in zip(df["player_ids"].to_list(), df["POSS"].to_list()):
        for pid in ids:
            poss_by_player[pid] = poss_by_player.get(pid, 0.0) + poss

    kept = sorted(pid for pid, p in poss_by_player.items() if p >= MIN_POSS)
    col_of = {pid: j for j, pid in enumerate(kept)}

    n, p = df.height, len(kept)
    X = np.zeros((n, p + 2))          # [intercept | players | replacement]
    X[:, 0] = 1.0
    for i, ids in enumerate(df["player_ids"].to_list()):
        for pid in ids:
            j = col_of.get(pid)
            if j is None:
                X[i, p + 1] += 1.0    # replacement count
            else:
                X[i, 1 + j] = 1.0

    y = df["net_100"].to_numpy().astype(float)
    w = df["POSS"].to_numpy().astype(float)
    return X, y, w, kept


def ridge_fit(X: np.ndarray, y: np.ndarray, w: np.ndarray, lam: float) -> np.ndarray:
    """Closed-form weighted ridge; intercept (column 0) unpenalized."""
    Xw = X * w[:, None]
    A = X.T @ Xw
    pen = np.full(X.shape[1], lam)
    pen[0] = 0.0
    A[np.diag_indices_from(A)] += pen
    return np.linalg.solve(A, Xw.T @ y)


def cv_lambda(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, pl.DataFrame]:
    """Deterministic 5-fold CV (fold = row index mod N_FOLDS on sorted rows)."""
    folds = np.arange(len(y)) % N_FOLDS
    rows = []
    for lam in LAMBDAS:
        sse = wsum = 0.0
        for k in range(N_FOLDS):
            tr, te = folds != k, folds == k
            beta = ridge_fit(X[tr], y[tr], w[tr], lam)
            resid = y[te] - X[te] @ beta
            sse += float(np.sum(w[te] * resid ** 2))
            wsum += float(np.sum(w[te]))
        rows.append({"lam": lam, "cv_wmse": sse / wsum})
    curve = pl.DataFrame(rows)
    best = float(curve.sort("cv_wmse")["lam"][0])
    return best, curve


# ------------------------------------------------------------------ main -----
def run(season: str) -> int:
    raw = RAW_ROOT / season
    out = ROOT / "output" / season
    fig_dir = ROOT / "figures" / season
    out.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    global OUT, FIG
    OUT, FIG = out, fig_dir

    print(f"=== {season} ===")
    df, df_full = load_lineups(raw)
    names = player_names(raw)
    print(f"{df.height:,} lineups, {df['TEAM_ID'].n_unique()} teams, "
          f"{df['POSS'].sum():,.0f} possessions")

    X, y, w, kept = build_design(df)
    print(f"design: {X.shape[0]:,} x {X.shape[1]} "
          f"({len(kept)} players >= {MIN_POSS} poss, + intercept + replacement)")

    best_lam, curve = cv_lambda(X, y, w)
    curve.write_csv(OUT / "cv_curve.csv")
    print(f"lambda by 5-fold CV: {best_lam:g}")

    beta = ridge_fit(X, y, w, best_lam)

    # Bootstrap over lineups at the chosen lambda. Resampling rows treats the
    # lineup-season as the sampling unit; the CI reflects how much each player's
    # estimate depends on which lineups happened to exist.
    rng = np.random.default_rng(SEED)
    boots = np.empty((BOOT_REPS, X.shape[1]))
    n = X.shape[0]
    for b in range(BOOT_REPS):
        idx = rng.integers(0, n, n)
        boots[b] = ridge_fit(X[idx], y[idx], w[idx], best_lam)
    lo, hi = np.quantile(boots, [0.025, 0.975], axis=0)

    poss_kept = {pid: 0.0 for pid in kept}
    for ids, poss in zip(df["player_ids"].to_list(), df["POSS"].to_list()):
        for pid in ids:
            if pid in poss_kept:
                poss_kept[pid] += poss

    values = (pl.DataFrame({
        "player_id": kept,
        "rapm_100": beta[1:1 + len(kept)],
        "ci_lo": lo[1:1 + len(kept)],
        "ci_hi": hi[1:1 + len(kept)],
        "poss": [poss_kept[pid] for pid in kept],
    })
        .join(names, on="player_id", how="left")
        .sort("rapm_100", descending=True)
        .with_row_index("rank", offset=1))
    values.write_csv(OUT / "player_values.csv")

    intercept, repl = beta[0], beta[-1]
    print(f"intercept {intercept:+.2f} (league avg), replacement effect {repl:+.2f}/100")

    # High-possession lineup table: the lineup-level deliverable.
    lineups = (df.filter(pl.col("POSS") >= 1000)
               .select("TEAM_ABBREVIATION", "GROUP_NAME", "MIN", "POSS",
                       "PLUS_MINUS", "net_100", "NET_RATING")
               .sort("net_100", descending=True))
    lineups.write_csv(OUT / "lineup_table.csv")

    # ---- external validation ------------------------------------------------
    teams = pl.read_parquet(raw / "team_stats.parquet")
    cov = (df_full.group_by("TEAM_ID")
           .agg(lineup_min=pl.col("MIN").sum(), lineup_pm=pl.col("PLUS_MINUS").sum())
           .join(teams.select("TEAM_ID", team_min=pl.col("MIN"),
                              team_pm=pl.col("PLUS_MINUS")), on="TEAM_ID")
           .with_columns(
               min_ratio=pl.col("lineup_min") / pl.col("team_min"),
               pm_err=(pl.col("lineup_pm") - pl.col("team_pm")).abs(),
           ))
    min_ratio_worst = float((cov["min_ratio"] - 1.0).abs().max())
    pm_err_max = float(cov["pm_err"].max())

    # our net_100 vs the NBA's own NET_RATING for the same lineups (they use
    # separate off/def possession denominators; agreement should be near 1)
    big = df.filter(pl.col("POSS") >= 200)
    r_net = float(np.corrcoef(big["net_100"].to_numpy(),
                              big["NET_RATING"].to_numpy())[0, 1])

    # face validity: rank-correlation with raw season plus-minus. Should be
    # clearly positive but NOT 1 - the daylight between them is the teammate
    # adjustment doing its work.
    vv = values.drop_nulls("season_pm").with_columns(
        pm_per_min=pl.col("season_pm") / pl.col("season_min"))
    def spearman(a: np.ndarray, b: np.ndarray) -> float:
        ra = np.argsort(np.argsort(a)).astype(float)
        rb = np.argsort(np.argsort(b)).astype(float)
        return float(np.corrcoef(ra, rb)[0, 1])
    rho_pm = spearman(vv["rapm_100"].to_numpy(), vv["pm_per_min"].to_numpy())

    validation = pl.DataFrame([
        {"check": "lineup minutes reconstruct team minutes",
         "value": min_ratio_worst, "threshold": 0.005, "pass": min_ratio_worst <= 0.005},
        {"check": "lineup plus-minus reconstructs team plus-minus (max abs pts)",
         "value": pm_err_max, "threshold": 1.0, "pass": pm_err_max <= 1.0},
        {"check": "net_100 vs NBA NET_RATING correlation (POSS>=200)",
         "value": r_net, "threshold": 0.97, "pass": r_net >= 0.97},
        {"check": "Spearman(RAPM, raw plus-minus per minute)",
         "value": rho_pm, "threshold": 0.5, "pass": rho_pm >= 0.5},
    ])
    validation.write_csv(OUT / "validation.csv")
    print(validation)
    if not validation["pass"].all():
        print("VALIDATION FAILED")
        return 1

    # ---- figures ------------------------------------------------------------
    top = values.head(15)
    bot = values.tail(15).sort("rapm_100")
    show = pl.concat([top, bot]).sort("rapm_100")
    fig = go.Figure(layout=base_layout(
        f"Regularized player value from {season} lineup data",
        f"Points per 100 possessions vs league average · ridge, lambda={best_lam:g} by CV · "
        f"95% bootstrap CI · players with {MIN_POSS}+ possessions",
        "Net points per 100 possessions (vs average)", ""))
    fig.update_layout(height=760, margin=dict(l=170, r=90, t=95, b=55))
    fig.add_vline(x=0, line=dict(color=AXIS, width=1))
    fig.add_trace(go.Scatter(
        x=show["rapm_100"].to_list(), y=show["player"].to_list(),
        mode="markers",
        error_x=dict(type="data", symmetric=False,
                     array=(show["ci_hi"] - show["rapm_100"]).to_list(),
                     arrayminus=(show["rapm_100"] - show["ci_lo"]).to_list(),
                     color=MUTED, thickness=1),
        marker=dict(size=8, color=[BLUE if v > 0 else ORANGE for v in show["rapm_100"]]),
        hovertemplate="%{y}: %{x:+.2f} per 100<extra></extra>"))
    save(fig, "fig1_player_values")  # season dir keeps them apart

    fig = go.Figure(layout=base_layout(
        "Ridge shrinks low-evidence players toward average",
        "Each dot is a player: naive per-possession impact vs the regularized estimate",
        "Naive on-court net per 100 (possession-weighted mean of lineups)",
        "Ridge estimate per 100"))
    naive = []
    for pid in kept:
        mask = np.array([pid in ids for ids in df["player_ids"].to_list()])
        naive.append(float(np.average(y[mask], weights=w[mask])))
    poss_arr = np.array([poss_kept[pid] for pid in kept])
    fig.add_trace(go.Scatter(
        x=naive, y=beta[1:1 + len(kept)], mode="markers",
        marker=dict(size=np.clip(np.sqrt(poss_arr) / 6, 3, 14), color=BLUE,
                    opacity=0.55, line=dict(width=0)),
        hoverinfo="skip"))
    lim = max(map(abs, naive)) * 1.05
    fig.add_trace(go.Scatter(x=[-lim, lim], y=[-lim, lim], mode="lines",
                             line=dict(color=AXIS, width=1, dash="dot"),
                             hoverinfo="skip"))
    fig.add_annotation(x=lim * 0.95, y=lim * 0.8, text="no shrinkage line",
                       showarrow=False, font=dict(color=MUTED, size=11), xanchor="right")
    save(fig, "fig2_shrinkage")

    fig = go.Figure(layout=base_layout(
        "Penalty chosen by cross-validation",
        "Possession-weighted MSE across 5 deterministic folds",
        "Ridge penalty (log scale)", "CV weighted MSE"))
    fig.add_trace(go.Scatter(x=curve["lam"].to_list(), y=curve["cv_wmse"].to_list(),
                             mode="lines+markers", line=dict(color=BLUE, width=2),
                             marker=dict(size=8, color=BLUE),
                             hovertemplate="lambda %{x:g}: %{y:.2f}<extra></extra>"))
    fig.update_xaxes(type="log")
    fig.add_vline(x=best_lam, line=dict(color=ORANGE, width=1.5, dash="dot"))
    save(fig, "fig3_cv_curve")

    print(f"\nTop 10:")
    for r in values.head(10).iter_rows(named=True):
        print(f"  {r['rank']:>3} {r['player']:<28} {r['rapm_100']:+.2f} "
              f"[{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}]  ({r['poss']:,.0f} poss)")
    print(f"\nWrote {OUT} and {FIG}")
    return 0


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+",
                    default=sorted(p.name for p in RAW_ROOT.iterdir() if p.is_dir()))
    args = ap.parse_args()
    rc = 0
    for season in args.seasons:
        rc |= run(season)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
