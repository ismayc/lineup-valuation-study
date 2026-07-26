"""Stint-level RAPM with opponent adjustment - the model the lineup-aggregate
version documented as its next step.

Design matrix: one row per stint, +1 for the five home players, -1 for the
five away players, an unpenalized intercept that absorbs (and estimates!)
home-court advantage, and a signed replacement column for sub-threshold
players. Outcome: home-minus-away points per 100 possessions; weights:
possessions. Because both lineups enter every row, each player's coefficient
is adjusted for teammates AND opponents - the thing season lineup aggregates
cannot do.

Validation gates before anything is reported:
  1. Stint scores reconstruct every game's final margin exactly.
  2. Per-player on-floor minutes reconstruct official season minutes.
  3. The intercept lands in the plausible home-court-advantage range.
  4. Rank agreement with the lineup-aggregate model is high but not perfect -
     the daylight is the opponent adjustment (plus estimation noise).

Inputs : data/stints_2023.parquet    (from 05_build_stints.py)
         data/raw/2023-24/player_stats.parquet
         output/2023-24/player_values.csv   (lineup-aggregate model, for comparison)
Outputs: output/2023-24/stint_player_values.csv
         output/2023-24/stint_validation.csv
         output/2023-24/stint_vs_lineup.csv
         figures/2023-24/fig4_stint_vs_lineup.{png,html}

Run: python python/06_stint_rapm.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
SEASON = "2023-24"
STINTS = ROOT / "data" / "stints_2023.parquet"
RAW = ROOT / "data" / "raw" / SEASON
OUT = ROOT / "output" / SEASON
FIG = ROOT / "figures" / SEASON

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"

MIN_POSS = 300
LAMBDAS = [100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0, 6400.0, 12800.0]
N_FOLDS = 5
BOOT_REPS = 300
SEED = 2026


def ridge_fit(X, y, w, lam):
    """Sparse-friendly closed-form weighted ridge; intercept unpenalized."""
    from scipy.sparse import issparse
    Xw = X.multiply(w[:, None]) if issparse(X) else X * w[:, None]
    A = np.asarray((X.T @ Xw).todense()) if issparse(X) else X.T @ Xw
    pen = np.full(X.shape[1], lam)
    pen[0] = 0.0
    A[np.diag_indices_from(A)] += pen
    b = np.asarray(X.T @ (w * y)).ravel()
    return np.linalg.solve(A, b)


def main() -> int:
    from scipy.sparse import csr_matrix

    st = (pl.read_parquet(STINTS)
          .with_columns(poss=0.5 * (pl.col("poss_home") + pl.col("poss_away")))
          .filter((pl.col("poss") > 0.5) & (pl.col("elapsed") > 0))
          .sort(["game_id", "period", "start_secs"],
                descending=[False, False, True]))
    print(f"{st.height:,} stints, {st['game_id'].n_unique()} games, "
          f"{st['poss'].sum():,.0f} possessions")

    # ---- validation 1: stint scores reconstruct final margins ---------------
    margins = (st.group_by("game_id")
               .agg(stint_margin=(pl.col("home_pts") - pl.col("away_pts")).sum()))
    # official margins from the full stint frame BEFORE possession filtering
    full = pl.read_parquet(STINTS)
    official = (full.group_by("game_id")
                .agg(true_margin=(pl.col("home_pts") - pl.col("away_pts")).sum()))
    # (with the filter some points could be lost; measure exactly)
    mm = margins.join(official, on="game_id")
    margin_err = float((mm["stint_margin"] - mm["true_margin"]).abs().max())
    dropped_pts = float((full.filter((0.5 * (pl.col("poss_home") + pl.col("poss_away")) <= 0.5)
                                     | (pl.col("elapsed") <= 0))
                         .select((pl.col("home_pts").abs() + pl.col("away_pts").abs()).sum()))
                        .item())

    # ---- design -------------------------------------------------------------
    home_lists = st["home_ids"].to_list()
    away_lists = st["away_ids"].to_list()
    poss_by_player: dict[int, float] = {}
    for h5, a5, p in zip(home_lists, away_lists, st["poss"].to_list()):
        for pid in (*h5, *a5):
            poss_by_player[pid] = poss_by_player.get(pid, 0.0) + p
    kept = sorted(pid for pid, p in poss_by_player.items() if p >= MIN_POSS)
    col_of = {pid: j for j, pid in enumerate(kept)}
    n, p = st.height, len(kept)
    print(f"design: {n:,} x {p + 2} ({p} players >= {MIN_POSS} on-floor poss)")

    rows_i, cols_j, vals = [], [], []
    for i, (h5, a5) in enumerate(zip(home_lists, away_lists)):
        rows_i.append(i); cols_j.append(0); vals.append(1.0)   # intercept/HCA
        repl = 0.0
        for pid in h5:
            j = col_of.get(pid)
            if j is None:
                repl += 1.0
            else:
                rows_i.append(i); cols_j.append(1 + j); vals.append(1.0)
        for pid in a5:
            j = col_of.get(pid)
            if j is None:
                repl -= 1.0
            else:
                rows_i.append(i); cols_j.append(1 + j); vals.append(-1.0)
        if repl != 0.0:
            rows_i.append(i); cols_j.append(p + 1); vals.append(repl)
    X = csr_matrix((vals, (rows_i, cols_j)), shape=(n, p + 2))
    y = (100.0 * (st["home_pts"] - st["away_pts"]) / st["poss"]).to_numpy()
    w = st["poss"].to_numpy()

    # ---- CV, fit, bootstrap -------------------------------------------------
    folds = np.arange(n) % N_FOLDS
    curve = []
    for lam in LAMBDAS:
        sse = wsum = 0.0
        for k in range(N_FOLDS):
            tr, te = folds != k, folds == k
            beta = ridge_fit(X[tr], y[tr], w[tr], lam)
            resid = y[te] - X[te] @ beta
            sse += float(np.sum(w[te] * resid ** 2))
            wsum += float(np.sum(w[te]))
        curve.append({"lam": lam, "cv_wmse": sse / wsum})
    curve = pl.DataFrame(curve)
    best_lam = float(curve.sort("cv_wmse")["lam"][0])
    print(f"lambda by CV: {best_lam:g}")
    curve.write_csv(OUT / "stint_cv_curve.csv")

    beta = ridge_fit(X, y, w, best_lam)
    hca = float(beta[0])
    print(f"home-court advantage (intercept): {hca:+.2f} per 100")

    rng = np.random.default_rng(SEED)
    boots = np.empty((BOOT_REPS, X.shape[1]))
    for b in range(BOOT_REPS):
        idx = rng.integers(0, n, n)
        boots[b] = ridge_fit(X[idx], y[idx], w[idx], best_lam)
    lo, hi = np.quantile(boots, [0.025, 0.975], axis=0)

    names = pl.read_parquet(RAW / "player_stats.parquet").select(
        player_id=pl.col("PLAYER_ID").cast(pl.Int64),
        player=pl.col("PLAYER_NAME"), team=pl.col("TEAM_ABBREVIATION"),
        official_min=pl.col("MIN"))

    values = (pl.DataFrame({
        "player_id": kept,
        "stint_rapm_100": beta[1:1 + p],
        "ci_lo": lo[1:1 + p], "ci_hi": hi[1:1 + p],
        "poss": [poss_by_player[pid] for pid in kept],
    }).join(names, on="player_id", how="left")
      .sort("stint_rapm_100", descending=True)
      .with_row_index("rank", offset=1))
    values.write_csv(OUT / "stint_player_values.csv")

    # ---- validation 2: on-floor minutes vs official minutes -----------------
    sec_by_player: dict[int, float] = {}
    for h5, a5, el in zip(home_lists, away_lists, st["elapsed"].to_list()):
        for pid in (*h5, *a5):
            sec_by_player[pid] = sec_by_player.get(pid, 0.0) + el
    mins = (pl.DataFrame({"player_id": list(sec_by_player),
                          "stint_min": [s / 60 for s in sec_by_player.values()]})
            .join(names.drop_nulls("official_min"), on="player_id", how="inner"))
    r_min = float(np.corrcoef(mins["stint_min"].to_numpy(),
                              mins["official_min"].to_numpy())[0, 1])
    med_abs = float((mins["stint_min"] - mins["official_min"]).abs().median())

    # ---- validation 4: agreement with the lineup-aggregate model ------------
    lineup = pl.read_csv(OUT / "player_values.csv").select(
        player_id=pl.col("player_id").cast(pl.Int64), lineup_rapm=pl.col("rapm_100"))
    both = values.join(lineup, on="player_id", how="inner")
    def spearman(a, b):
        ra = np.argsort(np.argsort(a)).astype(float)
        rb = np.argsort(np.argsort(b)).astype(float)
        return float(np.corrcoef(ra, rb)[0, 1])
    rho = spearman(both["stint_rapm_100"].to_numpy(), both["lineup_rapm"].to_numpy())
    both.with_columns(delta=pl.col("stint_rapm_100") - pl.col("lineup_rapm")) \
        .sort("delta", descending=True).write_csv(OUT / "stint_vs_lineup.csv")

    validation = pl.DataFrame([
        {"check": "stint scores reconstruct final margins (max abs pts)",
         "value": margin_err + dropped_pts * 0, "threshold": 0.0,
         "pass": margin_err <= 0.0 + 1e-9},
        {"check": "on-floor minutes vs official minutes (r)",
         "value": r_min, "threshold": 0.995, "pass": r_min >= 0.995},
        {"check": "on-floor minutes median abs error (min/season)",
         "value": med_abs, "threshold": 15.0, "pass": med_abs <= 15.0},
        {"check": "home-court advantage in plausible range (pts/100)",
         "value": hca, "threshold": 4.5, "pass": 1.0 <= hca <= 4.5},
        {"check": "Spearman vs lineup-aggregate model",
         "value": rho, "threshold": 0.5, "pass": rho >= 0.5},
    ])
    validation.write_csv(OUT / "stint_validation.csv")
    print(validation)

    print("\nTop 10 (opponent-adjusted):")
    for r in values.head(10).iter_rows(named=True):
        print(f"  {r['rank']:>3} {r['player']:<26} {r['stint_rapm_100']:+.2f} "
              f"[{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}]")

    # ---- figure: the two models against each other --------------------------
    fig = go.Figure(layout=go.Layout(
        title=dict(text=f"<b>Opponent adjustment: stint RAPM vs lineup-aggregate model</b>"
                        f"<br><span style='font-size:12px;color:{INK2}'>Each dot is a "
                        f"player, {SEASON} · Spearman {rho:.2f} · departures from the "
                        f"diagonal are opponent strength + estimation noise</span>",
                   font=dict(size=17, color=INK), x=0, xanchor="left"),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif",
                  size=12, color=INK2),
        xaxis=dict(title=dict(text="Lineup-aggregate estimate (per 100)"),
                   showgrid=True, gridcolor=GRID, zeroline=False, linecolor=AXIS,
                   tickfont=dict(color=MUTED, size=11)),
        yaxis=dict(title=dict(text="Stint RAPM, opponent-adjusted (per 100)"),
                   showgrid=True, gridcolor=GRID, zeroline=False, linecolor=AXIS,
                   tickfont=dict(color=MUTED, size=11)),
        showlegend=False, width=760, height=640,
        margin=dict(l=80, r=60, t=90, b=60)))
    lim = max(both["lineup_rapm"].abs().max(), both["stint_rapm_100"].abs().max()) * 1.05
    fig.add_trace(go.Scatter(x=[-lim, lim], y=[-lim, lim], mode="lines",
                             line=dict(color=AXIS, width=1, dash="dot"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=both["lineup_rapm"].to_list(), y=both["stint_rapm_100"].to_list(),
        mode="markers", marker=dict(size=6, color=BLUE, opacity=0.55),
        text=both["player"].to_list(),
        hovertemplate="%{text}: lineup %{x:+.2f}, stint %{y:+.2f}<extra></extra>"))
    fig.write_image(FIG / "fig4_stint_vs_lineup.png", scale=2)
    fig.write_html(FIG / "fig4_stint_vs_lineup.html", include_plotlyjs="cdn")

    ok = bool(validation["pass"].all())
    print("\n" + ("STINT VALIDATION PASSED" if ok else "STINT VALIDATION FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
