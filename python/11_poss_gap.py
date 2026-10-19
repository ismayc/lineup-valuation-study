"""Diagnose the gap between the NBA's lineup POSS and pbpstats' possessions,
and measure what it does to the model. Reports; changes nothing published.

Background
----------
10_pbpstats_lineup_check.py shows the NBA and pbpstats agree on every
lineup's minutes and plus-minus. Their possession counts do not agree: the
NBA's POSS runs a few percent above pbpstats' OffPoss. POSS is the model's
denominator (net_100 = 100 * PLUS_MINUS / POSS) and its weight, so the gap
can move both the magnitudes and the ordering of the published estimates.

What this script establishes
----------------------------
1. What POSS counts: it is the NBA's offensive-possession count. It equals
   100 * PTS / OFF_RATING on every lineup (to the rounding of OFF_RATING),
   and the NBA keeps a separate defensive count behind DEF_RATING.
2. Where the gap sits: league and per-team ratios, the ratio by lineup size,
   and (2023-24, where the committed stints exist) how far the number of
   times a lineup took the floor explains the gap compared with its minutes.
3. What it changes: the 02_analysis.py model refitted with the same
   definitions (same MIN_POSS, lambda grid, folds) under two alternative
   possession counts:
     pbp_off  - pbpstats' OffPoss per lineup as denominator and weight
     uniform  - NBA POSS divided by the season's league-wide ratio
   The refit's baseline reproduces output/<season>/player_values.csv
   exactly (asserted) before any comparison is made. No bootstrap: point
   estimates and the CV-chosen penalty only.

Inputs : data/raw/<season>/ (committed), data/reference/pbpstats_*.parquet
         (committed, from 10_pbpstats_lineup_check.py --fetch),
         data/stints_2023.parquet (committed)
Outputs: output/<season>/poss_gap.csv           one row per metric
         output/<season>/poss_gap_by_team.csv   per-team possession ratios
         output/<season>/poss_sensitivity.csv   per player, published vs refits
Run:     python python/11_poss_gap.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
SEASONS = ["2023-24", "2025-26"]


def _load(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


A = _load("python/02_analysis.py", "poss_gap_analysis")
C = _load("python/10_pbpstats_lineup_check.py", "poss_gap_check")


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def matched_frame(season: str) -> pl.DataFrame:
    """NBA lineup rows (02's model frame columns) with pbpstats' counts."""
    raw = A.RAW_ROOT / season
    adv = pl.concat([pl.read_parquet(f) for f in
                     sorted(raw.glob("lineups_advanced_*.parquet"))])
    base = pl.concat([pl.read_parquet(f) for f in
                      sorted(raw.glob("lineups_base_*.parquet"))])
    ref = pl.read_parquet(C.REF / "pbpstats_lineups.parquet").filter(
        pl.col("season") == season)
    return (base.select("GROUP_ID", "TEAM_ID", "TEAM_ABBREVIATION", "GROUP_NAME",
                        "MIN", "PLUS_MINUS", "PTS")
            .join(adv.select("GROUP_ID", "TEAM_ID", "POSS", "NET_RATING",
                             "OFF_RATING", "DEF_RATING"),
                  on=["GROUP_ID", "TEAM_ID"], how="inner")
            .with_columns(team_id=pl.col("TEAM_ID").cast(pl.Int64),
                          lineup_key=pl.col("GROUP_ID").map_elements(
                              C.lineup_key, return_dtype=pl.String))
            .join(ref.select("team_id", "lineup_key", "off_poss", "def_poss"),
                  on=["team_id", "lineup_key"], how="left"))


def entries_2023(frame: pl.DataFrame) -> pl.DataFrame:
    """Times each lineup took the floor (a new five for that team, or a new
    period), counted from the committed 2023-24 stints."""
    s = pl.read_parquet(ROOT / "data" / "stints_2023.parquet").sort(
        "game_id", "period", "start_secs", descending=[False, False, True])
    parts = []
    for side in ("home", "away"):
        d = s.select("game_id", "period",
                     k=pl.col(f"{side}_ids").list.sort().cast(pl.List(pl.String))
                     .list.join("-"))
        d = d.with_columns(new=(pl.col("k") != pl.col("k").shift(1))
                           | (pl.col("game_id") != pl.col("game_id").shift(1))
                           | (pl.col("period") != pl.col("period").shift(1)))
        parts.append(d.group_by("k").agg(entries=pl.col("new").sum()))
    ent = pl.concat(parts).group_by("k").agg(pl.col("entries").sum())
    return frame.join(ent.rename({"k": "lineup_key"}), on="lineup_key", how="left")


def fit(df: pl.DataFrame, poss_col: str) -> tuple[pl.DataFrame, float, pl.DataFrame]:
    """02_analysis.py's model on df with `poss_col` as POSS. Same definitions."""
    d = (df.with_columns(POSS=pl.col(poss_col).cast(pl.Float64))
         .filter(pl.col("POSS") > 0)
         .with_columns(net_100=100.0 * pl.col("PLUS_MINUS") / pl.col("POSS"),
                       player_ids=pl.col("GROUP_ID").str.strip_chars("-").str.split("-"))
         .sort(["GROUP_ID", "TEAM_ID"]))
    X, y, w, kept = A.build_design(d)
    lam, _ = A.cv_lambda(X, y, w)
    beta = A.ridge_fit(X, y, w, lam)
    vals = (pl.DataFrame({"player_id": kept, "rapm_100": beta[1:1 + len(kept)]})
            .sort("rapm_100", descending=True).with_row_index("rank", offset=1))
    return vals, lam, d


def run(season: str) -> None:
    out = ROOT / "output" / season
    f = matched_frame(season)
    metrics: list[tuple[str, float]] = []
    # 9 significant digits: drops float-summation noise between reruns
    put = lambda k, v: metrics.append((k, float(f"{float(v):.9g}")))  # noqa: E731

    # ---- 1. what POSS counts ----------------------------------------------
    big = f.filter((pl.col("POSS") >= 50) & (pl.col("OFF_RATING") > 0))
    off_imp = 100 * big["PTS"] / big["OFF_RATING"]
    put("poss_vs_off_rating_identity_max_abs", (big["POSS"] - off_imp).abs().max())
    put("poss_vs_off_rating_identity_lineups", big.height)

    # ---- 2. where the gap sits ---------------------------------------------
    m = f.filter(pl.col("off_poss").is_not_null())
    nba, off, dfp = (float(m[c].sum()) for c in ("POSS", "off_poss", "def_poss"))
    put("lineups_matched", m.height)
    put("nba_poss_sum", nba)
    put("pbp_off_poss_sum", off)
    put("pbp_def_poss_sum", dfp)
    ratio = nba / off
    put("league_ratio_nba_over_pbp_off", ratio)
    put("league_gap_pct", 100 * (ratio - 1))
    put("nba_poss_over_pbp_mean_off_def", nba / ((off + dfp) / 2))
    teams = (m.group_by("team_id", "TEAM_ABBREVIATION")
             .agg(nba_poss=pl.col("POSS").sum(), pbp_off=pl.col("off_poss").sum(),
                  minutes=pl.col("MIN").sum(), lineups=pl.len())
             .with_columns(ratio=pl.col("nba_poss") / pl.col("pbp_off"),
                           lineups_per_48=pl.col("lineups") / (pl.col("minutes") / 48))
             .sort("ratio", "team_id"))
    # polars' parallel group-by sums floats in a varying order; round so a
    # rerun rewrites the same bytes (minutes to 3 decimals, ratios to 6)
    (teams.with_columns(pl.col("minutes").round(3),
                        pl.col("ratio", "lineups_per_48").round(6))
     .write_csv(out / "poss_gap_by_team.csv"))
    put("team_ratio_min", teams["ratio"].min())
    put("team_ratio_max", teams["ratio"].max())
    put("team_gap_pct_min", 100 * (teams["ratio"].min() - 1))
    put("team_gap_pct_max", 100 * (teams["ratio"].max() - 1))
    put("team_ratio_vs_lineups_per_48_r",
        np.corrcoef(teams["ratio"].to_numpy(), teams["lineups_per_48"].to_numpy())[0, 1])
    for lo, hi, tag in [(0, 50, "lt50"), (50, 200, "50_200"), (200, 1e9, "ge200")]:
        b = m.filter((pl.col("POSS") >= lo) & (pl.col("POSS") < hi))
        put(f"ratio_poss_{tag}", b["POSS"].sum() / b["off_poss"].sum())
        put(f"share_of_gap_poss_{tag}",
            (b["POSS"].sum() - b["off_poss"].sum()) / (nba - off))

    if season == "2023-24":
        e = entries_2023(m).filter(pl.col("entries").is_not_null())
        y = (e["POSS"] - e["off_poss"]).to_numpy().astype(float)
        mins = e["MIN"].to_numpy()
        ent = e["entries"].to_numpy().astype(float)
        sst = float(np.sum((y - y.mean()) ** 2))
        for tag, X in [("minutes", mins[:, None]), ("entries", ent[:, None]),
                       ("both", np.c_[mins, ent])]:
            c, *_ = np.linalg.lstsq(X, y, rcond=None)
            put(f"gap_r2_{tag}", 1 - float(np.sum((y - X @ c) ** 2)) / sst)
            if tag == "both":
                put("gap_per_minute_both", c[0])
                put("gap_per_entry_both", c[1])
        one = e.filter(pl.col("entries") == 1)
        g1 = one["POSS"] - one["off_poss"]
        put("single_entry_lineups", one.height)
        put("single_entry_gap_mean", g1.mean())
        put("single_entry_gap_min", g1.min())
        put("single_entry_gap_max", g1.max())
        put("single_entry_gap_0_or_1_share", g1.is_in([0, 1]).mean())
        put("lineups_with_entries", e.height)
        put("gap_per_entry_overall", y.sum() / ent.sum())

    # ---- 3. what it changes --------------------------------------------------
    full = f.with_columns(POSS_nba=pl.col("POSS").cast(pl.Float64),
                          POSS_uniform=pl.col("POSS") / ratio,
                          POSS_pbp_off=pl.col("off_poss").fill_null(0.0))
    base, lam0, _ = fit(full, "POSS_nba")
    pub = pl.read_csv(out / "player_values.csv",
                      schema_overrides={"player_id": pl.String})
    chk = pub.select("player_id", pub_rapm="rapm_100").join(base, on="player_id")
    assert chk.height == pub.height == base.height, "baseline player set differs"
    assert float((chk["pub_rapm"] - chk["rapm_100"]).abs().max()) < 1e-9, \
        "refit baseline does not reproduce the published estimates"
    put("lambda_published", lam0)

    names = pub.select("player_id", "player")
    tab = pub.select("player_id", "player", rank_pub="rank", rapm_pub="rapm_100")
    for tag in ("uniform", "pbp_off"):
        v, lam, d = fit(full, f"POSS_{tag}")
        put(f"lambda_{tag}", lam)
        tab = tab.join(v.select("player_id", **{f"rank_{tag}": "rank",
                                                  f"rapm_{tag}": "rapm_100"}),
                       on="player_id", how="full", coalesce=True)
        both = tab.drop_nulls([f"rapm_{tag}", "rapm_pub"])
        a, b = both["rapm_pub"].to_numpy(), both[f"rapm_{tag}"].to_numpy()
        put(f"players_{tag}", v.height)
        put(f"players_common_{tag}", both.height)
        put(f"spearman_{tag}", spearman(a, b))
        put(f"slope_{tag}", np.polyfit(a, b, 1)[0])
        put(f"max_abs_change_{tag}", np.max(np.abs(b - a)))
        put(f"median_abs_change_{tag}", np.median(np.abs(b - a)))
        top_pub = set(pub.head(10)["player_id"].to_list())
        top_new = v.head(10)["player_id"].to_list()
        put(f"top10_overlap_{tag}", len(top_pub & set(top_new)))
        top_new_rank = (v.head(10).join(names, on="player_id", how="left"))
        put(f"top1_rapm_{tag}", top_new_rank["rapm_100"][0])
        for i, r in enumerate(top_new_rank.iter_rows(named=True), start=1):
            put(f"top10_{tag}_{i:02d}::{r['player']}", r["rapm_100"])
        lt = d.filter(pl.col("POSS") >= 1000).sort("net_100", descending=True)
        put(f"lineups_1000plus_{tag}", lt.height)
        if lt.height:
            put(f"best_lineup_net100_{tag}", lt["net_100"][0])
    put("top1_rapm_published", pub["rapm_100"][0])
    lt0 = pl.read_csv(out / "lineup_table.csv")
    put("best_lineup_net100_published", lt0["net_100"][0])
    put("lineups_1000plus_published", lt0.height)

    tab.sort("rank_pub", nulls_last=True).write_csv(out / "poss_sensitivity.csv")
    res = pl.DataFrame({"metric": [k for k, _ in metrics], "value": [v for _, v in metrics]})
    res.write_csv(out / "poss_gap.csv")
    print(f"=== {season} ===")
    with pl.Config(tbl_rows=200, fmt_str_lengths=60):
        print(res)


def main() -> int:
    for s in SEASONS:
        run(s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
