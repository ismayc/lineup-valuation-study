"""Gate: every player's season plus-minus and minutes from the stints vs the
NBA's own season totals (LeagueDashPlayerStats, data/raw/2023-24/player_stats.parquet).

Why
---
06_stint_rapm.py's gates compare league totals, which a wrong event order
and misplaced free throws leave intact. Per player they do not: the first
stint build (October 2026 audit) matched the official season PLUS_MINUS for
39 of 572 players (largest miss 77 points) and minutes within one for 46.
The corrected build matches 449 exactly, every player within 5, and
minutes within one for 99%. The remaining few-point gaps sit within the
per-game check's tolerance (08_stint_box_check.py) and are not chased here.

Gates (all players in either table)
-----
  * season plus-minus within 2 of the official for >= 95% of players
  * no player's season plus-minus off by more than 6
  * season minutes within 1 of the official for >= 98% of players
  * no player's minutes off by more than 3
Free: both inputs are committed, no network.

Output: output/2023-24/stint_season_check.csv (one row per player)
Run:    python python/07_stint_season_check.py
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
STINTS = ROOT / "data" / "stints_2023.parquet"
OFFICIAL = ROOT / "data" / "raw" / "2023-24" / "player_stats.parquet"
OUT = ROOT / "output" / "2023-24"

PM_TOL, PM_WITHIN_MIN, PM_MAX = 2.0, 0.95, 6.0
MIN_TOL, MIN_WITHIN_MIN, MIN_MAX = 1.0, 0.98, 3.0


def season_from_stints(stints: pl.DataFrame) -> pl.DataFrame:
    home = stints.select(pl.col("home_ids").alias("player_id"),
                         pm=pl.col("home_pts") - pl.col("away_pts"), sec="elapsed")
    away = stints.select(pl.col("away_ids").alias("player_id"),
                         pm=pl.col("away_pts") - pl.col("home_pts"), sec="elapsed")
    return (pl.concat([home, away]).explode("player_id", empty_as_null=True)
            .group_by("player_id")
            .agg(stint_pm=pl.col("pm").sum(), stint_min=pl.col("sec").sum() / 60))


def compare(stints: pl.DataFrame, official: pl.DataFrame) -> pl.DataFrame:
    off = official.select(pl.col("PLAYER_ID").cast(pl.Int64).alias("player_id"),
                          "PLAYER_NAME", pl.col("PLUS_MINUS").cast(pl.Float64),
                          pl.col("MIN").cast(pl.Float64))
    return (season_from_stints(stints)
            .join(off, on="player_id", how="full", coalesce=True)
            .with_columns(pl.col("stint_pm", "stint_min", "PLUS_MINUS", "MIN").fill_null(0.0))
            .with_columns(d_pm=pl.col("stint_pm") - pl.col("PLUS_MINUS"),
                          d_min=pl.col("stint_min") - pl.col("MIN"))
            .sort("player_id"))


def gates(t: pl.DataFrame) -> dict[str, bool]:
    pm_in = float((t["d_pm"].abs() <= PM_TOL).mean())
    min_in = float((t["d_min"].abs() <= MIN_TOL).mean())
    pm_max = float(t["d_pm"].abs().max())
    min_max = float(t["d_min"].abs().max())
    exact = int((t["d_pm"].abs() < 0.5).sum())
    return {
        f"plus-minus within {PM_TOL:.0f} for {pm_in:.1%} of {t.height} players "
        f"({exact} exact) >= {PM_WITHIN_MIN:.0%}": pm_in >= PM_WITHIN_MIN,
        f"max |plus-minus diff| {pm_max:.0f} <= {PM_MAX:.0f}": pm_max <= PM_MAX,
        f"minutes within {MIN_TOL:.0f} for {min_in:.1%} >= {MIN_WITHIN_MIN:.0%}":
            min_in >= MIN_WITHIN_MIN,
        f"max |minutes diff| {min_max:.2f} <= {MIN_MAX:.0f}": min_max <= MIN_MAX,
    }


def main() -> int:
    t = compare(pl.read_parquet(STINTS), pl.read_parquet(OFFICIAL))
    OUT.mkdir(parents=True, exist_ok=True)
    t.write_csv(OUT / "stint_season_check.csv")
    print("Stint season totals vs LeagueDashPlayerStats, 2023-24")
    g = gates(t)
    for name, ok in g.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    ok = all(g.values())
    print("STINT SEASON CHECK " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
