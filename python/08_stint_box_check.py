"""Gate: every player-game's stint plus-minus and seconds vs an independent feed.

Why
---
06_stint_rapm.py's own gates are league totals (points within 0.1%, minutes
within 2.5%), and they passed a stint layer whose event order was wrong and
whose free throws went to the wrong lineup. A per-player-game comparison
against an independent pipeline catches both. The reference here shares
nothing with the stints but the game itself:

  stints     bulk PlayByPlayV2 file (stats.nba.com) + nba-on-court lineups
  reference  NBA liveData play-by-play (cdn.nba.com) + pbpstats' lineup
             engine, which reports each player's PlusMinus and seconds
             (SecondsPlayedOff + SecondsPlayedDef)

On the 2026-10-19 audit sample, pbpstats on liveData agreed with the official
liveData box score, and the first stint build matched the box on 48 of 125
player-games (the corrected build: 125 of 125). The box score itself could
not be fetched for the full season that day (cdn.nba.com returned 403 to
every request), so the full-season gate uses the pbpstats reference.

Data
----
--build reads the liveData play-by-play cache that playbyplay-study keeps
(../playbyplay-study/data/pbpstats_live/pbp/, gitignored there) and writes
the compact committed table data/livedata_player_games_2023.parquet, so the
gate replays offline. Games pbpstats cannot give lineups for (it finds
fewer than five starters in a period) are listed in that table's
`skipped` file and left out of the gate, never silently.

Gates (player-games with reference seconds > 0)
-----
  * plus-minus exact on >= PM_EXACT_MIN of player-games
  * no player-game's plus-minus off by more than PM_MAX_ABS
  * seconds within SEC_TOL of the reference on >= SEC_WITHIN_MIN, and no
    player-game off by more than SEC_MAX_ABS. The bulk file's clock is whole
    seconds (MM:SS) while liveData's has tenths, so a late-period
    substitution can land up to a second off and a player with several of
    them a few seconds off (309 player-games at 1-3 s in 2023-24, mean
    difference 0.002 s); anything larger is a real error.
Every miss goes to output/2023-24/stint_box_misses.csv and the summary to
output/2023-24/stint_box_check.csv.

Run:  python python/08_stint_box_check.py --build   rebuild the reference, then check
      python python/08_stint_box_check.py           check (offline)
"""
from __future__ import annotations

import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
STINTS = ROOT / "data" / "stints_2023.parquet"
LIVE_CACHE = ROOT.parent / "playbyplay-study" / "data" / "pbpstats_live"
REFERENCE = ROOT / "data" / "livedata_player_games_2023.parquet"
SKIPPED = ROOT / "data" / "livedata_player_games_2023_skipped.csv"
OUT = ROOT / "output" / "2023-24"
WORKERS = 4

PM_EXACT_MIN = 0.995
PM_MAX_ABS = 4.0
SEC_TOL = 1.0
SEC_WITHIN_MIN = 0.98
SEC_MAX_ABS = 5.0


def reference_game(game_id: int) -> list[dict]:
    from pbpstats.client import Client
    c = Client({"dir": str(LIVE_CACHE),
                "Possessions": {"source": "file", "data_provider": "live"}})
    game = c.Game(f"{game_id:010d}")
    pm: dict[int, float] = defaultdict(float)
    sec: dict[int, float] = defaultdict(float)
    for r in game.possessions.player_stats:
        if r["stat_key"] == "PlusMinus":
            pm[int(r["player_id"])] += r["stat_value"]
        elif r["stat_key"] in ("SecondsPlayedOff", "SecondsPlayedDef"):
            sec[int(r["player_id"])] += r["stat_value"]
    return [{"game_id": game_id, "player_id": p, "plus_minus": pm.get(p, 0.0),
             "seconds": sec.get(p, 0.0)} for p in sorted(set(pm) | set(sec))]


def build(game_ids: list[int]) -> None:
    rows, skipped = [], []
    with ThreadPoolExecutor(WORKERS) as pool:
        futs = {pool.submit(reference_game, g): g for g in game_ids}
        for i, f in enumerate(as_completed(futs)):
            try:
                rows += f.result()
            except Exception as e:  # noqa: BLE001 - listed, never silent
                skipped.append({"game_id": futs[f], "error": f"{type(e).__name__}: {e}"[:200]})
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(game_ids)}", flush=True)
    pl.DataFrame(rows).sort(["game_id", "player_id"]).write_parquet(REFERENCE)
    (pl.DataFrame(skipped, schema={"game_id": pl.Int64, "error": pl.String})
     .sort("game_id").write_csv(SKIPPED))
    print(f"wrote {REFERENCE.name}: {len(rows):,} player-games; "
          f"{len(skipped)} games skipped (listed in {SKIPPED.name})")


def stint_player_games(stints: pl.DataFrame) -> pl.DataFrame:
    """Plus-minus and seconds per player-game from the stints."""
    home = stints.select("game_id", pl.col("home_ids").alias("player_id"),
                         pm=pl.col("home_pts") - pl.col("away_pts"), sec="elapsed")
    away = stints.select("game_id", pl.col("away_ids").alias("player_id"),
                         pm=pl.col("away_pts") - pl.col("home_pts"), sec="elapsed")
    return (pl.concat([home, away]).explode("player_id", empty_as_null=True)
            .group_by("game_id", "player_id")
            .agg(stint_pm=pl.col("pm").sum(), stint_seconds=pl.col("sec").sum()))


def compare(stints: pl.DataFrame, ref: pl.DataFrame,
            skipped: list[int]) -> tuple[dict, pl.DataFrame]:
    ref = ref.filter(pl.col("seconds") > 0)
    mine = stint_player_games(stints).filter(~pl.col("game_id").is_in(skipped))
    j = (ref.join(mine, on=["game_id", "player_id"], how="full", coalesce=True)
         .with_columns(pl.col("plus_minus", "seconds", "stint_pm", "stint_seconds")
                       .fill_null(0.0))
         .with_columns(d_pm=pl.col("stint_pm") - pl.col("plus_minus"),
                       d_sec=pl.col("stint_seconds") - pl.col("seconds")))
    n = j.height
    pm_exact = int((j["d_pm"].abs() < 0.5).sum())
    sec_ok = int((j["d_sec"].abs() <= SEC_TOL).sum())
    summary = {
        "games": j["game_id"].n_unique(),
        "games_skipped": len(skipped),
        "player_games": n,
        "pm_exact": pm_exact,
        "pm_exact_share": pm_exact / n,
        "pm_max_abs_diff": float(j["d_pm"].abs().max()),
        "seconds_within_1s": sec_ok,
        "seconds_within_share": sec_ok / n,
        "seconds_max_abs_diff": float(j["d_sec"].abs().max()),
    }
    misses = (j.filter((pl.col("d_pm").abs() >= 0.5) | (pl.col("d_sec").abs() > SEC_TOL))
              .sort(["game_id", "player_id"]))
    return summary, misses


def main(argv: list[str]) -> int:
    stints = pl.read_parquet(STINTS)
    if "--build" in argv:
        build(sorted(stints["game_id"].unique().to_list()))
    skipped = pl.read_csv(SKIPPED, schema_overrides={"game_id": pl.Int64})["game_id"].to_list()
    summary, misses = compare(stints, pl.read_parquet(REFERENCE), skipped)
    OUT.mkdir(parents=True, exist_ok=True)
    pl.DataFrame([summary]).write_csv(OUT / "stint_box_check.csv")
    misses.write_csv(OUT / "stint_box_misses.csv")

    s = summary
    gates = {
        f"plus-minus exact {s['pm_exact']:,}/{s['player_games']:,} "
        f"({s['pm_exact_share']:.2%}) >= {PM_EXACT_MIN:.1%}": s["pm_exact_share"] >= PM_EXACT_MIN,
        f"max |plus-minus diff| {s['pm_max_abs_diff']:.0f} <= {PM_MAX_ABS:.0f}":
            s["pm_max_abs_diff"] <= PM_MAX_ABS,
        f"seconds within {SEC_TOL:.0f} s {s['seconds_within_1s']:,}/{s['player_games']:,} "
        f"({s['seconds_within_share']:.2%}) >= {SEC_WITHIN_MIN:.1%}":
            s["seconds_within_share"] >= SEC_WITHIN_MIN,
        f"max |seconds diff| {s['seconds_max_abs_diff']:.1f} <= {SEC_MAX_ABS:.0f}":
            s["seconds_max_abs_diff"] <= SEC_MAX_ABS,
    }
    print(f"Stints vs liveData play-by-play (pbpstats lineups), {s['games']} games, "
          f"{s['player_games']:,} player-games")
    if skipped:
        print(f"  SKIPPED {len(skipped)} games pbpstats cannot give lineups for: {skipped}")
    for name, ok in gates.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"  {misses.height} player-game misses listed in {OUT.name}/stint_box_misses.csv")
    ok = all(gates.values())
    print("STINT BOX CHECK " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
