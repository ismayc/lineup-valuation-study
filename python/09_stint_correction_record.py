"""Record of the October 19, 2026 stint correction: first build vs corrected.

The first build is read from git (data/stints_2023.parquet at FIRST_BUILD),
so the comparison regenerates from committed files. Both builds are scored
with the same functions the gates use (07_stint_season_check.py,
08_stint_box_check.py), and 04_findings.py reads the table for the README's
correction note.

Output: output/2023-24/stint_correction.csv (metric, first_build, corrected)
Run:    python python/09_stint_correction_record.py
"""
from __future__ import annotations

import importlib.util
import io
import subprocess
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "2023-24"
FIRST_BUILD = "3c4fcaa"   # last commit with the EVENTNUM-ordered stints


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "python" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def git_file(rel: str) -> bytes:
    return subprocess.run(["git", "show", f"{FIRST_BUILD}:{rel}"], cwd=ROOT,
                          capture_output=True, check=True).stdout


def model_metrics(validation: pl.DataFrame, scale: pl.DataFrame) -> dict[str, float]:
    """What 06_stint_rapm.py reported for a build (its committed outputs)."""
    def check(name: str) -> float:
        return float(validation.filter(pl.col("check") == name)["value"][0])
    return {"stints_modeled": int(scale["stints_modeled"][0]),
            "spearman_vs_aggregate": check("Spearman vs lineup-aggregate model"),
            "home_court_per_100": check("home-court advantage in plausible range (pts/100)")}


def metrics(stints: pl.DataFrame, season, box, ref, skipped) -> dict[str, float]:
    t = season.compare(stints, pl.read_parquet(season.OFFICIAL))
    s, _ = box.compare(stints, ref, skipped)
    return {
        "stints_parsed": stints.height,
        "negative_elapsed_stints": stints.filter(pl.col("elapsed") < 0).height,
        "negative_points_stints": stints.filter((pl.col("home_pts") < 0)
                                                | (pl.col("away_pts") < 0)).height,
        "player_games_checked": s["player_games"],
        "player_game_pm_exact": s["pm_exact"],
        "player_game_pm_exact_share": s["pm_exact_share"],
        "player_game_seconds_within_1s_share": s["seconds_within_share"],
        "players_checked": t.height,
        "season_pm_exact_players": int((t["d_pm"].abs() < 0.5).sum()),
        "season_pm_max_abs_diff": float(t["d_pm"].abs().max()),
        "season_minutes_within_1_share": float((t["d_min"].abs() <= 1).mean()),
    }


def main() -> int:
    season, box = load("07_stint_season_check"), load("08_stint_box_check")
    ref = pl.read_parquet(box.REFERENCE)
    skipped = pl.read_csv(box.SKIPPED)["game_id"].to_list()
    first = metrics(pl.read_parquet(io.BytesIO(git_file("data/stints_2023.parquet"))),
                    season, box, ref, skipped)
    first |= model_metrics(
        pl.read_csv(io.BytesIO(git_file("output/2023-24/stint_validation.csv"))),
        pl.read_csv(io.BytesIO(git_file("output/2023-24/stint_scale.csv"))))
    now = metrics(pl.read_parquet(box.STINTS), season, box, ref, skipped)
    now |= model_metrics(pl.read_csv(OUT / "stint_validation.csv"),
                         pl.read_csv(OUT / "stint_scale.csv"))
    table = pl.DataFrame({"metric": list(first), "first_build": list(first.values()),
                          "corrected": [now[k] for k in first]},
                         schema={"metric": pl.String, "first_build": pl.Float64,
                                 "corrected": pl.Float64})
    table.write_csv(OUT / "stint_correction.csv")
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
