"""Harvest 2023-24 five-man lineup data + reference tables from stats.nba.com.

Endpoints (all public, via nba_api):
    LeagueDashLineups   per-team, group_quantity=5, Totals
                        - Base:     raw PLUS_MINUS, MIN per lineup
                        - Advanced: POSS, NET_RATING per lineup
    LeagueDashPlayerStats  per-player Totals (PLUS_MINUS, MIN) - external
                           reference for validation
    LeagueDashTeamStats    per-team Totals - validation that lineup rows
                           reconstruct team totals

The league-wide LeagueDashLineups call silently caps at 2,000 rows, which
truncates the long tail of rare lineups and would poison the coverage
validation. Queried per team (30 x 2 calls) instead; a team season is a few
hundred lineups, well under the cap.

Resumable: one parquet per team per measure per season; re-running skips
existing files.

Run:  python python/01_harvest_lineups.py [--seasons 2023-24 2025-26]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import polars as pl
from nba_api.stats.endpoints import (
    leaguedashlineups,
    leaguedashplayerstats,
    leaguedashteamstats,
)
from nba_api.stats.static import teams as static_teams

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

# 2023-24 matches the play-by-play study; 2025-26 is the most recent completed
# season - the endpoint is live, so the model runs on current basketball too.
DEFAULT_SEASONS = ["2023-24", "2025-26"]
PAUSE_S = 1.2  # stay polite; the endpoint rate-limits bursts


def fetch_team_lineups(season: str, team_id: int, measure: str) -> pd.DataFrame:
    return leaguedashlineups.LeagueDashLineups(
        season=season, group_quantity=5, per_mode_detailed="Totals",
        measure_type_detailed_defense=measure, team_id_nullable=team_id,
        timeout=60,
    ).get_data_frames()[0]


def harvest_season(season: str) -> None:
    raw = RAW / season
    raw.mkdir(parents=True, exist_ok=True)
    team_list = sorted(static_teams.get_teams(), key=lambda t: t["abbreviation"])
    assert len(team_list) == 30

    for team in team_list:
        for measure in ("Base", "Advanced"):
            out = raw / f"lineups_{measure.lower()}_{team['abbreviation']}.parquet"
            if out.exists():
                continue
            df = fetch_team_lineups(season, team["id"], measure)
            if df.empty:
                raise SystemExit(f"Empty lineup frame for {team['abbreviation']} {measure}")
            pl.from_pandas(df).write_parquet(out)
            print(f"  {season} {team['abbreviation']} {measure:8s}: {len(df):4d} lineups",
                  flush=True)
            time.sleep(PAUSE_S)

    ref_specs = {
        "player_stats.parquet": lambda: leaguedashplayerstats.LeagueDashPlayerStats(
            season=season, per_mode_detailed="Totals", timeout=60).get_data_frames()[0],
        "team_stats.parquet": lambda: leaguedashteamstats.LeagueDashTeamStats(
            season=season, per_mode_detailed="Totals", timeout=60).get_data_frames()[0],
    }
    for name, fetch in ref_specs.items():
        out = raw / name
        if not out.exists():
            pl.from_pandas(fetch()).write_parquet(out)
            print(f"  {season} {name}: written", flush=True)
            time.sleep(PAUSE_S)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", default=DEFAULT_SEASONS)
    args = ap.parse_args()
    for season in args.seasons:
        harvest_season(season)
    print("Harvest complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
