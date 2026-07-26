"""Build stint-level data: play-by-play with all ten on-court players.

This is the upgrade path the lineup-aggregate model documented: season lineup
aggregates cannot say who the OPPONENTS were, so opponent strength went
unadjusted. Stints — maximal runs of events with the same ten players on the
floor — carry both lineups, which is what full RAPM needs.

Sources (see ../docs/public-data-availability.md):
  data/pbp_bulk/nbastats_2023.csv   bulk PlayByPlayV2-format season file
                                    (shufinskiy/nba_data; one download instead
                                    of 1,230 rate-limited calls)
  nba-on-court                      fills the ten on-court player ids per
                                    event, OFFLINE - substitution logic only

Per stint: elapsed clock, home/away points (from the running SCORE), and a
possession estimate per side (FGA - OREB + TO + 0.44*FTA, the standard
box-score possession formula applied to the stint's events).

Output: data/stints_2023.parquet
Run:    python python/05_build_stints.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
BULK = ROOT / "data" / "pbp_bulk" / "nbastats_2023.csv"
OUT = ROOT / "data" / "stints_2023.parquet"

HOME_COLS = [f"HOME_PLAYER{i}" for i in range(1, 6)]
AWAY_COLS = [f"AWAY_PLAYER{i}" for i in range(1, 6)]

# EVENTMSGTYPE codes (PlayByPlayV2)
MADE, MISS, FT, REB, TO = 1, 2, 3, 4, 5


def secs_left(df: pd.DataFrame) -> pd.Series:
    """Seconds remaining in the period.

    nba-on-court REWRITES PCTIMESTRING from 'MM:SS' remaining to integer
    seconds ELAPSED in the period - discovered the hard way. Handle both:
    numeric means elapsed (subtract from the period length: 720 regulation,
    300 overtime); string means the raw 'MM:SS' remaining.
    """
    period_len = df["PERIOD"].map(lambda p: 720.0 if p <= 4 else 300.0)
    if pd.api.types.is_numeric_dtype(df["PCTIMESTRING"]):
        return period_len - df["PCTIMESTRING"].astype(float)
    parts = df["PCTIMESTRING"].astype(str).str.extract(r"^(\d+):(\d+)$")
    return (pd.to_numeric(parts[0], errors="coerce") * 60
            + pd.to_numeric(parts[1], errors="coerce"))


def game_stints(g: pd.DataFrame) -> list[dict]:
    """Split one filled game into stints and aggregate each."""
    g = g.sort_values(["PERIOD", "EVENTNUM"]).reset_index(drop=True)
    g["secs_left"] = secs_left(g)

    # Running score. SCORE arrives as 'visitor - home' on scoring events only.
    sc = g["SCORE"].str.extract(r"(\d+)\s*-\s*(\d+)")
    g["away_score"] = pd.to_numeric(sc[0]).ffill().fillna(0)
    g["home_score"] = pd.to_numeric(sc[1]).ffill().fillna(0)

    # Which team id is home? Any event whose PLAYER1 is in the home lineup
    # answers it.
    home_set = set(g[HOME_COLS].to_numpy().ravel())
    p1_in_home = g["PLAYER1_ID"].isin(home_set) & g["PLAYER1_TEAM_ID"].notna()
    if not p1_in_home.any():
        return []
    home_tid = float(g.loc[p1_in_home, "PLAYER1_TEAM_ID"].iloc[0])

    # Event team: PLAYER1_TEAM_ID for player events; team rebounds/turnovers
    # carry the team id in PLAYER1_ID itself.
    ev_team = g["PLAYER1_TEAM_ID"].fillna(g["PLAYER1_ID"])

    # Offensive rebound = rebound by the team that last missed (FG or FT).
    miss_team = ev_team.where(g["EVENTMSGTYPE"].isin([MISS, FT]))
    last_miss_team = miss_team.ffill()
    is_oreb = (g["EVENTMSGTYPE"] == REB) & (ev_team == last_miss_team)

    lineup_key = (g[HOME_COLS + AWAY_COLS].astype("int64").astype(str)
                  .agg("|".join, axis=1) + "@" + g["PERIOD"].astype(str))
    stint_id = (lineup_key != lineup_key.shift()).cumsum()

    rows = []
    prev_h = prev_a = 0.0
    for _, s in g.groupby(stint_id, sort=False):
        first, last = s.iloc[0], s.iloc[-1]
        ev = s["EVENTMSGTYPE"]
        team = ev_team.loc[s.index]
        oreb = is_oreb.loc[s.index]

        def poss(side_home: bool) -> float:
            mask = (team == home_tid) if side_home else (team != home_tid)
            fga = float(((ev == MADE) | (ev == MISS))[mask & team.notna()].sum())
            fta = float((ev == FT)[mask & team.notna()].sum())
            tov = float((ev == TO)[mask & team.notna()].sum())
            orb = float(oreb[mask & team.notna()].sum())
            return fga - orb + tov + 0.44 * fta

        h_pts = float(last["home_score"]) - prev_h
        a_pts = float(last["away_score"]) - prev_a
        prev_h, prev_a = float(last["home_score"]), float(last["away_score"])

        rows.append({
            "game_id": int(first["GAME_ID"]),
            "period": int(first["PERIOD"]),
            "start_secs": float(first["secs_left"]),
            "end_secs": float(last["secs_left"]),
            "elapsed": float(first["secs_left"] - last["secs_left"]),
            "home_pts": h_pts,
            "away_pts": a_pts,
            "poss_home": poss(True),
            "poss_away": poss(False),
            "n_events": int(len(s)),
            "home_ids": sorted(int(x) for x in first[HOME_COLS]),
            "away_ids": sorted(int(x) for x in first[AWAY_COLS]),
        })
    return rows


def main() -> int:
    """First run processes every game; later runs are RETRY passes that only
    process games missing from the existing parquet. nba-on-court is offline
    for most games but falls back to stats.nba.com for periods where a player
    logged no events - those requests can time out, so retries (with a pause
    between games) are part of the design, not an afterthought.
    """
    import time

    import nba_on_court as noc

    print("Reading bulk season file...")
    nba = pd.read_csv(BULK)
    game_ids = sorted(nba["GAME_ID"].unique())
    print(f"{len(nba):,} events, {len(game_ids)} games")

    existing = None
    todo = game_ids
    retry_mode = OUT.exists()
    if retry_mode:
        existing = pl.read_parquet(OUT)
        done = set(existing["game_id"].to_list())
        todo = [g for g in game_ids if g not in done]
        print(f"retry pass: {len(done)} games already built, {len(todo)} to do")
        if not todo:
            print("nothing to do")
            return 0

    all_rows: list[dict] = []
    skipped = 0
    for i, gid in enumerate(todo):
        g = nba[nba.GAME_ID == gid].reset_index(drop=True)
        try:
            # timeout kwarg reaches the BoxScoreTraditionalV2 fallback that
            # nba-on-court fires for periods where a player logged no events;
            # its 10s default reliably times out on this endpoint (and its
            # retry loop only catches ConnectionError, not ReadTimeout).
            filled = noc.players_on_court(g, timeout=60)
            all_rows.extend(game_stints(filled))
        except Exception as e:  # noqa: BLE001 - report and continue
            skipped += 1
            print(f"  SKIP {gid}: {type(e).__name__}: {e}")
        if retry_mode:
            time.sleep(1.0)   # the retry pass is here because of rate limits
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(todo)} games", flush=True)

    df = pl.DataFrame(all_rows)
    if existing is not None and df.height:
        df = pl.concat([existing, df]).sort(["game_id", "period"])
    elif existing is not None:
        df = existing
    df.write_parquet(OUT)
    print(f"wrote {OUT}: {df.height:,} stints from {df['game_id'].n_unique()} games"
          + (f" ({skipped} games still missing)" if skipped else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
