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

Three conventions, corrected October 19, 2026 (the first build had all three
wrong; 07_stint_season_check.py and 08_stint_box_check.py now gate them):

  * Events are processed in GAME-CLOCK order, ties broken by the bulk file's
    row order. EVENTNUM is not chronological: late-inserted events carry
    later numbers (3,966 clock reversals in 2,490 periods of 2023-24), which
    produced 123 stints with negative elapsed time and 165 with negative
    points. The same bug hit playbyplay-study's V3 windows.
  * Free-throw points belong to the lineup on the floor at the FOUL, which is
    how the official box plus-minus counts them; a substitution between foul
    and free throw no longer moves the points to the incoming lineup. (The
    0.44*FTA possession term stays with the stint holding the free throw.)
  * A stint runs from its first event to the NEXT stint's first event (or the
    period end), so the clock between one lineup's last event and the
    substitution counts. Stint seconds then sum to the full game.

Output: data/stints_2023.parquet
Run:    python python/05_build_stints.py            build missing games only
        python python/05_build_stints.py --rebuild  rebuild every game
"""
from __future__ import annotations

from pathlib import Path

import sys

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
BULK = ROOT / "data" / "pbp_bulk" / "nbastats_2023.csv"
OUT = ROOT / "data" / "stints_2023.parquet"

HOME_COLS = [f"HOME_PLAYER{i}" for i in range(1, 6)]
AWAY_COLS = [f"AWAY_PLAYER{i}" for i in range(1, 6)]

# EVENTMSGTYPE codes (PlayByPlayV2)
MADE, MISS, FT, REB, TO, FOUL = 1, 2, 3, 4, 5, 6
PERIOD_START, PERIOD_END = 12, 13


def period_end_elapsed(period: pd.Series) -> pd.Series:
    """Game seconds elapsed at the end of each period (720 s regulation
    periods, 300 s overtimes)."""
    return 720.0 * period.clip(upper=4) + 300.0 * (period - 4).clip(lower=0)


def secs_left(df: pd.DataFrame) -> pd.Series:
    """Seconds remaining in the period.

    nba-on-court REWRITES PCTIMESTRING from 'MM:SS' remaining to integer
    seconds ELAPSED IN THE GAME (not the period: the first build subtracted
    it from the period length, so every label after period 1 was off by
    whole periods; elapsed differences were unaffected). Handle both:
    numeric means game seconds elapsed; string means the raw 'MM:SS'
    remaining in the period.
    """
    if pd.api.types.is_numeric_dtype(df["PCTIMESTRING"]):
        return period_end_elapsed(df["PERIOD"]) - df["PCTIMESTRING"].astype(float)
    parts = df["PCTIMESTRING"].astype(str).str.extract(r"^(\d+):(\d+)$")
    return (pd.to_numeric(parts[0], errors="coerce") * 60
            + pd.to_numeric(parts[1], errors="coerce"))


def clock_order(g: pd.DataFrame) -> pd.DataFrame:
    """Events in game-clock order: period, then seconds left descending, then
    the bulk file's row order (_ROW) for events on the same second. Never
    EVENTNUM (see the module docstring)."""
    g = g.assign(secs_left=secs_left(g))
    tie = "_ROW" if "_ROW" in g.columns else None
    keys = ["PERIOD", "secs_left"] + ([tie] if tie else [])
    asc = [True, False] + ([True] if tie else [])
    return g.sort_values(keys, ascending=asc, kind="stable").reset_index(drop=True)


def point_owner(event_type: np.ndarray, stint_id: np.ndarray) -> np.ndarray:
    """The stint each event's points belong to: its own, except a free throw,
    which belongs to the stint of the most recent foul in the period."""
    owner = stint_id.copy()
    last_foul = None
    for i, t in enumerate(event_type):
        if t == FOUL:
            last_foul = stint_id[i]
        elif t in (PERIOD_START, PERIOD_END):
            last_foul = None
        elif t == FT and last_foul is not None:
            owner[i] = last_foul
    return owner


def game_stints(g: pd.DataFrame) -> list[dict]:
    """Split one filled game into stints and aggregate each."""
    g = clock_order(g)

    # Running score. SCORE arrives as 'visitor - home' on scoring events only.
    sc = g["SCORE"].str.extract(r"(\d+)\s*-\s*(\d+)")
    g["away_score"] = pd.to_numeric(sc[0]).ffill().fillna(0)
    g["home_score"] = pd.to_numeric(sc[1]).ffill().fillna(0)
    d_home = g["home_score"].diff().fillna(g["home_score"]).to_numpy()
    d_away = g["away_score"].diff().fillna(g["away_score"]).to_numpy()

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
    n_stints = int(stint_id.iloc[-1])
    owner = point_owner(g["EVENTMSGTYPE"].to_numpy(), stint_id.to_numpy())
    home_pts = np.bincount(owner, weights=d_home, minlength=n_stints + 1)
    away_pts = np.bincount(owner, weights=d_away, minlength=n_stints + 1)
    starts = g.groupby(stint_id, sort=True).agg(period=("PERIOD", "first"),
                                                start=("secs_left", "first"))
    # a stint ends where the next one in its period starts, or at 0 s left
    nxt_start = starts["start"].shift(-1)
    same_period = starts["period"].shift(-1) == starts["period"]
    ends = nxt_start.where(same_period, 0.0)

    rows = []
    for k, s in g.groupby(stint_id, sort=False):
        first = s.iloc[0]
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

        end = float(ends.loc[k])
        rows.append({
            "game_id": int(first["GAME_ID"]),
            "period": int(first["PERIOD"]),
            "start_secs": float(first["secs_left"]),
            "end_secs": end,
            "elapsed": float(first["secs_left"]) - end,
            "home_pts": float(home_pts[k]),
            "away_pts": float(away_pts[k]),
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

    if not hasattr(np, "in1d"):
        np.in1d = np.isin  # removed in NumPy 2.x; nba-on-court still calls it

    # stats.nba.com hangs requests whose User-Agent claims a real browser the
    # TLS fingerprint can't back up (nba_api ships Firefox/72). A generic
    # AppleWebKit UA passes; verified 2026-07-25 after 30/30 ReadTimeouts.
    from nba_api.stats.library import http as nba_http
    nba_http.STATS_HEADERS["User-Agent"] = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36")

    import nba_on_court as noc

    print("Reading bulk season file...")
    nba = pd.read_csv(BULK)
    nba["_ROW"] = np.arange(len(nba))  # same-second tiebreak, see clock_order
    game_ids = sorted(nba["GAME_ID"].unique())
    print(f"{len(nba):,} events, {len(game_ids)} games")

    existing = None
    todo = game_ids
    retry_mode = OUT.exists() and "--rebuild" not in sys.argv
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
