"""Unit tests for python/05_build_stints.py on a synthetic filled game.

The three conventions corrected on 2026-10-19: clock order (never EVENTNUM),
free-throw points to the lineup at the foul, and stints that run to the next
stint's first event. Plus the period labels nba-on-court's cumulative clock
broke.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import load_script

HOME, AWAY = 100, 200
OLD_HOME = [1, 2, 3, 4, 5]
NEW_HOME = [1, 2, 3, 4, 6]       # 6 in for 5 between the foul and the FTs
AWAY_IDS = [11, 12, 13, 14, 15]


@pytest.fixture(scope="module")
def b05():
    return load_script("python/05_build_stints.py", "lineup_build_stints")


def _game(rows: list[tuple]) -> pd.DataFrame:
    """rows: (elapsed game seconds, EVENTNUM, type, player1, team1, SCORE, home five)."""
    out = []
    for i, (t, num, typ, p1, team, score, home) in enumerate(rows):
        r = {"GAME_ID": 22300001, "PERIOD": 1 if t <= 720 else 2,
             "PCTIMESTRING": float(t), "EVENTNUM": num, "_ROW": i,
             "EVENTMSGTYPE": typ, "PLAYER1_ID": p1, "PLAYER1_TEAM_ID": team,
             "SCORE": score}
        r.update({f"HOME_PLAYER{k + 1}": v for k, v in enumerate(home)})
        r.update({f"AWAY_PLAYER{k + 1}": v for k, v in enumerate(AWAY_IDS)})
        out.append(r)
    return pd.DataFrame(out)


def _foul_then_sub_then_fts() -> pd.DataFrame:
    return _game([
        (0, 1, 12, np.nan, np.nan, None, OLD_HOME),        # start of period
        (30, 2, 1, 1, HOME, "0 - 2", OLD_HOME),            # home basket
        (100, 3, 6, 11, AWAY, None, OLD_HOME),             # away foul
        (100, 4, 8, 5, HOME, None, OLD_HOME),              # sub: 6 for 5
        (100, 5, 3, 2, HOME, "0 - 3", NEW_HOME),           # FT made
        (100, 6, 3, 2, HOME, "0 - 4", NEW_HOME),           # FT made
        (200, 7, 2, 11, AWAY, None, NEW_HOME),             # away miss
        (720, 8, 13, np.nan, np.nan, None, NEW_HOME),      # end of period
    ])


def test_free_throws_go_to_the_lineup_at_the_foul(b05):
    first, second = b05.game_stints(_foul_then_sub_then_fts())
    assert sorted(first["home_ids"]) == OLD_HOME
    assert first["home_pts"] == 4.0          # basket + both free throws
    assert second["home_pts"] == 0.0         # the incoming lineup gets none
    assert first["away_pts"] == second["away_pts"] == 0.0


def test_stints_run_to_the_next_stints_first_event(b05):
    # the old five's last event is the basket at 30 s elapsed; the sub row at
    # 100 s already carries the new five, so 70 s fall between the stints
    g = _game([
        (0, 1, 12, np.nan, np.nan, None, OLD_HOME),
        (30, 2, 1, 1, HOME, "0 - 2", OLD_HOME),
        (100, 3, 8, 5, HOME, None, NEW_HOME),
        (200, 4, 2, 11, AWAY, None, NEW_HOME),
        (720, 5, 13, np.nan, np.nan, None, NEW_HOME),
    ])
    first, second = b05.game_stints(g)
    assert (first["start_secs"], first["end_secs"], first["elapsed"]) == (720.0, 620.0, 100.0)
    assert (second["start_secs"], second["end_secs"], second["elapsed"]) == (620.0, 0.0, 620.0)
    assert first["elapsed"] + second["elapsed"] == 720.0


def test_events_are_ordered_by_clock_not_eventnum(b05):
    g = _foul_then_sub_then_fts()
    # a late-inserted event: the basket at 30 s carries the highest EVENTNUM
    # and arrives last in the file
    g.loc[1, "EVENTNUM"] = 99
    shuffled = pd.concat([g.drop(index=1), g.loc[[1]]]).reset_index(drop=True)
    shuffled["_ROW"] = np.arange(len(shuffled))
    assert b05.game_stints(shuffled) == b05.game_stints(_foul_then_sub_then_fts())


def test_same_second_ties_keep_file_order(b05):
    ordered = b05.clock_order(_foul_then_sub_then_fts())
    at_100 = ordered[ordered["PCTIMESTRING"] == 100.0]["EVENTMSGTYPE"].tolist()
    assert at_100 == [6, 8, 3, 3]            # foul, sub, FT, FT


def test_secs_left_reads_nba_on_courts_cumulative_clock(b05):
    df = pd.DataFrame({"PERIOD": [1, 2, 4, 5, 6],
                       "PCTIMESTRING": [100.0, 820.0, 2880.0, 2880.0, 3300.0]})
    assert b05.secs_left(df).tolist() == [620.0, 620.0, 0.0, 300.0, 180.0]


def test_secs_left_reads_raw_mmss(b05):
    df = pd.DataFrame({"PERIOD": [1, 5], "PCTIMESTRING": ["11:40", "4:05"]})
    assert b05.secs_left(df).tolist() == [700.0, 245.0]


def test_point_owner_resets_at_period_boundaries(b05):
    # a free throw after a period start, with no foul since, stays put
    types = np.array([6, 13, 12, 3])
    stints = np.array([1, 1, 2, 3])
    assert b05.point_owner(types, stints).tolist() == [1, 1, 2, 3]
