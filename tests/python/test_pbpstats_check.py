"""Unit tests for python/10_pbpstats_lineup_check.py on synthetic lineups.

The gate joins the NBA's lineup rows to pbpstats' on an order-free key and
must fail on each error class it exists to catch: a plus-minus miss, a
minutes miss, a lineup missing from one feed, and season totals that drift.
"""
from __future__ import annotations

import polars as pl
import pytest

from conftest import load_script


@pytest.fixture(scope="module")
def c10():
    return load_script("python/10_pbpstats_lineup_check.py", "lineup_pbpstats_check")


def _feeds(c10, n_lineups: int = 200):
    nba, pbp = [], []
    for i in range(n_lineups):
        ids = [i * 10 + k for k in (5, 1, 4, 2, 3)]          # unsorted on purpose
        team = 1 if i % 2 else 2
        mins, pm = 10.0 + i, (i % 7) - 3
        nba.append({"GROUP_ID": "-" + "-".join(map(str, ids)) + "-", "TEAM_ID": team,
                    "TEAM_ABBREVIATION": "AAA" if team == 1 else "BBB",
                    "GROUP_NAME": f"L{i}", "MIN": mins, "PLUS_MINUS": pm,
                    "POSS": 2 * mins})
        pbp.append({"team_id": team, "lineup_key": "-".join(map(str, sorted(ids))),
                    "seconds": mins * 60, "plus_minus": float(pm),
                    "off_poss": 2 * mins, "def_poss": 2 * mins})
    n = pl.DataFrame(nba).with_columns(
        team_id=pl.col("TEAM_ID").cast(pl.Int64),
        lineup_key=pl.col("GROUP_ID").map_elements(c10.lineup_key, return_dtype=pl.String))
    return n, pl.DataFrame(pbp)


def _teams(pbp: pl.DataFrame, n_teams: int = 30) -> pl.DataFrame:
    """pbpstats team totals that agree with the lineup rows (30 teams, as the
    real gate requires; teams with no synthetic lineups total zero)."""
    t = pbp.group_by("team_id").agg(pl.col("seconds", "plus_minus").sum())
    extra = pl.DataFrame({"team_id": list(range(3, n_teams + 1)),
                          "seconds": [0.0] * (n_teams - 2), "plus_minus": [0.0] * (n_teams - 2)})
    return pl.concat([t, extra.cast(t.schema)])


def test_lineup_key_is_order_free(c10):
    assert c10.lineup_key("-5-1-4-2-3-") == c10.lineup_key("1-2-3-4-5") == "1-2-3-4-5"


def test_identical_feeds_pass(c10):
    n, p = _feeds(c10)
    t = c10.compare(n, p)
    assert t.height == 200 and (t["in_nba"] & t["in_pbp"]).all()
    g = c10.gates(t, _teams(p))
    assert all(g.values()), g
    assert c10.misses(t).height == 0


def test_plus_minus_miss_fails(c10):
    n, p = _feeds(c10)
    p = p.with_columns(plus_minus=pl.when(pl.col("lineup_key") == p["lineup_key"][7])
                       .then(pl.col("plus_minus") + 5).otherwise(pl.col("plus_minus")))
    t = c10.compare(n, p)
    g = c10.gates(t)
    assert not g[next(k for k in g if k.startswith("plus-minus within"))]   # 1 of 200
    assert not g[next(k for k in g if k.startswith("every team"))]          # unbalanced
    assert c10.misses(t).height == 1


def test_large_plus_minus_miss_fails_max(c10):
    n, p = _feeds(c10, 5000)
    p = p.with_columns(plus_minus=pl.when(pl.col("lineup_key") == p["lineup_key"][7])
                       .then(pl.col("plus_minus") + 12).otherwise(pl.col("plus_minus")))
    g = c10.gates(c10.compare(n, p))
    assert g[next(k for k in g if k.startswith("plus-minus within"))]       # 1 of 5000 is rare
    assert not g[next(k for k in g if k.startswith("max |plus-minus"))]


def test_balanced_swap_passes_totals(c10):
    """A substitution-timing disagreement moves points between two lineups of
    one team; team totals still agree (the 2025-26 LAC case)."""
    n, p = _feeds(c10, 5000)
    a, b = p.filter(pl.col("team_id") == 1)["lineup_key"][:2]
    p = p.with_columns(plus_minus=pl.when(pl.col("lineup_key") == a).then(pl.col("plus_minus") + 8)
                       .when(pl.col("lineup_key") == b).then(pl.col("plus_minus") - 8)
                       .otherwise(pl.col("plus_minus")))
    g = c10.gates(c10.compare(n, p))
    assert all(g.values()), g


def test_many_small_plus_minus_misses_fail_exact_share(c10):
    n, p = _feeds(c10)
    p = p.with_columns(plus_minus=pl.col("plus_minus")
                       + (pl.int_range(pl.len()) % 50 == 0).cast(pl.Float64))   # 4 of 200 off by 1
    g = c10.gates(c10.compare(n, p))
    assert not g[next(k for k in g if k.startswith("plus-minus exact"))]


def test_minutes_miss_fails(c10):
    n, p = _feeds(c10)
    p = p.with_columns(seconds=pl.when(pl.col("lineup_key") == p["lineup_key"][3])
                       .then(pl.col("seconds") + 180).otherwise(pl.col("seconds")))
    g = c10.gates(c10.compare(n, p))
    assert not g[next(k for k in g if k.startswith("max |minutes"))]


def test_lineup_missing_from_one_feed_fails(c10):
    n, p = _feeds(c10)
    big = p.sort("seconds", descending=True)["lineup_key"][0]
    t = c10.compare(n, p.filter(pl.col("lineup_key") != big))
    assert int((~t["in_pbp"]).sum()) == 1
    g = c10.gates(t)
    assert not g[next(k for k in g if k.startswith("unmatched"))]
    assert c10.misses(t).height == 1


def test_team_totals_mismatch_fails(c10):
    n, p = _feeds(c10)
    tm = _teams(p).with_columns(plus_minus=pl.when(pl.col("team_id") == 1)
                                .then(pl.col("plus_minus") + 10).otherwise(pl.col("plus_minus")))
    g = c10.gates(c10.compare(n, p), tm)
    assert not g[next(k for k in g if k.startswith("pbpstats lineup rows sum"))]


def test_truncated_team_list_fails(c10):
    n, p = _feeds(c10)
    g = c10.gates(c10.compare(n, p), _teams(p).head(29))
    assert not g[next(k for k in g if k.startswith("pbpstats lineup rows sum"))]


def test_malformed_pbpstats_row_is_reported_and_excluded(c10):
    n, p = _feeds(c10)
    dup = pl.DataFrame({"team_id": [1], "lineup_key": ["7-7-8-9-10"], "seconds": [315.0],
                        "plus_minus": [4.0], "off_poss": [11.0], "def_poss": [11.0]})
    t = c10.compare(n, pl.concat([p, dup]))
    assert int(t["pbp_malformed"].sum()) == 1
    g = c10.gates(t, _teams(p))       # team totals exclude it, as pbpstats' do
    assert all(g.values()), g
    assert c10.misses(t).height == 1
    big = dup.with_columns(seconds=pl.lit(1e6))
    g = c10.gates(c10.compare(n, pl.concat([p, big])), _teams(p))
    assert not g[next(k for k in g if k.startswith("malformed"))]
