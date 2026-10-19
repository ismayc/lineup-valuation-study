"""Gate: every NBA five-man lineup row vs an independent feed, api.pbpstats.com.

Why
---
02_analysis.py's gates check that the lineup rows reconstruct the NBA's own
team totals. They cannot catch an error the NBA's lineup and team endpoints
share, because both come from the same source. pbpstats parses the
play-by-play itself and publishes per-lineup totals, so it is an
independent check on the two inputs the model takes from each row: the
minutes and the plus-minus (net_100 = 100 * PLUS_MINUS / POSS).

What the October 19, 2026 fetch showed (per-lineup rows in
output/<season>/pbpstats_lineup_check.csv): in both seasons more than 99.5%
of the lineups in both feeds agree exactly on plus-minus, and more than
99.8% agree on minutes to within 3 seconds. The possession counts do not
agree; that is a difference of definition, reported by 11_poss_gap.py and
in the README Findings, and it is not gated here.

Gates, per season (observed values are printed next to each limit)
-----------------
  * pbpstats rows that do not name five distinct players (a parse defect:
    2023-24 has one ATL row listing Bogdan Bogdanovic twice, 5.25 minutes)
    carry <= 0.1% of minutes. They are reported and left out of the rest.
  * lineup sets agree: lineups in only one feed carry <= 0.1% of minutes
    (observed: about 0.005% in 2023-24, under 0.0001% in 2025-26)
  * plus-minus exact for >= 99% of matched lineups (observed 99.6% and
    99.7%), within 2 points for >= 99.9% (all but 2 of 35,369), and no
    lineup off by more than 10. The 2 are one 2025-26 LAC pair that differs
    only by Dunn or Sanders in the fifth slot and trades 8 points and 1.2
    minutes between the two rows: the feeds disagree on one substitution's
    timing, and which feed is right is unverified. The team totals still
    agree, which is why a pair like it passes and an unbalanced error does not.
  * minutes within 0.05 (3 seconds) for >= 99% of matched lineups, and no
    lineup off by more than 2 minutes (observed worst: that LAC pair, 1.18)
  * totals: the NBA's summed lineup minutes within 0.1% and plus-minus within
    2 points of pbpstats' own team totals, for the league and every team
  * completeness: pbpstats' well-formed lineup rows sum to its team totals
    for all 30 teams (within 1 minute and 1 point), so no capped call or
    date window went missing or was counted twice

Inputs
------
  data/raw/<season>/lineups_base_<TEAM>.parquet   (committed)
  data/reference/pbpstats_lineups.parquet         (committed, from --fetch)
  data/reference/pbpstats_teams.parquet           (committed, from --fetch)

--fetch rebuilds the two reference tables from api.pbpstats.com (one call
per team per season plus one team-totals call per season, a pause between
calls, raw JSON cached under data/pbpstats_cache/, which is gitignored).
get-totals returns at most 500 rows per call, and the 500 are not the
top 500 by minutes, so a team-season that hits the cap is refetched in date
windows (three per season, halved while a window still hits the cap) and
summed per lineup. The team-totals gate below proves the sum is complete.
Without --fetch the gate replays offline.

Output: output/<season>/pbpstats_lineup_check.csv   one row per lineup (gitignored:
                                                     ~3 MB, rewritten every run)
        output/<season>/pbpstats_lineup_misses.csv  rows outside tolerance
        output/<season>/pbpstats_lineup_summary.csv counts the README quotes
Run:    python python/10_pbpstats_lineup_check.py [--fetch]
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
REF = ROOT / "data" / "reference"
CACHE = ROOT / "data" / "pbpstats_cache"
SEASONS = ["2023-24", "2025-26"]

API = "https://api.pbpstats.com/get-totals/nba"
PAUSE_S = 3.0

MALFORMED_MIN_SHARE = UNMATCHED_MIN_SHARE = 0.001
PM_EXACT_MIN, PM_NEAR, PM_NEAR_MIN, PM_MAX = 0.99, 2.0, 0.999, 10.0
MIN_TOL, MIN_WITHIN_MIN, MIN_MAX = 0.05, 0.99, 2.0
TOTAL_MIN_REL, TOTAL_PM_ABS = 0.001, 2.0


# ------------------------------------------------------------------ fetch ----
def _get(url: str, dest: Path) -> dict:
    if dest.exists():
        return json.loads(dest.read_text())
    req = urllib.request.Request(url, headers={"User-Agent": "lineup-valuation-study"})
    with urllib.request.urlopen(req, timeout=120) as r:
        body = r.read()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(body)
    time.sleep(PAUSE_S)
    return json.loads(body)


def lineup_key(ids: str) -> str:
    """Order-free key for a five-man lineup: sorted player ids joined by '-'.

    The NBA writes GROUP_ID as '-id-id-id-id-id-'; pbpstats writes EntityId
    as 'id-id-id-id-id'. Neither promises an order.
    """
    parts = sorted(x for x in ids.strip("-").split("-") if x)
    return "-".join(parts)


ROW_CAP = 500  # get-totals returns at most 500 rows per call, not by minutes


def _windows(season: str) -> list[tuple[date, date]]:
    y = int(season[:4])
    return [(date(y, 10, 1), date(y, 12, 31)), (date(y + 1, 1, 1), date(y + 1, 2, 28 if
             (y + 1) % 4 else 29)), (date(y + 1, 3, 1), date(y + 1, 4, 30))]


def _team_rows(season: str, tid: int) -> list[dict]:
    """Every lineup row for one team-season. A full-season call that comes
    back at the row cap is replaced by date windows (halved again while a
    window hits the cap) whose additive totals are summed per lineup; the
    team-totals gate proves nothing was lost or counted twice."""
    q = f"Season={season}&SeasonType=Regular%2BSeason&TeamId={tid}&Type=Lineup"
    rows = _get(f"{API}?{q}", CACHE / season / f"lineups_{tid}.json")["multi_row_table_data"]
    if len(rows) < ROW_CAP:
        return rows
    todo, parts = _windows(season), []
    while todo:
        lo, hi = todo.pop(0)
        d = _get(f"{API}?{q}&FromDate={lo}&ToDate={hi}",
                 CACHE / season / f"lineups_{tid}_{lo}_{hi}.json")["multi_row_table_data"]
        if len(d) >= ROW_CAP:
            assert lo < hi, (season, tid, lo)
            mid = lo + (hi - lo) // 2
            todo[:0] = [(lo, mid), (mid + timedelta(days=1), hi)]
            continue
        parts.extend(d)
    acc: dict[str, dict] = {}
    for x in parts:
        k0 = lineup_key(x["EntityId"])
        a = acc.setdefault(k0, {"EntityId": k0})
        for k in ("SecondsPlayed", "PlusMinus", "OffPoss", "DefPoss"):
            a[k] = a.get(k, 0) + x.get(k, 0)
    return list(acc.values())


def fetch(seasons: list[str]) -> None:
    lineups, teams = [], []
    for season in seasons:
        q = f"Season={season}&SeasonType=Regular%2BSeason"
        tdata = _get(f"{API}?{q}&Type=Team", CACHE / season / "teams.json")
        for t in tdata["multi_row_table_data"]:
            teams.append({"season": season, "team_id": int(t["EntityId"]),
                          "team": t["Name"],
                          "seconds": float(t.get("SecondsPlayed", 0)),
                          "plus_minus": float(t.get("PlusMinus", 0)),
                          "off_poss": float(t.get("OffPoss", 0)),
                          "def_poss": float(t.get("DefPoss", 0))})
        tids = pl.read_parquet(RAW / season / "team_stats.parquet")["TEAM_ID"].to_list()
        assert len(tids) == 30, (season, len(tids))
        for tid in sorted(tids):
            rows = _team_rows(season, tid)
            for x in rows:
                lineups.append({"season": season, "team_id": int(tid),
                                "lineup_key": lineup_key(x["EntityId"]),
                                "seconds": float(x.get("SecondsPlayed", 0)),
                                "plus_minus": float(x.get("PlusMinus", 0)),
                                "off_poss": float(x.get("OffPoss", 0)),
                                "def_poss": float(x.get("DefPoss", 0))})
            print(f"  {season} {tid}: {len(rows)} lineups", flush=True)
    REF.mkdir(parents=True, exist_ok=True)
    (pl.DataFrame(lineups).sort("season", "team_id", "lineup_key")
     .write_parquet(REF / "pbpstats_lineups.parquet"))
    (pl.DataFrame(teams).sort("season", "team_id")
     .write_parquet(REF / "pbpstats_teams.parquet"))
    print(f"wrote {len(lineups)} lineup rows and {len(teams)} team rows to {REF}")


# ------------------------------------------------------------------ compare --
def nba_lineups(season: str) -> pl.DataFrame:
    raw = RAW / season
    base = pl.concat([pl.read_parquet(f) for f in sorted(raw.glob("lineups_base_*.parquet"))])
    adv = pl.concat([pl.read_parquet(f) for f in sorted(raw.glob("lineups_advanced_*.parquet"))])
    return (base.select("GROUP_ID", "TEAM_ID", "TEAM_ABBREVIATION", "GROUP_NAME",
                        "MIN", "PLUS_MINUS")
            .join(adv.select("GROUP_ID", "TEAM_ID", "POSS"), on=["GROUP_ID", "TEAM_ID"],
                  how="left")
            .with_columns(team_id=pl.col("TEAM_ID").cast(pl.Int64),
                          lineup_key=pl.col("GROUP_ID").map_elements(
                              lineup_key, return_dtype=pl.String)))


def compare(nba: pl.DataFrame, pbp: pl.DataFrame) -> pl.DataFrame:
    """Full outer join on (team, lineup); one row per lineup in either feed.

    `pbp_malformed` marks pbpstats rows that do not name five distinct
    players; they are kept in the table but left out of every comparison.
    """
    p = pbp.select("team_id", "lineup_key",
                   pbp_min=pl.col("seconds") / 60, pbp_pm="plus_minus",
                   pbp_off_poss="off_poss", pbp_def_poss="def_poss")
    n = nba.select("team_id", "lineup_key", "TEAM_ABBREVIATION", "GROUP_NAME",
                   nba_min=pl.col("MIN").cast(pl.Float64),
                   nba_pm=pl.col("PLUS_MINUS").cast(pl.Float64),
                   nba_poss=pl.col("POSS").cast(pl.Float64))
    return (n.join(p, on=["team_id", "lineup_key"], how="full", coalesce=True)
            .with_columns(in_nba=pl.col("nba_min").is_not_null(),
                          in_pbp=pl.col("pbp_min").is_not_null(),
                          pbp_malformed=pl.col("pbp_min").is_not_null()
                          & (pl.col("lineup_key").str.split("-").list.unique().list.len() != 5))
            .with_columns(d_min=pl.col("nba_min") - pl.col("pbp_min"),
                          d_pm=pl.col("nba_pm") - pl.col("pbp_pm"))
            .sort("team_id", "lineup_key"))


def _wellformed(t: pl.DataFrame) -> pl.DataFrame:
    return t.filter(~pl.col("pbp_malformed"))


def team_totals(t: pl.DataFrame, pbp_teams: pl.DataFrame | None = None) -> pl.DataFrame:
    """NBA summed lineup rows per team vs pbpstats' team totals (or, without
    them, pbpstats' summed well-formed lineup rows)."""
    w = _wellformed(t)
    tt = w.group_by("team_id").agg(nba_min=pl.col("nba_min").sum(),
                                   nba_pm=pl.col("nba_pm").sum(),
                                   lu_min=pl.col("pbp_min").sum(),
                                   lu_pm=pl.col("pbp_pm").sum())
    if pbp_teams is None:
        tt = tt.with_columns(pbp_min=pl.col("lu_min"), pbp_pm=pl.col("lu_pm"))
    else:
        tt = tt.join(pbp_teams.select("team_id", pbp_min=pl.col("seconds") / 60,
                                      pbp_pm="plus_minus"),
                     on="team_id", how="full", coalesce=True).fill_null(0.0)
    return (tt.with_columns(d_min_rel=(pl.col("nba_min") / pl.col("pbp_min") - 1).abs(),
                            d_pm=(pl.col("nba_pm") - pl.col("pbp_pm")).abs())
            .sort("team_id"))


def gates(t: pl.DataFrame, pbp_teams: pl.DataFrame | None = None) -> dict[str, bool]:
    total_min = float(t["nba_min"].fill_null(0).sum())
    bad = t.filter(pl.col("pbp_malformed"))
    bad_share = float(bad["pbp_min"].sum()) / total_min if total_min else 1.0
    w = _wellformed(t)
    m = w.filter(pl.col("in_nba") & pl.col("in_pbp"))
    only_nba = float(w.filter(~pl.col("in_pbp"))["nba_min"].sum())
    only_pbp = float(w.filter(~pl.col("in_nba"))["pbp_min"].sum())
    um_share = (only_nba + only_pbp) / total_min if total_min else 1.0
    n_um = int((~(w["in_nba"] & w["in_pbp"])).sum())
    pm_exact = float((m["d_pm"].abs() < 0.5).mean()) if m.height else 0.0
    pm_near = float((m["d_pm"].abs() <= PM_NEAR).mean()) if m.height else 0.0
    n_far = int((m["d_pm"].abs() > PM_NEAR).sum())
    pm_max = float(m["d_pm"].abs().max() or 0.0)
    min_in = float((m["d_min"].abs() <= MIN_TOL).mean()) if m.height else 0.0
    min_max = float(m["d_min"].abs().max() or 0.0)
    tt = team_totals(t, pbp_teams)
    lg_min_rel = abs(float(tt["nba_min"].sum()) / float(tt["pbp_min"].sum()) - 1)
    lg_pm = abs(float(tt["nba_pm"].sum()) - float(tt["pbp_pm"].sum()))
    team_min_worst = float(tt["d_min_rel"].max())
    team_pm_worst = float(tt["d_pm"].max())
    g = {
        f"malformed pbpstats rows ({bad.height}) carry {bad_share:.4%} of minutes "
        f"<= {MALFORMED_MIN_SHARE:.1%}": bad_share <= MALFORMED_MIN_SHARE,
        f"unmatched lineups ({n_um}) carry {um_share:.4%} of minutes "
        f"<= {UNMATCHED_MIN_SHARE:.1%}": um_share <= UNMATCHED_MIN_SHARE,
        f"plus-minus exact for {pm_exact:.2%} of {m.height} matched lineups "
        f">= {PM_EXACT_MIN:.0%}": m.height > 0 and pm_exact >= PM_EXACT_MIN,
        f"plus-minus within {PM_NEAR:.0f} for {pm_near:.3%} ({n_far} beyond) "
        f">= {PM_NEAR_MIN:.1%}": m.height > 0 and pm_near >= PM_NEAR_MIN,
        f"max |plus-minus diff| {pm_max:.0f} <= {PM_MAX:.0f}": pm_max <= PM_MAX,
        f"minutes within {MIN_TOL} for {min_in:.2%} >= {MIN_WITHIN_MIN:.0%}":
            m.height > 0 and min_in >= MIN_WITHIN_MIN,
        f"max |minutes diff| {min_max:.3f} <= {MIN_MAX:.0f}": min_max <= MIN_MAX,
        f"league minutes within {lg_min_rel:.5%} <= {TOTAL_MIN_REL:.1%}, "
        f"plus-minus within {lg_pm:.0f} <= {TOTAL_PM_ABS:.0f}":
            lg_min_rel <= TOTAL_MIN_REL and lg_pm <= TOTAL_PM_ABS,
        f"every team ({tt.height}): minutes within {team_min_worst:.5%}, plus-minus "
        f"within {team_pm_worst:.0f}":
            team_min_worst <= TOTAL_MIN_REL and team_pm_worst <= TOTAL_PM_ABS,
    }
    if pbp_teams is not None:
        # pbpstats' own team totals vs its well-formed lineup rows: proves the
        # reference table is complete (no capped call or date window missing
        # or counted twice).
        w_min = float((tt["pbp_min"] - tt["lu_min"]).abs().max())
        w_pm = float((tt["pbp_pm"] - tt["lu_pm"]).abs().max())
        g[f"pbpstats lineup rows sum to its team totals ({tt.height} teams; worst "
          f"{w_min:.2f} min, {w_pm:.0f} pts)"] = tt.height == 30 and w_min <= 1.0 and w_pm <= 1.0
    return g


def misses(t: pl.DataFrame) -> pl.DataFrame:
    return t.filter(pl.col("pbp_malformed") | ~pl.col("in_nba") | ~pl.col("in_pbp")
                    | (pl.col("d_pm").abs() >= 0.5) | (pl.col("d_min").abs() > MIN_TOL))


def run(season: str, ref: pl.DataFrame, ref_teams: pl.DataFrame) -> bool:
    t = compare(nba_lineups(season), ref.filter(pl.col("season") == season))
    out = ROOT / "output" / season
    out.mkdir(parents=True, exist_ok=True)
    t.write_csv(out / "pbpstats_lineup_check.csv")
    mm = misses(t)
    mm.write_csv(out / "pbpstats_lineup_misses.csv")
    print(f"{season}: {t.height} lineups in either feed, "
          f"{int((t['in_nba'] & t['in_pbp']).sum())} matched, {mm.height} outside tolerance")
    g = gates(t, ref_teams.filter(pl.col("season") == season))
    w = t.filter(~pl.col("pbp_malformed"))
    m = w.filter(pl.col("in_nba") & pl.col("in_pbp"))
    pl.DataFrame({"metric": ["lineups_either_feed", "lineups_matched", "pm_exact",
                             "pm_beyond_2", "minutes_within_3s", "only_nba", "only_pbp",
                             "pbp_malformed", "gate_passed"],
                  "value": [float(t.height), float(m.height),
                            float((m["d_pm"].abs() < 0.5).sum()),
                            float((m["d_pm"].abs() > PM_NEAR).sum()),
                            float((m["d_min"].abs() <= MIN_TOL).sum()),
                            float((~w["in_pbp"]).sum()), float((~w["in_nba"]).sum()),
                            float(t["pbp_malformed"].sum()), float(all(g.values()))]}
                 ).write_csv(out / "pbpstats_lineup_summary.csv")
    for name, ok in g.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if mm.height:
        print(mm.sort(pl.col("nba_min").fill_null(0), descending=True).head(5)
              .select("TEAM_ABBREVIATION", "lineup_key", "nba_min", "pbp_min",
                      "nba_pm", "pbp_pm"))
    return all(g.values())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true",
                    help="rebuild data/reference from api.pbpstats.com first")
    ap.add_argument("--seasons", nargs="+", default=SEASONS)
    args = ap.parse_args()
    if args.fetch:
        fetch(args.seasons)
    ref = pl.read_parquet(REF / "pbpstats_lineups.parquet")
    ref_teams = pl.read_parquet(REF / "pbpstats_teams.parquet")
    ok = all([run(s, ref, ref_teams) for s in args.seasons])
    print("PBPSTATS LINEUP CHECK " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
