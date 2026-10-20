"""Explainer diagrams for the story page, drawn with the family's famdiagram
(../basketball-analysis-tools/famdiagram.py) and saved to figures/<ref>.svg
(plus a <ref>_narrow.svg phone layout). report_builder.py inlines them.

Every number drawn is read from committed files: the model scripts'
settings, the output tables, and the committed lineup and stint rows of the
two examples (data/raw/2023-24/, data/stints_2023.parquet).

Diagrams (all 2023-24):
  dg1_two_rows    how each model turns basketball into one regression row:
                  a real five-man lineup's season (lineup model) and a real
                  stint with the same five on the floor (stint model)
                  (data/raw/2023-24/lineups_advanced_<team>.parquet,
                  data/raw/2023-24/player_stats.parquet,
                  data/stints_2023.parquet, output/2023-24/scale.csv,
                  output/2023-24/stint_scale.csv,
                  output/2023-24/stint_correction.csv)
  dg2_two_builds  the two independent builds, their audits, and where they
                  meet (output/2023-24/scale.csv, validation.csv,
                  stint_scale.csv, stint_box_check.csv, stint_validation.csv,
                  player_values.csv, stint_player_values.csv,
                  stint_vs_lineup.csv)
  dg3_tiers       what the intervals allow: how many players are clearly
                  above or below average, and which ranks the 5th-ranked
                  player's interval cannot separate from him
                  (output/2023-24/player_values.csv)

Run:     python python/12_diagrams.py
--check: rebuild every SVG in memory and fail if a committed one differs.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import tempfile
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
SEASON = "2023-24"
OUT = ROOT / "output" / SEASON
RAW = ROOT / "data" / "raw" / SEASON
FIG = ROOT / "figures"
sys.path.insert(0, str(ROOT.parent / "basketball-analysis-tools"))
import famdiagram as fd  # noqa: E402


def _load(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


MIN_POSS = _load("python/02_analysis.py", "lineup_02_for_diagrams").MIN_POSS

# Text keys only. dg1's lineup: the best 1,000+ possession unit the story
# names (the first row of output/2023-24/lineup_table.csv); its stint: the
# one with the most possessions in which those five shared the floor.
# dg3's reference rank: the story's "5th-best".
REF_RANK = 5
OTHER_RANK = 30   # the story's "30th"
SHOW_RANKS = 70
WORD = {1: "first", 2: "second", 3: "third", 4: "fourth"}


def wrap(s: str, width: float, size: float) -> list[str]:
    per = size * 0.55
    out, cur = [], ""
    for w in s.split():
        t = f"{cur} {w}".strip()
        if len(t) * per > width and cur:
            out.append(cur)
            cur = w
        else:
            cur = t
    if cur:
        out.append(cur)
    return out


def para(d: fd.Diagram, x, y, s, width, size, lh=None, **kw) -> float:
    """Wrapped text; returns the baseline of the last line."""
    lh = lh or size * 1.35
    lines = wrap(s, width, size)
    for i, line in enumerate(lines):
        d.text(x, y + i * lh, line, size=size, **kw)
    return y + (len(lines) - 1) * lh


def sgn(v: float, dgt: int = 1) -> str:
    return f"{v:+.{dgt}f}".replace("-", "−")


def clock(secs: float) -> str:
    s = int(secs)
    return f"{s // 60}:{s % 60:02d}"


def last_name(full: str) -> str:
    parts = full.split()
    return " ".join(parts[1:]) if len(parts) > 1 else full


# ---------------------------------------------------------------- dg1
def examples() -> dict:
    top = pl.read_csv(OUT / "lineup_table.csv").row(0, named=True)
    team = top["TEAM_ABBREVIATION"]
    lu = (pl.read_parquet(RAW / f"lineups_advanced_{team}.parquet")
            .filter(pl.col("GROUP_NAME") == top["GROUP_NAME"]).row(0, named=True))
    ids = [int(t) for t in lu["GROUP_ID"].strip("-").split("-")]
    names = dict(pl.read_parquet(RAW / "player_stats.parquet")
                   .select(pl.col("PLAYER_ID").cast(pl.Int64), "PLAYER_NAME").iter_rows())
    teams = dict(pl.read_parquet(RAW / "player_stats.parquet")
                   .select(pl.col("PLAYER_ID").cast(pl.Int64), "TEAM_ABBREVIATION").iter_rows())
    # the lineup's names in the feed's own order (GROUP_NAME)
    order = [n.strip() for n in top["GROUP_NAME"].split(" - ")]
    by_short = {f"{names[i].split()[0][0]}. {last_name(names[i])}": i for i in ids}
    ids = [by_short[n] for n in order]
    st = (pl.read_parquet(ROOT / "data" / "stints_2023.parquet")
            .with_columns(poss=0.5 * (pl.col("poss_home") + pl.col("poss_away")))
            .filter(pl.col("home_ids").list.set_intersection(ids).list.len().eq(5)
                    | pl.col("away_ids").list.set_intersection(ids).list.len().eq(5))
            .sort(["poss", "game_id", "period"], descending=[True, False, False]))
    s = st.row(0, named=True)
    side = "home" if set(ids) <= set(s["home_ids"]) else "away"
    opp_ids = s["away_ids"] if side == "home" else s["home_ids"]
    opp_team = pl.Series([teams[i] for i in opp_ids]).mode().sort()[0]
    return {"top": top, "team": team, "ids": ids, "names": names, "stint": s,
            "side": side, "opp_ids": sorted(opp_ids, key=lambda i: last_name(names[i])),
            "opp_team": opp_team}


def two_rows(narrow: bool) -> fd.Diagram:
    ex = examples()
    top, team, names, s = ex["top"], ex["team"], ex["names"], ex["stint"]
    scale = pl.read_csv(OUT / "scale.csv").row(0, named=True)
    sscale = pl.read_csv(OUT / "stint_scale.csv").row(0, named=True)
    home_team = team if ex["side"] == "home" else ex["opp_team"]
    away_team = ex["opp_team"] if ex["side"] == "home" else team
    home_ids = ex["ids"] if ex["side"] == "home" else ex["opp_ids"]
    away_ids = ex["opp_ids"] if ex["side"] == "home" else ex["ids"]
    margin = s["home_pts"] - s["away_pts"]
    y_stint = 100 * margin / s["poss"]

    W = 360 if narrow else 760
    fs, small, head = (13, 12, 17) if narrow else (15, 14, 18)
    pad = 12
    d = fd.Diagram(W, 10, "How each model turns basketball into one row of a regression")
    colw = (W - 2 * pad) if narrow else (W - 3 * pad) / 2
    y0 = 14
    bottoms = []

    # ---- lineup model panel
    px, py = pad, y0
    lines_y = py + 28
    d.text(px + 12, lines_y, "Lineup model", size=head, display=True, weight="700", fill="nba")
    ly = para(d, px + 12, lines_y + 22, "One row per five-man lineup, summed over the season",
              colw - 24, small, fill="ink2") + 30
    d.text(px + 12, ly, f"{team}'s best high-volume lineup", size=fs, fill="ink", weight="700")
    ly += fs * 1.6
    for i in ex["ids"]:
        d.text(px + 14, ly, "+1", size=fs, fill="nba", weight="700")
        d.text(px + 46, ly, names[i], size=fs, fill="ink")
        ly += fs * 1.45
    d.text(px + 14, ly, "0", size=fs, fill="ink3", weight="700")
    d.text(px + 46, ly, "every other player", size=fs, fill="ink3")
    ly += fs * 2
    ly = para(d, px + 12, ly, f"Outcome: {sgn(top['PLUS_MINUS'], 0)} over "
              f"{top['POSS']:,} possessions = {sgn(top['net_100'])} per 100",
              colw - 24, fs, fill="ink") + fs * 1.45
    d.text(px + 12, ly, f"Weight: {top['POSS']:,} possessions", size=fs, fill="ink2")
    ly += fs * 1.7
    ly = para(d, px + 12, ly, "Opponents: not in the season totals, so not adjusted for",
              colw - 24, small, fill="c8", italic=True) + 22
    lineup_bottom = ly
    panel_a = (px, py, lineup_bottom - py)

    # ---- stint model panel
    if narrow:
        px, py = pad, lineup_bottom + 18
    else:
        px, py = pad + colw + pad, y0
    sy = py + 28
    d.text(px + 12, sy, "Stint model", size=head, display=True, weight="700", fill="nba")
    sy = para(d, px + 12, sy + 22, "One row per stint: no substitution by either team",
              colw - 24, small, fill="ink2") + 30
    d.text(px + 12, sy, f"{away_team} at {home_team}, {WORD[s['period']]} quarter, "
           f"{clock(s['start_secs'])} to {clock(s['end_secs'])}", size=fs if not narrow else small,
           fill="ink", weight="700")
    sy += fs * 1.6
    half = (colw - 24) / 2
    for k, (tm, ids, sign) in enumerate(((home_team, home_ids, "+1"),
                                         (away_team, away_ids, "−1"))):
        cx = px + 12 + k * half
        d.text(cx, sy, f"{tm} ({'home' if k == 0 else 'away'})", size=small, fill="ink2")
        for j, i in enumerate(ids):
            yy = sy + small * 1.5 + j * fs * 1.45
            d.text(cx + 2, yy, sign, size=fs, fill="nba" if k == 0 else "c8", weight="700")
            d.text(cx + 28, yy, last_name(names[i]), size=fs, fill="ink")
    sy += small * 1.5 + 5 * fs * 1.45 + fs * 0.6
    d.text(px + 14, sy, "1", size=fs, fill="ink", weight="700")
    d.text(px + 40, sy, "intercept: home-court advantage", size=fs, fill="ink2")
    sy += fs * 2
    sy = para(d, px + 12, sy, f"Outcome: {home_team} {int(s['home_pts'])}, {away_team} "
              f"{int(s['away_pts'])} over {s['poss']:.1f} possessions = "
              f"{sgn(y_stint)} per 100", colw - 24, fs, fill="ink") + fs * 1.45
    d.text(px + 12, sy, f"Weight: {s['poss']:.1f} possessions", size=fs, fill="ink2")
    sy += fs * 1.7
    sy = para(d, px + 12, sy, "Opponents: on the same row, so every player is adjusted "
              "for the other nine", colw - 24, small, fill="c3", italic=True) + 22
    panel_b = (px, py, sy - py)
    if not narrow:
        h = max(panel_a[2], panel_b[2])
        panel_a, panel_b = (panel_a[0], panel_a[1], h), (panel_b[0], panel_b[1], h)
    # panels drawn under the text: prepend their rects
    rects = []
    for (x0, y_, h) in (panel_a, panel_b):
        rects.append(f'<rect x="{x0:g}" y="{y_:g}" width="{colw:g}" height="{h:g}" '
                     f'rx="8" stroke-width="1.5" class="dg-f-surface2 dg-s-rule"/>')
    d.parts[0:0] = rects
    yb = max(panel_a[1] + panel_a[2], panel_b[1] + panel_b[2]) + 30
    yb = para(d, pad, yb, "Both models stack their rows and fit a possession-weighted ridge "
              "regression, with the penalty chosen by cross-validation. Each player's "
              "coefficient is his net points per 100 possessions with the other players "
              "on the floor accounted for.", W - 2 * pad, fs, fill="ink2") + fs * 1.7
    yb = para(d, pad, yb, f"Rows in 2023-24: {scale['model_rows']:,} lineups; "
              f"{sscale['stints_modeled']:,} stints, after slivers such as a lineup on "
              f"the floor for one free throw were merged into the next stint "
              f"({sscale['stints_parsed']:,} parsed).", W - 2 * pad, small, fill="ink3")
    d.h = yb + 14
    return d


# ---------------------------------------------------------------- dg2
def two_builds(narrow: bool) -> fd.Diagram:
    scale = pl.read_csv(OUT / "scale.csv").row(0, named=True)
    val = {r["check"]: r for r in pl.read_csv(OUT / "validation.csv").iter_rows(named=True)}
    sscale = pl.read_csv(OUT / "stint_scale.csv").row(0, named=True)
    box = pl.read_csv(OUT / "stint_box_check.csv").row(0, named=True)
    sval = {r["check"]: r for r in pl.read_csv(OUT / "stint_validation.csv").iter_rows(named=True)}
    n_lin = pl.read_csv(OUT / "player_values.csv").height
    n_st = pl.read_csv(OUT / "stint_player_values.csv").height
    n_both = pl.read_csv(OUT / "stint_vs_lineup.csv").height
    rho = sval["Spearman vs lineup-aggregate model"]["value"]
    hca = sval["home-court advantage in plausible range (pts/100)"]["value"]
    pm_gap = val["lineup plus-minus reconstructs team plus-minus (max abs pts)"]["value"]
    assert all(r["pass"] for r in val.values()) and all(r["pass"] for r in sval.values())

    left = [("Source", "the NBA's season lineup totals (stats.nba.com)"),
            ("Unit", f"{scale['lineup_rows']:,} five-man lineups"),
            ("Audit", f"lineup plus-minus re-adds to every team's season total "
                      f"(largest miss: {pm_gap:.0f} points)"),
            ("Model", "ridge on each lineup's five players"),
            ("Result", f"{n_lin} player values")]
    right = [("Source", "the season's play-by-play, event by event"),
             ("Unit", f"{sscale['stints_parsed']:,} stints from "
                      f"{sscale['games']:,} games"),
             ("Audit", f"plus-minus matches the NBA's own feed in "
                       f"{box['pm_exact_share']:.1%} of player-games"),
             ("Model", "ridge on both fives, with opponents"),
             ("Result", f"{n_st} player values, and home court worth "
                        f"{sgn(hca)} per 100 as a byproduct")]

    W = 360 if narrow else 760
    fs, small, head = (13, 12, 17) if narrow else (15, 14, 18)
    pad = 12
    d = fd.Diagram(W, 10, "Two independent builds of player value and where they meet")
    gap = 40
    colw = (W - 2 * pad) if narrow else (W - 2 * pad - gap) / 2

    def column(x0: float, y: float, title: str, steps) -> tuple[float, list[float]]:
        d.text(x0, y, title, size=head, display=True, weight="700", fill="nba")
        y += 16
        mids = []
        for k, (lab, txt) in enumerate(steps):
            lines = wrap(txt, colw - 24, fs)
            bh = 30 + len(lines) * fs * 1.35
            fill = "surface2"
            d.rect(x0, y, colw, bh, fill=fill, stroke="c3" if lab == "Audit" else "rule",
                   rx=6, sw=1.5 if lab == "Audit" else 1)
            d.text(x0 + 12, y + 20, lab.upper(), size=small,
                   fill="c3" if lab == "Audit" else "ink3", weight="700")
            para(d, x0 + 12, y + 20 + fs * 1.4, txt, colw - 24, fs, fill="ink",
                 weight="700" if lab == "Result" else None)
            mids.append(y + bh)
            y += bh
            if k < len(steps) - 1:
                d.line(x0 + colw / 2, y + 2, x0 + colw / 2, y + 16, stroke="ink3", arrow=True)
                y += 18
        return y, mids

    y = 26
    if narrow:
        yl, _ = column(pad, y, "Lineup model", left)
        yr, _ = column(pad, yl + 40, "Stint model", right)
        bottom = yr
        joins = [(pad + colw / 2, yr)]
    else:
        yl, _ = column(pad, y, "Lineup model", left)
        yr, _ = column(pad + colw + gap, y, "Stint model", right)
        bottom = max(yl, yr)
        joins = [(pad + colw / 2, yl), (pad + colw + gap + colw / 2, yr)]
    my = bottom + 40
    for jx, jy in joins:
        d.line(jx, jy + 2, jx if narrow else W / 2 + (-30 if jx < W / 2 else 30), my - 4,
               stroke="ink3", arrow=True)
    mw = (W - 2 * pad) if narrow else 460
    mx = (W - mw) / 2
    msg = (f"{n_both} players are in both. Ranked by each model, their orders agree "
           f"with a Spearman correlation of {rho:.2f}.")
    lines = wrap(msg, mw - 28, fs)
    mh = 34 + len(lines) * fs * 1.4
    d.rect(mx, my, mw, mh, fill="surface2", stroke="nba", rx=8, sw=2)
    d.text(mx + 14, my + 24, "WHERE THEY MEET", size=small, fill="nba", weight="700")
    para(d, mx + 14, my + 24 + fs * 1.45, msg, mw - 28, fs, fill="ink")
    yb = my + mh + 30
    yb = para(d, pad, yb, "No code path and no data feed is shared between the two "
              "columns, so their agreement is evidence about the players, not the "
              "plumbing.", W - 2 * pad, small, fill="ink2")
    d.h = yb + 14
    return d


# ---------------------------------------------------------------- dg3
def tiers(narrow: bool) -> fd.Diagram:
    v = pl.read_csv(OUT / "player_values.csv").sort("rank")
    n = v.height
    above = v.filter(pl.col("ci_lo") > 0).height
    below = v.filter(pl.col("ci_hi") < 0).height
    middle = n - above - below
    ref = v.filter(pl.col("rank") == REF_RANK).row(0, named=True)
    inside = v.filter(pl.col("rapm_100").is_between(ref["ci_lo"], ref["ci_hi"]))
    lo_rank, hi_rank = inside["rank"].min(), inside["rank"].max()
    assert inside.height == hi_rank - lo_rank + 1   # a contiguous run of ranks
    assert lo_rank <= OTHER_RANK <= hi_rank
    other = v.filter(pl.col("rank") == OTHER_RANK).row(0, named=True)
    show = v.filter(pl.col("rank") <= SHOW_RANKS)

    W = 360 if narrow else 760
    fs, small, head = (13, 12, 17) if narrow else (15, 14, 18)
    pad = 12
    d = fd.Diagram(W, 10, "What one season's intervals can and cannot separate")
    y = 28
    d.text(pad, y, "1. Three tiers", size=head, display=True, weight="600")
    y = para(d, pad, y + 20, f"All {n} players with {MIN_POSS}+ possessions in 2023-24, "
             "sorted by where their 95% interval sits against zero (league average)",
             W - 2 * pad, small, fill="ink2") + 18
    bw = W - 2 * pad
    bh = 34
    parts = [(above, "nba", 0.9, "clearly above average"),
             (middle, "ink3", 0.35, "cannot tell from average"),
             (below, "c8", 0.75, "clearly below average")]
    x = pad
    for cnt, col, op, lab in parts:
        w = bw * cnt / n
        d.rect(x, y, w, bh, fill=col, opacity=op)
        x += w
    d.rect(pad, y, bw, bh, fill="none", stroke="rule", sw=1)
    ly = y + bh + 24
    if narrow:
        for cnt, col, op, lab in parts:
            d.rect(pad, ly - 11, 12, 12, fill=col, opacity=op, rx=2)
            d.text(pad + 18, ly, f"{cnt}", size=fs, fill="ink", weight="700")
            d.text(pad + 54, ly, lab, size=fs, fill="ink2")
            ly += fs * 1.6
    else:
        d.text(pad, ly, f"{above} {parts[0][3]}", size=fs, fill="nba", weight="700")
        d.text(pad + bw / 2, ly, f"{middle} {parts[1][3]}", size=fs, fill="ink",
               anchor="middle", weight="700")
        d.text(pad + bw, ly, f"{below} {parts[2][3]}", size=fs, fill="c8", anchor="end",
               weight="700")
        ly += fs * 1.6
    y = ly + 30

    d.text(pad, y, f"2. Ranks the {REF_RANK}th-ranked player cannot be separated from",
           size=head, display=True, weight="600") if not narrow else para(
        d, pad, y, f"2. Ranks the {REF_RANK}th-ranked player cannot be separated from",
        W - 2 * pad, head, display=True, weight="600")
    if narrow:
        y += head * 1.35
    y = para(d, pad, y + 20, f"Each dot is one of the top {SHOW_RANKS} players' estimates. "
             f"The band is {ref['player']}'s 95% interval "
             f"({sgn(ref['ci_lo'])} to {sgn(ref['ci_hi'])} per 100).",
             W - 2 * pad, small, fill="ink2") + 20
    ph = 150 if not narrow else 130
    left_pad = 40 if not narrow else 34
    px0, px1 = pad + left_pad, W - pad - 6
    vmax = max(show["rapm_100"].max(), ref["ci_hi"]) + 0.3
    vmin = min(0.0, show["rapm_100"].min()) - 0.3

    def vy(val: float) -> float:
        return y + (vmax - val) / (vmax - vmin) * ph

    def rx(rank: int) -> float:
        return px0 + (rank - 1) / (SHOW_RANKS - 1) * (px1 - px0)

    d.rect(px0 - 6, vy(ref["ci_hi"]), px1 - px0 + 12, vy(ref["ci_lo"]) - vy(ref["ci_hi"]),
           fill="nba", opacity=0.14)
    d.line(px0 - 6, vy(0), px1 + 6, vy(0), stroke="rule", sw=1)
    d.text(px0 - 10, vy(0) + 5, "0", size=small, fill="ink3", anchor="end")
    for val in (ref["ci_hi"], ref["ci_lo"]):
        d.line(px0 - 6, vy(val), px1 + 6, vy(val), stroke="nba", sw=1, dash="4 3")
    for r in show.iter_rows(named=True):
        inb = lo_rank <= r["rank"] <= hi_rank
        d.circle(rx(r["rank"]), vy(r["rapm_100"]), 3.6 if not narrow else 2.6,
                 fill="nba" if inb else "ink3", opacity=None if inb else 0.6)
    for r, anchor in ((ref, "start"), (other, "start")):
        cx, cy = rx(r["rank"]), vy(r["rapm_100"])
        d.circle(cx, cy, 6 if not narrow else 4.5, fill="none", stroke="ink", sw=1.5)
    # labels above the band for the two marked players
    lab_y = vy(ref["ci_hi"]) - 10
    d.text(rx(ref["rank"]) - 4, lab_y, f"#{ref['rank']} {last_name(ref['player'])}",
           size=small, fill="ink", weight="700")
    d.text(rx(other["rank"]), vy(other["rapm_100"]) + (26 if not narrow else 22),
           f"#{other['rank']} {last_name(other['player'])}", size=small, fill="ink",
           weight="700", anchor="middle")
    xa = (rx(lo_rank) + rx(hi_rank)) / 2
    by = y + ph + 22
    d.line(rx(lo_rank), by - 8, rx(lo_rank), by, stroke="nba", sw=1.5)
    d.line(rx(hi_rank), by - 8, rx(hi_rank), by, stroke="nba", sw=1.5)
    d.line(rx(lo_rank), by, rx(hi_rank), by, stroke="nba", sw=1.5)
    d.text(xa, by + 20, f"ranks {lo_rank} to {hi_rank}: inside the band",
           size=small, fill="nba", anchor="middle", weight="700")
    d.text(rx(SHOW_RANKS), by + 20 + (0 if not narrow else small * 1.5),
           f"rank {SHOW_RANKS}", size=small, fill="ink3", anchor="end")
    yb = by + 20 + (small * 1.5 if narrow else 0) + 34
    yb = para(d, pad, yb, "Trust the tier, not the rank: the data place these players near "
              "the top, but cannot order them within it.", W - 2 * pad, fs, fill="ink2")
    d.h = yb + 14
    return d


DIAGRAMS = {"dg1_two_rows": two_rows, "dg2_two_builds": two_builds, "dg3_tiers": tiers}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        fails = []
        with tempfile.TemporaryDirectory() as tmp:
            for ref, fn in DIAGRAMS.items():
                for p in fd.save(Path(tmp), ref, fn(False), fn(True)):
                    old = FIG / p.name
                    if not old.exists() or old.read_text() != p.read_text():
                        fails.append(p.name)
        for f in fails:
            print("FAIL", f, "differs from a fresh build")
        print("DIAGRAMS", "FAILED" if fails else
              f"PASSED: {len(DIAGRAMS)} diagrams rebuild identically")
        return 1 if fails else 0
    for ref, fn in DIAGRAMS.items():
        for p in fd.save(FIG, ref, fn(False), fn(True)):
            print("wrote", p.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
