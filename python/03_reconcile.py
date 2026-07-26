"""Verify the R and Python valuation implementations agree, per season.

Point estimates, the CV curve, the lineup table, and the validation checks
must match to tight numeric tolerance; bootstrap CIs use each language's RNG
and are compared for overlap on every player instead.

Run:  python python/03_reconcile.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"

TOL_EST = 1e-6   # ridge solutions via different linear-algebra backends
TOL_TAB = 1e-9   # raw table arithmetic


def check_season(season: str) -> bool:
    out = OUT / season
    ok = True
    print(f"=== {season} ===")

    py = pl.read_csv(out / "player_values.csv")
    r = pl.read_csv(out / "player_values_r.csv")
    m = py.join(r, on="player_id", how="full", suffix="_r", coalesce=True)
    complete = m.height == py.height == r.height
    est_diff = (m["rapm_100"] - m["rapm_100_r"]).abs().max()
    poss_diff = (m["poss"] - m["poss_r"]).abs().max()
    overlap = ((m["ci_lo"] <= m["ci_hi_r"]) & (m["ci_lo_r"] <= m["ci_hi"])).all()
    good = complete and est_diff <= TOL_EST and poss_diff <= TOL_TAB and overlap
    ok &= good
    print(f"  {'PASS' if good else 'FAIL'}  player_values: {m.height} players, "
          f"max est diff {est_diff:.2e}, CI overlap {'all' if overlap else 'NOT all'}")

    cpy = pl.read_csv(out / "cv_curve.csv")
    cr = pl.read_csv(out / "cv_curve_r.csv")
    cm = cpy.join(cr, on="lam", suffix="_r")
    cv_diff = (cm["cv_wmse"] - cm["cv_wmse_r"]).abs().max()
    good = cm.height == cpy.height == cr.height and cv_diff <= TOL_EST
    ok &= good
    print(f"  {'PASS' if good else 'FAIL'}  cv_curve: max diff {cv_diff:.2e}")

    lpy = pl.read_csv(out / "lineup_table.csv")
    lr = pl.read_csv(out / "lineup_table_r.csv")
    lm = lpy.join(lr, on=["TEAM_ABBREVIATION", "GROUP_NAME"], suffix="_r")
    ln_diff = (lm["net_100"] - lm["net_100_r"]).abs().max()
    good = lm.height == lpy.height == lr.height and ln_diff <= TOL_TAB
    ok &= good
    print(f"  {'PASS' if good else 'FAIL'}  lineup_table: {lm.height} lineups, "
          f"max diff {ln_diff:.2e}")

    vpy = pl.read_csv(out / "validation.csv")
    vr = pl.read_csv(out / "validation_r.csv")
    vm = vpy.join(vr, on="check", suffix="_r")
    v_diff = (vm["value"] - vm["value_r"]).abs().max()
    good = vm.height == 4 and v_diff <= TOL_EST and vm["pass"].all() and vm["pass_r"].all()
    ok &= good
    print(f"  {'PASS' if good else 'FAIL'}  validation: max diff {v_diff:.2e}, all pass")

    return bool(ok)


def main() -> int:
    seasons = sorted(p.name for p in OUT.iterdir() if p.is_dir())
    all_ok = all([check_season(s) for s in seasons])
    print("\n" + ("ALL CHECKS PASS" if all_ok else "MISMATCHES FOUND"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
