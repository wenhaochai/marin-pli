"""Share of an arm's gain over baseline that a control removes, f = (ctrl - arm) / (pool - arm), with a 95% CI.

    .venv/bin/python scripts/della/objective_fraction.py --pool=<sfx,...> --arm=<sfx,...> --ctrl=<sfx,...>

Pre-registered for the document-boundary-fix control (ledger ``prereg_for_n4``, 2026-09-22 00:40), BEFORE its
seeds 1-3 finished: judged on code; a 95% CI that excludes f >= 0.5 means the fix removes less than half of the
gain (survived); one that excludes f <= 0.5 means more than half of the gain was the packing splice (collapsed);
otherwise report f with its interval and stop. Redpajama is the confirming axis, macro is descriptive.

Two intervals are printed so the choice of method is not made after seeing the numbers:
* delta method on the ratio of two mean differences that share the arm mean (Cov(N, D) = Var(arm mean)), with a
  t critical value at df = min(n) - 1 (conservative);
* percentile bootstrap resampling RUNS within each of the three arms (10000 draws).
With a single control run neither interval exists (no within-arm variance); the script then prints the point
estimate only and says so rather than borrowing a spread from another arm. Runs are admitted through
objective_compare.collect, so nothing mid-flight or off-budget can enter.
"""
import argparse

import numpy as np
import wandb

from objective_compare import MACRO, clean, collect, dom_key, series, tcrit

AXES = [("code", dom_key("dolma_100_programing_languages")), ("redpajama", dom_key("redpajama")), ("macro", MACRO)]


def fraction(ctrl, arm, pool, B=10000, seed=0):
    ctrl, arm, pool = (np.asarray(x, float) for x in (ctrl, arm, pool))
    mc, ma, mp = ctrl.mean(), arm.mean(), pool.mean()
    N, D = mc - ma, mp - ma
    f = N / D
    if min(len(ctrl), len(arm), len(pool)) < 2:
        return f, (np.nan, np.nan), (np.nan, np.nan), "n=1 somewhere: point estimate only, no interval"
    vc, va, vp = (x.var(ddof=1) / len(x) for x in (ctrl, arm, pool))
    var_f = (vc + va - 2 * f * va + f * f * (vp + va)) / (D * D)  # delta method; N and D share -ma
    t = tcrit(min(len(ctrl), len(arm), len(pool)) - 1)
    delta_ci = (f - t * np.sqrt(var_f), f + t * np.sqrt(var_f))
    rng = np.random.default_rng(seed)
    fb = np.empty(B)
    for b in range(B):
        c = rng.choice(ctrl, len(ctrl)); a = rng.choice(arm, len(arm)); p = rng.choice(pool, len(pool))
        d = p.mean() - a.mean()
        fb[b] = (c.mean() - a.mean()) / d if d != 0 else np.nan
    boot_ci = tuple(np.nanpercentile(fb, [2.5, 97.5]))
    return f, delta_ci, boot_ci, ""


def verdict(lo, hi):
    if not np.isfinite(lo):
        return "no interval at n=1 -> no verdict"
    if hi < 0.5:
        return "SURVIVED  (CI excludes f >= 0.5: the fix removes less than half the gain)"
    if lo > 0.5:
        return "COLLAPSED (CI excludes f <= 0.5: more than half the gain was the splice)"
    return "REPORT f  (CI straddles 0.5; per the pre-registration, report and stop)"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size", default="130m"); p.add_argument("--device", default="h100")
    p.add_argument("--pool", required=True); p.add_argument("--arm", required=True); p.add_argument("--ctrl", required=True)
    a = p.parse_args()
    api = wandb.Api(timeout=60)
    prefix = f"reself/marin-della/muonh-qwen3-{a.size}-della4x{a.device}"
    pool, bp = collect(api, prefix, clean(a.pool), "pool"); arm, ba = collect(api, prefix, clean(a.arm), "arm"); ctrl, bc = collect(api, prefix, clean(a.ctrl), "ctrl")
    if len(bp | ba | bc) > 1:
        print(f"  !! ABORT: arms do not share a training budget: {sorted(bp | ba | bc)}"); return
    print(f"\nf = (ctrl - arm) / (pool - arm)   ctrl n={len(ctrl)}  arm n={len(arm)}  pool n={len(pool)}")
    for name, k in AXES:
        c, r, q = series(ctrl, k), series(arm, k), series(pool, k)
        f, (dl, dh), (bl, bh), note = fraction(c, r, q)
        print(f"  {name:9s} f {f:+.3f}   delta 95% [{dl:+.3f},{dh:+.3f}]   bootstrap 95% [{bl:+.3f},{bh:+.3f}]   "
              f"(ctrl {c.mean():.5f}  arm {r.mean():.5f}  pool {q.mean():.5f})  {note}")
        if name == "code":
            print(f"  {'':9s} -> {verdict(dl, dh)}  [delta]   |   {verdict(bl, bh)}  [bootstrap]")
    print("\nVerdict axis is code; the two intervals must agree on which side of 0.5 they exclude for a call to be made.")


if __name__ == "__main__":
    main()
