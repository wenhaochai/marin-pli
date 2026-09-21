"""Compare a candidate cell against a baseline pool on Paloma, with mandatory correctness guards.

    .venv/bin/python scripts/della/objective_compare.py --pool=<suffix>[,...] --cell=<suffix>[,...] [--cell2=...]

Use the ``--flag=value`` form: suffixes start with a hyphen and argparse would otherwise read them as options.

Every guard here exists because the corresponding mistake was actually made in this project:

* A run still training reports a mid-flight summary far above its final values, so runs are admitted only at
  ``trainer.num_train_steps - 1`` AND only when the run state is finished.
* Pool and cell must share a training budget; a run from a different schedule would pass a per-run step check
  while contributing a loss from a different experiment entirely.
* Three of the sixteen Paloma domains are logged under unexpected names, so the domain list is discovered from
  the runs rather than hardcoded, taking the union over the pool and checking completeness on BOTH arms.
* ``d/se`` is a Welch t statistic but only means something with its degrees of freedom, so df, p and a
  t-based interval are printed rather than a normal-style ``2*se``.
* Sixteen domain tests per invocation produce about one false positive under the null, so the Bonferroni bar is
  printed and the table is ordered by |t| rather than by raw delta (which favours loss-scale-heavy domains).
* A repeated or mistyped suffix silently duplicates or drops a run, so suffixes are stripped, de-duplicated, and
  the two arms are checked for overlap.
* Welch is undefined for n=1; a single-run cell is compared with the one-versus-sample se, sd*sqrt(1+1/n).
"""
import argparse
import re

import numpy as np
import wandb

try:
    from scipy import stats as _st
except Exception:  # scipy is optional; without it we print df but not p
    _st = None

MACRO = "eval/paloma/macro_loss"
MACRO_BPB = "eval/macro_bpb"  # NOTE: not in the eval/paloma namespace; may not be the same 16-domain macro
C4_BPB = "eval/paloma/c4_en-marin-tokenizer/bpb"
DOM_RE = re.compile(r"^eval/paloma/(.+)-marin-tokenizer/loss$")


def dom_key(d: str) -> str:
    return f"eval/paloma/{d}-marin-tokenizer/loss"


def discover_domains(summaries) -> list:
    """Union of Paloma domains over every summary, so one run missing a domain cannot hide it."""
    out: set = set()
    for s in summaries:
        out |= {m.group(1) for k in s.keys() if (m := DOM_RE.match(k))}
    return sorted(out)


def collect(api, prefix, suffixes, label, verbose=True):
    """Return (summaries, budgets) for runs that genuinely finished their configured schedule."""
    kept, budgets, dropped = [], set(), []
    for sfx in suffixes:
        rid = prefix + sfx
        try:
            run = api.run(rid)
        except Exception as e:  # distinguish a typo from a transport failure
            dropped.append((sfx, f"unreachable ({type(e).__name__})")); continue
        step = run.summary.get("_step")
        want = run.config.get("trainer", {}).get("num_train_steps")
        if not isinstance(want, int):
            dropped.append((sfx, f"no int trainer.num_train_steps (got {want!r}) -- refusing to guess")); continue
        if step != want - 1:
            dropped.append((sfx, f"_step={step} of {want} ({run.state})")); continue
        if run.state != "finished":
            dropped.append((sfx, f"reached final step but state={run.state}; summary may be stale")); continue
        kept.append(run.summary); budgets.add(want)
    if verbose:
        for sfx, why in dropped:
            print(f"  [{label}] DROP '{sfx or '(seed0)'}': {why}")
        print(f"  [{label}] admitted {len(kept)}/{len(suffixes)}")
    return kept, budgets


def welch(a, b):
    """Return (diff, se, t, df). For a single-observation cell use the one-versus-sample se."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    d = a.mean() - b.mean()
    if na < 2:  # one observation against a sample: sd*sqrt(1 + 1/nb), df = nb - 1
        sb = b.std(ddof=1)
        se = sb * np.sqrt(1.0 + 1.0 / nb)
        return d, se, (d / se if se > 0 else np.nan), nb - 1
    va, vb = a.std(ddof=1) ** 2 / na, b.std(ddof=1) ** 2 / nb
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (na - 1) + vb ** 2 / (nb - 1)) if se > 0 else np.nan
    return d, se, (d / se if se > 0 else np.nan), df


def pval(t, df):
    if _st is None or not np.isfinite(t) or not np.isfinite(df):
        return np.nan
    return 2 * _st.t.sf(abs(t), df)


def tcrit(df, q=0.975):
    if _st is None or not np.isfinite(df):
        return 2.0
    return float(_st.t.ppf(q, df))


def series(rows, key):
    return np.array([r[key] for r in rows if r.get(key) is not None], float)


def clean(s):
    """Strip, drop blanks that are not the intentional seed-0 empty suffix, and de-duplicate in order."""
    if not s:
        return []
    out, seen = [], set()
    for x in s.split(","):
        x = x.strip()
        if x in seen:
            print(f"  !! duplicate suffix '{x or '(seed0)'}' ignored (would have counted one run twice)")
            continue
        seen.add(x); out.append(x)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size", default="130m")
    p.add_argument("--device", default="h100")
    p.add_argument("--pool", required=True)
    p.add_argument("--cell", required=True)
    p.add_argument("--cell2", default="")
    p.add_argument("--name", default="cell")
    p.add_argument("--name2", default="control")
    p.add_argument("--domains", default="auto")
    a = p.parse_args()

    api = wandb.Api(timeout=60)
    prefix = f"reself/marin-della/muonh-qwen3-{a.size}-della4x{a.device}"
    sp, sc, s2 = clean(a.pool), clean(a.cell), clean(a.cell2)
    for x in set(sp) & set(sc):
        print(f"  !! '{x or '(seed0)'}' is in BOTH pool and cell; Welch assumes independent arms")

    print("guards: finished-at-final-step, shared training budget, domain union over both arms, Welch df")
    pool, bp = collect(api, prefix, sp, "pool")
    cell, bc = collect(api, prefix, sc, a.name)
    ctrl, b2 = collect(api, prefix, s2, a.name2) if s2 else ([], set())
    budgets = bp | bc | b2
    if len(budgets) > 1:
        print(f"  !! ABORT: arms do not share a training budget: num_train_steps = {sorted(budgets)}"); return
    if len(pool) < 2 or not cell:
        print("not enough finished runs to compare"); return
    if len(cell) < 2:
        print(f"  note: {a.name} has n=1; using the one-versus-sample se sd*sqrt(1+1/n_pool), df=n_pool-1")

    print(f"\n=== {a.name} n={len(cell)} vs pool n={len(pool)}  (budget {sorted(budgets)[0]} steps) ===")
    for nm, k in [("macro_loss", MACRO), ("macro_bpb", MACRO_BPB), ("c4_en bpb", C4_BPB)]:
        d, se, t, df = welch(series(cell, k), series(pool, k))
        lo, hi = d - tcrit(df) * se, d + tcrit(df) * se
        print(f"  {nm:11s} {d:+.5f}  se {se:.5f}  t {t:+.2f}  df {df:.1f}  p {pval(t, df):.4f}  95%CI [{lo:+.5f},{hi:+.5f}]")
    mc, mp = series(cell, MACRO), series(pool, MACRO)
    print(f"  below pool mean: {(mc < mp.mean()).sum()}/{len(mc)}")
    if ctrl:
        d, se, t, df = welch(series(ctrl, MACRO), series(pool, MACRO))
        print(f"\n=== {a.name2} n={len(ctrl)} vs pool ===\n  macro_loss {d:+.5f}  se {se:.5f}  t {t:+.2f}  df {df:.1f}  p {pval(t, df):.4f}")

    doms = discover_domains(pool + cell + ctrl) if a.domains == "auto" else clean(a.domains)
    rows, ntests = [], len(doms) + 3
    bonf = tcrit(max(len(pool) + len(cell) - 2, 1), 1 - 0.05 / (2 * ntests))
    for dm in doms:
        k = dom_key(dm)
        b, c = series(pool, k), series(cell, k)
        if len(b) < len(pool) or len(c) < len(cell):
            print(f"  !! {dm}: pool {len(b)}/{len(pool)}, {a.name} {len(c)}/{len(cell)} -- skipped, incomplete")
            continue
        d, se, t, df = welch(c, b)
        rows.append((dm, b.std(ddof=1), d, t, df))

    print(f"\n{len(rows)} domains, ordered by |t|. {ntests} UNCORRECTED tests per run; Bonferroni bar |t| > {bonf:.2f}")
    hdr = f"{'domain':34s} {'pool sd':>8s} | {a.name[:9]:>9s} {'t':>6s} {'':>3s}"
    if ctrl:
        hdr += f" | {a.name2[:9]:>9s} {'t':>6s} | {'gap/sd':>6s}"
    print(hdr)
    for dm, sd, d, t, df in sorted(rows, key=lambda r: -abs(r[3])):
        mark = "**" if abs(t) > bonf else ("*" if abs(t) > 2 else "")
        line = f"{dm:34s} {sd:8.4f} | {d:+9.4f} {t:6.2f} {mark:>3s}"
        if ctrl:
            dc, _s, tc, _d = welch(series(ctrl, dom_key(dm)), series(pool, dom_key(dm)))
            line += f" | {dc:+9.4f} {tc:6.2f} | {(dc - d) / sd:6.2f}"
        print(line)
    print("  ** clears Bonferroni over all tests in this run;  * nominal |t|>2 only")

    print(f"\n{a.name} macro:", np.round(mc, 5).tolist())
    print("pool  macro:", np.round(mp, 5).tolist())
    if ctrl:
        print(f"{a.name2} macro:", np.round(series(ctrl, MACRO), 5).tolist())


if __name__ == "__main__":
    main()
