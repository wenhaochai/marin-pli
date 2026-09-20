"""Compare a candidate cell against a baseline pool on Paloma, with a mandatory completion filter.

    .venv/bin/python scripts/della/objective_compare.py --pool=<suffix>[,...] --cell=<suffix>[,...] [--cell2=...]

Use the ``--flag=value`` form: suffixes start with a hyphen and argparse would otherwise read them as options.

Two silent failure modes this script exists to prevent, both hit on 2026-09-20:

* A run that is still training reports a mid-flight summary whose losses sit far above its final ones. One such run
  in a cell inverts the comparison without any error. Every run is therefore admitted only at its configured last
  step (``trainer.num_train_steps - 1``), and each dropped run is printed with the reason.
* Three of the sixteen Paloma domains are logged under names that do not match their common spelling
  (``m2d2_s2orc_unsplit``, ``m2d2_wikipedia_unsplit``, ``manosphere_meta_sep``). A hardcoded domain list skipped
  them and reported a 13-domain table as if it were complete, so the list is read off the run instead.

Suffixes are appended to ``muonh-qwen3-{size}-della4x{device}``; the empty string is the seed-0 run of a family.
"""
import argparse
import re

import numpy as np
import wandb

MACRO = "eval/paloma/macro_loss"
MACRO_BPB = "eval/macro_bpb"
C4_BPB = "eval/paloma/c4_en-marin-tokenizer/bpb"
DOM_RE = re.compile(r"^eval/paloma/(.+)-marin-tokenizer/loss$")


def dom_key(d: str) -> str:
    return f"eval/paloma/{d}-marin-tokenizer/loss"


def discover_domains(summary) -> list:
    """Read the Paloma domain list off a run summary rather than hardcoding it."""
    return sorted({m.group(1) for k in summary.keys() if (m := DOM_RE.match(k))})


def collect(api, prefix, suffixes, label, verbose=True):
    """Return the summaries of the runs that actually finished; print and drop the rest."""
    kept, dropped = [], []
    for sfx in suffixes:
        rid = prefix + sfx
        try:
            run = api.run(rid)
        except Exception:
            dropped.append((sfx, "missing")); continue
        step = run.summary.get("_step")
        want = run.config.get("trainer", {}).get("num_train_steps")
        want = (want - 1) if isinstance(want, int) else None
        if want is not None and step != want:
            dropped.append((sfx, f"_step={step} of {want} ({run.state})")); continue
        if want is None and run.state != "finished":
            dropped.append((sfx, f"state={run.state}, no num_train_steps")); continue
        kept.append(run.summary)
    if verbose:
        for sfx, why in dropped:
            print(f"  [{label}] DROP '{sfx or '(seed0)'}': {why}")
        print(f"  [{label}] admitted {len(kept)}/{len(suffixes)}")
    return kept


def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a.mean() - b.mean()
    se = np.sqrt(a.std(ddof=1) ** 2 / len(a) + b.std(ddof=1) ** 2 / len(b))
    return d, se, (d / se if se > 0 else np.nan)


def series(rows, key):
    return np.array([r[key] for r in rows if r.get(key) is not None], float)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size", default="130m")
    p.add_argument("--device", default="h100")
    p.add_argument("--pool", required=True, help="comma-separated run-id suffixes of the baseline pool")
    p.add_argument("--cell", required=True, help="comma-separated run-id suffixes of the candidate cell")
    p.add_argument("--cell2", default="", help="optional second cell (e.g. a control arm)")
    p.add_argument("--name", default="cell")
    p.add_argument("--name2", default="control")
    p.add_argument("--domains", default="auto", help="'auto' reads the domain list off the pool runs")
    a = p.parse_args()

    api = wandb.Api(timeout=60)
    prefix = f"reself/marin-della/muonh-qwen3-{a.size}-della4x{a.device}"
    split = lambda s: [x for x in s.split(",")] if s else []

    print("completion filter: a run counts only at its configured final step")
    pool = collect(api, prefix, split(a.pool), "pool")
    cell = collect(api, prefix, split(a.cell), a.name)
    ctrl = collect(api, prefix, split(a.cell2), a.name2) if a.cell2 else []
    if len(pool) < 2 or len(cell) < 2:
        print("not enough finished runs to compare"); return

    print(f"\n=== {a.name} n={len(cell)} vs pool n={len(pool)} ===")
    for nm, k in [("macro_loss", MACRO), ("macro_bpb", MACRO_BPB), ("c4_en bpb", C4_BPB)]:
        d, se, t = welch(series(cell, k), series(pool, k))
        print(f"  {nm:11s} {d:+.5f}  se {se:.5f}  t {t:+.2f}  95%CI [{d - 2 * se:+.5f},{d + 2 * se:+.5f}]")
    print(f"  below pool mean: {(series(cell, MACRO) < series(pool, MACRO).mean()).sum()}/{len(cell)}")
    if ctrl:
        d, se, t = welch(series(ctrl, MACRO), series(pool, MACRO))
        print(f"\n=== {a.name2} n={len(ctrl)} vs pool ===\n  macro_loss {d:+.5f}  se {se:.5f}  t {t:+.2f}")

    doms = discover_domains(pool[0]) if a.domains == "auto" else split(a.domains)
    rows = []
    for dm in doms:
        b = series(pool, dom_key(dm))
        if len(b) < len(pool):
            print(f"  !! {dm}: only {len(b)}/{len(pool)} pool runs report it"); continue
        rows.append((dm, b.std(ddof=1)) + welch(series(cell, dom_key(dm)), b))

    hdr = f"\n{len(rows)} domains, best first\n{'domain':34s} {'pool sd':>8s} | {a.name[:9]:>9s} {'t':>6s}"
    if ctrl:
        hdr += f" | {a.name2[:9]:>9s} {'t':>6s} | {'gap/sd':>6s}"
    print(hdr)
    for dm, sd, de, _see, te in sorted(rows, key=lambda r: r[2]):
        line = f"{dm:34s} {sd:8.4f} | {de:+9.4f} {te:6.2f}"
        if ctrl:
            dc, _sec, tc = welch(series(ctrl, dom_key(dm)), series(pool, dom_key(dm)))
            line += f" | {dc:+9.4f} {tc:6.2f} | {(dc - de) / sd:6.2f}"
        print(line)

    print(f"\n{a.name} macro:", np.round(series(cell, MACRO), 5).tolist())
    print("pool  macro:", np.round(series(pool, MACRO), 5).tolist())
    if ctrl:
        print(f"{a.name2} macro:", np.round(series(ctrl, MACRO), 5).tolist())


if __name__ == "__main__":
    main()
