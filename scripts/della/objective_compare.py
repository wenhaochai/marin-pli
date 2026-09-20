"""Compare a candidate cell against a baseline pool on Paloma, with a mandatory completion filter.

    .venv/bin/python scripts/della/objective_compare.py --pool=<suffix>[,...] --cell=<suffix>[,...] [--cell2=...]

Use the ``--flag=value`` form: suffixes start with a hyphen and argparse would otherwise read them as options.

Every run is admitted only if it reached its configured last step (``trainer.num_train_steps - 1``). A run that is
still training reports a mid-flight summary whose losses are far above the final ones; mixing one into a cell silently
inverts the comparison. This happened once (2026-09-20, sr-gated control) and the filter exists so it cannot recur.

Suffixes are appended to ``muonh-qwen3-{size}-della4x{device}``; the empty string is the seed-0 run of a family.
"""
import argparse

import numpy as np
import wandb

MACRO = "eval/paloma/macro_loss"
MACRO_BPB = "eval/macro_bpb"
DOMAINS = [
    "dolma_100_programing_languages", "redpajama", "dolma-v1_5", "mc4", "c4_en", "ptb",
    "twitterAAE_HELM_fixed", "gab", "4chan", "manosphere", "m2d2_s2orc", "m2d2_wikipedia",
    "wikitext_103", "falcon-refinedweb", "dolma_100_subreddits", "c4_100_domains",
]


def dom_key(d: str) -> str:
    return f"eval/paloma/{d}-marin-tokenizer/loss"


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
    p.add_argument("--domains", default=",".join(DOMAINS[:8]))
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
    for nm, k in [("macro_loss", MACRO), ("macro_bpb", MACRO_BPB), ("c4_en bpb", "eval/paloma/c4_en-marin-tokenizer/bpb")]:
        d, se, t = welch(series(cell, k), series(pool, k))
        print(f"  {nm:11s} {d:+.5f}  se {se:.5f}  t {t:+.2f}  95%CI [{d - 2 * se:+.5f},{d + 2 * se:+.5f}]")
    below = (series(cell, MACRO) < series(pool, MACRO).mean()).sum()
    print(f"  below pool mean: {below}/{len(cell)}")
    if ctrl:
        d, se, t = welch(series(ctrl, MACRO), series(pool, MACRO))
        print(f"\n=== {a.name2} n={len(ctrl)} vs pool ===\n  macro_loss {d:+.5f}  se {se:.5f}  t {t:+.2f}")

    hdr = f"\n{'domain':34s} {'pool sd':>8s} | {a.name[:9]:>9s} {'t':>6s}"
    if ctrl:
        hdr += f" | {a.name2[:9]:>9s} {'t':>6s} | {'gap/sd':>6s}"
    print(hdr)
    for dm in split(a.domains):
        k = dom_key(dm)
        sd = series(pool, k).std(ddof=1)
        de, see, te = welch(series(cell, k), series(pool, k))
        line = f"{dm:34s} {sd:8.4f} | {de:+9.4f} {te:6.2f}"
        if ctrl:
            dc, sec, tc = welch(series(ctrl, k), series(pool, k))
            line += f" | {dc:+9.4f} {tc:6.2f} | {(dc - de) / sd:6.2f}"
        print(line)

    print(f"\n{a.name} macro:", np.round(series(cell, MACRO), 5).tolist())
    print(f"pool  macro:", np.round(series(pool, MACRO), 5).tolist())
    if ctrl:
        print(f"{a.name2} macro:", np.round(series(ctrl, MACRO), 5).tolist())


if __name__ == "__main__":
    main()
