"""Consolidated arm table for the objective hill-climb close-out report (run from scripts/della/closeout:
    ../../../.venv/bin/python objective_arms_table.py [out.json]). Every run enters through objective_compare.collect (finished at
the final step, shared budget); statistics through objective_compare.welch / pval / tcrit. Writes arms.json."""
import json, sys
import numpy as np
import wandb
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from objective_compare import MACRO, C4_BPB, collect, dom_key, series, welch, pval, tcrit

api = wandb.Api(timeout=90)
P130 = "reself/marin-della/muonh-qwen3-130m-della4xh100"
P300 = "reself/marin-della/muonh-qwen3-300m-della4xh100"
POOL130 = ["", "-s1", "-s2", "-s3", "-ema0.999", "-ema0.999-s1", "-ema0.999-s2", "-ema0.999-s3"]
POOL300 = ["", "-s1", "-s2", "-s3"]
S4 = ["", "-s1", "-s2", "-s3"]
S8 = S4 + ["-s4", "-s5", "-s6", "-s7"]
CODE, RP, DOL = dom_key("dolma_100_programing_languages"), dom_key("redpajama"), dom_key("dolma-v1_5")
PROSE = [dom_key(d) for d in ("c4_en", "falcon-refinedweb", "ptb", "wikitext_103", "m2d2_wikipedia_unsplit")]
# (family, label, base suffix, seed suffixes, note); fixbnd seeds carry the tag AFTER the seed
ARMS = [
    ("twin", "twin w0.1 (free head)", "-twin0.1-fh", [""], "backward-model state matching"),
    ("sr", "sr w0.1 (free head)", "-sr0.9w0.1-fh", [""], "successor-representation TD"),
    ("sr", "sr w0.03", "-sr0.9w0.03-fh", [""], ""),
    ("sr", "sr w0.1 gated 4.0:3.6", "-sr0.9w0.1-g4-3.6-fh", S4, "control arm of the ladder"),
    ("pi", "pi k4 w0.1", "-pik4w0.1-fh", [""], "predictive-information InfoNCE"),
    ("pi", "pi k4 w0.03", "-pik4w0.03-fh", [""], ""),
    ("pi", "pi k4 w0.1 at L3", "-pik4w0.1-L3-fh", [""], ""),
    ("eos", "eos w0.1", "-eosw0.1-fh", [""], "distance to document end"),
    ("eos", "eos w0.03", "-eosw0.03-fh", [""], ""),
    ("eos", "eos w0.1 gated", "-eosw0.1-g4-3.6-fh", [""], ""),
    ("mtp", "mtp k2 w0.1", "-mtpk2w0.1-fh", [""], "multi-token prediction"),
    ("mtp", "mtp k2 w0.1 gated", "-mtpk2w0.1-g4-3.6-fh", [""], ""),
    ("dn", "dn w0.1 gated", "-dnr0.5w0.1-g4-3.6-fh", S4, "denoising NTP on own-sample-corrupted input"),
    ("dn", "dn w0.3 gated", "-dnr0.5w0.3-g4-3.6-fh", [""], ""),
    ("ebm", "ebm w0.1 ungated", "-ebmr0.5w0.1-fh", [""], "first ebm run"),
    ("ebm", "ebm w0.03 ungated", "-ebmr0.5w0.03-fh", S4, "ladder: discriminator without gate"),
    ("ebm", "ebm w0.03 gated (pre-fix code)", "-ebmr0.5w0.03-g4-3.6-fh", S8, "the standing positive before the bug fix"),
    ("ebm", "ebm w0.03 gated (FIXED code)", "-ebmr0.5w0.03-g4-3.6-fh", ["-fixbnd", "-s1-fixbnd", "-s2-fixbnd", "-s3-fixbnd"], "document-boundary bug fixed"),
    ("ebm", "ebm w0.015 gated", "-ebmr0.5w0.015-g4-3.6-fh", S4, "dose-response"),
    ("ebm", "ebm w0.05 gated", "-ebmr0.5w0.05-g4-3.6-fh", [""], "dose-response"),
    ("ebm", "ebm w0.1 gated", "-ebmr0.5w0.1-g4-3.6-fh", [""], "dose-response"),
    ("ebm", "ebm w0.03 T2 gated", "-ebmr0.5w0.03T2-g4-3.6-fh", S4, "sampling temperature 2"),
    ("ebm", "ebm rho0.8 gated", "-ebmr0.8w0.03-g4-3.6-fh", [""], "corruption rate"),
    ("ebm", "ebm rho1.0 gated", "-ebmr1w0.03-g4-3.6-fh", [""], "corruption rate"),
    ("ebm", "ebm w0.03 at L3", "-ebmr0.5w0.03-L3-fh", [""], "readout placement"),
    ("ebm", "ebm w0.03 gate 4.4:3.9", "-ebmr0.5w0.03-g4.4-3.9-fh", [""], "gate schedule"),
    ("ebm", "ebm w0.03 gate 3.8:3.4", "-ebmr0.5w0.03-g3.8-3.4-fh", [""], "gate schedule"),
    ("ebm", "ebm + dn gated", "-ebmr0.5w0.03-dn0.1-g4-3.6-fh", [""], "stacking"),
    ("swap", "swap w0.03 gated", "-sww0.03-g4-3.6-fh", [""], "real text, wrong context"),
    ("adv", "adv w0.03 gated", "-ebmr0.5w0.03-adv0.03-g4-3.6-fh", [""], "discriminator as REINFORCE reward"),
    ("adv", "adv w0.03 ungated", "-ebmr0.5w0.03-adv0.03-fh", [""], ""),
]


def stats(cell, pool, key):
    c, p = series(cell, key), series(pool, key)
    if len(c) == 0 or len(p) < 2:
        return None
    d, se, t, df = welch(c, p)
    return dict(d=float(d), se=float(se), t=float(t), df=float(df), p=float(pval(t, df)), lo=float(d - tcrit(df) * se), hi=float(d + tcrit(df) * se), sd_pool=float(p.std(ddof=1)))


def arm_row(prefix, pool, fam, label, base, seeds, note, size):
    cell, _ = collect(api, prefix, [base + s if not s.endswith("fixbnd") else base + s for s in seeds], label, verbose=False)
    if not cell:
        return dict(family=fam, label=label, n=0, note=note, size=size, missing=True)
    row = dict(family=fam, label=label, size=size, n=len(cell), note=note,
               run_ids=[(prefix.split("/")[-1] + base + s) for s in seeds],
               macro_abs=[float(x) for x in series(cell, MACRO)])
    for nm, k in [("macro", MACRO), ("c4_bpb", C4_BPB), ("code", CODE), ("redpajama", RP), ("dolma", DOL)]:
        row[nm] = stats(cell, pool, k)
    row["prose_worse_2sd"] = int(sum(1 for k in PROSE if (s := stats(cell, pool, k)) and s["d"] > 2 * s["sd_pool"]))
    row["prose_better_2sd"] = int(sum(1 for k in PROSE if (s := stats(cell, pool, k)) and s["d"] < -2 * s["sd_pool"]))
    return row


pool130, b = collect(api, P130, POOL130, "pool130", verbose=True)
pool300, b3 = collect(api, P300, POOL300, "pool300", verbose=True)
out = dict(pool130=dict(n=len(pool130), macro=[float(x) for x in series(pool130, MACRO)], budget=sorted(b)),
           pool300=dict(n=len(pool300), macro=[float(x) for x in series(pool300, MACRO)], budget=sorted(b3)), arms=[])
for fam, label, base, seeds, note in ARMS:
    out["arms"].append(arm_row(P130, pool130, fam, label, base, seeds, note, "130m"))
out["arms"].append(arm_row(P300, pool300, "ebm", "ebm w0.03 gated 3.65:3.23 (300m escalation)", "-ebmr0.5w0.03-g3.65-3.23-fh", S4, "pre-registered bar: code <= -0.05 and macro < 0", "300m"))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../docs/della/objective-hillclimb-arms.json")
json.dump(out, open(OUT, "w"), indent=1)
pm = np.array(out["pool130"]["macro"]); print(f"pool130 n={len(pm)} macro mean {pm.mean():.5f} sd {pm.std(ddof=1):.5f} budget {out['pool130']['budget']}")
pm3 = np.array(out["pool300"]["macro"]); print(f"pool300 n={len(pm3)} macro mean {pm3.mean():.5f} sd {pm3.std(ddof=1):.5f} budget {out['pool300']['budget']}")
print(f"{'arm':44s} {'n':>2s} {'macro d':>8s} {'t':>6s} {'p':>6s} | {'code d':>7s} {'t':>6s} | {'rp d':>7s} {'t':>6s} | {'c4bpb d':>8s} {'t':>6s} | prose+/-")
for r in out["arms"]:
    if r.get("missing"):
        print(f"{r['label']:44s} MISSING"); continue
    m, c, rp, c4 = r["macro"], r["code"], r["redpajama"], r["c4_bpb"]
    print(f"{r['label'][:44]:44s} {r['n']:2d} {m['d']:+8.5f} {m['t']:+6.2f} {m['p']:6.3f} | {c['d']:+7.4f} {c['t']:+6.2f} | {rp['d']:+7.4f} {rp['t']:+6.2f} | {c4['d']:+8.5f} {c4['t']:+6.2f} | {r['prose_better_2sd']}/{r['prose_worse_2sd']}")
