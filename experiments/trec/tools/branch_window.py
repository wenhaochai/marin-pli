"""Verification 2026-11-01. Effect of the scored-window offset on the Figure 10 numbers: checkpoint step-k already holds the update of batch k,
so the batches read between the step-(b-248) and step-b checkpoints are b-247..b, and b is not scored. Compares the
page's window (b-248..b-1, 248 batches) with b-247..b-1 (247 batches) for every branch point and both momenta."""
import sys
import numpy as np
sys.path.insert(0, "/scratch/gpfs/GROUP/USER/project/marin-trec/experiments/trec")
import build_site_data as b

orig = b._range_score


def shifted(d, bp=2480):
    import glob, os
    files = sorted(glob.glob(os.path.join(d, "[0-9]*.npz")))
    if not files:
        return None
    idx = np.concatenate([np.load(f)["index"] for f in files])
    loss = np.concatenate([np.load(f)["seq_loss"].sum(1) / np.load(f)["seq_weight"].sum(1) for f in files])
    tr = (idx > bp - 248) & (idx < bp)
    if tr.sum() != 247 or (idx >= 4959).sum() != 256:
        return None
    return loss[idx >= 4959].mean(), loss[tr][np.argsort(idx[tr])]


def spread(arms, p):
    cs = [(np.log(np.maximum(a[1:, 0], 1) * lr ** p), np.log(a[1:, 1])) for lr, a in arms]
    lo, hi = max(c[0].min() for c in cs), min(c[0].max() for c in cs)
    if hi <= lo:
        return float("nan")
    g = np.linspace(lo, hi, 50)
    ys = np.array([np.interp(g, x, y) for x, y in cs])
    return float(np.mean((ys - ys.mean(0)) ** 2))


def summary(bp, tag):
    cur = b.branch_curves(bp, tag)
    arms = [(lr, a) for _, lr, _, a in cur]
    ps = np.arange(0.5, 2.001, 0.01)
    sp = np.array([spread(arms, p) for p in ps])
    k = int(np.nanargmin(sp)); ok = ps[sp <= 2 * sp[k]]
    hl = [float(np.interp(0.5, a[::-1, 1], a[::-1, 0])) for _, a in arms]
    at248 = [float(np.interp(248, a[:, 0], a[:, 1])) for _, a in arms]
    end = [float(a[-1, 1]) for _, a in arms]
    return dict(p=round(float(ps[k]), 2), lo=round(float(ok.min()), 2), hi=round(float(ok.max()), 2),
                s1=spread(arms, 1), s2=spread(arms, 2), s0=spread(arms, 0), best=float(sp[k]),
                hl=[round(h, 1) for h in hl], at248=[round(v, 3) for v in at248], end=[round(v, 3) for v in end],
                halving=[round(hl[i + 1] / hl[i], 2) for i in range(2)])


def e0(bp, fn):
    import os
    base = os.path.join(b.OUT, "ck", "branch", "base")
    pre, at = fn(os.path.join(base, f"step-{bp - 248}"), bp), fn(os.path.join(base, f"step-{bp}"), bp)
    return float(((at[0] - at[1]) - (pre[0] - pre[1])).mean())


for name, fn in (("page window b-248..b-1", orig), ("shifted b-247..b-1", shifted)):
    b._range_score = fn
    print("==", name)
    for bp, tag in ((1240, ""), (2480, ""), (3720, ""), (2480, "-mom0.98")):
        s = summary(bp, tag)
        print(f"{bp}{tag}: E0 {e0(bp, fn):.4f} p {s['p']} ({s['lo']}-{s['hi']}) gaps best {s['best']:.1e} p1 {s['s1']:.1e} p2 {s['s2']:.1e}"
              f" steps {s['s0']:.2e} | half-life {s['hl']} halving {s['halving']} | f@248 {s['at248']} end {s['end']}")
b._range_score = orig
