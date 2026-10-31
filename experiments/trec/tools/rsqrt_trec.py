"""TREC summary of d12 and the two 1/sqrt(t) runs (wq w465, w590, w591): unread level, quarter means minus unread,
first 5% minus unread, low point of a 100- and a 500-step centred moving mean; and the logged lr against the formula."""
import glob, json, os
import numpy as np
OUT = "/scratch/gpfs/GROUP/USER/project/trec/out"; CACHE = "/scratch/gpfs/GROUP/USER/project/trec/cache/wandb"; T = 4959
def load(r):
    idx, l = [], []
    for f in sorted(glob.glob(f"{OUT}/{r}/[0-9]*.npz")):
        z = np.load(f); idx.append(z["index"]); l.append(z["seq_loss"].sum(1) / z["seq_weight"].sum(1))
    idx, l = np.concatenate(idx), np.concatenate(l)
    tr = idx < T; o = np.argsort(idx[tr])
    assert np.array_equal(idx[tr][o], np.arange(T)), r
    return l[tr][o], l[~tr]
def sm(v, w):
    c = np.concatenate([[0.0], np.cumsum(v)]); i = np.arange(len(v)); lo, hi = np.maximum(i - w // 2, 0), np.minimum(i + w - w // 2, len(v))
    return (c[hi] - c[lo]) / (hi - lo)
for r in ["muonh-qwen3-130m-della4xh100-d12", "muonh-qwen3-130m-della4xh100-d12-rsqrt", "muonh-qwen3-130m-della4xh100-d12-rsqrtcd"]:
    tr, un = load(r); u, use = un.mean(), un.std(ddof=1) / np.sqrt(len(un))
    q = [tr[int(T * i / 4):int(T * (i + 1) / 4)] for i in range(4)]
    qs = " ".join(f"{x.mean() - u:+.4f}" for x in q)
    se = np.sqrt(max(x.std(ddof=1) ** 2 / len(x) for x in q) + use ** 2)
    f5 = tr[:int(0.05 * T)].mean() - u
    lows = []
    for w in (100, 500):
        s = sm(tr, w); k = int(np.argmin(s)); lows.append(f"{(k + 1) / T:.3f} ({s[k] - u:+.4f})")
    print(f"{r.replace('muonh-qwen3-130m-della4xh100-', ''):10s} n_unread {len(un)} unread {u:.4f}+-{use:.4f} | quarters-unread {qs} (+-{se:.4f}) | first5% {f5:+.4f} | low100 {lows[0]} low500 {lows[1]} | last batch {tr[-1]:.4f}")
    if r.endswith(("rsqrt", "rsqrtcd")):
        d = json.load(open(f"{CACHE}/{r}.json")); st, v = np.array(d["optim/learning_rate"]).T
        t = np.array([0, 100, 991, 2480, 3967, 4500, 4958]); ref = 0.02 / np.sqrt(1 + 0.0092 * t)
        if r.endswith("cd"): ref = np.where(t < 3967, ref, 0.02 / np.sqrt(1 + 0.0092 * 3967) * np.clip((T - t) / (T - 3967), 0, 1))
        got = np.interp(t, st, v)
        print("   lr logged", np.round(got, 6), "max rel err vs formula", float(np.max(np.abs(got - ref) / np.maximum(ref, 1e-6))))
        tl = np.array(d["train/loss"]); print("   train loss: max jump step-to-step", round(float(np.max(np.diff(tl[:, 1]))), 3), "nan", bool(np.isnan(tl[:, 1]).any()))

# paired against d12: the runs read the same batches and are scored on the same unread ones
base_tr, base_un = load("muonh-qwen3-130m-della4xh100-d12")
for r in ["muonh-qwen3-130m-della4xh100-d12-rsqrt", "muonh-qwen3-130m-della4xh100-d12-rsqrtcd"]:
    tr, un = load(r); du = un - base_un
    out = []
    for i in range(4):
        sl = slice(int(T * i / 4), int(T * (i + 1) / 4)); d = tr[sl] - base_tr[sl]
        diff = d.mean() - du.mean(); se = np.sqrt(d.var(ddof=1) / len(d) + du.var(ddof=1) / len(du))
        out.append(f"Q{i+1} {diff:+.4f}+-{se:.4f} (abs {d.mean():+.4f})")
    print(f"{r[-8:]:>8s} minus d12, relative to own unread: " + " | ".join(out) + f" | unread {du.mean():+.4f}+-{du.std(ddof=1)/np.sqrt(len(du)):.4f}")
