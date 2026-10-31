"""Imprint and decay (wq w642 d12-ck248, w643 rsqrtcd-ck248). Per checkpoint s and sampled block b (8 steps every 128):
excess fit e_s(b) = [U_s - L_s(b)] - [U_p - L_p(b)], with U the unread mean, L the block mean, p the last checkpoint
before b was read (block difficulty and the general gain cancel). Imprint = e at the first checkpoint after the read;
retention = e at later checkpoints."""
import glob, os, json
import numpy as np
import sys
OUT = "/scratch/gpfs/GROUP/USER/project/trec/out/ck"; T = int(sys.argv[2]) if len(sys.argv) > 2 else 4959
res = {}
for run in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["muonh-qwen3-130m-della4xh100-d12-ck248", "muonh-qwen3-130m-della4xh100-d12-rsqrtcd-ck248"]):
    steps = sorted(int(d.split("-")[-1]) for d in glob.glob(f"{OUT}/{run}/step-*"))
    blocks = np.arange(0, T, 128)
    U, L = [], []
    for s in steps:
        idx, l = [], []
        for f in glob.glob(f"{OUT}/{run}/step-{s}/[0-9]*.npz"):
            z = np.load(f); idx.append(z["index"]); l.append(z["seq_loss"].sum(1) / z["seq_weight"].sum(1))
        idx, l = np.concatenate(idx), np.concatenate(l)
        tail = idx >= T
        # the final checkpoint has 512 tail batches, the others 256: use the 256 that all share (the scorer spreads them evenly)
        ti = idx[tail]; tl = l[tail]
        U.append(tl.mean() if s != steps[-1] else tl[np.isin(ti, common)].mean() if 'common' in dir() else tl.mean())
        if s == steps[0]: common = ti
        L.append([l[(idx >= b) & (idx < b + 8)].mean() for b in blocks])
    U, L = np.array(U), np.array(L)               # [S], [S, B]
    st = np.array(steps)
    g = U[:, None] - L                             # below-unread at each checkpoint
    E = np.full_like(g, np.nan)
    for j, b in enumerate(blocks):
        before = np.flatnonzero(st < b)            # checkpoints that had not read block b (block ends at b+7 < next ckpt)
        if not len(before) or b + 8 > st[-1]:
            continue
        p = before[-1]
        E[:, j] = g[:, j] - g[p, j]
        E[: p + 1, j] = np.nan
    read_q = blocks / T
    first = np.array([E[np.flatnonzero(~np.isnan(E[:, j]))[0], j] if (~np.isnan(E[:, j])).any() else np.nan for j in range(len(blocks))])
    final = E[-1]
    print(run.replace("muonh-qwen3-130m-della4xh100-", ""), "steps", st[0], "..", st[-1], len(st), "| unread at ckpts", np.round(U[[0, 4, 9, 14, 19]], 3))
    for lo, hi in ((0, .25), (.25, .5), (.5, .75), (.75, .95)):
        m = (read_q >= lo) & (read_q < hi) & ~np.isnan(first)
        print(f"   read {lo:.2f}-{hi:.2f}: imprint {np.nanmean(first[m]):+.4f}  left at end {np.nanmean(final[m]):+.4f}  ratio {np.nanmean(final[m]) / np.nanmean(first[m]):.2f}  (n {m.sum()})")
    # decay curve: mean E as a function of age (s - read) for blocks read in the first half vs second half
    for lo, hi in ((0, .5), (.5, .8)):
        m = (read_q >= lo) & (read_q < hi)
        ages = {}
        for j in np.flatnonzero(m):
            for i in np.flatnonzero(~np.isnan(E[:, j])):
                a = int(round((st[i] - blocks[j]) / (T / 10)))
                ages.setdefault(a, []).append(E[i, j])
        print(f"   decay, read {lo}-{hi}: " + " ".join(f"age~{a/10:.1f}T:{np.mean(v):+.4f}" for a, v in sorted(ages.items()) if a <= 8))
    res[run] = dict(steps=st.tolist(), blocks=blocks.tolist(), U=U.tolist(), E=np.where(np.isnan(E), None, np.round(E, 5)).tolist(), g=np.round(g, 5).tolist())
json.dump(res, open(f"{OUT}/imprint.json" if len(sys.argv) < 2 else f"{OUT}/imprint_{T}.json", "w"))
