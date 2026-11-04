"""Read ce_by_target<tag>.npz. Same training tokens in every run (128 batches of 2 x 1024). A position p predicts tok[p + 1].
1. Mean CE per head. 2. How much of the mean CE at the first heads comes from tokens with CE above 20 nats, per run. 3. Which targets those tokens have. 4. Their state norm: are they the blown tokens? 5. The gap w810 - HC per head, split by those tokens and by newline targets."""
import glob, numpy as np
from transformers import AutoTokenizer
tz = AutoTokenizer.from_pretrained(sorted(glob.glob("/scratch/gpfs/GROUP/USER/cache/huggingface/hub/models--marin-community--marin-tokenizer/snapshots/*"))[-1], local_files_only=True)
R = {"w810": "-sep-ahc-hcid-fa4thd-vs-d48", "w893": "-sep-ahc-hcid-fa4thd-vs-d48-s1", "HC": "-sep-ahc-fa4thd-vs-d48"}
D = {k: np.load(f"fw_ce_by_target{t}.npz") for k, t in R.items()}
heads = [int(h) for h in D["w810"]["heads"]]
tok = D["w810"]["tok"]
tgt = tok[:, 1:]
CE = {k: z["ce"][:, :, :-1] for k, z in D.items()}
NR = {k: z["nr"][:, :, :-1] for k, z in D.items()}
for k in R:
    print(k, D[k]["ck"].item().rsplit("/", 1)[1], "tokens", tgt.size)
print("head:      " + " ".join(f"L{h + 1:<6d}" for h in heads))
for k in R:
    print(f"{k:5s} mean  " + " ".join(f"{CE[k][i].mean():7.3f}" for i in range(len(heads))))
for thr in (20, 30):
    print(f"\nshare of tokens with CE above {thr} nats, and their part of the mean CE")
    for k in R:
        print(f"{k:5s} share " + " ".join(f"{(CE[k][i] > thr).mean():7.4f}" for i in range(len(heads))))
        print(f"{k:5s} part  " + " ".join(f"{(CE[k][i] * (CE[k][i] > thr)).sum() / CE[k][i].size:7.3f}" for i in range(len(heads))))
vals, cnt = np.unique(tgt, return_counts=True)
freq = dict(zip(vals.tolist(), cnt.tolist()))
top = vals[np.argsort(-cnt)[:12]]
print("\nmost frequent targets: count; CE at L1 for w810 / w893 / HC; CE at L48 for w810 / w893 / HC")
for t in top:
    m = tgt == t
    print(f"  {tz.decode([int(t)])!r:14s} {m.sum():6d}; " + " ".join(f"{CE[k][0][m].mean():6.2f}" for k in R) + " ; " + " ".join(f"{CE[k][-1][m].mean():6.2f}" for k in R))
for k in R:
    m = CE[k][0] > 20
    v, c = np.unique(tgt[m], return_counts=True)
    o = np.argsort(-c)[:12]
    print(f"\n{k}: tokens with CE > 20 at L1: {m.sum()}; top targets (text, count among them, count in the text, mean CE at L1, L48):")
    for i in o:
        t = v[i]
        q = m & (tgt == t)
        print(f"   {tz.decode([int(t)])!r:14s} {c[i]:5d} {freq[int(t)]:6d}   {CE[k][0][q].mean():6.2f} {CE[k][-1][q].mean():6.2f}")
    med = np.median(NR[k], axis=(1, 2))[:, None, None]
    blown = (NR[k] > 100 * med).any(0)
    b10 = (NR[k] > 10 * med).any(0)
    print(f"   blown tokens anywhere (> 100 x median at some layer): {blown.mean():.4f} of all; > 10 x: {b10.mean():.4f}; among CE > 20 at L1: {blown[m].mean():.3f}, {b10[m].mean():.3f}; CE > 20 at L1 among the blown: {m[blown].mean():.3f}; mean CE at L1, L48 of the blown: {CE[k][0][blown].mean() if blown.any() else 0:.2f}, {CE[k][-1][blown].mean() if blown.any() else 0:.2f}")
print("\ngap w810 - HC and w893 - HC per head (w893 at step 4,550, HC at step 4,958): all; from tokens with CE > 20 at that head in the run; from newline targets; rest")
nl = np.isin(tgt, [i for i in vals.tolist() if "\n" in tz.decode([i])])
print("newline targets share of tokens", round(float(nl.mean()), 4))
for r in ("w810", "w893"):
    for i, h in enumerate(heads):
        d = CE[r][i] - CE["HC"][i]
        hi = CE[r][i] > 20
        print(f"  {r} L{h + 1:<3d} all {d.mean():+.3f}; CE>20 {(d * hi).sum() / d.size:+.3f}; newline targets {(d * nl).sum() / d.size:+.3f} (CE>20 among them {(d * (nl & hi)).sum() / d.size:+.3f}); other {(d * (~nl & ~hi)).sum() / d.size:+.3f}")
