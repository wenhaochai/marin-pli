"""Read fw_ce_by_target<tag>.npz: 128 random windows of 4,096 tokens of the real training data (FineWeb-Edu 10B), same windows in every run, compute bfloat16. A position p predicts tok[p + 1].
1. Mean CE per head. 2. Blown share by layer. 3. The gap to HC per head, split by the run's state norm relative to the median at that head. 4. Blown tokens by first blown layer: CE before and after the jump, in the run and in HC on the same tokens. 5. Which tokens blow up."""
import glob, numpy as np
from transformers import AutoTokenizer
tz = AutoTokenizer.from_pretrained(sorted(glob.glob("/scratch/gpfs/GROUP/USER/cache/huggingface/hub/models--marin-community--marin-tokenizer/snapshots/*"))[-1], local_files_only=True)
R = {"w810": "-sep-ahc-hcid-fa4thd-vs-d48", "w893": "-sep-ahc-hcid-fa4thd-vs-d48-s1", "HC": "-sep-ahc-fa4thd-vs-d48"}
D = {k: np.load(f"fw_ce_by_target{t}.npz") for k, t in R.items()}
heads = [int(h) for h in D["w810"]["heads"]]
tok = D["w810"]["tok"]
ctx, tgt = tok[:, :-1], tok[:, 1:]
CE = {k: z["ce"][:, :, :-1] for k, z in D.items()}
NR = {k: z["nr"][:, :, :-1] for k, z in D.items()}
N = tgt.size
for k in R:
    print(k, D[k]["ck"].item().rsplit("/", 1)[1], "tokens", N)
print("head:       " + " ".join(f"L{h + 1:<6d}" for h in heads))
for k in R:
    print(f"{k:5s} mean  " + " ".join(f"{CE[k][i].mean():7.3f}" for i in range(len(heads))))
med = {k: np.median(NR[k], axis=(1, 2))[:, None, None] for k in R}
LS = [0, 3, 4, 5, 7, 9, 10, 11, 12, 13, 15, 23, 47]
print("\nshare of tokens above 10 x / 100 x the median state norm, layers " + " ".join(f"L{l + 1}" for l in LS))
for k in R:
    print(f"{k:5s}", " ".join(f"{(NR[k][l] > 10 * med[k][l]).mean():.3f}/{(NR[k][l] > 100 * med[k][l]).mean():.3f}" for l in LS))
E = [0, 1.5, 3, 10, 100, np.inf]
for r in ("w810", "w893"):
    print(f"\n{r} minus HC, by |s| / median of {r} at that head (share of tokens: part of the gap)")
    for i, h in enumerate(heads):
        rat = NR[r][h] / med[r][h]
        d = CE[r][i] - CE["HC"][i]
        o = f"  L{h + 1:<3d} gap {d.mean():+.3f}, se {d.std() / np.sqrt(N):.3f}:"
        for lo, hi in zip(E[:-1], E[1:]):
            m = (rat >= lo) & (rat < hi)
            o += f"  [{lo:g}-{hi:g}] {m.mean():.3f}: {(d * m).sum() / N:+.3f}"
        print(o)
for r in ("w810", "w893"):
    over = NR[r] > 100 * med[r]
    ever = over.any(0)
    first = np.where(ever, over.argmax(0), -1)
    print(f"\n{r}: tokens above 100 x median at some layer {ever.mean():.4f}; by first layer (1-based) with count; CE in {r} / in HC at heads " + " ".join(f"L{h + 1}" for h in heads))
    v, c = np.unique(first[ever], return_counts=True)
    for l, n in sorted(zip(v, c), key=lambda x: -x[1])[:6]:
        m = first == l
        print(f"  first L{l + 1:<3d} n {n:6d}: " + " ".join(f"{CE[r][i][m].mean():.2f}/{CE['HC'][i][m].mean():.2f}" for i in range(len(heads))))
    print("  not blown:  " + " ".join(f"{CE[r][i][~ever].mean():.2f}/{CE['HC'][i][~ever].mean():.2f}" for i in range(len(heads))))
    vals, cnt = np.unique(ctx, return_counts=True)
    fr = dict(zip(vals.tolist(), cnt.tolist()))
    v2, c2 = np.unique(ctx[ever], return_counts=True)
    o = np.argsort(-c2)[:14]
    print("  context tokens among the blown (text, blown count, count in the text):", [(tz.decode([int(v2[j])]), int(c2[j]), fr[int(v2[j])]) for j in o])
    enr = [(int(c2[j]) / fr[int(v2[j])], tz.decode([int(v2[j])]), int(c2[j]), fr[int(v2[j])]) for j in range(len(v2)) if fr[int(v2[j])] >= 50]
    print("  highest blown rate among context tokens seen at least 50 times:", [(round(a, 2), b, x, y) for a, b, x, y in sorted(enr, reverse=True)[:14]])
    vt, ct = np.unique(tgt, return_counts=True)
    ft = dict(zip(vt.tolist(), ct.tolist()))
    v3, c3 = np.unique(tgt[ever], return_counts=True)
    enr = [(int(c3[j]) / ft[int(v3[j])], tz.decode([int(v3[j])]), int(c3[j]), ft[int(v3[j])]) for j in range(len(v3)) if ft[int(v3[j])] >= 50]
    print("  highest blown rate among target tokens seen at least 50 times:", [(round(a, 2), b, x, y) for a, b, x, y in sorted(enr, reverse=True)[:14]])
    pos = np.tile(np.arange(tgt.shape[1]), (tgt.shape[0], 1))
    print("  blown rate by position bin (0-63, 64-255, 256-1023, 1024-4095):", [round(float(ever[(pos >= a) & (pos < b)].mean()), 4) for a, b in ((0, 64), (64, 256), (256, 1024), (1024, 4095))])
