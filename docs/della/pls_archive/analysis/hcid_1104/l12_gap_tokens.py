"""Where does the layer 12 gap between HC and mixing = I (ordinary models) sit? Per-token CE on the same FineWeb-Edu windows (ce_by_target.py). Bins: mean CE of both models at layer 12 (no selection on one model), and whether the target token already occurred earlier in the same window (a proxy for copying from context)."""
import sys, numpy as np
A, B = sys.argv[1], sys.argv[2]
a = np.load(f"fw_ce_by_target{A}.npz"); b = np.load(f"fw_ce_by_target{B}.npz")
H = list(a["heads"]); tok = a["tok"]; assert (tok == b["tok"]).all()
ca, cb = a["ce"], b["ce"]    # (heads, batches, positions)
print("shape", ca.shape, "heads", [int(h) + 1 for h in H])
for i, h in enumerate(H):
    print(f"  L{int(h) + 1}: {A[:40]} {ca[i].mean():.3f}  {B[:40]} {cb[i].mean():.3f}  diff {ca[i].mean() - cb[i].mean():+.3f}")
i12 = H.index(11); x, y = ca[i12], cb[i12]
print("mean CE by position of the target in the window is flat?", [round(float((x - y)[:, s:s + 512].mean()), 3) for s in range(0, tok.shape[1], 512)][:8])
# target seen earlier in the same window (positions shift: CE at position p predicts tok[p + 1])
seen = np.zeros_like(tok, bool)
for r in range(tok.shape[0]):
    s = set()
    for p in range(tok.shape[1]):
        t = tok[r, p]
        seen[r, p] = t in s
        s.add(t)
tar_seen = np.roll(seen, -1, axis=1)
print(f"target seen earlier in the window: {tar_seen.mean():.3f} of tokens")
for name, m in (("seen", tar_seen), ("new", ~tar_seen)):
    print(f"  {name}: {A[:24]} {x[m].mean():.3f} {B[:24]} {y[m].mean():.3f}, diff {(x - y)[m].mean():+.3f}, contribution to the mean {(x - y)[m].sum() / x.size:+.3f}")
z = (x + y) / 2; q = np.quantile(z, [.25, .5, .75])
for n, m in (("Q1", z < q[0]), ("Q2", (z >= q[0]) & (z < q[1])), ("Q3", (z >= q[1]) & (z < q[2])), ("Q4", z >= q[2])):
    print(f"  mean-CE {n}: {z[m].mean():.3f}; diff {(x - y)[m].mean():+.3f}; contribution {(x - y)[m].sum() / x.size:+.3f}")
