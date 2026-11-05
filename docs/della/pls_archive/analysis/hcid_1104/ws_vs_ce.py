"""Does the layer's readable write size order the layer's probe CE across ordinary designs? rel_write (pls_write_size.py, tmp/pls_paths/ws-sep-bbfrozen-*.json, final checkpoints) against train CE (d48g4.npz). Spearman over the designs at layers 12, 24, 36."""
import glob, json, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
z = np.load("/scratch/gpfs/GROUP/USER/tmp/pls_blog/d48g4.npz", allow_pickle=True)
rows = []
for f in sorted(glob.glob("/scratch/gpfs/GROUP/USER/tmp/pls_paths/ws-sep-bbfrozen-*.json")):
    t = "-" + Path(f).name[3:-5]
    if t + "/final" not in z.files:
        continue
    d = json.load(open(f))
    rows.append((t, np.array(d["rel_write"]), np.array(d["write"]), np.array(d["norm"]), z[t + "/final"]))
print(len(rows), "designs")
for l in (6, 12, 18, 24, 30, 36):
    i = l - 1
    x = np.array([r[1][i] for r in rows]); x3 = np.array([np.mean(r[1][max(i - 2, 0):i + 1]) for r in rows]); y = np.array([r[4][i] for r in rows])
    print(f"L{l}: Spearman(rel_write at L, CE at L) {spearmanr(x, y).statistic:+.2f}; mean of last 3 layers {spearmanr(x3, y).statistic:+.2f}")
print("tag  rel_write L12 (mean L10-12), norm L12, CE L12 | rel_write L24, CE L24")
for t, r, w, n, c in sorted(rows, key=lambda r: r[4][11]):
    print(f"{t.replace('-sep-bbfrozen-', ''):40s} {r[11]:6.3f} {r[9:12].mean():6.3f} {n[11]:9.3g} {c[11]:6.3f} | {r[23]:6.3f} {c[23]:6.3f}")
