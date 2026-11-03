"""HC's static mixing on the stream sum: geometric mean per layer of sum_gain (two sublayers multiplied) over layers 2-11, 12-24
and 25-47, and the cumulative product to layers 12, 24, 48, for the ordinary and separate-heads checkpoints with a mix-*.json.
Also the rank correlation of mid-layer readability with the cost over all design pairs, now with w810/w811, and without HC."""
import json, glob, os, numpy as np
from scipy.stats import spearmanr
D = "/scratch/gpfs/GROUP/USER/tmp/pls_paths/"
print(f"{'checkpoint':34s} L2-11 L12-24 L25-47 | cum log10 L12 L24 L48")
for f in sorted(glob.glob(D + "mix-*.json")):
    r = json.load(open(f))["rows"]; g = np.array([x["sum_gain"] for x in r])
    lay = np.array([g[2*l] * g[2*l+1] for l in range(48)])
    gm = lambda a, b: np.exp(np.log(np.abs(lay[a:b])).mean())
    c = lambda n: np.log10(np.abs(lay[:n])).sum()
    print(f"{os.path.basename(f)[4:-5]:34s} {gm(1,11):5.2f} {gm(11,24):5.2f} {gm(24,47):5.2f}  | {c(12):6.2f} {c(24):6.2f} {c(48):6.2f}")
z = np.load("/scratch/gpfs/GROUP/USER/tmp/pls_blog/d48g4.npz", allow_pickle=True)
rows = []
for k in z.files:
    if k.startswith("-sep-bbfrozen") and k.endswith("/final") and z[k].shape[0] == 48:
        t = k[len("-sep-bbfrozen"):-6]
        if "-sep" + t + "/final" not in z.files: continue
        p, q = z[k], z["-sep" + t + "/final"]
        rows.append((t, float(p[11:36].mean()), float(q[47]-p[47]), float(z["-sep"+t+"/eval"][47]-z[k[:-6]+"/eval"][47]), float(p[47]), float(q[47])))
drop = ("-adeepnorm-vs-d48-g4", "-aattnres-vs-d48-g4")
for name, T in (("all without 2 outliers", [r for r in rows if r[0] not in drop]), ("HC family", [r for r in rows if "-ahc" in r[0]]), ("not HC, without 2 outliers", [r for r in rows if "-ahc" not in r[0] and r[0] not in drop])):
    x = [r[1] for r in T]
    print(name, "n", len(T), "mid vs cost", round(spearmanr(x, [r[2] for r in T]).correlation, 2), "p", round(spearmanr(x, [r[2] for r in T]).pvalue, 4), "mid vs eval cost", round(spearmanr(x, [r[3] for r in T]).correlation, 2), "mid vs L48 ord", round(spearmanr(x, [r[4] for r in T]).correlation, 2), "mid vs L48 sep", round(spearmanr(x, [r[5] for r in T]).correlation, 2))
print("costs, largest first", [(r[0][1:-8], round(r[2], 3)) for r in sorted(rows, key=lambda r: -r[2])[:6]])
print("layer-48 separate heads, largest first", [(r[0][1:-8], round(r[5], 3)) for r in sorted(rows, key=lambda r: -r[5])[:8]])
