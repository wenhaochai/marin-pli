"""Does the ordinary model's state composition predict the cost of separate heads across designs? Cost = final L48 train CE, separate heads minus ordinary (d48g4.npz). Composition: state_share<tag>.npz of the ordinary model, mean over layers 2-36 and at single layers."""
import glob, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
HERE = Path(__file__).parent
z = np.load("/scratch/gpfs/GROUP/USER/tmp/pls_blog/d48g4.npz", allow_pickle=True)
rows = []
for f in sorted(glob.glob(str(HERE / "state_share-sep-bbfrozen*.npz"))):
    t = Path(f).name[len("state_share"):-4]; s = t.replace("-sep-bbfrozen", "-sep")
    if t + "/final" not in z.files or s + "/final" not in z.files:
        continue
    d = np.load(f); ct, cs = z[t + "/final"], z[s + "/final"]
    rows.append(dict(tag=t.replace("-sep-bbfrozen-", ""), cost=cs[47] - ct[47], sep48=cs[47], ord48=ct[47], type_mean=d["type_share"][1:36].mean(), type12=d["type_share"][11], type24=d["type_share"][23], rel_mean=d["rel"][1:36].mean(), ce12=ct[11], ce24=ct[23], ce_mean=ct[1:36].mean(), gain=(ct[1:47] - cs[1:47]).mean()))
print(len(rows), "pairs")
print("design                                      cost   sep48  ord48  type12 type24 typemean  CE12  CE24 meanCE(2-36) mean probe gap")
for r in sorted(rows, key=lambda r: r["cost"]):
    print(f"{r['tag']:42s} {r['cost']:6.3f} {r['sep48']:6.3f} {r['ord48']:6.3f} {r['type12']:5.2f} {r['type24']:6.2f} {r['type_mean']:6.2f} {r['ce12']:6.2f} {r['ce24']:5.2f} {r['ce_mean']:6.2f} {r['gain']:6.2f}")
c = np.array([r["cost"] for r in rows])
for k in ("type12", "type24", "type_mean", "rel_mean", "ce12", "ce24", "ce_mean", "gain", "ord48"):
    print(f"Spearman(cost, {k}) {spearmanr(c, [r[k] for r in rows]).statistic:+.2f}")
s = np.array([r["sep48"] for r in rows])
for k in ("type12", "type24", "type_mean", "ce12", "ce_mean"):
    print(f"Spearman(separate-heads L48, {k}) {spearmanr(s, [r[k] for r in rows]).statistic:+.2f}")
