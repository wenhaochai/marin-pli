"""Across ordinary designs and layers: the probe CE at a layer against what the state is made of there (state_share<tag>.npz, final or newest checkpoint; CE: d48g4.npz final, or the newest 50 steps of live_1104d.npz for running runs, marked with *; a running run's state comes from a checkpoint up to 30 min older than its CE)."""
import glob, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
HERE = Path(__file__).parent
z = np.load("/scratch/gpfs/GROUP/USER/tmp/pls_blog/d48g4.npz", allow_pickle=True)
L = np.load(HERE / "live_1104d.npz", allow_pickle=True); lay = [int(v) for v in L["lay"]]
LIVE = {"-sep-bbfrozen-ahc-hcshrink-fa4thd-vs-d48": "w972 only shrinks", "-sep-bbfrozen-ahc-hcshrink-sp0.5-fa4thd-vs-d48": "w1002 half dose", "-sep-bbfrozen-ahc-hcsum-hcf32-fa4thd-vs-d48": "w1006 keeps the sum"}
rows = {}
for f in sorted(glob.glob(str(HERE / "state_share-sep-bbfrozen*.npz"))):
    t = Path(f).name[len("state_share"):-4]; d = np.load(f)
    if t + "/final" in z.files:
        ce = z[t + "/final"]; star = ""
    elif t in LIVE:
        ce = {l: L[LIVE[t] + "/ce"][-50:, i].mean() for i, l in enumerate(lay)}; star = "*"
    else:
        continue
    rows[t] = (d, ce, star, int(d["step"]))
print("design                                         step    L12: CE type rel cos_e keep | L24: CE type rel cos_e keep")
for t, (d, ce, star, st) in sorted(rows.items(), key=lambda kv: kv[1][1][12 if kv[1][2] else 11]):
    g = lambda l: ce[l] if star else ce[l - 1]
    print(f"{t.replace('-sep-bbfrozen-', ''):44s}{star:1s} {st:5d} {g(12):6.3f} {d['type_share'][11]:5.2f} {d['rel'][11]:5.2f} {d['cos_e'][11]:5.2f} {d['keep'][11]:5.2f} | {g(24):6.3f} {d['type_share'][23]:5.2f} {d['rel'][23]:5.2f} {d['cos_e'][23]:5.2f} {d['keep'][23]:5.2f}")
for l in (12, 24, 36):
    fin = [(t, r) for t, r in rows.items() if not r[2]]
    y = np.array([r[1][l - 1] for _, r in fin])
    print(f"L{l}, finished runs ({len(fin)}): Spearman of CE with type share {spearmanr([r[0]['type_share'][l - 1] for _, r in fin], y).statistic:+.2f}, rel {spearmanr([r[0]['rel'][l - 1] for _, r in fin], y).statistic:+.2f}, cos_e {spearmanr([r[0]['cos_e'][l - 1] for _, r in fin], y).statistic:+.2f}, keep {spearmanr([r[0]['keep'][l - 1] for _, r in fin], y).statistic:+.2f}")
