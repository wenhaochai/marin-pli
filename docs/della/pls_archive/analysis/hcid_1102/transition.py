"""Where does each ordinary model's probe loss fall? For each design: the drop of probe train loss over any 4 layers (largest
drop and its layers), and for HC checkpoints the static mixing per layer around that drop (sum_gain, leak, mean of the two
sublayers; mix-*.json). Source: d48g4.npz, ../pls_paths/mix-*.json."""
import json, os, numpy as np
z = np.load("/scratch/gpfs/GROUP/USER/tmp/pls_blog/d48g4.npz", allow_pickle=True)
P = "-sep-bbfrozen"
rows = []
for k in z.files:
    if k.startswith(P) and k.endswith("/final") and z[k].shape[0] == 48:
        f = z[k]; d = f[:-4] - f[4:]; i = int(np.argmax(d[1:])) + 1
        rows.append((i + 1, k[len(P):-6], float(d[i]), float(f[i]), float(f[i + 4]), float(f[47])))
print("design, layer where the largest 4-layer fall starts (after layer 2), fall, loss before, after, layer 48")
for i, t, d, a, b, c in sorted(rows):
    print(f"{t:38s} L{i:<3d} fall {d:5.2f}  {a:5.2f} -> {b:5.2f}  L48 {c:5.2f}")
for t in ("-ahc-fa4thd-vs-d48", "-ahc-vs-d48-g4", "-ahc-vs-d48-s1", "-ahc-rstatic-vs-d48", "-ahc-hcs8-vs-d48", "-ahc-hcrwsig-vs-d48"):
    p = f"/scratch/gpfs/GROUP/USER/tmp/pls_paths/mix{P}{t}.json"
    f = z[P + t + "/final"]
    if not os.path.exists(p): continue
    r = json.load(open(p))["rows"]
    g = np.array([x["sum_gain"] for x in r]); lk = np.array([x["leak"] for x in r]); dg = np.array([x["diff_gain"] for x in r])
    print(t); print(" layer " + " ".join(f"{l+1:5d}" for l in range(0, 24)))
    print(" CE    " + " ".join(f"{f[l]:5.2f}" for l in range(0, 24)))
    print(" sum   " + " ".join(f"{g[2*l:2*l+2].prod():5.2f}" for l in range(0, 24)))
    print(" leak  " + " ".join(f"{lk[2*l:2*l+2].mean():5.2f}" for l in range(0, 24)))
    print(" diffg " + " ".join(f"{dg[2*l:2*l+2].mean():5.2f}" for l in range(0, 24)))
