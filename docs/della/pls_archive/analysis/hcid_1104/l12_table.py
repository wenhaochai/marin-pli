"""Layer 12, 24, 36, 48 train CE (final = mean of the last 50 logged steps) of every ordinary (-sep-bbfrozen) 48-layer run in d48g4.npz, sorted by layer 12."""
import numpy as np
z = np.load("/scratch/gpfs/GROUP/USER/tmp/pls_blog/d48g4.npz", allow_pickle=True)
rows = []
for k in z.files:
    if k.endswith("/final") and "bbfrozen" in k and "d48" in k:
        t = k[:-6]; f = z[k]
        if len(f) == 48:
            rows.append((float(f[11]), t.replace("-sep-bbfrozen-", ""), [float(f[i]) for i in (1, 11, 23, 35, 47)]))
print("tag  L2 L12 L24 L36 L48")
for _, t, v in sorted(rows):
    print(f"{t:44s} " + " ".join(f"{x:6.3f}" for x in v))
