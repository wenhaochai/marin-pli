"""Every finished 48-layer ordinary model and its separate-heads twin: last-layer final train and held-out loss, sorted by held-out loss."""
import numpy as np
z = np.load("/scratch/gpfs/GROUP/USER/tmp/pls_blog/d48g4.npz", allow_pickle=True)
rows = []
for k in z.files:
    if k.startswith("-sep-bbfrozen") and k.endswith("/final") and "d48" in k:
        t = k[len("-sep-bbfrozen-"):-6]; f = z[k]
        if len(f) != 48:
            continue
        e = float(z[k[:-6] + "/eval"][47]); s = "-sep-" + t
        st = z[s + "/final"][47] if s + "/final" in z.files else np.nan; se = z[s + "/eval"][47] if s + "/eval" in z.files else np.nan
        p = float(z[k[:-6] + "/params"]) if k[:-6] + "/params" in z.files else np.nan
        rows.append((e, float(f[47]), t, float(st), float(se), p))
print(" held-out  train | sep held-out  sep train | params | design")
for e, f, t, st, se, p in sorted(rows):
    print(f"  {e:7.3f} {f:7.3f} | {se:7.3f} {st:7.3f} | {p / 1e6:7.1f}M | {t}")
