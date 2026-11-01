"""All 48-layer separate-heads costs in tmp/pls_blog/d48g4.npz: sep L48 final minus probes (-sep-bbfrozen) L48 final.
Read-only; prints one row per tag that has both runs, sorted by cost."""
import numpy as np
d = dict(np.load("/scratch/gpfs/GROUP/USER/tmp/pls_blog/d48g4.npz", allow_pickle=True))
tags = sorted({k[len("-sep-"):-len("/final")] for k in d if k.startswith("-sep-") and not k.startswith("-sep-bbfrozen-") and k.endswith("/final")})
rows = []
for t in tags:
    a, b = d.get(f"-sep-{t}/final"), d.get(f"-sep-bbfrozen-{t}/final")
    if a is None or b is None or len(a) < 48 or len(b) < 48:
        continue
    rows.append((float(a[47] - b[47]), t, float(a[47]), float(b[47])))
for c, t, a, b in sorted(rows):
    print(f"{t:45s} sep {a:.3f} probes {b:.3f} cost {c:+.3f}")
