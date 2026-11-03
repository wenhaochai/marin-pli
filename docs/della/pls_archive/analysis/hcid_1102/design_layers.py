"""Probe train loss (nats) at selected layers for every 48-layer ordinary-model run in d48g4.npz, sorted by layer 24, with
layer-48 train and eval. Which designs read well at the middle layers? Source: ../pls_blog/d48g4.npz."""
import numpy as np
z = np.load("/scratch/gpfs/GROUP/USER/tmp/pls_blog/d48g4.npz", allow_pickle=True)
P = "-sep-bbfrozen"
L = [0, 3, 7, 11, 15, 23, 31, 35, 39, 43, 47]
rows = []
for k in z.files:
    if k.startswith(P) and k.endswith("/final") and z[k].shape[0] == 48:
        f = z[k]; e = z[k[:-6] + "/eval"]
        rows.append((float(f[23]), k[len(P):-6], f, e))
print(" " * 40 + " ".join(f"{l+1:5d}" for l in L) + "  evalL48  evalL24")
for _, t, f, e in sorted(rows, key=lambda r: r[0]):
    print(f"{t:40s}" + " ".join(f"{f[l]:5.2f}" for l in L) + f"  {e[47]:.3f}  {e[23]:.3f}")
