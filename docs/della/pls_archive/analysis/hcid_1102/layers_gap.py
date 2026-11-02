"""All 48 layers: probe train loss of w811 minus mHC identity + linear, and minus HC (d48g4.npz, final = last 50 steps)."""
import numpy as np
z = np.load("/scratch/gpfs/GROUP/USER/tmp/pls_blog/d48g4.npz", allow_pickle=True)
P = "-sep-bbfrozen-"
I = z[P + "ahc-hcid-fa4thd-vs-d48/final"]
M = z[P + "amhc-mhcid-mhcrwlin-fa4thd-vs-d48/final"]
H = z[P + "ahc-fa4thd-vs-d48/final"]
d = I - M
print("w811-mHC by layer", " ".join(f"{k+1}:{v:+.2f}" for k, v in enumerate(d)))
print("mHC gap L12-48: max", round(float(d[11:].max()), 3), "at", int(d[11:].argmax()) + 12, "min", round(float(d[11:].min()), 3), "at", int(d[11:].argmin()) + 12, "mean abs", round(float(np.abs(d[11:]).mean()), 3))
e = I - H
print("w811-HC by layer", " ".join(f"{k+1}:{v:+.2f}" for k, v in enumerate(e)))
print("HC gap L12-30: min", round(float(e[11:30].min()), 3), "max", round(float(e[11:30].max()), 3))
print("HC L42", H[40:44].round(3).tolist())
