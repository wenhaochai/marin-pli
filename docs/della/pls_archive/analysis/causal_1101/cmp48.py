"""L48 train CE (final = mean of the last 50 logged steps) of every 48-layer run in tmp/pls_blog/d48g4.npz, and the
separate-heads cost (sep minus its probes run) where both exist. Read-only; prints a table."""
import numpy as np
d = dict(np.load("/scratch/gpfs/GROUP/USER/tmp/pls_blog/d48g4.npz", allow_pickle=True))
final = {k[:-len("/final")]: float(np.asarray(v)[-1]) for k, v in d.items() if k.endswith("/final")}
print("probes runs (L48)")
for t in sorted(final):
    if "bbfrozen" in t:
        print(f"  {t:62s} {final[t]:.3f}")
print("sep runs: sep, probes, cost")
for t in sorted(final):
    if "bbfrozen" in t:
        continue
    p = final.get(t.replace("-sep-", "-sep-bbfrozen-", 1))
    print(f"  {t:62s} {final[t]:.3f}  " + (f"{p:.3f}  {final[t] - p:+.3f}" if p is not None else "no probes run"))
