import numpy as np
D = "/scratch/gpfs/GROUP/USER/tmp/pls/dyn/"
b = np.load(D + "base.npz")
arms = {"共享头": "-pls1", "只读共享头": "-pls1-dh", "独立头": "-pls1-sep"}
A = {n: np.load(D + s + ".npz") for n, s in arms.items()}
st = b["step"]
for n, a in A.items():
    assert (a["step"] == st).all()
print("paired check (same init + data order?): first steps, final-layer CE minus baseline train/loss")
for n, a in A.items():
    d = a["train/pls/L5"] - b["train/loss"]
    print(f"  {n}: " + " ".join(f"{int(s)}:{x:+.4f}" for s, x in zip(st[:8], d[:8])))
print("\nwindowed mean of d(t) = CE_final(arm) - CE(baseline), same batches")
edges = [0, 50, 100, 150, 200, 300, 400, 500, 700, 1000, 1500, 2000, 3000, 4000, 4960]
print("steps        " + "".join(f"{n:>12}" for n in A))
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (st > lo) & (st <= hi)
    print(f"{lo:5d}-{hi:5d}  " + "".join(f"{np.nanmean(a['train/pls/L5'][m] - b['train/loss'][m]):>+12.4f}" for a in A.values()))
