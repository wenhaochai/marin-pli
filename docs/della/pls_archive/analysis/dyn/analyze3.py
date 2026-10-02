import numpy as np
D = "/scratch/gpfs/GROUP/USER/tmp/pls/dyn/"
L = np.load(D + "seeds_trainloss.npy", allow_pickle=True).item()
steps = sorted(L["base"])
b = np.load(D + "base.npz")
arms = {n: np.load(D + s + ".npz") for n, s in [("shared", "-pls1"), ("sep", "-pls1-sep")]}
edges = [0, 50, 100, 150, 200, 300, 500, 1000, 2000, 4960]
print("window       s1-base  s2-base  s3-base | ema-base | shared-base  sep-base")
for lo, hi in zip(edges[:-1], edges[1:]):
    ss = [s for s in steps if lo < s <= hi]
    d = [np.mean([L[k][s] - L["base"][s] for s in ss if s in L[k]]) for k in ["-s1", "-s2", "-s3", "-ema0.999"]]
    m = (b["step"] > lo) & (b["step"] <= hi)
    arm = [np.mean(a["train/pls/L5"][m] - b["train/loss"][m]) for a in arms.values()]
    print(f"{lo:5d}-{hi:5d}  " + " ".join(f"{x:+.4f}" for x in d[:3]) + f" | {d[3]:+.4f}  | " + "     ".join(f"{x:+.4f}" for x in arm))
print("\nper-10-step loss drop, baseline vs arms (steps 120-260): base ds, shared ds, sep ds")
for s in range(130, 270, 10):
    i = int(np.where(b["step"] == s)[0][0])
    print(s, f"{b['train/loss'][i-1]-b['train/loss'][i]:+.3f}", " ".join(f"{a['train/pls/L5'][i-1]-a['train/pls/L5'][i]:+.3f}" for a in arms.values()))
