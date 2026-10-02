import numpy as np
D = "/scratch/gpfs/GROUP/USER/tmp/pls/dyn/"
b = np.load(D + "base.npz")
arms = {"shared": "-pls1", "readonly": "-pls1-dh", "sep": "-pls1-sep"}
A = {n: np.load(D + s + ".npz") for n, s in arms.items()}
st = b["step"]
sel = st <= 400
for n, a in A.items():
    print(f"\n== {n}: step | d=CE5-base | CE0..CE5 | gap45=CE4-CE5 gap05=CE0-CE5 | blockgrad ratio arm/base L0..L5 | gradtot arm base")
    for i in np.where(sel)[0]:
        ce = [a[f"train/pls/L{k}"][i] for k in range(6)]
        r = a["block_grad"][i] / b["block_grad"][i]
        print(f"{int(st[i]):4d} {ce[5]-b['train/loss'][i]:+.3f} | " + " ".join(f"{x:.2f}" for x in ce)
              + f" | {ce[4]-ce[5]:+.3f} {ce[0]-ce[5]:+.3f} | " + " ".join(f"{x:.2f}" for x in r)
              + f" | {a['grad/norm/total'][i]:.2f} {b['grad/norm/total'][i]:.2f}")
