import numpy as np
for t in ("", "-bbfrozen"):
    P = np.load(f"paths-sep{t}-amhc-mhcid-vs-d48.npz")["pi_sub"]
    for th in (0.03, 0.05, 0.1):
        n = [(P[:, :, c] >= th).sum() for c in range(4)]
        print(t or "sep", th, n)
    # top paths
    idx = np.dstack(np.unravel_index(np.argsort(-P.ravel())[:12], P.shape))[0]
    print([(int(v), int(u) - 1, int(c), round(float(P[v, u, c]), 3)) for v, u, c in idx])
