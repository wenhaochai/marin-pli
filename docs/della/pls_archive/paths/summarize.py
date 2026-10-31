import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=160)
R = {k: np.load(f"paths-sep{t}-amhc-mhcid-vs-d48.npz") for k, t in (("sep", ""), ("probes", "-bbfrozen"))}
for k, d in R.items():
    P = d["pi_sub"]; V, U, m = P.shape   # U = 2L + 1 writers, index 0 = embedding (sublayer -1)
    usub = np.arange(U) - 1
    skip = (np.arange(V)[:, None] - usub[None]) / 2.0          # layers skipped
    print(f"== {k}  tokens {int(d['tokens'])}")
    # per stream: total share and share-weighted mean skip, over all readers
    for c in range(m):
        w = P[:, :, c]; tot = w.sum()
        print(f" stream {c}: share {tot / V:.3f}  mean skip {(w * skip).sum() / tot:5.2f} layers  emb share {w[:, 0].sum() / V:.3f}")
    Pt = P.sum(2)
    print(" overall mean skip", round(float((Pt * skip).sum() / Pt.sum()), 2), " embedding share of sublayer inputs by layer 0/12/24/36/47:",
          [round(float(Pt[2 * l, 0]), 3) for l in (0, 12, 24, 36, 47)])
    ro = d["pi_ro"].sum(2)
    print(" readout share from embedding by layer 1/12/24/36/48:", [round(float(ro[l, 0]), 3) for l in (0, 11, 23, 35, 47)])
    print(" readout share from own layer's 2 sublayers:", [round(float(ro[l, 2 * l + 1] + ro[l, 2 * l + 2]), 3) for l in (0, 11, 23, 35, 47)])
    print(" readout share from last 4 layers:", [round(float(ro[l, max(1, 2 * l - 5): 2 * l + 3].sum()), 3) for l in (0, 11, 23, 35, 47)])
    print(" mean read per stream, layer 0/12/24/36/47 (attn):\n", d["read_mean"][[0, 24, 48, 72, 94]])
    print(" mean write per stream, same:\n", d["write_mean"][[0, 24, 48, 72, 94]])
Ps, Pp = R["sep"]["pi_sub"].sum(2), R["probes"]["pi_sub"].sum(2)
V, U = Ps.shape; skip = (np.arange(V)[:, None] - (np.arange(U) - 1)[None]) / 2.0
D = Ps - Pp
for name, sel in (("skip<=1", skip <= 1), ("1<skip<=6", (skip > 1) & (skip <= 6)), ("6<skip<=24", (skip > 6) & (skip <= 24)), ("skip>24", skip > 24), ("embedding", np.eye(1, U, 0, dtype=bool).repeat(V, 0))):
    print(f"delta (sep - probes) summed over readers / {V}: {name}: {D[sel].sum() / V:+.3f}")
