"""How close is each run's state to the float32 limit of RMSNorm's sum of squares (|s| about 1.8e19)? fw_ce_by_target<tag>.npz. Per run: median and max state norm at chosen layers, the share of tokens above 1e18 and 1.8e19 at the last layers, and for w890 whether tokens above the limit are the tokens at the uniform loss 11.76 nats."""
import glob, numpy as np, re
for f in sorted(glob.glob("fw_ce_by_target*.npz")):
    z = np.load(f)
    nr = z["nr"][:, :, :-1]
    ce = z["ce"][:, :, :-1]
    L = [11, 23, 35, 41, 43, 45, 46, 47]
    print(f.replace("fw_ce_by_target", "").replace(".npz", ""), z["ck"].item().rsplit("/", 1)[1], "tokens", ce[0].size)
    print("   median |s|: " + " ".join(f"L{l + 1} {np.median(nr[l]):.2g}" for l in L))
    print("   max |s|:    " + " ".join(f"L{l + 1} {nr[l].max():.2g}" for l in L))
    print(f"   share above 1e18 at L46, L47, L48: {(nr[45] > 1e18).mean():.4f} {(nr[46] > 1e18).mean():.4f} {(nr[47] > 1e18).mean():.4f}; above 1.8e19: {(nr[45] > 1.8e19).mean():.4f} {(nr[46] > 1.8e19).mean():.4f} {(nr[47] > 1.8e19).mean():.4f}; L48 mean CE {ce[-1].mean():.3f}")
    u = np.abs(ce[-1] - 11.7618) < 0.01
    if u.any():
        o = nr[47] > 1.8e19
        print(f"   tokens at the uniform loss at L48: {u.mean():.4f}; among them above 1.8e19: {o[u].mean():.4f}; among the tokens above 1.8e19, at the uniform loss: {u[o].mean() if o.any() else 0:.4f}; median |s| of the others {np.median(nr[47][~u]):.3g}")
