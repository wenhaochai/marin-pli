"""w973: ln_2 norm at layers 12, 14, 17 and gradient norm of layer 12's MLP down_proj, by step, W&B cache (every 10 steps), with layer 48 CE. Compare with HC separate heads."""
import numpy as np
for t in ("sep-ahc-hcshrink-fa4thd-vs-d48", "sep-ahc-fa4thd-vs-d48"):
    d = np.load(f"cache/{t}.npz"); s, K, v = d["step"], list(d["keys"]), d["v"]
    col = lambda k: v[:, K.index(k)]
    ln = {l: col(f"params/norm/transformer.layers.{l - 1}.ln_2.weight") for l in (12, 14, 17)}
    gd = col("grad/norm/transformer.layers.11.mlp.down_proj.weight"); gt = col("grad/norm/total"); ce = col("train/pls/L47"); c12 = col("train/pls/L11")
    print(t, "last", s[-1])
    print(" step  ln_2 L12 L14 L17 | grad L12 down | grad total | L48 | L12")
    for lo in list(range(600, 2000, 100)):
        m = (s >= lo) & (s < lo + 100)
        if m.sum() == 0:
            continue
        print(f" {lo:5d} {np.nanmedian(ln[12][m]):6.2f} {np.nanmedian(ln[14][m]):6.2f} {np.nanmedian(ln[17][m]):6.2f} | {np.nanmedian(gd[m]):7.3f} {np.nanmax(gd[m]):8.3f} | {np.nanmedian(gt[m]):7.2f} {np.nanmax(gt[m]):8.2f} | {np.nanmean(ce[m]):.3f} | {np.nanmean(c12[m]):.3f}")
