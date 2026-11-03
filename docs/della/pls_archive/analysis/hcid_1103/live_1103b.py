"""Interim read of w959/w960 (HC, mixing keeps the sum / mixing = a scalar; probes) and w893 (HC mixing = I, seed 1, separate heads) on W&B against
finished references at the same steps (d48g4.npz): 100-step mean of the train CE at chosen layers. Loss window and grad rows over the last 600 steps,
and for w893 the grad median per 250-step bin next to w810 (seed 0, HC mixing = I, separate heads) and HC separate heads."""
import numpy as np, wandb
api = wandb.Api(timeout=300)
P = "reself/marin-della/muonh-qwen3-130m-della4xh100-pls1-"
z = np.load("/scratch/gpfs/GROUP/USER/tmp/pls_blog/d48g4.npz", allow_pickle=True)
LAY = [2, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48]
K = {l - 1: f"train/pls/L{l - 1}" for l in LAY}
RUNS = [("w959 HC mixing keeps the sum, ordinary", "sep-bbfrozen-ahc-hcsum-fa4thd-vs-d48", "-sep-bbfrozen-ahc-fa4thd-vs-d48", "-sep-bbfrozen-ahc-hcid-fa4thd-vs-d48"),
        ("w960 HC mixing = scalar, ordinary", "sep-bbfrozen-ahc-hcscalar-fa4thd-vs-d48", "-sep-bbfrozen-ahc-fa4thd-vs-d48", "-sep-bbfrozen-ahc-hcid-fa4thd-vs-d48"),
        ("w893 HC I seed 1, sep", "sep-ahc-hcid-fa4thd-vs-d48-s1", "-sep-ahc-fa4thd-vs-d48", "-sep-ahc-hcid-fa4thd-vs-d48"),
        ("w892 mHC I lin seed 1, ordinary", "sep-bbfrozen-amhc-mhcid-mhcrwlin-fa4thd-vs-d48-s1", "-sep-bbfrozen-ahc-fa4thd-vs-d48", "-sep-bbfrozen-amhc-mhcid-mhcrwlin-fa4thd-vs-d48")]
for name, tag, ref1, ref2 in RUNS:
    r = api.run(P + tag)
    last = int(r.summary.get("_step", -1))
    by = {}
    for x in r.scan_history(keys=["_step", *K.values()], min_step=max(0, last - 400), max_step=last + 1, page_size=2000):
        by[x["_step"]] = x
    st = np.array(sorted(by))
    if len(st) < 20:
        print(name, "last", last, "rows", len(st)); continue
    win = st[st > st[-1] - 100]
    print(f"{name}: step {last} state {r.state}; rows {len(win)}; brackets: minus {ref1[1:]} / minus {ref2[1:]}")
    for l, k in K.items():
        vals = [by[s][k] for s in win if by[s].get(k) is not None]
        if not vals:
            print(f"   L{l + 1}: no values"); continue
        v = np.mean(vals)
        o = f"   L{l + 1:>2} {v:7.3f}"
        for ref in (ref1, ref2):
            rs, rc = z[ref + "/step"], z[ref + "/ce"]
            m = np.isin(rs, win)
            o += f" ({v - rc[m, l].mean():+.3f})" if m.any() else " (-)"
        print(o, flush=True)
    rows = []
    for x in r.scan_history(keys=["_step", "train/loss", "grad/norm/total"], page_size=2000):
        rows.append((x["_step"], x["train/loss"], x["grad/norm/total"]))
    a = np.array(sorted(rows))
    for lo in range(0, int(a[-1, 0]) + 1, 250):
        m = (a[:, 0] >= lo) & (a[:, 0] < lo + 250)
        if m.any():
            print(f"     step {lo:>5}: loss median {np.median(a[m, 1]):8.2f} max {a[m, 1].max():8.1f}; grad median {np.median(a[m, 2]):8.2f} max {a[m, 2].max():10.1f}")
    print(flush=True)
