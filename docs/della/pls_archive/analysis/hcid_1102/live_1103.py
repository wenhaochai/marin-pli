"""Interim read of the runs in training (W&B) against finished references on the same stack at the same step (d48g4.npz): 100-step mean of the
train CE at layers 12, 24, 36, 48, minus the reference. Also total grad norm rows above 1.5 x median and the maximum, every 10 steps."""
import numpy as np, wandb
api = wandb.Api(timeout=300)
P = "reself/marin-della/muonh-qwen3-130m-della4xh100-pls1-"
z = np.load("/scratch/gpfs/GROUP/USER/tmp/pls_blog/d48g4.npz", allow_pickle=True)
K = {11: "train/pls/L11", 23: "train/pls/L23", 35: "train/pls/L35", 47: "train/pls/L47"}
RUNS = [("w889 HC mix + joint rw, ordinary", "sep-bbfrozen-ahc-hcrwj-fa4thd-vs-d48", "-sep-bbfrozen-ahc-fa4thd-vs-d48", "-sep-bbfrozen-amhc-mhcid-mhcrwlin-fa4thd-vs-d48"),
        ("w888 HC mix + joint rw, sep", "sep-ahc-hcrwj-fa4thd-vs-d48", "-sep-ahc-fa4thd-vs-d48", "-sep-amhc-mhcid-mhcrwlin-fa4thd-vs-d48"),
        ("w890 HC seed 1, ordinary", "sep-bbfrozen-ahc-fa4thd-vs-d48-s1", "-sep-bbfrozen-ahc-fa4thd-vs-d48", "-sep-bbfrozen-amhc-mhcid-mhcrwlin-fa4thd-vs-d48"),
        ("w891 HC I seed 1, ordinary", "sep-bbfrozen-ahc-hcid-fa4thd-vs-d48-s1", "-sep-bbfrozen-ahc-fa4thd-vs-d48", "-sep-bbfrozen-ahc-hcid-fa4thd-vs-d48"),
        ("w892 mHC I lin seed 1, ordinary", "sep-bbfrozen-amhc-mhcid-mhcrwlin-fa4thd-vs-d48-s1", "-sep-bbfrozen-ahc-fa4thd-vs-d48", "-sep-bbfrozen-amhc-mhcid-mhcrwlin-fa4thd-vs-d48"),
        ("w893 HC I seed 1, sep", "sep-ahc-hcid-fa4thd-vs-d48-s1", "-sep-ahc-fa4thd-vs-d48", "-sep-ahc-hcid-fa4thd-vs-d48")]
for name, tag, ref1, ref2 in RUNS:
    try:
        r = api.run(P + tag)
    except Exception as e:
        print(name, "no run:", str(e)[:80]); continue
    last = int(r.summary.get("_step", -1))
    by = {}
    for x in r.scan_history(keys=["_step", *K.values()], min_step=max(0, last - 600), max_step=last + 1, page_size=2000):
        by[x["_step"]] = x
    st = np.array(sorted(by))
    if len(st) < 20:
        print(name, "last", last, "rows", len(st)); continue
    win = st[st > st[-1] - 100]
    out = []
    for l, k in K.items():
        v = np.mean([by[s][k] for s in win])
        o = f"L{l+1} {v:.3f}"
        for ref in (ref1, ref2):
            rs, rc = z[ref + "/step"], z[ref + "/ce"]
            m = np.isin(rs, win)
            o += f" ({v - rc[m, l].mean():+.3f})" if m.any() else " (-)"
        out.append(o)
    g = {}
    for x in r.scan_history(keys=["_step", "grad/norm/total"], min_step=max(0, last - 600), max_step=last + 1, page_size=2000):
        g[x["_step"]] = x["grad/norm/total"]
    gv = np.array(list(g.values()))
    print(f"{name}: step {last} state {r.state}; {ref1[1:]} / {ref2[1:]} gaps in brackets\n   " + "; ".join(out) + f"\n   grad median {np.median(gv):.2f} max {gv.max():.1f} rows>1.5xmed {int((gv > 1.5 * np.median(gv)).sum())}/{len(gv)}", flush=True)
