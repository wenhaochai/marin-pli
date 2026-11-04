"""Step-100 check of a new design: W&B state, the design options in the config, and the train CE at layers 2, 12, 24, 48 over 25-step bins against HC and HC mixing = I (w810/w811) at the same steps from d48g4.npz.
Usage: check100.py <sep|bbfrozen tag after -pls1, e.g. -sep-bbfrozen-ahc-hcmove-fa4thd-vs-d48> ..."""
import sys, numpy as np, wandb
api = wandb.Api(timeout=300)
P = "reself/marin-della/muonh-qwen3-130m-della4xh100-pls1"
z = np.load("/scratch/gpfs/GROUP/USER/tmp/pls_blog/d48g4.npz", allow_pickle=True)
OPT = ["depth_arch", "depth_hc_res", "depth_hc_rwdyn", "depth_shrink_power", "depth_preln_leak", "depth_hc_state_f32", "num_layers", "hc_streams"]
LAY = [2, 12, 24, 48]
K = {l: f"train/pls/L{l - 1}" for l in LAY}
def find(c, k):
    if isinstance(c, dict):
        for a, b in c.items():
            if a == k: return b
            v = find(b, k)
            if v is not None: return v
    return None
for tag in sys.argv[1:]:
    r = api.run(P + tag)
    cfg = dict(r.config)
    last = int(r.summary.get("_step", -1))
    print(f"== {tag} state {r.state} step {last}; " + "; ".join(f"{o}={find(cfg, o)}" for o in OPT), flush=True)
    by = {x["_step"]: x for x in r.scan_history(keys=["_step", "train/loss", "grad/norm/total", *K.values()], page_size=2000)} if False else {}
    for x in r.scan_history(keys=["_step", *K.values()], page_size=2000):
        by[x["_step"]] = x
    g = {x["_step"]: x["grad/norm/total"] for x in r.scan_history(keys=["_step", "grad/norm/total"], page_size=2000)}
    st = np.array(sorted(by))
    setup = "-sep-bbfrozen" if "-bbfrozen" in tag else "-sep"
    refs = {"HC": setup + "-ahc-fa4thd-vs-d48", "hcid": setup + "-ahc-hcid-fa4thd-vs-d48"}
    print("  bin      " + "".join(f"  L{l:<5d}" for l in LAY) + "  |  " + "  ".join(f"{n} L12, L48" for n in refs) + "  | grad norm median")
    for lo in range(0, int(st[-1]) + 1, 25):
        m = st[(st >= lo) & (st < lo + 25)]
        if not len(m): continue
        o = f"  {lo:>4}-{lo + 24:<4}" + "".join(f" {np.mean([by[s][K[l]] for s in m]):7.3f}" for l in LAY) + "  |"
        for n, t in refs.items():
            rs, rc = z[t + "/step"], z[t + "/ce"]
            q = np.isin(rs, m)
            o += f"  {rc[q, 11].mean():6.3f} {rc[q, 47].mean():6.3f}"
        gv = [g[s] for s in g if lo <= s < lo + 25]
        o += f"  | {np.median(gv):7.2f}" if gv else "  |"
        print(o, flush=True)
