"""Interim read of the running 2 x 2 runs at step windows, next to finished references at the same steps (d48g4.npz); design options from W&B; MLP gain from the newest checkpoint (mlp_spike_all.py's gain: weights only).
Usage: live_1104b.py"""
import glob, numpy as np, tensorstore as ts, wandb
api = wandb.Api(timeout=300)
P = "reself/marin-della/muonh-qwen3-130m-della4xh100-pls1"
z = np.load("/scratch/gpfs/GROUP/USER/tmp/pls_blog/d48g4.npz", allow_pickle=True)
ROOT = "/scratch/gpfs/GROUP/USER/marin_store_big"
SP, TP = ROOT + "/speedrun/muonh-qwen3-130m-della4xh100-pls1", ROOT + "/tmp/checkpoints-temp/scratch/gpfs/GROUP/USER/marin_store_big/speedrun/muonh-qwen3-130m-della4xh100-pls1"
OPT = ["depth_arch", "depth_hc_res", "depth_shrink_power", "depth_preln_leak", "depth_hc_state_f32"]
LAY = [2, 12, 24, 48]
K = {l: f"train/pls/L{l - 1}" for l in LAY}
HCO, IDO, HCS, IDS = "-sep-bbfrozen-ahc-fa4thd-vs-d48", "-sep-bbfrozen-ahc-hcid-fa4thd-vs-d48", "-sep-ahc-fa4thd-vs-d48", "-sep-ahc-hcid-fa4thd-vs-d48"
PLO, PLS = "-sep-bbfrozen-apreln-fa4thd-vs-d48", "-sep-apreln-fa4thd-vs-d48"
RUNS = [("w1000 only moves, ordinary", "-sep-bbfrozen-ahc-hcmove-fa4thd-vs-d48", [("HC", HCO), ("mixing = I", IDO)]),
        ("w1001 only moves, separate heads", "-sep-ahc-hcmove-fa4thd-vs-d48", [("HC", HCS), ("mixing = I", IDS)]),
        ("w1002 half dose, ordinary", "-sep-bbfrozen-ahc-hcshrink-sp0.5-fa4thd-vs-d48", [("HC", HCO), ("mixing = I", IDO)]),
        ("w1003 half dose, separate heads", "-sep-ahc-hcshrink-sp0.5-fa4thd-vs-d48", [("HC", HCS), ("mixing = I", IDS)]),
        ("w1004 Pre-LN fixed gains, ordinary", "-sep-bbfrozen-apreln-plleak-fa4thd-vs-d48", [("Pre-LN", PLO)]),
        ("w1005 Pre-LN fixed gains, separate heads", "-sep-apreln-plleak-fa4thd-vs-d48", [("Pre-LN", PLS)]),
        ("w962 learned scalar, separate heads", "-sep-ahc-hcscalar-fa4thd-vs-d48", [("HC", HCS), ("mixing = I", IDS)])]
WIN = [(100, 125), (200, 300), (500, 600), (1000, 1100), (1500, 1600), (2100, 2197), (2500, 2600)]
def find(c, k):
    if isinstance(c, dict):
        for a, b in c.items():
            if a == k:
                return b
            v = find(b, k)
            if v is not None:
                return v
    return None
def gains(tag):
    c = glob.glob(f"{SP}{tag}/*/checkpoints/step-*") + glob.glob(f"{TP}{tag}/*/checkpoints/step-*")
    ck = sorted(c, key=lambda p: int(p.rsplit("-", 1)[1]))[-1]
    def rd(key):
        s = ts.open({"driver": "zarr3", "kvstore": {"driver": "ocdbt", "base": {"driver": "file", "path": ck.rstrip("/") + "/"}, "path": f"model/transformer/layers/stacked/{key}/"}}).result()
        return np.asarray(s.read().result(), np.float64)
    g, u, d, ln = rd("mlp/gate_proj/weight"), rd("mlp/up_proj/weight"), rd("mlp/down_proj/weight"), rd("ln_2/weight")
    a = np.array([np.linalg.norm(g[l] * ln[l], 2) * np.linalg.norm(u[l] * ln[l], 2) * np.linalg.norm(d[l], 2) for l in range(g.shape[0])])
    return int(ck.rsplit("-", 1)[1]), a, ln.max()
for name, tag, refs in RUNS:
    r = api.run(P + tag)
    by = {}
    for x in r.scan_history(keys=["_step", *K.values()], page_size=2000):
        by[x["_step"]] = x
    st = np.array(sorted(by))
    ce = np.array([[by[s][K[l]] for l in LAY] for s in st], float)
    print(f"== {name}: state {r.state}, last step {st[-1]}; " + "; ".join(f"{o}={find(dict(r.config), o)}" for o in OPT), flush=True)
    step, a, lnmax = gains(tag)
    top = [(int(i) + 1, int(a[i])) for i in np.argsort(-a)[:4]]
    print(f"   MLP gain at step {step}: largest {top}, median {np.median(a):.1f}, layers above 200: {int((a > 200).sum())}, max ln_2 {lnmax:.2f}", flush=True)
    print("   window    " + " ".join(f"L{l:<6d}" for l in LAY) + " | " + " | ".join(n + " " + " ".join(f"L{l}" for l in LAY) for n, _ in refs))
    for lo, hi in WIN:
        m = (st >= lo) & (st < hi)
        if m.sum() < 0.5 * (hi - lo):
            continue
        o = f"   {lo:>5}-{hi - 1:<5}" + " ".join(f"{ce[m, i].mean():7.3f}" for i in range(4))
        for _, t in refs:
            rs, rc = z[t + "/step"], z[t + "/ce"]
            q = (rs >= lo) & (rs < hi)
            o += " | " + " ".join(f"{rc[q, l - 1].mean():.3f}" for l in LAY)
        print(o, flush=True)
