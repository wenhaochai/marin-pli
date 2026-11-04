"""Interim read of every running run of the 2 x 2 on HC's mixing: train CE at step windows next to finished references (d48g4.npz), MLP gain from the newest checkpoint (weights only), series saved to live_1104c.npz for the status figure. W&B state may read "crashed" for a live run.
Usage: live_1104c.py"""
import glob, numpy as np, tensorstore as ts, wandb
from pathlib import Path
api = wandb.Api(timeout=300)
P = "reself/marin-della/muonh-qwen3-130m-della4xh100-pls1"
z = np.load("/scratch/gpfs/GROUP/USER/tmp/pls_blog/d48g4.npz", allow_pickle=True)
ROOT = "/scratch/gpfs/GROUP/USER/marin_store_big"
SP, TP = ROOT + "/speedrun/muonh-qwen3-130m-della4xh100-pls1", ROOT + "/tmp/checkpoints-temp/scratch/gpfs/GROUP/USER/marin_store_big/speedrun/muonh-qwen3-130m-della4xh100-pls1"
LAY = [2, 12, 24, 36, 48]
K = {l: f"train/pls/L{l - 1}" for l in LAY}
HCO, IDO, HCS, IDS = "-sep-bbfrozen-ahc-fa4thd-vs-d48", "-sep-bbfrozen-ahc-hcid-fa4thd-vs-d48", "-sep-ahc-fa4thd-vs-d48", "-sep-ahc-hcid-fa4thd-vs-d48"
PLS = "-sep-apreln-fa4thd-vs-d48"
ORD = [("w972 only shrinks", "-sep-bbfrozen-ahc-hcshrink-fa4thd-vs-d48"), ("w1002 half dose", "-sep-bbfrozen-ahc-hcshrink-sp0.5-fa4thd-vs-d48"), ("w1000 only moves", "-sep-bbfrozen-ahc-hcmove-fa4thd-vs-d48"), ("w1006 keeps the sum", "-sep-bbfrozen-ahc-hcsum-hcf32-fa4thd-vs-d48")]
SEP = [("w973 only shrinks", "-sep-ahc-hcshrink-fa4thd-vs-d48"), ("w1003 half dose", "-sep-ahc-hcshrink-sp0.5-fa4thd-vs-d48"), ("w1001 only moves", "-sep-ahc-hcmove-fa4thd-vs-d48"), ("w1007 keeps the sum", "-sep-ahc-hcsum-hcf32-fa4thd-vs-d48"), ("w962 learned scalar", "-sep-ahc-hcscalar-fa4thd-vs-d48"), ("w1091 mixing = I, bound", "-sep-ahc-hcid-fb100-fa4thd-vs-d48"), ("w1005 Pre-LN fixed gains", "-sep-apreln-plleak-fa4thd-vs-d48")]
WIN = [(100, 200), (200, 300), (300, 400), (400, 500), (500, 600), (1000, 1100), (1500, 1600), (2100, 2197), (2500, 2600), (3000, 3100)]
def gains(tag):
    c = glob.glob(f"{SP}{tag}/*/checkpoints/step-*") + glob.glob(f"{TP}{tag}/*/checkpoints/step-*")
    ck = sorted(c, key=lambda p: int(p.rsplit("-", 1)[1]))[-1]
    def rd(key):
        s = ts.open({"driver": "zarr3", "kvstore": {"driver": "ocdbt", "base": {"driver": "file", "path": ck.rstrip("/") + "/"}, "path": f"model/transformer/layers/stacked/{key}/"}}).result()
        return np.asarray(s.read().result(), np.float64)
    g, u, d, ln = rd("mlp/gate_proj/weight"), rd("mlp/up_proj/weight"), rd("mlp/down_proj/weight"), rd("ln_2/weight")
    a = np.array([np.linalg.norm(g[l] * ln[l], 2) * np.linalg.norm(u[l] * ln[l], 2) * np.linalg.norm(d[l], 2) for l in range(g.shape[0])])
    return int(ck.rsplit("-", 1)[1]), a, ln.max()
save = {}
for group, refs, runs in (("ordinary", [("HC", HCO), ("I", IDO)], ORD), ("separate heads", [("HC", HCS), ("I", IDS)], SEP)):
    print(f"\n#### {group}")
    D = {}
    for name, tag in runs:
        r = api.run(P + tag)
        by = {}
        for x in r.scan_history(keys=["_step", *K.values()], page_size=2000):
            by[x["_step"]] = x
        st = np.array(sorted(by))
        ce = np.array([[by[s][K[l]] for l in LAY] for s in st], float)
        D[name] = (st, ce)
        save[name + "/step"], save[name + "/ce"] = st, ce
        step, a, lnmax = gains(tag)
        print(f"{name}: last step {st[-1]}; gain at step {step}: {[(int(i) + 1, int(a[i])) for i in np.argsort(-a)[:3]]}, median {np.median(a):.0f}, above 200: {int((a > 200).sum())}, max ln_2 {lnmax:.2f}", flush=True)
    np.savez(Path(__file__).parent / "live_1104c.npz", lay=np.array(LAY), **save)
    for li, l in enumerate(LAY):
        if l not in (12, 48):
            continue
        print(f"-- layer {l}: window | " + " ".join(f"{n:>8s}" for n, _ in refs) + " | " + " ".join(f"{n.split()[0]:>7s}" for n, _ in runs))
        for lo, hi in WIN:
            o = f"{lo:>5}-{hi - 1:<5}| "
            o += " ".join(f"{z[t + '/ce'][(z[t + '/step'] >= lo) & (z[t + '/step'] < hi), l - 1].mean():8.3f}" for _, t in refs) + " | "
            vals = []
            for n, _ in runs:
                st, ce = D[n]
                m = (st >= lo) & (st < hi)
                vals.append(f"{ce[m, li].mean():7.3f}" if m.sum() >= 0.5 * (hi - lo) else "      -")
            print(o + " ".join(vals), flush=True)
print("\ngroup of separate heads, cost so far at layer 48 (separate heads minus ordinary, same steps)")
for a, b, n in (("w973 only shrinks", "w972 only shrinks", "only shrinks"), ("w1003 half dose", "w1002 half dose", "half dose"), ("w1001 only moves", "w1000 only moves", "only moves"), ("w1007 keeps the sum", "w1006 keeps the sum", "keeps the sum")):
    sa, ca = save[a + "/step"], save[a + "/ce"][:, 4]
    sb, cb = save[b + "/step"], save[b + "/ce"][:, 4]
    o = f"  {n:14s}"
    for lo, hi in WIN:
        ma, mb = (sa >= lo) & (sa < hi), (sb >= lo) & (sb < hi)
        if ma.sum() >= 0.5 * (hi - lo) and mb.sum() >= 0.5 * (hi - lo):
            o += f" {lo}: {ca[ma].mean() - cb[mb].mean():+.3f}"
    print(o, flush=True)
