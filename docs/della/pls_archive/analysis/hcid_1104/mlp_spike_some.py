"""MLP gain (mlp_spike_all.py's definition, weights only) of the named runs at their newest checkpoint. Usage: mlp_spike_some.py <tag after -pls1> ..."""
import glob, sys, numpy as np, tensorstore as ts
ROOT = "/scratch/gpfs/GROUP/USER/marin_store_big"
SP, TP = ROOT + "/speedrun/muonh-qwen3-130m-della4xh100-pls1", ROOT + "/tmp/checkpoints-temp/scratch/gpfs/GROUP/USER/marin_store_big/speedrun/muonh-qwen3-130m-della4xh100-pls1"
for tag in sys.argv[1:]:
    c = glob.glob(f"{SP}{tag}/*/checkpoints/step-*") + glob.glob(f"{TP}{tag}/*/checkpoints/step-*")
    ck = sorted(c, key=lambda p: int(p.rsplit("-", 1)[1]))[-1]
    def rd(k):
        s = ts.open({"driver": "zarr3", "kvstore": {"driver": "ocdbt", "base": {"driver": "file", "path": ck.rstrip("/") + "/"}, "path": f"model/transformer/layers/stacked/{k}/"}}).result()
        return np.asarray(s.read().result(), np.float64)
    g, u, d, ln = rd("mlp/gate_proj/weight"), rd("mlp/up_proj/weight"), rd("mlp/down_proj/weight"), rd("ln_2/weight")
    a = np.array([np.linalg.norm(g[l] * ln[l], 2) * np.linalg.norm(u[l] * ln[l], 2) * np.linalg.norm(d[l], 2) for l in range(g.shape[0])])
    print(f"{tag} {ck.rsplit('-', 1)[1]}: top gains {[(int(i) + 1, int(a[i])) for i in np.argsort(-a)[:5]]}, median {np.median(a):.1f}, layers above 200: {int((a > 200).sum())}, max ln_2 {ln.max():.2f}", flush=True)
