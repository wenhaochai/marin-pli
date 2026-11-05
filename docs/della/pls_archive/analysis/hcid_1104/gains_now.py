"""MLP gain per layer from the newest checkpoint, by power iteration on 4 threads: sigma_max(gate diag(ln_2)) sigma_max(up diag(ln_2)) sigma_max(down). Usage: gains_now.py TAG ..."""
import glob, os, sys
os.environ.setdefault("OMP_NUM_THREADS", "4"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import numpy as np, tensorstore as ts
ROOT = "/scratch/gpfs/GROUP/USER/marin_store_big"
SP, TP = ROOT + "/speedrun/muonh-qwen3-130m-della4xh100-pls1", ROOT + "/tmp/checkpoints-temp/scratch/gpfs/GROUP/USER/marin_store_big/speedrun/muonh-qwen3-130m-della4xh100-pls1"
def smax(M):
    v = np.ones(M.shape[1]) / np.sqrt(M.shape[1]) + 1e-3 * np.cos(np.arange(M.shape[1]))
    for _ in range(40):
        v = M.T @ (M @ v); v /= np.linalg.norm(v)
    return np.linalg.norm(M @ v)
for tag in sys.argv[1:]:
    c = glob.glob(f"{SP}{tag}/*/checkpoints/step-*") + glob.glob(f"{TP}{tag}/*/checkpoints/step-*")
    ck = sorted(c, key=lambda p: int(p.rsplit("-", 1)[1]))[-1]
    def rd(key):
        s = ts.open({"driver": "zarr3", "kvstore": {"driver": "ocdbt", "base": {"driver": "file", "path": ck.rstrip("/") + "/"}, "path": f"model/transformer/layers/stacked/{key}/"}}).result()
        return np.asarray(s.read().result(), np.float32)
    g, u, d, ln = rd("mlp/gate_proj/weight"), rd("mlp/up_proj/weight"), rd("mlp/down_proj/weight"), rd("ln_2/weight")
    a = np.array([smax(g[l] * ln[l]) * smax(u[l] * ln[l]) * smax(d[l]) for l in range(g.shape[0])])
    top = np.argsort(-a)[:5]
    print(f"{tag} {ck.rsplit('/', 1)[1]}: top {[(int(i) + 1, int(a[i])) for i in top]}, median {np.median(a):.0f}, above 200: {int((a > 200).sum())}, max ln_2 {ln.max():.2f} at layer {int(np.unravel_index(ln.argmax(), ln.shape)[0]) + 1}", flush=True)
