"""Static row sums of HC's learned mixing, r_i = sum_o A0[i, o] with A0 = I + res, per sublayer, at the final checkpoint. The mean g is the uniform shrink; the spread r_i - g is what moves stream differences into the sum (sum' = g sum + Sigma_i (r_i - g) H_i). Usage: row_sums.py TAG ..."""
import glob, sys, numpy as np, tensorstore as ts
ROOT = "/scratch/gpfs/GROUP/USER/marin_store_big"
SP, TP = ROOT + "/speedrun/muonh-qwen3-130m-della4xh100-pls1", ROOT + "/tmp/checkpoints-temp/scratch/gpfs/GROUP/USER/marin_store_big/speedrun/muonh-qwen3-130m-della4xh100-pls1"
def rd(ck, key):
    s = ts.open({"driver": "zarr3", "kvstore": {"driver": "ocdbt", "base": {"driver": "file", "path": ck.rstrip("/") + "/"}, "path": f"model/transformer/layers/stacked/{key}/"}}).result()
    return np.asarray(s.read().result(), np.float64)
for tag in sys.argv[1:]:
    c = glob.glob(f"{SP}{tag}/*/checkpoints/step-*") + glob.glob(f"{TP}{tag}/*/checkpoints/step-*")
    ck = sorted(c, key=lambda p: int(p.rsplit("-", 1)[1]))[-1]
    R = np.stack([rd(ck, f"hyper/{j}/res") for j in (0, 1)], 1)   # (layer, j, stream, stream_out)
    m = R.shape[-1]; A = R + np.eye(m)
    r = A.sum(-1)   # (layer, j, stream): outflow of input stream i
    print(f"{tag} {ck.rsplit('/', 1)[1]} streams {m}")
    print(" layer: j=0 g, spread (max - min of r_i) | j=1 g, spread | cumulative log10 g")
    cg = np.cumsum(np.log10(np.abs(r.mean(-1))), axis=None).reshape(r.shape[0], 2)
    for l in list(range(1, 17)) + [20, 24, 28, 32, 36, 40, 44, 46, 47, 48]:
        i = l - 1
        print(f"  L{l:<3d} {r[i, 0].mean():6.3f} {r[i, 0].max() - r[i, 0].min():6.3f} | {r[i, 1].mean():6.3f} {r[i, 1].max() - r[i, 1].min():6.3f} | {cg[i, 1]:7.2f}   r(j=1) {np.round(r[i, 1], 2).tolist()}")
