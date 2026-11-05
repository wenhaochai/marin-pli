"""w1006 (keeps the sum, ordinary): static mixing at the newest checkpoint, A = A0 - (row sum of A0 - 1) / m, A0 = I + res. Off-diagonal positive and negative mass per row by layer group, next to HC's final mixing (no normalisation)."""
import glob, numpy as np, tensorstore as ts
ROOT = "/scratch/gpfs/GROUP/USER/marin_store_big"
SP, TP = ROOT + "/speedrun/muonh-qwen3-130m-della4xh100-pls1", ROOT + "/tmp/checkpoints-temp/scratch/gpfs/GROUP/USER/marin_store_big/speedrun/muonh-qwen3-130m-della4xh100-pls1"
def rd(ck, key):
    s = ts.open({"driver": "zarr3", "kvstore": {"driver": "ocdbt", "base": {"driver": "file", "path": ck.rstrip("/") + "/"}, "path": f"model/transformer/layers/stacked/{key}/"}}).result()
    return np.asarray(s.read().result(), np.float64)
for tag, norm in (("-sep-bbfrozen-ahc-hcsum-hcf32-fa4thd-vs-d48", True), ("-sep-bbfrozen-ahc-fa4thd-vs-d48", False)):
    ck = sorted(glob.glob(f"{SP}{tag}/*/checkpoints/step-*") + glob.glob(f"{TP}{tag}/*/checkpoints/step-*"), key=lambda p: int(p.rsplit("-", 1)[1]))[-1]
    R = np.stack([rd(ck, f"hyper/{j}/res") for j in (0, 1)], 1); m = R.shape[-1]
    A = R + np.eye(m)
    if norm:
        A = A - (A.sum(-1, keepdims=True) - 1) / m
    off = A - A * np.eye(m)
    print(tag, ck.rsplit("/", 1)[1], "group: off+ off- mean diag mean |row sum - 1|")
    for lo, hi in ((1, 1), (2, 6), (7, 11), (12, 12), (13, 24), (25, 36), (37, 48)):
        a, o = A[lo - 1:hi], off[lo - 1:hi]
        print(f"  L{lo}-{hi:<3d} {np.maximum(o, 0).sum(-1).mean():6.3f} {-np.minimum(o, 0).sum(-1).mean():6.3f} {np.diagonal(a, axis1=-2, axis2=-1).mean():6.3f} {np.abs(a.sum(-1) - 1).mean():6.3f}")
