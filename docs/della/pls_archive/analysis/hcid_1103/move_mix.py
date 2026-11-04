"""Static mixing read from a checkpoint without building the model (only the arrays hyper/{0,1}/res, a few KB).
depth_hc_res="move": A = sp(I + res) row-normalised, sp(a) = softplus(beta (a - shift)) / beta + 1e-6 I (depth_arch_qwen3.py), beta 50, shift 0.1. --hc: the plain formula A = I + res.
Prints by layer group: positive off-diagonal mass (sum over o != i of max(A[i, o], 0), mean over rows and sublayers), negative off-diagonal mass, share of off-diagonal entries of a = I + res
below the closing line (a < shift - 2 / beta, where softplus(beta (a - shift)) / beta is below 1.3e-3), mean diagonal, mean row sum, and the gain of the stream differences (second largest singular
value of A, mean over sublayers). Also saves the res arrays to mix_<tag>_<step>.npz beside this script, because a temporary checkpoint is deleted when the next one is saved.
Usage: move_mix.py [--hc] <tag after -pls1> ..."""
import glob, sys
from pathlib import Path
import numpy as np, tensorstore as ts
SP = "/scratch/gpfs/GROUP/USER/marin_store_big/speedrun/muonh-qwen3-130m-della4xh100-pls1"
TP = "/scratch/gpfs/GROUP/USER/marin_store_big/tmp/checkpoints-temp" + SP
BETA, SHIFT = 50.0, 0.1
GROUPS = [(1, 1), (2, 6), (7, 11), (12, 12), (13, 24), (25, 36), (37, 48)]
hc = "--hc" in sys.argv
for tag in [a for a in sys.argv[1:] if a != "--hc"]:
    cks = sorted(glob.glob(f"{TP}{tag}/*/checkpoints/step-*") + glob.glob(f"{SP}{tag}/*/checkpoints/step-*"), key=lambda p: int(p.rsplit("-", 1)[1]))
    if not cks:
        print(tag, "no checkpoint"); continue
    ck = cks[-1]
    step = int(ck.rsplit("-", 1)[1])
    res = []
    for j in (0, 1):
        s = ts.open({"driver": "zarr3", "kvstore": {"driver": "ocdbt", "base": {"driver": "file", "path": ck.rstrip("/") + "/"}, "path": f"model/transformer/layers/stacked/hyper/{j}/res/"}}).result()
        res.append(np.asarray(s.read().result(), np.float64))
    res = np.stack(res, 1)   # (layer, sublayer, s, so)
    np.savez(Path(__file__).parent / f"mix_{tag.strip('-')}_{step}.npz", res=res)
    m = res.shape[-1]; I = np.eye(m)
    print(f"== {tag} step {step} res {res.shape} ({'HC formula' if hc else 'move formula'})")
    print(" group: off+  off-  entries below closing line  mean diagonal  mean row sum  difference gain")
    for lo, hi in GROUPS:
        out = []
        for l in range(lo - 1, hi):
            for j in range(res.shape[1]):
                a0 = I + res[l, j]
                if hc:
                    A = a0
                else:
                    pos = np.logaddexp(0, BETA * (a0 - SHIFT)) / BETA + 1e-6 * I
                    A = pos / pos.sum(1, keepdims=True)
                off = A - np.diag(np.diag(A))
                low = (a0 < SHIFT - 2 / BETA)[~np.eye(m, dtype=bool)].mean()
                out.append((np.maximum(off, 0).sum() / m, -np.minimum(off, 0).sum() / m, low, np.diag(A).mean(), A.sum(1).mean(), np.linalg.svd(A, compute_uv=False)[1]))
        r = np.mean(out, 0)
        print(f"  L{lo}-{hi}: {r[0]:.3f} {r[1]:.3f} {r[2]:.2f} {r[3]:.3f} {r[4]:.3f} {r[5]:.3f}", flush=True)
