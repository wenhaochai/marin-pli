"""The static part of HC's learned mixing at layer 12, A0 = I + res, both sublayers, final checkpoint of the ordinary 48-layer run; used by the blog's schematic and film. Row i, column o: how much of input stream i goes into output stream o."""
import glob, json, numpy as np, tensorstore as ts
ROOT = "/scratch/gpfs/GROUP/USER/marin_store_big"
SP, TP = ROOT + "/speedrun/muonh-qwen3-130m-della4xh100-pls1", ROOT + "/tmp/checkpoints-temp/scratch/gpfs/GROUP/USER/marin_store_big/speedrun/muonh-qwen3-130m-della4xh100-pls1"
t = "-sep-bbfrozen-ahc-fa4thd-vs-d48"
ck = sorted(glob.glob(f"{SP}{t}/*/checkpoints/step-*") + glob.glob(f"{TP}{t}/*/checkpoints/step-*"), key=lambda p: int(p.rsplit("-", 1)[1]))[-1]
print(ck.rsplit("/", 1)[1])
import subprocess
out = {}
for j, name in enumerate(("attention", "mlp")):
    for key in (f"hyper/{j}/res", f"hyper_{j}/res", f"hyper/item_{j}/res"):
        try:
            s = ts.open({"driver": "zarr3", "kvstore": {"driver": "ocdbt", "base": {"driver": "file", "path": ck.rstrip("/") + "/"}, "path": f"model/transformer/layers/stacked/{key}/"}}).result()
            a = np.asarray(s.read().result(), np.float64); print(key, a.shape)
            A = np.eye(4) + a[11]
            out[name] = np.round(A, 3).tolist(); print(name, "row sums", np.round(A.sum(1), 2).tolist()); print(np.round(A, 2)); break
        except Exception as e:
            continue
json.dump(out, open("a0_l12.json", "w"))
