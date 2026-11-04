"""Random windows of the real training data of the 130m runs: FineWeb-Edu 10B (W&B config data.train_weights {"fineweb-edu-10B": 1}), same windows for every run. Returns N * B windows of T tokens, flat.
The zarr array is declared 1.8e16 long and reads as zeros past the end; offsets[0] is the number of documents and offsets[offsets[0]] the number of tokens (10,000,000,738)."""
import numpy as np, tensorstore as ts
P = "/scratch/gpfs/GROUP/USER/marin_store_big/fineweb-edu-10B/2026.06.28/train/input_ids"
def load(B, T, N, seed=0):
    a = ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": P + "/data"}}).result()
    o = ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": P + "/offsets"}}).result()
    n = int(np.asarray(o[int(np.asarray(o[0].read().result()))].read().result()))
    off = np.random.default_rng(seed).integers(0, n - T, size=N * B)
    x = np.concatenate([np.asarray(a[int(i): int(i) + T].read().result(), np.int64) for i in off])
    assert x.size == N * B * T and (x == 0).mean() < 0.01, (x.size, float((x == 0).mean()))
    return x
