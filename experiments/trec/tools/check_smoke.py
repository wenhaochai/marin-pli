"""Compare the replayed loss of the smoke's training batches with the W&B train/loss of the same steps."""
import glob, json, sys
import numpy as np
OUT = sys.argv[1]
CACHE = "/scratch/gpfs/GROUP/USER/project/trec/cache/wandb"
for run in ("muonh-qwen3-130m-della4xh100-d12", "muonh-qwen3-300m-della4xh100-s1-rerun"):
    z = np.load(sorted(glob.glob(f"{OUT}/{run}/*.npz"))[0])
    idx, L, W = z["index"], z["seq_loss"], z["seq_weight"]
    T = int(z["total_steps"])
    mean = L.sum(1) / W.sum(1)
    d = json.load(open(f"{CACHE}/{run}.json"))
    s, v = np.array(d["train/loss"]).T
    wb = dict(zip(s.astype(int), v))
    print(run, "T", T)
    diffs = []
    for i, m in zip(idx, mean):
        tag = "train" if i < T else "tail"
        w = wb.get(int(i))
        if w is None:
            print(f"  {tag} {i:6d} replay {m:.4f}")
        else:
            diffs.append((i, m - w))
            print(f"  {tag} {i:6d} replay {m:.4f} wandb {w:.4f} diff {m - w:+.4f}")
    last = [dd for i, dd in diffs if i >= T - 12]
    first = [dd for i, dd in diffs if i < 4]
    print(f"  last 12: mean diff {np.mean(last):+.4f}, max |diff| {np.max(np.abs(last)):.4f}; first 4: {np.round(first, 4)}")
