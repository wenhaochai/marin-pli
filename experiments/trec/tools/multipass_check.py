"""Fig 7 check: per multi-pass run, batch size, T, scored index range, and whether batch i and batch i + N/B (the same
sequences one pass later, if every pass reads the stream in the same order) have the same final-model loss."""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, "/scratch/gpfs/GROUP/USER/project/marin-trec/experiments/trec")
import build_site_data as b, predict
for _, names in b.MULTIPASS:
    for name in names:
        idx, loss = [], []
        for f in sorted(glob.glob(os.path.join(b.OUT, "runs", name, "[0-9]*.npz"))):
            z = np.load(f); idx.append(z["index"]); loss.append(z["seq_loss"].sum(1) / z["seq_weight"].sum(1)); T = int(z["total_steps"])
        idx, loss = np.concatenate(idx), np.concatenate(loss)
        cfg = json.load(open(os.path.join(predict.CACHE, f"{name}.json")))["config"]
        B = cfg["trainer"]["train_batch_size"]; P = b.N_SEQ / B
        d = dict(zip(idx.tolist(), loss.tolist()))
        for lag in (int(np.floor(P)), int(np.round(P)), int(np.ceil(P)), 64):
            pr = [(d[i], d[i + lag]) for i in idx.tolist() if i + lag in d]
            if pr:
                a = np.array(pr); print(f"  lag {lag}: n {len(a)} r {np.corrcoef(a.T)[0,1]:.4f} mean|diff| {np.abs(a[:,0]-a[:,1]).mean():.5f}")
        tl, reads = b.last_read(name, idx, T)
        print(name, "B", B, "seq", cfg.get("model", {}).get("max_seq_len") or cfg.get("model", {}).get("seq_len"), "T", T, "N/B", round(P, 1),
              "passes", round(T * B / b.N_SEQ, 3), "idx", idx.min(), idx.max(), "n>=T", int((idx >= T).sum()),
              "reads", {int(r): int((reads == r).sum()) for r in np.unique(reads)}, "last-read range", round(tl.min(), 3), round(tl.max(), 3), flush=True)
