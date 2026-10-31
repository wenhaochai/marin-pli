"""Fig 5 second seed (owner approved 2026-10-27): per-seed low point of unique minus 8-pass, as fig_tokenizers."""
import sys
import numpy as np
sys.path.insert(0, "/scratch/gpfs/GROUP/USER/project/marin-trec/experiments/trec")
import build_site_data as b, predict
P = "muonh-qwen3-300m-della4xh100"
PAIRS = [("300m", "128K s1", "-s1-rerun", "-rep8-s1", 128000), ("300m-8k", "8K s1", "-v8k-s1", "-v8k-rep8-s1", 8000),
         ("300m-8k", "8K s0", "-v8k", "-v8k-rep8", 8000), ("300m-bytes", "bytes s0", "-bytes", "-bytes-rep8", 256),
         ("300m-bytes", "bytes s1", "-bytes-s1", "-bytes-rep8-s1", 256)]
cur = {}
for model, lab, u, r, k in PAIRS:
    a, z = np.load(f"{b.W438}/{P}{u}.npz"), np.load(f"{b.W438}/{P}{r}.npz")
    byf = np.load(b.BYTES.format(k)); by = byf["seq_bytes"]
    steps, T = a["steps"], int(a["total_steps"])
    assert np.array_equal(steps, z["steps"]) and np.array_equal(steps, byf["steps"]) and T == predict.TOTAL[model]
    bpb = lambda q: q["seq_loss"].sum(1) / np.log(2) / by.sum(1)
    d = bpb(a) - bpb(z)
    tr = steps < T
    unread = (steps >= T) & (steps != steps[steps >= T].min())
    x = steps[tr] / T
    sm = np.convolve(d[tr], np.ones(16) / 16, mode="valid") - d[unread].mean()
    xm = np.convolve(x, np.ones(16) / 16, mode="valid")
    f = xm > 0.125 + 8 / len(x)
    xm, sm = xm[f], sm[f]
    i = int(np.argmin(sm))
    c = predict.coefficients(predict.lr_schedule(model), "momentum", model)
    print(f"{lab:9s} low t {xm[i]:.3f} at {sm[i]:+.4f} bpb; unread d {d[unread].mean():+.4f} (sd/sqrt n {d[unread].std(ddof=1)/np.sqrt(unread.sum()):.4f}); pred low (m 3.3) {predict.valley(c, 3.3):.3f}; first after subset {sm[:8].mean():+.4f}, last {sm[-8:].mean():+.4f}")
    cur[lab] = (xm, sm)
for t in ("8K", "bytes"):
    (x0, s0), (x1, s1) = [cur[k] for k in cur if k.startswith(t)]
    print(t, "seed curves corr", np.corrcoef(s0, s1)[0, 1].round(3), "rms diff", np.sqrt(np.mean((s0 - s1) ** 2)).round(4), "mean curve low", x0[np.argmin((s0 + s1) / 2)].round(3))
