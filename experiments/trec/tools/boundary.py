"""Figure 7 read-count edge (owner popup 2026-10-28): for each run that reads its data more than once, the loss of the
batches last read just below and just above the edge where the read count changes (batch-level last read,
build_site_data.last_read), in windows of 0.01 and 0.02 of training, mean +- SE and the difference; and the range of
the by-t moving mean. Source of the page's edge numbers (1.2b 0.833: <= 0.003 nats within 1.4 SE; 130m 0.769: <= 0.0013).

    python tools/boundary.py     # vis node
"""
import sys, glob, os, numpy as np
sys.path.insert(0, "/scratch/gpfs/GROUP/USER/project/marin-trec/experiments/trec")
import build_site_data as b
for name, _ in [("muonh-qwen3-1_2b-della8xh100-pls0-noev",0),("muonh-qwen3-130m-della4xh100-tpp200",0),("muonh-qwen3-130m-della4xh100-tpp200-s1",0),("muonh-qwen3-520m-della8xh100-pls0-noev",0)]:
    idx, loss = [], []
    for f in sorted(glob.glob(os.path.join(b.OUT, "runs", name, "[0-9]*.npz"))):
        z = np.load(f); idx.append(z["index"]); loss.append(z["seq_loss"].sum(1) / z["seq_weight"].sum(1)); T = int(z["total_steps"])
    idx, loss = np.concatenate(idx), np.concatenate(loss)
    tl, r = b.last_read(name, idx, T)
    edge = tl[r == r.max()].min()
    for wdt in (0.01, 0.02):
        lo = loss[(tl >= edge - wdt) & (tl < edge)]; hi = loss[(tl >= edge) & (tl < edge + wdt)]
        print(name[12:], f"edge {edge:.3f} w {wdt}: below {lo.mean():.4f}+-{lo.std(ddof=1)/np.sqrt(len(lo)):.4f} (n {len(lo)}) above {hi.mean():.4f}+-{hi.std(ddof=1)/np.sqrt(len(hi)):.4f} (n {len(hi)}) diff {hi.mean()-lo.mean():+.4f}")
    # by-t step for tpp200 at t ~ 0.23
    x = idx / T
    print("   by t, range", np.ptp(np.convolve(loss[np.argsort(tl)], np.ones(64)/64, mode='valid')).round(4))
