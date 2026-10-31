"""Q8 first look: fraction of the imprint of batches 2232-2479 left in each branch (w658 a, w659 b, w660 c when done)."""
import glob, os, sys
import numpy as np
sys.path.insert(0, "/scratch/gpfs/GROUP/USER/project/marin-trec/experiments/trec")
import build_site_data as b
base = os.path.join(b.OUT, "ck", "branch")
pre, at = b._range_score(f"{base}/base/step-2232"), b._range_score(f"{base}/base/step-2480")
e0 = ((at[0] - at[1]) - (pre[0] - pre[1])).mean()
print(f"imprint at the branch point: {e0:.4f} nats")
for run, label, lr, _ in b.BRANCH:
    rows = []
    for st in sorted(int(d.split("-")[-1]) for d in glob.glob(f"{base}/{run}/step-*")):
        r = b._range_score(f"{base}/{run}/step-{st}")
        if r is None: break
        a = st - 2480; rows.append(f"{a}:{((r[0] - r[1]) - (pre[0] - pre[1])).mean() / e0:.3f}(Σlr {a * lr:.2f})")
    print(label, " ".join(rows))
