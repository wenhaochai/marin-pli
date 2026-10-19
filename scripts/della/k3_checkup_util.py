"""GPU utilization of a k3_checkup_v3 run: OUT/gpu_util.csv (nvidia-smi timestamp, index, utilization.gpu, memory.used)
against the layer loop's wall-clock window from the rank files (t_start of layer 1, so layer 0's one-time kernel
compile is left out, to the last t_end over ranks). Prints the whole-run mean, the in-loop mean, and per GPU.
Usage: python k3_checkup_util.py OUT_DIR"""
import json
import os
import sys
from datetime import datetime

out = sys.argv[1]
rows = []
for line in open(os.path.join(out, "gpu_util.csv")):
    p = [x.strip() for x in line.split(",")]
    try:
        rows.append((datetime.strptime(p[0], "%Y/%m/%d %H:%M:%S.%f").timestamp(), int(p[1]), float(p[2])))
    except (ValueError, IndexError):
        continue
ranks = [json.load(open(os.path.join(out, f))) for f in os.listdir(out) if f.startswith("rank") and f.endswith(".json")]
lay = [r["layers"] for r in ranks if len(r["layers"]) > 1]
mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
print(f"GPU utilization whole run: {mean([u for _, _, u in rows]):.1f}% ({len(rows)} samples)")
if lay:
    a = min(l[1]["t_start"] for l in lay)
    b = max(l[-1]["t_end"] for l in lay)
    win = [(g, u) for t, g, u in rows if a <= t <= b]
    per = {g: mean([u for gg, u in win if gg == g]) for g in sorted({g for g, _ in win})}
    n = len(lay[0]) - 1
    print(f"GPU utilization in the layer loop (layers 1-{n}, {b - a:.0f} s, {(b - a) / max(n, 1):.1f} s per layer): "
          f"{mean([u for _, u in win]):.1f}% ({len(win)} samples); per GPU " + " ".join(f"{g}:{v:.0f}" for g, v in per.items()))
