"""Small device-to-host copies on GPU 0 of a profile (speed work 2026-10-26): a host-driven while loop reads its
predicate back every iteration (a 1-byte D2H), which leaves the GPU idle between iterations; captured loops (CUDA-graph
conditional nodes) do not. Counts the D2H copies by size and by the name / hlo_op of the copy, per step of the trace.
Usage: trace_d2h.py <profile dir> [steps]"""
import collections
import re
import sys
from pathlib import Path

from jax.profiler import ProfileData

prof = Path(sys.argv[1])
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 3
xp = sorted(prof.rglob("*.xplane.pb"), key=lambda p: p.stat().st_mtime)[-1]
pd = ProfileData.from_file(str(xp))
gpu = next(p for p in pd.planes if re.search(r"/device:GPU:0\b", p.name))
cnt = collections.Counter()
for line in gpu.lines:
    for e in line.events:
        if not e.name.startswith("MemcpyD2H"):
            continue
        st = dict(e.stats)
        size = re.search(r"size:(\d+)", str(st.get("memcpy_details", "")))
        key = (size.group(1) if size else "?", str(st.get("hlo_op", ""))[:40], str(st.get("name", ""))[-70:])
        cnt[key] += 1
for (size, op, name), n in cnt.most_common(25):
    print(f"{n / steps:7.1f} per step  size {size:>6}  {op:40s} {name}")
