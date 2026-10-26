"""Where the small device-to-host copies come from (speed work 2026-10-26): joins each 4-byte D2H copy on GPU 0 to the
host API call with the same correlation id, then names the host events that enclose that call on its thread (XLA thunk
and executable annotations). Usage: trace_d2h_origin.py <profile dir> [steps]"""
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
want = {}
for line in gpu.lines:
    for e in line.events:
        if e.name.startswith("MemcpyD2H"):
            st = dict(e.stats)
            if "size:4 " in str(st.get("memcpy_details", "")) + " ":
                want[str(st.get("correlation_id"))] = e.start_ns
print(len(want), "4-byte D2H copies on GPU 0")
origin = collections.Counter()
apis = collections.Counter()
for p in pd.planes:
    if not p.name.startswith("/host:CPU"):
        continue
    for line in p.lines:
        evs = sorted(((e.start_ns, e.start_ns + e.duration_ns, e.name, dict(e.stats)) for e in line.events),
                     key=lambda x: (x[0], -x[1]))
        stack = []
        for s, t, n, st in evs:
            while stack and stack[-1][1] <= s:
                stack.pop()
            cid = str(st.get("correlation_id", ""))
            if cid in want:
                apis[n[:40]] += 1
                encl = [x[2][:60] for x in stack if not x[2].startswith("$")][-3:]
                origin[" < ".join(reversed(encl))] += 1
            stack.append((s, t, n))
print("host API calls:", apis.most_common(5))
for k, v in origin.most_common(20):
    print(f"{v / steps:7.1f} per step  {k}")
