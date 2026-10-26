"""The GPU-0 work around each small device-to-host copy (speed work 2026-10-26): for the 4-byte D2H copies of one
step, prints the copy's time, the kernel that ends just before it on any stream, the idle time after it, and the kernel
that starts next. Usage: trace_d2h_context.py <profile dir> [first] [count]"""
import re
import sys
from pathlib import Path

from jax.profiler import ProfileData

prof = Path(sys.argv[1])
first = int(sys.argv[2]) if len(sys.argv) > 2 else 0
count = int(sys.argv[3]) if len(sys.argv) > 3 else 40
xp = sorted(prof.rglob("*.xplane.pb"), key=lambda p: p.stat().st_mtime)[-1]
pd = ProfileData.from_file(str(xp))
gpu = next(p for p in pd.planes if re.search(r"/device:GPU:0\b", p.name))
ev, d2h = [], []
for line in gpu.lines:
    for e in line.events:
        st = dict(e.stats)
        tag = str(st.get("hlo_op", "")) or e.name
        ev.append((e.start_ns, e.start_ns + e.duration_ns, f"{e.name[:38]} [{str(st.get('hlo_op', ''))[:28]}] <{line.name[:12]}>"))
        if e.name.startswith("MemcpyD2H") and "size:4 " in str(st.get("memcpy_details", "")) + " ":
            d2h.append((e.start_ns, e.start_ns + e.duration_ns))
ev.sort()
t0 = ev[0][0]
d2h.sort()
for s, t in d2h[first:first + count]:
    before = max((x for x in ev if x[1] <= s and not x[2].startswith("MemcpyD2H")), key=lambda x: x[1], default=None)
    after = min((x for x in ev if x[0] >= t and not x[2].startswith("MemcpyD2H")), key=lambda x: x[0], default=None)
    print(f"{(s - t0) / 1e6:9.2f} ms  before: {before[2] if before else '-':80s} gap {((after[0] - t) / 1e3) if after else -1:7.1f} us  after: {after[2] if after else '-'}")
