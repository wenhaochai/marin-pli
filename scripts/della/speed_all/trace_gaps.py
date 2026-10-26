"""Idle gaps on GPU 0 of a JAX profiler trace (speed work 2026-10-26): where the 8% idle of the production-flag profile
(w286, CUDA graphs on) sits. Merges all kernel / memcpy intervals on the device's streams into busy spans, then prints the
gaps longer than MIN_US: their length, their start relative to the trace start, the event that ends before and the one
that starts after (name and hlo_op), and the sum of gaps by size class. Usage: trace_gaps.py <profile dir> [min_us]"""
import re
import sys
from pathlib import Path

from jax.profiler import ProfileData

prof = Path(sys.argv[1])
min_us = float(sys.argv[2]) if len(sys.argv) > 2 else 200.0
xp = sorted(prof.rglob("*.xplane.pb"), key=lambda p: p.stat().st_mtime)[-1]
pd = ProfileData.from_file(str(xp))
plane = next(p for p in pd.planes if re.search(r"/device:GPU:0\b", p.name))
ev = []
for line in plane.lines:
    if not line.name.startswith("Stream"):
        continue
    for e in line.events:
        st = dict(e.stats)
        ev.append((e.start_ns, e.start_ns + e.duration_ns, e.name[:60], str(st.get("hlo_op", ""))[:40]))
ev.sort()
t0 = ev[0][0]
spans, cur = [], list(ev[0])
for s, t, n, h in ev[1:]:
    if s > cur[1]:
        spans.append(tuple(cur))
        cur = [s, t, n, h]
    elif t > cur[1]:
        cur[1], cur[2], cur[3] = t, n, h
spans.append(tuple(cur))
gaps = []
for (s0, t0_, n0, h0), (s1, t1, n1, h1) in zip(spans, spans[1:]):
    g = (s1 - t0_) / 1e3
    if g >= min_us:
        gaps.append((g, (t0_ - t0) / 1e6, n0, h0, n1, h1))
total = (spans[-1][1] - spans[0][0]) / 1e6
idle = sum((b[0] - a[1]) for a, b in zip(spans, spans[1:])) / 1e6
print(f"{xp}\ntrace {total:.1f} ms, idle {idle:.1f} ms ({idle / total:.1%}), {len(spans)} busy spans")
for lo, hi in ((0, 0.05), (0.05, 0.2), (0.2, 1), (1, 5), (5, 1e9)):
    sel = [(b[0] - a[1]) / 1e6 for a, b in zip(spans, spans[1:]) if lo <= (b[0] - a[1]) / 1e6 < hi]
    print(f"  gaps {lo}-{hi} ms: {len(sel)} gaps, {sum(sel):.1f} ms")
print(f"gaps >= {min_us:.0f} us (ms, at ms, before -> after):")
for g, at, n0, h0, n1, h1 in sorted(gaps, reverse=True)[:25]:
    print(f"  {g / 1e3:7.2f} at {at:8.1f}: {n0} [{h0}] -> {n1} [{h1}]")
