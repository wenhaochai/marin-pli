"""What the host does while GPU 0 sits idle (speed work 2026-10-26): the 4-GPU profile w313 shows ~65 ms of idle per
step at the step boundary, the same with the async host loop, so the host is blocked somewhere before it issues the
next step. For each GPU-0 gap longer than MIN_MS, prints the host events (TraceMe, Python tracer) that overlap the gap,
per host thread, with their start relative to the gap start and their length. Usage: trace_host_gaps.py <profile dir>
[min_ms] [max_gaps]"""
import re
import sys
from pathlib import Path

from jax.profiler import ProfileData

prof = Path(sys.argv[1])
min_ms = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
max_gaps = int(sys.argv[3]) if len(sys.argv) > 3 else 4
xp = sorted(prof.rglob("*.xplane.pb"), key=lambda p: p.stat().st_mtime)[-1]
pd = ProfileData.from_file(str(xp))
print(xp)
print("planes:", [p.name for p in pd.planes])
gpu = next(p for p in pd.planes if re.search(r"/device:GPU:0\b", p.name))
ev = sorted((e.start_ns, e.start_ns + e.duration_ns, e.name[:50]) for line in gpu.lines if line.name.startswith("Stream")
            for e in line.events)
spans, cur = [], list(ev[0])
for s, t, n in ev[1:]:
    if s > cur[1]:
        spans.append(tuple(cur))
        cur = [s, t, n]
    elif t > cur[1]:
        cur[1], cur[2] = t, n
spans.append(tuple(cur))
gaps = [(a[1], b[0], a[2], b[2]) for a, b in zip(spans, spans[1:]) if (b[0] - a[1]) / 1e6 >= min_ms]
host = [p for p in pd.planes if p.name.startswith("/host:")]
for g0, g1, before, after in gaps[:max_gaps]:
    print(f"\n=== GPU-0 gap {(g1 - g0) / 1e6:.1f} ms: {before} -> {after}")
    for p in host:
        for line in p.lines:
            rows = [(e.start_ns, e.duration_ns, e.name) for e in line.events
                    if e.start_ns < g1 + 2e6 and e.start_ns + e.duration_ns > g0 - 2e6 and e.duration_ns > 2e5]
            if not rows:
                continue
            print(f"  [{p.name} | {line.name}] {len(rows)} events > 0.2 ms")
            for s, d, n in sorted(rows)[:40]:
                print(f"    {(s - g0) / 1e6:+8.2f} ms  {d / 1e6:7.2f} ms  {n[:110]}")
