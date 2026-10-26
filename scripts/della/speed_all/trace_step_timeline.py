"""One merged timeline of a profile window (speed work 2026-10-26): the main Python thread's frames (nested by
containment, down to MAX_DEPTH, longer than MIN_MS), the data-loader thread's top frames, and GPU 0's busy spans and
idle gaps longer than MIN_MS with the HLO module of the work on each side. Shows what the host does at each step
boundary while the GPU waits. Usage: trace_step_timeline.py <profile dir> <from_ms> <to_ms> [min_ms] [max_depth]"""
import re
import sys
from pathlib import Path

from jax.profiler import ProfileData

prof = Path(sys.argv[1])
lo_ms, hi_ms = float(sys.argv[2]), float(sys.argv[3])
min_ms = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
max_depth = int(sys.argv[5]) if len(sys.argv) > 5 else 6
xp = sorted(prof.rglob("*.xplane.pb"), key=lambda p: p.stat().st_mtime)[-1]
pd = ProfileData.from_file(str(xp))
gpu = next(p for p in pd.planes if re.search(r"/device:GPU:0\b", p.name))
ev = []
for line in gpu.lines:
    if not line.name.startswith("Stream"):
        continue
    for e in line.events:
        st = dict(e.stats)
        mod = str(st.get("hlo_module", st.get("hlo_module_name", "")))[:40]
        ev.append((e.start_ns, e.start_ns + e.duration_ns, mod or e.name[:40]))
ev.sort()
t0 = ev[0][0]
lo, hi = t0 + lo_ms * 1e6, t0 + hi_ms * 1e6
rows = []
# GPU busy spans (merged) and gaps
spans, cur = [], [ev[0][0], ev[0][1], {ev[0][2]}]
for s, t, m in ev[1:]:
    if s > cur[1]:
        spans.append(cur)
        cur = [s, t, {m}]
    else:
        cur[1] = max(cur[1], t)
        cur[2].add(m)
spans.append(cur)
for a, b in zip(spans, spans[1:]):
    if a[1] >= lo and b[0] <= hi and (b[0] - a[1]) / 1e6 >= min_ms:
        rows.append((a[1], 0, f"GPU0 IDLE {(b[0] - a[1]) / 1e6:6.1f} ms   (before: {sorted(a[2])[:2]}, after: {sorted(b[2])[:2]})"))
for s in spans:
    if s[1] >= lo and s[0] <= hi and (s[1] - s[0]) / 1e6 >= 5 * min_ms:
        rows.append((s[0], 0, f"GPU0 BUSY {(s[1] - s[0]) / 1e6:6.1f} ms   {sorted(s[2])[:3]}"))
# host python threads: nest by containment
for p in pd.planes:
    if not p.name.startswith("/host:CPU"):
        continue
    for line in p.lines:
        evs = sorted(((e.start_ns, e.start_ns + e.duration_ns, e.name) for e in line.events), key=lambda x: (x[0], -x[1]))
        if not any("trainer.py" in n or "loader.py" in n for _, _, n in evs):
            continue
        stack = []
        for s, t, n in evs:
            while stack and stack[-1] <= s:
                stack.pop()
            depth = len(stack)
            stack.append(t)
            if t < lo or s > hi or (t - s) / 1e6 < min_ms or depth > max_depth:
                continue
            tag = "MAIN" if any("trainer.py" in x for _, _, x in evs[:50]) else "LOAD"
            rows.append((s, 1 + depth, f"{tag} {'  ' * depth}{n[:90]}  [{(t - s) / 1e6:.1f} ms]"))
for s, _, txt in sorted(rows):
    print(f"{(s - t0) / 1e6:9.1f}  {txt}")
