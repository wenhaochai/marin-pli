"""Host lines of a profile with event counts and their most common event names (speed work 2026-10-26).
Usage: trace_lines.py <profile dir> [pattern]"""
import collections
import re
import sys
from pathlib import Path

from jax.profiler import ProfileData

prof = Path(sys.argv[1])
pat = re.compile(sys.argv[2]) if len(sys.argv) > 2 else None
xp = sorted(prof.rglob("*.xplane.pb"), key=lambda p: p.stat().st_mtime)[-1]
pd = ProfileData.from_file(str(xp))
for p in pd.planes:
    if not p.name.startswith("/host"):
        continue
    for line in p.lines:
        names = collections.Counter(e.name[:50] for e in line.events)
        if pat and not any(pat.search(n) for n in names):
            continue
        top = [f"{n} x{c}" for n, c in names.most_common(6)]
        print(f"[{p.name} | {line.name[:40]}] {sum(names.values())} events: {top}")
