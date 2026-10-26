import re, sys
from pathlib import Path
from jax.profiler import ProfileData
prof = Path(sys.argv[1])
xp = sorted(prof.rglob("*.xplane.pb"), key=lambda p: p.stat().st_mtime)[-1]
pd = ProfileData.from_file(str(xp))
gpu = next(p for p in pd.planes if re.search(r"/device:GPU:0\b", p.name))
runs = {}
for line in gpu.lines:
    if not line.name.startswith("Stream"):
        continue
    for i, e in enumerate(line.events):
        st = dict(e.stats)
        if i < 2:
            print(line.name, e.name[:40], {k: str(v)[:40] for k, v in st.items()})
        key = (str(st.get("hlo_module", "")), str(st.get("run_id", st.get("correlation_id", ""))))
        r = runs.setdefault(key, [e.start_ns, e.start_ns + e.duration_ns, 0, 0.0])
        r[0] = min(r[0], e.start_ns); r[1] = max(r[1], e.start_ns + e.duration_ns); r[2] += 1; r[3] += e.duration_ns / 1e6
t0 = min(r[0] for r in runs.values())
for (m, rid), (s, t, n, busy) in sorted(runs.items(), key=lambda kv: kv[1][0]):
    if n > 3 or busy > 1:
        print(f"{(s - t0) / 1e6:9.1f} -> {(t - t0) / 1e6:9.1f}  span {(t - s) / 1e6:7.1f} ms  busy {busy:7.1f} ms  n={n:5d}  {m[:30]} run={rid}")
