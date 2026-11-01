"""Stream-difference use (layer-48 CE rise when the stream differences are deleted after layer l, mean over l <= 16, as
in tmp/pls_paths/plot/hidden_vs_cost.py) on the separate-heads (sep) and probes checkpoints, mean-collapse (ha-) and
shuffle (has-) modes, next to the separate-heads cost. Inputs: tmp/pls_paths/ha[s]-sep[-bbfrozen]-<tag>.json. Read-only."""
import json
from pathlib import Path
O = Path("/scratch/gpfs/GROUP/USER/tmp/pls_paths")
PTS = [("ahc-vs-d48-g4", "HC", 0.048), ("ahc-vs-d48-s1", "HC seed 1", 0.034), ("ahc-hcrwsig-vs-d48", "HC sigmoid rw", 0.044),
       ("ahc-hcs8-vs-d48", "HC 8 streams", 0.049), ("ahc-hcs2-vs-d48", "HC 2 streams", 0.074), ("amhc-vs-d48-g4", "mHC", 0.102),
       ("amhc-mhcid-vs-d48", "mHC id", 0.100), ("amhc-mhcrwlin-vs-d48", "mHC lin", 0.080),
       ("amhc-mhcid-mhcrwlin-fa4thd-vs-d48", "NEW mHC id+lin", None), ("ahc-hcwsh-fa4thd-vs-d48", "NEW HC shared write", None)]


def early(p):
    if not p.exists():
        return None
    d = json.loads(p.read_text())["dce"]
    v = [x for k, x in d.items() if int(k) <= 16]
    return sum(v) / len(v)


def f(x):
    return f"{x:7.3f}" if x is not None else "      -"


print(f"{'design':22s} {'cost':>6s}   {'sep mean':>8s} {'sep shuf':>8s}   {'prb mean':>8s} {'prb shuf':>8s}")
for tag, lab, c in PTS:
    vals = [early(O / f"{m}-sep{arm}-{tag}.json") for arm in ("", "-bbfrozen") for m in ("ha", "has")]
    print(f"{lab:22s} {f(c) if c is not None else '     ?'}   {f(vals[0])}  {f(vals[1])}   {f(vals[2])}  {f(vals[3])}")
