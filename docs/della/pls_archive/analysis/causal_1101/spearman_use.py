"""Spearman correlation between the separate-heads cost and the stream-difference use on the separate-heads checkpoint
(mean collapse, mean over l <= 16; tmp/pls_paths/ha-sep-<tag>.json), over the 10 designs of Fig 49 (one-stream designs at
x = 0), then with the causal-test designs added as they finish. Read-only. 2026-11-01."""
import json, sys
from pathlib import Path
from scipy.stats import spearmanr
O = Path("/scratch/gpfs/GROUP/USER/tmp/pls_paths")
OLD = [("ahc-vs-d48-g4", 0.048), ("ahc-vs-d48-s1", 0.034), ("ahc-hcrwsig-vs-d48", 0.044), ("ahc-hcs8-vs-d48", 0.049),
       ("ahc-hcs2-vs-d48", 0.074), ("amhc-vs-d48-g4", 0.102), ("amhc-mhcid-vs-d48", 0.100), ("amhc-mhcrwlin-vs-d48", 0.080),
       (None, 0.125), (None, 0.089)]
NEW = [("amhc-mhcid-mhcrwlin-fa4thd-vs-d48", 0.085)] + [(a.split("=")[0], float(a.split("=")[1])) for a in sys.argv[1:]]


def use(tag, prefix="ha"):
    if tag is None:
        return 0.0
    d = json.loads((O / f"{prefix}-sep-{tag}.json").read_text())["dce"]
    v = [x for k, x in d.items() if int(k) <= 16]
    return sum(v) / len(v)


for prefix in ("ha", "has"):
    try:
        for name, pts in (("old 10", OLD), ("old + new", OLD + NEW)):
            x = [use(t, prefix) for t, _ in pts]; y = [c for _, c in pts]
            r = spearmanr(x, y)
            print(f"{prefix:4s} {name:10s} n={len(pts):2d} spearman {r.statistic:+.3f} p {r.pvalue:.3f}")
    except FileNotFoundError as e:
        print(prefix, "missing", e.filename)
