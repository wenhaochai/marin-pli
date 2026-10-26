"""GPU time per kernel family from a JAX profiler perfetto trace (speed-all profile job, 2026-10-25).

Usage: trace_breakdown.py <dir with perfetto_trace.json.gz somewhere below> [steps]
Reads the trace, keeps the complete events ('ph' == 'X') of the process named like /device:GPU:0 (one device; FSDP
devices are symmetric), and prints: the traced wall time and GPU-busy time of that device (union of its kernel
intervals), the summed kernel time per family, and the 40 kernels with the most summed time, each with its HLO op and
scope when the trace carries them. Times are per traced step when [steps] (default 3) is given.
Families come from the kernel name and the scope (tf_op / long_name / hlo_op args): attention, cross-entropy and head,
optimizer (Muon / Newton-Schulz / Adam), collectives (NCCL), GEMM (cuBLAS / CUTLASS / nvjet) and the rest.
"""
import gzip
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1])
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 3
paths = sorted(root.rglob("*perfetto_trace.json.gz")) or sorted(root.rglob("*.trace.json.gz"))
if not paths:
    sys.exit(f"no perfetto trace below {root}")
with gzip.open(paths[-1], "rt") as f:
    data = json.load(f)
ev = data["traceEvents"] if isinstance(data, dict) else data
pname, tname = {}, {}
for e in ev:
    if e.get("ph") == "M" and e.get("name") == "process_name":
        pname[e["pid"]] = e["args"]["name"]
    if e.get("ph") == "M" and e.get("name") == "thread_name":
        tname[(e["pid"], e["tid"])] = e["args"]["name"]
gpu = [p for p, n in pname.items() if re.search(r"/device:GPU:0\b", n)]
if not gpu:
    sys.exit(f"no GPU:0 process; processes: {sorted(set(pname.values()))[:20]}")
g = gpu[0]
ks = [e for e in ev if e.get("pid") == g and e.get("ph") == "X" and e.get("dur", 0) > 0]
threads = defaultdict(float)
for e in ks:
    threads[tname.get((g, e["tid"]), str(e["tid"]))] += e["dur"]
# drop summary/aggregate lanes (XLA Ops, Steps, ...) that duplicate the kernel streams: keep streams with "Stream" in the name
stream_tids = {t for (p, t), n in tname.items() if p == g and re.search(r"[Ss]tream", n)}
if stream_tids:
    ks = [e for e in ks if e["tid"] in stream_tids]
t0, t1 = min(e["ts"] for e in ks), max(e["ts"] + e["dur"] for e in ks)
iv = sorted((e["ts"], e["ts"] + e["dur"]) for e in ks)
busy, cur_s, cur_e = 0.0, None, None
for s, t in iv:
    if cur_e is None or s > cur_e:
        if cur_e is not None:
            busy += cur_e - cur_s
        cur_s, cur_e = s, t
    else:
        cur_e = max(cur_e, t)
busy += cur_e - cur_s


def scope(e):
    a = e.get("args", {})
    return " ".join(str(a.get(k, "")) for k in ("tf_op", "long_name", "hlo_op", "name_scope", "kernel_details") if a.get(k))


def family(e):
    n, s = e["name"].lower(), scope(e).lower()
    both = n + " " + s
    if re.search(r"nccl|all-reduce|all-gather|reduce-scatter|allreduce|allgather|reducescatter|collective", both):
        return "collectives"
    if re.search(r"fa4|flash|attention|attn|cute", both):
        return "attention"
    if re.search(r"cross_entropy|next_token_loss|fused_cross|logsumexp|\blse\b|lm_head|batched_xla|b_tiled|readout", both):
        return "cross-entropy and head"
    if re.search(r"muon|newton|schulz|zeropower|orthogon|optax|adam|optim|update", both):
        return "optimizer"
    if re.search(r"memcpy|memset", n):
        return "memcpy/memset"
    if re.search(r"gemm|cutlass|nvjet|xmma|cublas|matmul|dot", both):
        return "other GEMM"
    return "other elementwise/fusion"


fam, kern = defaultdict(float), defaultdict(lambda: [0.0, 0, ""])
for e in ks:
    fam[family(e)] += e["dur"]
    k = kern[e["name"]]
    k[0] += e["dur"]; k[1] += 1; k[2] = k[2] or scope(e)[:160]
ms = lambda us: us / 1e3 / steps
print(f"trace {paths[-1]}\nprocess {pname[g]}; lanes kept: {len(stream_tids) or 'all'}; per step over {steps} steps")
print(f"wall {ms(t1 - t0):.1f} ms, GPU busy {ms(busy):.1f} ms ({busy / (t1 - t0):.0%}), summed kernel time {ms(sum(e['dur'] for e in ks)):.1f} ms")
tot = sum(fam.values())
for k, v in sorted(fam.items(), key=lambda x: -x[1]):
    print(f"  {k:28s} {ms(v):8.1f} ms  {v / tot:6.1%}")
print("top kernels (summed ms per step, calls per step, family, scope):")
for n, (d, c, s) in sorted(kern.items(), key=lambda x: -x[1][0])[:40]:
    print(f"  {ms(d):8.2f}  {c / steps:7.1f}  {family({'name': n, 'args': {'tf_op': s}}):24s} {n[:90]}  |  {s[:120]}")
