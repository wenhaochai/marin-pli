"""GPU time per part of the train step, from a JAX profiler xplane and the step's optimized HLO (speed work, 2026-10-25).

Usage: trace_families.py <profile dir> <hlo dump dir> [steps]
The run must have had XLA_FLAGS with --xla_gpu_enable_command_buffer= (no CUDA graphs: each kernel then carries its own
HLO instruction in the trace instead of the command buffer's) and --xla_dump_to=<hlo dump dir> --xla_dump_hlo_as_text.
Reads the device plane /device:GPU:0 of the newest *.xplane.pb below <profile dir> (FSDP devices are symmetric), maps each
kernel's hlo_op to that instruction's op_name metadata in the largest jit__train_step after_optimizations module, and
sums kernel time per family of the op_name (the JAX name stack): attention kernel (flash forward and backward), the
remat recompute (anything under rematted_computation), attention projections, MLP, other layer ops (norms, residuals),
layer-scan bookkeeping, the output head and cross-entropy (the fused CE loop outside the transformer), embedding,
optimizer and update (everything after the gradient), collectives. Prints, per traced step: wall, GPU-busy time, each
family's time and share, and the 30 largest instructions with their op_name.
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

from jax.profiler import ProfileData

prof, hlo_dir = Path(sys.argv[1]), Path(sys.argv[2])
steps = int(sys.argv[3]) if len(sys.argv) > 3 else 3
mods = sorted(hlo_dir.glob("*jit__train_step*.after_optimizations.txt"), key=lambda p: p.stat().st_size)
if not mods:
    sys.exit(f"no jit__train_step after_optimizations module in {hlo_dir}")
OPNAME, COMP_OP, CALLS = {}, {}, {}
INSTR = re.compile(r'^\s*(?:ROOT\s+)?%?([\w.\-]+) = ')
META = re.compile(r'metadata=\{[^}]*?op_name="([^"]*)"')
COMP = re.compile(r'^(?:ENTRY\s+)?%?([\w.\-]+) \(.*\{\s*$')
comp = None
for line in mods[-1].read_text(errors="replace").splitlines():
    c = COMP.match(line)
    if c:
        comp = c.group(1)
        continue
    m = INSTR.match(line)
    if not m:
        continue
    meta = META.search(line)
    if meta:
        OPNAME[m.group(1)] = meta.group(1)
        COMP_OP.setdefault(comp, meta.group(1))   # a computation's first op_name stands for it
    else:
        k = re.search(r"calls=%([\w.\-]+)", line)
        if k:
            CALLS[m.group(1)] = k.group(1)
for name, called in CALLS.items():   # a fusion with no metadata of its own: the op_name of the computation it calls
    if name not in OPNAME and called in COMP_OP:
        OPNAME[name] = COMP_OP[called]
xp = sorted(prof.rglob("*.xplane.pb"), key=lambda p: p.stat().st_mtime)
if not xp:
    sys.exit(f"no xplane.pb below {prof}")
pd = ProfileData.from_file(str(xp[-1]))
plane = next(p for p in pd.planes if re.search(r"/device:GPU:0\b", p.name))


def family(hlo, name):
    n = name.lower()
    if re.search(r"nccl|all-reduce|all-gather|reduce-scatter|collective-permute|allreduce|allgather|reducescatter", hlo + " " + n):
        return "collectives"
    op = OPNAME.get(hlo)
    if op is None:
        return "memcpy/memset" if re.search(r"memcpy|memset", n) else "outside the train step"
    if "rematted_computation" in op:
        return "remat recompute"
    if re.search(r"flash_attention|dot_product_attention|fa4|cudnn|_attention_forward|_attention_backward", op):
        return "attention kernel"
    if "Attention" in op:
        return "attention projections"
    if re.search(r"Mlp", op):
        return "MLP"
    if re.search(r"scan\(layer\)/while/body/jit\(dynamic_(update_)?index_in_dim\)", op):
        return "layer-scan bookkeeping"
    if re.search(r"DecoderLayer|scan\(layer\)", op):
        return "other layer ops (norms, residuals)"
    if re.search(r"Embedding", op):
        return "embedding"
    if re.search(r"lm_head|cross_entropy|fused|logsumexp|next_token_loss|^jvp\(\)|^transpose\(jvp\(\)\)|LmHead|get_lm_head", op.removeprefix("jit(_train_step)/")):
        return "output head and cross-entropy"
    if not re.search(r"jvp\(|transpose\(", op):
        return "optimizer and update"
    return "other"


tot, per_op, ivs = defaultdict(float), defaultdict(float), []
for line in plane.lines:
    if not line.name.startswith("Stream"):
        continue
    for e in line.events:
        st = dict(e.stats)
        hlo = str(st.get("hlo_op", ""))
        d = e.duration_ns / 1e6
        tot[family(hlo, e.name)] += d
        per_op[hlo or e.name[:40]] += d
        ivs.append((e.start_ns, e.start_ns + e.duration_ns))
ivs.sort()
busy, cs, ce = 0.0, None, None
for s, t in ivs:
    if ce is None or s > ce:
        busy += 0 if ce is None else ce - cs
        cs, ce = s, t
    else:
        ce = max(ce, t)
busy += (ce - cs) if ce is not None else 0
wall = (ivs[-1][1] - ivs[0][0]) if ivs else 0
summed = sum(tot.values())
print(f"xplane {xp[-1]}\nhlo {mods[-1].name} ({len(OPNAME)} instructions with op_name); per step over {steps} steps")
print(f"wall {wall / 1e6 / steps:.1f} ms, GPU busy {busy / 1e6 / steps:.1f} ms ({busy / max(wall, 1):.0%}), summed kernel time {summed / steps:.1f} ms")
for k, v in sorted(tot.items(), key=lambda x: -x[1]):
    print(f"  {k:38s} {v / steps:8.1f} ms  {v / summed:6.1%}")
print("largest instructions (ms per step, family, op_name tail):")
for hlo, v in sorted(per_op.items(), key=lambda x: -x[1])[:30]:
    print(f"  {v / steps:8.2f}  {hlo[:34]:34s} {family(hlo, hlo):30s} {OPNAME.get(hlo, '')[-110:]}")
