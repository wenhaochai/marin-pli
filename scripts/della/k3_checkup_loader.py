"""Node loader for k3_checkup_v2.py: reads each K3 safetensors shard from GPFS ONCE per node into /dev/shm, with
THREADS parallel striped preads (one GPFS stream gives 0.1-2 GB/s; striped reads several GB/s), so the 8 ranks no
longer read the same 16 GB each. In K3 each decoder layer i lives alone in shard i+1; the head shard holds the
embedding, lm_head, final norm and the output AttnRes query.
Protocol (all files in SHM):
  <shard name>               a shard ready to read (written as <name>.tmp, then renamed);
  done.<layer>.<rank>        written by rank <rank> once it has copied layer <layer> out of SHM;
  head.<rank>                written by rank <rank> once it has copied what it needs from the head shard.
The loader stages the head shard, then layers 0..L-1 in order, keeping at most AHEAD layer shards in SHM, and deletes
a shard once every rank has marked it. It logs read seconds and GB/s per shard.
Usage: python k3_checkup_loader.py MODEL_DIR SHM NRANKS L [AHEAD=3] [THREADS=16]
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

MODEL, SHM, NR, L = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
AHEAD = int(sys.argv[5]) if len(sys.argv) > 5 else 3
THREADS = int(sys.argv[6]) if len(sys.argv) > 6 else 16
CH = 64 << 20
os.makedirs(SHM, exist_ok=True)
wm = json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))["weight_map"]
shard_of = {}
for k, f in wm.items():
    if k.startswith("language_model.model.layers."):
        shard_of.setdefault(int(k.split(".")[3]), set()).add(f)
assert all(len(s) == 1 for s in shard_of.values()), "a layer spans several shards"
shard_of = {i: next(iter(s)) for i, s in shard_of.items()}
head = wm["language_model.lm_head.weight"]
pool = ThreadPoolExecutor(THREADS)


def stage(f):
    src, dst = os.path.join(MODEL, f), os.path.join(SHM, f)
    size = os.path.getsize(src)
    t0 = time.time()
    fi = os.open(src, os.O_RDONLY)
    fo = os.open(dst + ".tmp", os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
    os.ftruncate(fo, size)

    def part(off):
        n = min(CH, size - off)
        got = 0
        while got < n:
            b = os.pread(fi, n - got, off + got)
            if not b:
                raise IOError(f"short read {src} at {off + got}")
            os.pwrite(fo, b, off + got)
            got += len(b)
    list(pool.map(part, range(0, size, CH)))
    os.close(fi)
    os.close(fo)
    os.rename(dst + ".tmp", dst)
    dt = time.time() - t0
    return dt, size / 1e9 / dt


def marked(prefix):
    return sum(os.path.exists(os.path.join(SHM, f"{prefix}.{r}")) for r in range(NR)) == NR


t_start = time.time()
dt, gbs = stage(head)
print(f"loader head {head}: {dt:.1f}s {gbs:.2f} GB/s", flush=True)
staged = []   # layers in SHM, not yet released by every rank
head_live = True
stats = []
for i in range(L):
    while True:   # free shards every rank has copied; wait while AHEAD layers are staged
        for j in list(staged):
            if marked(f"done.{j}"):
                os.remove(os.path.join(SHM, shard_of[j]))
                for r in range(NR):
                    os.remove(os.path.join(SHM, f"done.{j}.{r}"))
                staged.remove(j)
        if head_live and marked("head"):
            os.remove(os.path.join(SHM, head))
            head_live = False
        if len(staged) < AHEAD:
            break
        time.sleep(0.2)
    t_wait = time.time()
    dt, gbs = stage(shard_of[i])
    staged.append(i)
    stats.append(dict(layer=i, read_s=round(dt, 2), gbps=round(gbs, 2)))
    print(f"loader layer {i}: read {dt:.1f}s {gbs:.2f} GB/s (staged {staged}) t={time.time() - t_start:.0f}s", flush=True)
while staged or head_live:   # final cleanup once every rank has copied the last layers
    for j in list(staged):
        if marked(f"done.{j}"):
            os.remove(os.path.join(SHM, shard_of[j]))
            for r in range(NR):
                os.remove(os.path.join(SHM, f"done.{j}.{r}"))
            staged.remove(j)
    if head_live and marked("head"):
        os.remove(os.path.join(SHM, head))
        head_live = False
    time.sleep(0.5)
print(f"loader done in {(time.time() - t_start) / 60:.1f} min; mean read {sum(s['read_s'] for s in stats) / max(len(stats), 1):.1f}s", flush=True)
