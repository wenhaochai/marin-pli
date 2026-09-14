"""Download a seeded uniform file sample of Nemotron-CC from data.commoncrawl.org, per quality/kind/kind2 group.

    python scripts/della/nemotron_sample.py <full_paths.txt> <out_root> --fraction 0.01 [--exclude used.txt ...]

Every group (e.g. quality=medium/kind=actual/kind2=actual) contributes ceil(fraction x its files), drawn uniformly at
random from the files not listed in any --exclude manifest, so all CC snapshots stay represented in proportion.
Files land at <out_root>/<path>, the layout the Marin tokenize step globs
(raw/nemotro-cc-eeb783/contrib/Nemotron/Nemotron-CC/data-jsonl/...), and the chosen list is written to
<out_root>/sample_paths.txt.

To grow the corpus later, pass every manifest in scripts/della/data/ as --exclude so the new draw is disjoint from
the files already tokenized (and from the early subsets used by the discarded runs), then tokenize the union.
"""

import argparse
import collections
import concurrent.futures as cf
import math
import os
import random
import re
import subprocess

GROUP = re.compile(r"quality=[^/]+/kind=[^/]+/kind2=[^/]+")

ap = argparse.ArgumentParser()
ap.add_argument("paths_file")
ap.add_argument("out_root")
ap.add_argument("--fraction", type=float, default=0.01)
ap.add_argument("--seed", type=int, default=20260914)
ap.add_argument("--exclude", action="append", default=[], help="manifest of files to leave out (one path per line)")
ap.add_argument("--workers", type=int, default=8)
args = ap.parse_args()

paths = [line.strip() for line in open(args.paths_file) if line.strip()]
excluded = set()
for m in args.exclude:
    excluded |= {line.strip() for line in open(m) if line.strip()}
groups = collections.defaultdict(list)
for p in paths:
    if p not in excluded:
        groups[GROUP.search(p).group(0)].append(p)

rng = random.Random(args.seed)
picked = []
for g in sorted(groups):
    files = sorted(groups[g])
    n = min(len(files), math.ceil(len(files) * args.fraction))
    chosen = rng.sample(files, n)
    snaps = {re.search(r"CC-MAIN-\d{4}-\d{2}", c).group(0) for c in chosen}
    print(f"{g:60s} {n:4d}/{len(files)} files, {len(snaps)} snapshots", flush=True)
    picked += chosen
os.makedirs(args.out_root, exist_ok=True)
with open(os.path.join(args.out_root, "sample_paths.txt"), "w") as f:
    f.write("\n".join(sorted(picked)) + "\n")


def fetch(p):
    dest = os.path.join(args.out_root, p)
    if os.path.exists(dest):
        return p, "cached"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    cmd = ["curl", "-sfL", "--retry", "10", "--retry-all-errors", "--retry-delay", "20", "-o", dest + ".part", "https://data.commoncrawl.org/" + p]
    if subprocess.run(cmd).returncode != 0:
        return p, "FAILED"
    os.rename(dest + ".part", dest)
    return p, "ok"


done = failed = 0
with cf.ThreadPoolExecutor(args.workers) as ex:
    for p, status in ex.map(fetch, picked):
        done += 1
        failed += status == "FAILED"
        if status == "FAILED" or done % 25 == 0:
            print(f"{done}/{len(picked)} {status} {p}", flush=True)
print(f"DONE {done} files, {failed} failed", flush=True)
