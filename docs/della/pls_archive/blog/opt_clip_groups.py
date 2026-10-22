"""L6 on the logs: how much the readouts' gradients changed the clipping of the backbone's own parameters in the 48-layer
runs (-vs-d48-g4). Each optimizer group clips by its own global norm (scripts/della/pls_opt_effective.py, check 4), and the
readouts share two groups with the backbone: AdamH (lm_head with the 47 aux_lm_heads) and Adam (embeddings, every norm
gain and the final norm, with the 47 aux_norms). MuonH and the designs' adam_new hold no readout parameter.
Source: the local W&B files (wandb-runs/wandb/run-*-<run>/*.wandb, every resumed segment; the last row of a step wins),
key grad/norm/<path>: the raw gradient's norm per parameter, logged every 10 steps before clipping (levanter watch).
Per logged step and group: G = sqrt(sum of squares) with the readouts and G0 without; the clip factor is min(1, 1/G)
(max_grad_norm 1.0); the readouts change the backbone parameters' factor by min(1, 1/G) / min(1, 1/G0).
Prints, per run and group, the share of logged steps where the group clips with and without the readouts, and the median
and minimum of that ratio. Run on a vis node: nice -n 19 $PY opt_clip_groups.py [arch ...]
"""
import glob
import json
import math
import re
import sys
from collections import defaultdict

from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal import datastore

ROOT = "/scratch/gpfs/GROUP/USER/wandb-runs/wandb"
ARCHS = sys.argv[1:] or ["preln", "lns", "hc", "attnres"]
MUONH = re.compile(r"_proj\.weight$")
NEW = ("hyper.", "query", "key_gain", "src_logit")


def group(path):
    if "lm_head" in path:
        return "adamh"
    if MUONH.search(path):
        return "muonh"
    if any(f in path for f in NEW):
        return "adam_new"
    return "adam"


def rows(run):
    by = {}
    for f in sorted(glob.glob(f"{ROOT}/run-*-{run}/run-*.wandb")):
        ds = datastore.DataStore()
        ds.open_for_scan(f)
        while True:
            try:
                data = ds.scan_data()
            except AssertionError:
                break
            if data is None:
                break
            rec = wandb_internal_pb2.Record()
            rec.ParseFromString(data)
            if rec.WhichOneof("record_type") != "history":
                continue
            items = {(i.key or "/".join(i.nested_key)): i.value_json for i in rec.history.item}
            if not any(k.startswith("grad/norm/") for k in items):
                continue
            step = int(json.loads(items["_step"]))
            by[step] = {k[10:]: float(json.loads(v)) for k, v in items.items()
                        if k.startswith("grad/norm/") and "layers.stacked" not in k and k != "grad/norm/total"}
    return by


for arch in ARCHS:
    for bb in ("-bbfrozen", ""):
        run = f"muonh-qwen3-130m-della4xh100-pls1-sep{bb}-a{arch}-vs-d48-g4"
        by = rows(run)
        if not by:
            print(f"{run}: no local W&B rows")
            continue
        stat = defaultdict(lambda: [0, 0, []])   # clips with readouts, clips without, ratios
        for step, g in by.items():
            ss, ss0 = defaultdict(float), defaultdict(float)
            for p, n in g.items():
                k = group(p)
                ss[k] += n * n
                if not p.startswith("aux_"):
                    ss0[k] += n * n
            for k in ss:
                G, G0 = math.sqrt(ss[k]), math.sqrt(ss0[k])
                st = stat[k]
                st[0] += G > 1
                st[1] += G0 > 1
                st[2].append(min(1, 1 / G) / min(1, 1 / G0) if G0 > 0 else 1.0)
        n = len(by)
        print(f"{run}: {n} logged steps ({min(by)}..{max(by)})")
        for k in ("muonh", "adamh", "adam", "adam_new"):
            if k in stat:
                c, c0, r = stat[k]
                r = sorted(r)
                print(f"  {k:8s} clips {100 * c / n:5.1f}% of steps (without readouts {100 * c0 / n:5.1f}%); "
                      f"backbone clip factor x median {r[len(r) // 2]:.3f}, min {r[0]:.3f}, share < 0.9 {100 * sum(x < 0.9 for x in r) / n:.1f}%")
