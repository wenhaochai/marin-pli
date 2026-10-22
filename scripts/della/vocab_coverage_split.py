"""Survey diagnostic (owner popup 2026-10-21): how much of the cost of 8 passes comes from Paloma targets that the repeated
1/8 subset never or rarely contains (not seen: coverage) and how much from targets it holds often (seen: memorisation).

For each seed-matched pair of per-position Paloma losses (vocab_token_losses.py outputs, as vocab_cost_groups.py), the
positions are grouped by how often the predicted token (and, separately, the input token) occurs in the 8-pass run's
subset, i.e. pass 1 of materialized/<size>[-v<K>k]-rep8-rs (the subset in the run's order; data does not depend on the
init seed). Bins of the subset count: 0, 1, 2-9, 10-99, 100-999, 1000+. Per bin: share of the scored positions, pooled
cost in bits per byte of those positions, and its contribution to the pair's pooled total (the bin's loss difference over
all scored bytes), so the contributions add up to the total. CPU only; run on a vis node:
    nice -n 19 .venv/bin/python scripts/della/vocab_coverage_split.py
Writes /scratch/gpfs/GROUP/USER/tmp_plots/vocab/coverage_split.json."""
import json
import os

import numpy as np
from levanter.store.cache import TreeCache

PREFIX = "/scratch/gpfs/GROUP/USER/marin_store_big"
LOSSES = "/scratch/gpfs/GROUP/USER/project/marin/logs/vocab_token_losses"
OUT = "/scratch/gpfs/GROUP/USER/tmp_plots/vocab/coverage_split.json"
TOKJ = {128000: "/scratch/gpfs/GROUP/USER/cache/huggingface/hub/models--marin-community--marin-tokenizer/snapshots/a5ca45f2feb6c959bd87b81689aa7279b5bdcaa2/tokenizer.json",
        8000: f"{PREFIX}/tokenizers/marin-small/v8000/tokenizer.json"}
SUBSET = {128000: f"{PREFIX}/materialized/300m-rep8-rs", 8000: f"{PREFIX}/materialized/300m-v8k-rep8-rs"}
PAIRS = {"128K-s1": (128000, "muonh-qwen3-300m-della4xh100-s1-rerun", "muonh-qwen3-300m-della4xh100-rep8-s1"),
         "8K-s1": (8000, "muonh-qwen3-300m-della4xh100-v8k-s1", "muonh-qwen3-300m-della4xh100-v8k-rep8-s1"),
         "128K-cvd-s1": (128000, "muonh-qwen3-300m-della4xh100-cvd8k-p0.5-s1", "muonh-qwen3-300m-della4xh100-cvd8k-p0.5-rep8-s1")}
EDGES = [0, 1, 2, 10, 100, 1000, 10 ** 15]
NAMES = ["0", "1", "2-9", "10-99", "100-999", "1000+"]
EX = {"input_ids": np.zeros((0,), dtype=np.int32)}


def nbytes_of(k):
    t = json.load(open(TOKJ[k]))
    n = max(max(t["model"]["vocab"].values()), max(a["id"] for a in t["added_tokens"])) + 1
    nb = np.zeros(n, np.int64)
    for s, i in t["model"]["vocab"].items():
        nb[i] = len(s)
    return nb, n


def subset_counts(k, n_vocab):
    path = f"{LOSSES}/subset_counts_{k}.npy"
    if os.path.exists(path):
        return np.load(path)
    meta = json.load(open(SUBSET[k] + "/meta.json"))
    n = meta["subset_sequences"]
    c = TreeCache.load(SUBSET[k] + "/train", EX)
    counts = np.zeros(n_vocab, np.int64)
    for a in range(0, n, 4096):
        rows = c.get_batch_sync(list(range(a, min(n, a + 4096))))
        counts += np.bincount(np.concatenate([np.asarray(x["input_ids"]) for x in rows]), minlength=n_vocab)
    assert counts.sum() == n * 4096
    np.save(path, counts)
    return counts


res = {}
for pair, (k, full, rep) in PAIRS.items():
    f, r = np.load(f"{LOSSES}/{full}.npz"), np.load(f"{LOSSES}/{rep}.npz")
    nb, nv = nbytes_of(k)
    cnt = subset_counts(k, nv)
    subs = sorted(x[:-len("/loss")] for x in f.files if x.endswith("/loss"))
    tok = np.concatenate([f[f"{s}/tokens"] for s in subs]); assert all(np.array_equal(f[f"{s}/tokens"], r[f"{s}/tokens"]) for s in subs)
    d = np.concatenate([r[f"{s}/loss"].astype(np.float64) - f[f"{s}/loss"] for s in subs])
    w = np.concatenate([f[f"{s}/weight"].astype(np.float64) for s in subs])
    tgt = np.roll(tok, -1, axis=1)
    b = nb[tgt].astype(np.float64)
    total_bytes = (w * b).sum()
    out = {"pooled_total": float((w * d).sum() / np.log(2) / total_bytes)}
    for side, ids in (("target", tgt), ("input", tok)):
        g = np.searchsorted(EDGES, cnt[ids], side="right") - 1
        rows = []
        for i, name in enumerate(NAMES):
            m = (g == i) & (w > 0)
            share = float(w[m].sum() / w[w > 0].sum())
            cost = float((w * d)[m].sum() / np.log(2) / max((w * b)[m].sum(), 1))
            contrib = float((w * d)[m].sum() / np.log(2) / total_bytes)
            rows.append({"bin": name, "share": round(share, 4), "cost": round(cost, 4), "contribution": round(contrib, 4)})
        out[side] = rows
    res[pair] = out
    print(pair, "pooled total", round(out["pooled_total"], 4))
    for side in ("target", "input"):
        print("  ", side, " | ".join(f'{x["bin"]}: share {x["share"]:.3f} cost {x["cost"]:+.3f} contrib {x["contribution"]:+.4f}' for x in out[side]))
json.dump(res, open(OUT, "w"), indent=1)
print("written", OUT)
