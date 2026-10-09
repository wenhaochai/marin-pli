"""Grouped cost of 8 passes for blogs/vocab-overfitting.html Q4-Q6, from vocab_token_losses.py outputs.

For a tokenizer's pair (full-data run, 8-pass run), the per-position Paloma losses line up position by position. The
cost of a group of positions in one Paloma subset is sum(loss_rep - loss_full) / ln 2 / sum(bytes of the predicted
tokens), in bits per byte (positions with loss weight 0 excluded); the reported cost is the mean over the 16 subsets
(macro, as the page's bars and W&B's paloma macro), and the cost over all positions pooled is kept as "pooled". The
128K-vs-8K comparison uses seed-matched pairs (seed 1 for both); the 8K seed-0 pair is kept as a check. Groups:
  input:   frequency decile of the input token at the position (1 = rarest), by its count in the first 200M tokens of
           the tokenizer's fineweb-edu-10B training cache; deciles hold equal numbers of positions over all subsets,
           ties at a decile boundary broken at random (fixed seed);
  target:  the same for the token being predicted;
  context: bytes of context before the predicted token (from the window start or the last end-of-text token, whichever
           is later), in bins 0-64, 64-128, ... doubling, 16K and above.
Byte counts: a byte-level BPE token's string has one character per byte; special tokens count 0.

Run on a vis node: nice -n 19 .venv/bin/python scripts/della/vocab_cost_groups.py
Writes /scratch/gpfs/GROUP/USER/tmp_plots/vocab/cost_groups.json: {pair: {input|target|context: {edges, cost, n}}}.
"""
import asyncio
import glob
import json
import os
import sys

import numpy as np

PREFIX = "/scratch/gpfs/GROUP/USER/marin_store_big"
LOSSES = "/scratch/gpfs/GROUP/USER/project/marin/logs/vocab_token_losses"
OUT = "/scratch/gpfs/GROUP/USER/tmp_plots/vocab/cost_groups.json"
TOK = {128000: os.path.dirname(glob.glob("/scratch/gpfs/GROUP/USER/cache/huggingface/hub/models--marin-community--marin-tokenizer/snapshots/*/tokenizer.json")[0]),
       8000: f"{PREFIX}/tokenizers/marin-small/v8000", 256: f"{PREFIX}/tokenizers/marin-small/v256"}
TRAIN = {128000: f"{PREFIX}/fineweb-edu-10B/2026.06.28/train", 8000: f"{PREFIX}/fineweb-edu-10B-v8000/train",
         256: f"{PREFIX}/fineweb-edu-10B-v256/train"}
# pair name -> (tokenizer K, full-data run, 8-pass run); a pair is skipped until both npz files exist
PAIRS = {
    "128K": (128000, "muonh-qwen3-300m-della4xh100-s1-rerun", "muonh-qwen3-300m-della4xh100-rep8-s1"),
    "8K": (8000, "muonh-qwen3-300m-della4xh100-v8k", "muonh-qwen3-300m-della4xh100-v8k-rep8"),
    "8K-s1": (8000, "muonh-qwen3-300m-della4xh100-v8k-s1", "muonh-qwen3-300m-della4xh100-v8k-rep8-s1"),
    # bytes (K=256: ids 0-255 one byte each, specials 256-511 count 0 bytes); 256 types, so equal-count deciles split
    # single byte values at the edges (seeded tie-break): read its frequency panels as coarse
    "bytes": (256, "muonh-qwen3-300m-della4xh100-bytes", "muonh-qwen3-300m-della4xh100-bytes-rep8"),
}
CTX_EDGES = [0, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 10 ** 9]


def tables(k):
    t = json.load(open(os.path.join(TOK[k], "tokenizer.json")))
    n = max(max(t["model"]["vocab"].values()), max(a["id"] for a in t["added_tokens"])) + 1
    nbytes = np.zeros(n, np.int64)
    for s, i in t["model"]["vocab"].items():
        nbytes[i] = len(s)
    eos = [a["id"] for a in t["added_tokens"] if a["content"] == "<|end_of_text|>"][0]
    return nbytes, eos, n


def train_counts(k, n_vocab, n_tokens=200_000_000):
    """Token counts in the first n_tokens of the training cache (cached as an npy beside the outputs)."""
    path = f"{LOSSES}/train_counts_{k}.npy"
    if os.path.exists(path):
        return np.load(path)
    import jax
    from haliax import Axis
    from levanter.data.text.datasets import DatasetComponent, LmDataConfig, UrlDatasetSourceConfig
    from levanter.data.text.formats import TextLmDatasetFormat
    src = UrlDatasetSourceConfig(tags=[], train_urls=[], validation_urls=[], cache_dir=os.path.dirname(TRAIN[k]), format=TextLmDatasetFormat())
    comp = DatasetComponent(source=src, cache_dir=src.cache_dir, format=src.format, tags=[])
    cfg = LmDataConfig(components={"t": comp}, train_weights={"t": 1.0}, tokenizer=TOK[k], cache_dir=None, shuffle=False)
    ds = cfg.train_sets(Axis("position", 4096), key=jax.random.PRNGKey(0))["t"]
    counts = np.zeros(n_vocab, np.int64)

    async def run():
        for a in range(0, n_tokens // 4096, 2048):
            b = await ds.get_batch(list(range(a, a + 2048)))
            counts[:] += np.bincount(np.concatenate([np.asarray(x["input_ids"] if isinstance(x, dict) else x.tokens) for x in b]), minlength=n_vocab)
    asyncio.run(run())
    np.save(path, counts)
    return counts


def pooled(npz_full, npz_rep):
    """Concatenated (tokens [n,T], loss diff [n,T], weight [n,T], subset index [n]) over the 16 subsets, checking the
    windows match."""
    f, r = np.load(npz_full), np.load(npz_rep)
    subs = sorted(k[:-len("/loss")] for k in f.files if k.endswith("/loss"))
    tok, d, w, sub = [], [], [], []
    for i, s in enumerate(subs):
        assert np.array_equal(f[f"{s}/tokens"], r[f"{s}/tokens"]), f"{s}: the two runs saw different windows"
        tok.append(f[f"{s}/tokens"]); d.append(r[f"{s}/loss"].astype(np.float64) - f[f"{s}/loss"]); w.append(f[f"{s}/weight"].astype(np.float64))
        sub.append(np.full(len(tok[-1]), i))
    assert len(subs) == 16, subs
    return np.concatenate(tok), np.concatenate(d), np.concatenate(w), np.concatenate(sub)


def group_cost(key, d, w, b, sub, edges=None, deciles=False):
    """Cost in bits per byte of each group of positions: macro (mean over subsets of each subset's cost) and pooled.
    key: the grouping value per position; sub: the subset index per position."""
    m = w > 0
    key, d, w, b, sub = key[m], d[m], w[m], b[m], sub[m]
    if deciles:   # equal numbers of positions, rarest first; random order among equal keys
        order = np.lexsort((np.random.default_rng(0).random(len(key)), key))
        gid = np.empty(len(key), np.int64)
        gid[order] = np.arange(len(key)) * 10 // len(key)
        G = 10
    else:
        gid = np.searchsorted(edges, key, side="right") - 1
        G = len(edges) - 1
    num = np.bincount(gid, weights=w * d, minlength=G) / np.log(2)
    den = np.bincount(gid, weights=w * b, minlength=G)
    n = np.bincount(gid, minlength=G)
    S = int(sub.max()) + 1
    cell = gid * S + sub
    snum = np.bincount(cell, weights=w * d, minlength=G * S).reshape(G, S) / np.log(2)
    sden = np.bincount(cell, weights=w * b, minlength=G * S).reshape(G, S)
    macro = [float(np.mean(snum[g][sden[g] > 0] / sden[g][sden[g] > 0])) if (sden[g] > 0).any() else None for g in range(G)]
    return macro, [float(x) for x in num / np.maximum(den, 1)], [int(x) for x in n]


res = json.load(open(OUT)) if os.path.exists(OUT) else {}
for pair, (k, full, rep) in PAIRS.items():
    pf, pr = f"{LOSSES}/{full}.npz", f"{LOSSES}/{rep}.npz"
    if not (os.path.exists(pf) and os.path.exists(pr)):
        print(f"{pair}: waiting for {[p for p in (pf, pr) if not os.path.exists(p)]}")
        continue
    nbytes, eos, nv = tables(k)
    counts = train_counts(k, nv)
    tok, d, w, sub = pooled(pf, pr)
    sub = np.repeat(sub[:, None], tok.shape[1], axis=1)
    tgt = np.roll(tok, -1, axis=1)                  # loss at position t predicts token t+1
    b_tgt = nbytes[tgt].astype(np.float64)
    # bytes of context before the predicted token: bytes of tokens 0..t since the window start or the last end-of-text
    nb = nbytes[tok]
    cum = np.cumsum(nb, axis=1)
    reset = np.where(tok == eos, cum, 0)
    last = np.maximum.accumulate(reset, axis=1)
    ctx = (cum - last).astype(np.float64)
    keys = ("cost", "pooled", "n")
    out = {"input": dict(zip(keys, group_cost(counts[tok].astype(np.float64), d, w, b_tgt, sub, deciles=True))),
           "target": dict(zip(keys, group_cost(counts[tgt].astype(np.float64), d, w, b_tgt, sub, deciles=True))),
           "context": dict(zip(keys, group_cost(ctx, d, w, b_tgt, sub, edges=CTX_EDGES)), edges=CTX_EDGES[:-1]),
           "pooled_total": float((w * d).sum() / np.log(2) / (w * b_tgt).sum())}
    m = w > 0
    per_sub = [((w * d)[(sub == i) & m].sum() / np.log(2)) / (w * b_tgt)[(sub == i) & m].sum() for i in range(16)]
    out["macro_total"] = float(np.mean(per_sub))
    res[pair] = out
    print(pair, "macro total", round(out["macro_total"], 4), "pooled", round(out["pooled_total"], 4), "| input", [round(x, 3) for x in out["input"]["cost"]], "| target", [round(x, 3) for x in out["target"]["cost"]],
          "| context", [round(x, 3) for x in out["context"]["cost"]], flush=True)
json.dump(res, open(OUT, "w"), indent=1)
print("written", OUT)
