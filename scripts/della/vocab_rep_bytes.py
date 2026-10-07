"""Q3 of blogs/vocab-overfitting.html: how many bytes of text does each tokenizer's 8-pass part hold?

The 8-pass runs train on the first round(steps / 8) batches of the shuffled training sequences (levanter
max_train_batches, linear permutation, data seed 42). This script builds those sequences with levanter's own data code,
exactly as train_lm does (train_set: split(PRNGKey(42)) -> shuffle key; train_sets with the run's batch of 128 and
max_train_batches), for the Marin tokenizer (128K, 11,444 steps) and its 8K truncation (15,464 steps), and counts their
UTF-8 bytes: a byte-level BPE token's string has one character per byte, special tokens count 0. The two parts are
different random selections (the sequence count differs, so the permutation differs); only their sizes should match.

Run on a vis node: nice -n 19 .venv/bin/python scripts/della/vocab_rep_bytes.py
"""
import asyncio
import json
import os

import jax
import numpy as np

from haliax import Axis
from levanter.data.text.datasets import DatasetComponent, LmDataConfig, UrlDatasetSourceConfig
from levanter.data.text.formats import TextLmDatasetFormat

PREFIX = "/scratch/gpfs/GROUP/USER/marin_store_big"
SRC = {"128K": (f"{PREFIX}/fineweb-edu-10B/2026.06.28/train", f"{PREFIX}/../cache/huggingface/hub/models--marin-community--marin-tokenizer", 11444),
       "8K": (f"{PREFIX}/fineweb-edu-10B-v8000/train", f"{PREFIX}/tokenizers/marin-small/v8000", 15464)}
EPOCHS, BATCH, SEQ = 8, 128, 4096


def token_bytes(tok_dir: str) -> np.ndarray:
    import glob
    path = (glob.glob(f"{tok_dir}/snapshots/*/tokenizer.json") or [f"{tok_dir}/tokenizer.json"])[0]
    t = json.load(open(path))
    vocab = t["model"]["vocab"]
    n = max(max(vocab.values()), max(a["id"] for a in t["added_tokens"])) + 1
    out = np.zeros(n, np.int64)
    for s, i in vocab.items():
        out[i] = len(s)   # byte-level alphabet: one character per byte
    for a in t["added_tokens"]:
        out[a["id"]] = 0
    return out


def part(path: str, steps: int):
    src = UrlDatasetSourceConfig(tags=[], train_urls=[], validation_urls=[], cache_dir=os.path.dirname(path), format=TextLmDatasetFormat())
    comp = DatasetComponent(source=src, cache_dir=src.cache_dir, format=src.format, tags=[])
    cfg = LmDataConfig(components={"fineweb-edu-10B": comp}, train_weights={"fineweb-edu-10B": 1.0}, tokenizer=None, cache_dir=None,
                       shuffle=True, permutation_type="linear", max_train_batches={"fineweb-edu-10B": round(steps / EPOCHS)})
    _, shuffle_key = jax.random.split(jax.random.PRNGKey(42))
    return cfg.train_sets(Axis("position", SEQ), initial_batch_size=BATCH, key=shuffle_key)["fineweb-edu-10B"]


async def count(ds, table):
    n = await ds.async_len()
    total, toks = 0, 0
    for a in range(0, n, 2048):
        batch = await ds.get_batch(list(range(a, min(n, a + 2048))))
        ids = np.concatenate([np.asarray(b["input_ids"] if isinstance(b, dict) else b.tokens) for b in batch])
        total += int(table[ids].sum())
        toks += ids.size
    return n, toks, total


res = {}
for name, (path, tok_dir, steps) in SRC.items():
    ds = part(path, steps)
    n, toks, b = asyncio.run(count(ds, token_bytes(tok_dir)))
    res[name] = dict(sequences=n, tokens=toks, bytes=b, bytes_per_token=b / toks)
    print(name, res[name], flush=True)
print(f"8K / 128K bytes: {res['8K']['bytes'] / res['128K']['bytes']:.4f}")
json.dump(res, open("/scratch/gpfs/GROUP/USER/tmp_plots/vocab/rep_bytes.json", "w"), indent=1)
