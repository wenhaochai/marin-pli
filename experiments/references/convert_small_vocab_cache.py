# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Converts the Marin-tokenized fineweb-edu-10B training cache and the Paloma validation caches to the K-token truncated
tokenizer (experiments/references/small_vocab_tokenizer.py), document by document, through the exact id-level expansion.

Writes MARIN_PREFIX/fineweb-edu-10B-v{K}/train and MARIN_PREFIX/tokenized/paloma-v{K}/<subset>/validation (Levanter
TreeCaches, field input_ids). Afterwards it checks: same document counts as the source; for CHECK_DOCS sampled documents
the same decoded text, and for the canonically tokenized ones the same ids as re-encoding with the truncated tokenizer
(see check()). Prints the token
ratio (converted / source) that sets the step count of the fixed-text comparison.

    K=32000 [LIMIT=n] [SKIP_TRAIN=1] python -m experiments.references.convert_small_vocab_cache
"""

import json
import os
import time

import numpy as np
from tokenizers import Tokenizer

from levanter.store.cache import CacheMetadata, SerialCacheWriter, TreeCache

from experiments.references.small_vocab_tokenizer import N_ORDINARY, SRC, expansion

K = int(os.environ["K"])
PREFIX = os.environ.get("MARIN_PREFIX", "/scratch/gpfs/GROUP/USER/marin_store_big")
TOK_DIR = f"{PREFIX}/tokenizers/marin-small/v{K}"
LIMIT = int(os.environ.get("LIMIT", "0"))   # documents per cache, for a quick test (0 = all)
CHUNK = 32768
CHECK_DOCS = int(os.environ.get("CHECK_DOCS", "300"))
EX = {"input_ids": np.zeros((0,), dtype=np.int32)}


def convert_docs(docs, flat, offs):
    """Lists of full-tokenizer ids -> lists of truncated ids (one vectorised gather per chunk)."""
    lens_in = np.fromiter((len(d) for d in docs), np.int64, len(docs))
    ids = np.concatenate([np.asarray(d, np.int64) for d in docs]) if len(docs) else np.zeros(0, np.int64)
    e_len = offs[ids + 1] - offs[ids]
    starts = np.repeat(offs[ids], e_len)
    within = np.arange(e_len.sum()) - np.repeat(np.cumsum(e_len) - e_len, e_len)
    out = flat[starts + within].astype(np.int32)
    cs, ends = np.concatenate([[0], np.cumsum(e_len)]), np.cumsum(lens_in)
    doc_out_len = cs[ends] - cs[ends - lens_in]   # per-document output lengths; an empty document gets 0
    return np.split(out, np.cumsum(doc_out_len)[:-1]), int(ids.size), int(out.size)


def convert_cache(src_dir, dst_dir, flat, offs):
    src = TreeCache.load(src_dir, EX)
    n = len(src) if not LIMIT else min(LIMIT, len(src))
    t0, n_in, n_out = time.time(), 0, 0
    with SerialCacheWriter(dst_dir, EX, metadata=CacheMetadata.empty()) as w:
        for a in range(0, n, CHUNK):
            docs = [d["input_ids"] for d in src.get_batch_sync(list(range(a, min(n, a + CHUNK))))]
            new, i, o = convert_docs(docs, flat, offs)
            w.write_batch([{"input_ids": x} for x in new])
            n_in += i; n_out += o
            if (a // CHUNK) % 20 == 0:
                print(f"  {dst_dir}: {a + len(docs)}/{n} docs, {n_in / 1e9:.2f}B -> {n_out / 1e9:.2f}B tokens, {time.time() - t0:.0f}s", flush=True)
    return n, n_in, n_out


def check(src_dir, dst_dir, n):
    """Same document count, and for sampled documents: (a) the converted ids decode to exactly the same text as the source
    ids; (b) where the source ids are the canonical encoding of their text (full encode(decode) == ids), the converted ids
    equal encoding that text with the truncated tokenizer. Some caches (Paloma) were tokenized in newline-split chunks, so
    their ids are not canonical at chunk edges; the conversion keeps those edges, as tokenizing the same chunks with the
    truncated tokenizer would."""
    full, small = Tokenizer.from_file(os.path.join(SRC, "tokenizer.json")), Tokenizer.from_file(os.path.join(TOK_DIR, "tokenizer.json"))
    src, dst = TreeCache.load(src_dir, EX), TreeCache.load(dst_dir, EX)
    assert len(dst) == n, (len(dst), n)
    idx = np.random.default_rng(0).choice(n, size=min(CHECK_DOCS, n), replace=False).tolist()
    shift = N_ORDINARY - K
    canon = 0
    for a, b in zip(src.get_batch_sync(idx), dst.get_batch_sync(idx)):
        a, b = np.asarray(a["input_ids"]).tolist(), np.asarray(b["input_ids"]).tolist()
        assert [t - shift for t in a if t >= N_ORDINARY] == [t for t in b if t >= K], "special tokens differ"
        ta = full.decode([t for t in a if t < N_ORDINARY], skip_special_tokens=False)
        tb = small.decode([t for t in b if t < K], skip_special_tokens=False)
        assert ta == tb, (src_dir, "decoded text differs")
        body = [t for t in a if t < N_ORDINARY]
        if full.encode(ta, add_special_tokens=False).ids == body:
            canon += 1
            assert [t for t in b if t < K] == small.encode(ta, add_special_tokens=False).ids, (src_dir, "canonical document re-encodes differently")
    return len(idx), canon


def main():
    flat, offs = expansion(K, TOK_DIR)
    out = {"K": K}
    jobs = []
    if not os.environ.get("SKIP_TRAIN"):
        jobs.append(("fineweb-edu-10B", f"{PREFIX}/fineweb-edu-10B/2026.06.28/train", f"{PREFIX}/fineweb-edu-10B-v{K}/train"))
    pal = f"{PREFIX}/tokenized/paloma"
    for d in sorted(os.listdir(pal)):
        if os.path.isdir(os.path.join(pal, d, "validation")):
            jobs.append((d.rsplit("-", 1)[0], os.path.join(pal, d, "validation"), f"{PREFIX}/tokenized/paloma-v{K}/{d.rsplit('-', 1)[0]}/validation"))
    for name, s, d in jobs:
        n, i, o = convert_cache(s, d, flat, offs)
        c, canon = check(s, d, n)
        out[name] = dict(docs=n, tokens_in=i, tokens_out=o, ratio=o / i, checked=c, canonical=canon)
        print(f"{name}: {n} docs, {i} -> {o} tokens (x{o / i:.4f}); {c} sampled documents decode to the same text, {canon} canonical ones equal re-encoding", flush=True)
    tag = f"_limit{LIMIT}" if LIMIT else ""
    path = f"{PREFIX}/tokenized/convert_v{K}{tag}.json"
    json.dump(out, open(path, "w"), indent=1)
    print("written", path)


if __name__ == "__main__":
    main()
