# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Byte caches that carry the 128K Marin tokenizer's token starts, for the H-Net with route="bpe" (hnet_qwen3.py; owner popup
2026-11-03): does 128K's multi-epoch regret come from its segmentation or from its lookup rows?

Converts the Marin-tokenized fineweb-edu-10B training cache and the 16 Paloma validation caches document by document,
through the same id-level expansion as the v256 byte caches (convert_small_vocab_cache.py with K=256: each Marin token
becomes its bytes, a special token its v256 id), so the text, the document order and the byte ids are those of v256. The
first byte of every ordinary Marin token is stored as its id + BPE_FLAG (512, the byte vocabulary's size); special
tokens (BOS 256, EOS 257) stay as they are, so the data pipeline still finds EOS. The model strips the flag
(HNetQwen3LMHeadModel.split_ids), so the labels and the loss are the byte task's.

Writes MARIN_PREFIX/fineweb-edu-10B-v256ts/train and MARIN_PREFIX/tokenized/paloma-v256ts/<subset>/validation (new
directories; no existing cache is touched), then checks each cache:
  (a) same document count as the source; for every document of a LIMIT test (CHECK_DOCS sampled ones otherwise), the
      stripped ids equal the v256 cache's document;
  (b) for the same documents, the flagged positions are the token starts of the source: the cumulative UTF-8 byte
      lengths of the Marin tokens, read from the tokenizer's own vocabulary strings (not from the expansion table).
Prints the bytes per token start (the H-Net's bytes per chunk under route="bpe") and writes convert_v256ts.json.

    [LIMIT=n] [SKIP_TRAIN=1] [OUT_TAG=v256ts] python -m experiments.references.convert_token_start_cache
"""

import json
import os
import time

import numpy as np

from levanter.store.cache import CacheMetadata, SerialCacheWriter, TreeCache

from experiments.references.hnet_qwen3 import BPE_FLAG, N_BYTES
from experiments.references.small_vocab_tokenizer import N_ORDINARY, SRC, expansion

PREFIX = os.environ.get("MARIN_PREFIX", "/scratch/gpfs/GROUP/USER/marin_store_big")
OUT_PREFIX = os.environ.get("OUT_PREFIX", PREFIX)   # a test writes elsewhere
OUT_TAG = os.environ.get("OUT_TAG", "v256ts")
V256_DIR = f"{PREFIX}/tokenizers/marin-small/v256"
LIMIT = int(os.environ.get("LIMIT", "0"))   # documents per cache, for a test (0 = all)
CHECK_DOCS = int(os.environ.get("CHECK_DOCS", "300"))
CHUNK = 32768
EX = {"input_ids": np.zeros((0,), dtype=np.int32)}


def flagged_docs(docs, flat, offs):
    """Lists of Marin ids -> lists of v256 ids with the first byte of every ordinary token + BPE_FLAG (one gather per
    chunk, as convert_small_vocab_cache.convert_docs)."""
    lens_in = np.fromiter((len(d) for d in docs), np.int64, len(docs))
    ids = np.concatenate([np.asarray(d, np.int64) for d in docs]) if len(docs) else np.zeros(0, np.int64)
    e_len = offs[ids + 1] - offs[ids]
    first = np.cumsum(e_len) - e_len
    starts = np.repeat(offs[ids], e_len)
    within = np.arange(e_len.sum()) - np.repeat(first, e_len)
    out = flat[starts + within].astype(np.int32)
    opens = (within == 0) & (np.repeat(ids, e_len) < N_ORDINARY)
    assert np.all(out[opens] < N_BYTES), "an ordinary token expanded to a special id"
    out[opens] += BPE_FLAG
    cs, ends = np.concatenate([[0], np.cumsum(e_len)]), np.cumsum(lens_in)
    doc_out_len = cs[ends] - cs[ends - lens_in]
    return np.split(out, np.cumsum(doc_out_len)[:-1]), int(ids.size), int(out.size), int(opens.sum())


def convert_cache(src_dir, dst_dir, flat, offs):
    src = TreeCache.load(src_dir, EX)
    n = len(src) if not LIMIT else min(LIMIT, len(src))
    t0, n_in, n_out, n_open = time.time(), 0, 0, 0
    with SerialCacheWriter(dst_dir, EX, metadata=CacheMetadata.empty()) as w:
        for a in range(0, n, CHUNK):
            docs = [d["input_ids"] for d in src.get_batch_sync(list(range(a, min(n, a + CHUNK))))]
            new, i, o, k = flagged_docs(docs, flat, offs)
            w.write_batch([{"input_ids": x} for x in new])
            n_in += i; n_out += o; n_open += k
            if (a // CHUNK) % 20 == 0:
                print(f"  {dst_dir}: {a + len(docs)}/{n} docs, {n_in / 1e9:.3f}B tokens -> {n_out / 1e9:.3f}B bytes, {time.time() - t0:.0f}s", flush=True)
    return n, n_in, n_out, n_open


def token_byte_lengths():
    """UTF-8 bytes of every Marin id, from the vocabulary strings (byte-level alphabet: one character per byte)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    nchar = len(bs) + sum(1 for b in range(256) if b not in bs)
    assert nchar == 256
    vocab = json.load(open(os.path.join(SRC, "tokenizer.json")))["model"]["vocab"]
    out = np.zeros(N_ORDINARY + 256, np.int64)
    for s, i in vocab.items():
        if i < N_ORDINARY:
            out[i] = len(s)
    assert np.all(out[:N_ORDINARY] > 0)
    return out


def check(src_dir, v256_dir, dst_dir, n, tok_len):
    src, v256, dst = TreeCache.load(src_dir, EX), TreeCache.load(v256_dir, EX), TreeCache.load(dst_dir, EX)
    assert len(dst) == n and (LIMIT or len(v256) == n), (len(dst), len(v256), n)
    idx = list(range(n)) if LIMIT else sorted(np.random.default_rng(0).choice(n, size=min(CHECK_DOCS, n), replace=False).tolist())
    for a in range(0, len(idx), 4096):
        part = idx[a:a + 4096]
        for s, v, d in zip(src.get_batch_sync(part), v256.get_batch_sync(part), dst.get_batch_sync(part)):
            s, v, d = (np.asarray(x["input_ids"], np.int64) for x in (s, v, d))
            flag = d >= BPE_FLAG
            assert np.array_equal(np.where(flag, d - BPE_FLAG, d), v), (dst_dir, "stripped ids differ from v256")
            # token starts of the source: each Marin token is tok_len bytes (a special token one id, not a start)
            ln = np.where(s < N_ORDINARY, tok_len[s], 1)
            pos = np.cumsum(ln) - ln
            want = np.zeros(len(v), bool)
            want[pos[s < N_ORDINARY]] = True
            assert ln.sum() == len(v) and np.array_equal(flag, want), (dst_dir, "flags differ from the source's token starts")
    return len(idx)


def main():
    flat, offs = expansion(256, V256_DIR)
    tok_len = token_byte_lengths()
    out = {"flag": BPE_FLAG, "limit": LIMIT}
    jobs = []
    if not os.environ.get("SKIP_TRAIN"):
        jobs.append(("fineweb-edu-10B", f"{PREFIX}/fineweb-edu-10B/2026.06.28/train", f"{PREFIX}/fineweb-edu-10B-v256/train",
                     f"{OUT_PREFIX}/fineweb-edu-10B-{OUT_TAG}/train"))
    pal = f"{PREFIX}/tokenized/paloma"
    for d in sorted(os.listdir(pal)):
        if os.path.isdir(os.path.join(pal, d, "validation")):
            sub = d.rsplit("-", 1)[0]
            jobs.append((sub, os.path.join(pal, d, "validation"), f"{PREFIX}/tokenized/paloma-v256/{sub}/validation",
                         f"{OUT_PREFIX}/tokenized/paloma-{OUT_TAG}/{sub}/validation"))
    assert len(jobs) == 16 + (0 if os.environ.get("SKIP_TRAIN") else 1), [j[0] for j in jobs]
    for name, s, v, d in jobs:
        if os.path.exists(d):
            raise FileExistsError(f"{d} exists: this script writes new caches only")
        n, i, o, k = convert_cache(s, d, flat, offs)
        c = check(s, v, d, n, tok_len)
        out[name] = dict(docs=n, tokens_in=i, bytes_out=o, token_starts=k, bytes_per_start=o / k, checked=c)
        print(f"{name}: {n} docs, {i} tokens -> {o} bytes, {k} token starts ({o / k:.4f} bytes per start); {c} documents equal v256 "
              f"after stripping, flags equal the source's token starts", flush=True)
    path = f"{OUT_PREFIX}/tokenized/convert_{OUT_TAG}" + (f"_limit{LIMIT}" if LIMIT else "") + ".json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(out, open(path, "w"), indent=1)
    print("written", path, flush=True)


if __name__ == "__main__":
    main()
