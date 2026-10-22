# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Builds the 8 passes of a repeated run's training subset as one Levanter cache (vocabulary Q10, owner 2026-10-11).

The baseline's 8-pass run reads the first N = steps/8 x batch sequences of its shuffled training set and repeats them in
the same order every pass (MixtureDataset RESTART). This script takes exactly those N sequences (the run's own data
config: same caches, linear permutation, data seed, max_train_batches) and writes
  pass 1: the N sequences in the run's order;
  passes 2-8 (MODE=rs): the same sequences, each pass in a new random order (seed 10000 + pass);
  passes 2-8 (MODE=bpd): each pass in the same new order as rs, every sequence re-segmented with BPE-dropout P (the
           Marin tokenizer with ignore_merges off, so a word in the vocabulary can split too); the re-segmented text is
           concatenated and cut into SEQ_LEN windows again (more tokens than the original, so more windows);
  passes 2-8 (MODE=repack, survey idea 5, owner popup 2026-10-21): the subset's windows are cut into pieces before every
           BOS token (a window edge also ends a piece, since neighbouring windows are not neighbouring text), the pieces
           of each pass are shuffled (seed 20000 + pass) and packed into SEQ_LEN windows again, so the documents meet new
           neighbours and new window edges; a pass has exactly the subset's tokens and windows;
  passes 2-8 (MODE=rsplit, survey idea 3, owner popup 2026-10-21): in the same new order as rs, every rare token (ids
           8000 and up, specials excluded) is replaced with probability P by the sequence of its 8K pieces (the 8K map
           of the tokenizer, marin-small/v8000/expand_full.npz or expand_v<K>k.npz), so inputs and targets change
           together; the longer stream is cut into SEQ_LEN windows again (more windows, as bpd).
Each written document is one SEQ_LEN window, so Levanter's token dataset reads them back window for window. The training
run (della_muonh_qwen3_scaling.py PASS_DATA=rs | bpd<P>) reads this cache in order with no shuffle.

    SIZE=300m [VOCAB_K=8000] MODE=rs|repack|bpd|rsplit [P=0.1] python -m experiments.references.vocab_materialize_passes
Writes MARIN_PREFIX/materialized/<size>[-v<K>k|-bytes]-rep8-<rs|bpd<P>>/{train,meta.json}. Checks at the end: pass 1 equals the
subset; every rs pass is a permutation of it; a sample of bpd windows decodes to the original text (window edges aside).
"""

import asyncio
import json
import os
import time
from unittest import mock

import jax
import numpy as np

MODE = os.environ["MODE"]
P = float(os.environ.get("P", "0.1"))
SIZE = os.environ.get("SIZE", "300m")
K = int(os.environ.get("VOCAB_K", "0"))
PASSES, SEQ = 8, 4096
if MODE not in ("rs", "bpd", "repack", "rsplit") or (MODE == "bpd" and (K or not 0 < P < 1)) or (MODE == "rsplit" and (K not in (0, 16000, 32000, 64000) or not 0 < P <= 1)):
    raise ValueError("MODE=rs or repack (any vocabulary), MODE=bpd with 0 < P < 1 (the Marin tokenizer), MODE=rsplit with 0 < P <= 1 (Marin or its 64K/32K/16K truncation)")
TAG = MODE if MODE in ("rs", "repack") else f"{MODE}{P:g}"
BOS = (K or 128000) if K != 256 else 256   # the first special of each tokenizer opens a document (checked in the subset)
# the run's own data config, without PASS_DATA (that is what reads this cache)
for k in ("PASS_DATA", "VARIANT", "SMOKE_STEPS", "TOTAL_STEPS", "SEQ_LEN", "BATCH", "TPP", "SEED"):
    os.environ.pop(k, None)
os.environ.update(SIZE=SIZE, DATA_EPOCHS="8", DRY_RUN="1", MEMSTATS_EVERY="0")
if K:
    os.environ["VOCAB_K"] = str(K)

from haliax import Axis  # noqa: E402
from levanter.store.cache import CacheMetadata, SerialCacheWriter, TreeCache  # noqa: E402

from experiments.references import della_muonh_qwen3_scaling as scaling  # noqa: E402

PREFIX = scaling._PREFIX
OUT = f"{PREFIX}/materialized/{SIZE}" + ("-bytes" if K == 256 else f"-v{K // 1000}k" if K else "") + f"-rep8-{TAG}"   # same rule as _pass_data_dir in della_muonh_qwen3_scaling.py
EX = {"input_ids": np.zeros((0,), dtype=np.int32)}


def subset():
    """The run's repeated subset, [N, SEQ] int32, in the order the run reads pass 1."""
    tc = scaling.muonh_qwen3_run(SIZE).build_config(mock.MagicMock()).train_config
    batch, data = tc.trainer.train_batch_size, tc.data
    (name, n_batches), = data.max_train_batches.items()
    if not K:   # the 128K component's path is resolved by the executor at run time; point it at the same prebuilt cache
        import dataclasses
        import glob
        tok = os.path.dirname(glob.glob("/scratch/gpfs/GROUP/USER/cache/huggingface/hub/models--marin-community--marin-tokenizer/snapshots/*/tokenizer.json")[0])
        data = dataclasses.replace(data, tokenizer=tok, components={**data.components, name: scaling._cache_component(f"{PREFIX}/fineweb-edu-10B/2026.06.28/train")})
    _, shuffle_key = jax.random.split(jax.random.PRNGKey(tc.data_seed))   # train_lm: data_key = PRNGKey(data_seed); train_set splits it
    ds = data.train_sets(Axis("position", SEQ), key=shuffle_key, initial_batch_size=batch)[name]
    n = len(ds.as_sync_dataset())
    assert n == n_batches * batch, (n, n_batches, batch)
    out = np.zeros((n, SEQ), np.int32)

    async def read():
        for a in range(0, n, 2048):
            b = await ds.get_batch(list(range(a, min(n, a + 2048))))
            out[a:a + len(b)] = np.stack([np.asarray(x["input_ids"] if isinstance(x, dict) else x.tokens) for x in b])
    asyncio.run(read())
    return out, tc


def order(e, n):
    return np.arange(n) if e == 1 else np.random.default_rng(10000 + e).permutation(n)


def repack(rows, e):
    """The windows ``rows`` [n, SEQ] cut before every BOS, the pieces shuffled with seed 20000 + e and packed into n
    windows again (the same tokens, new neighbours and new window edges)."""
    n = len(rows)
    flat = rows.reshape(-1)
    starts = np.union1d(np.flatnonzero(flat == BOS), np.arange(0, n * SEQ, SEQ))
    ends = np.append(starts[1:], n * SEQ)
    perm = np.random.default_rng(20000 + e).permutation(len(starts))
    out = np.empty_like(flat)
    pos = 0
    for i in perm:
        a, b = starts[i], ends[i]
        out[pos:pos + b - a] = flat[a:b]
        pos += b - a
    assert pos == n * SEQ
    return out.reshape(n, SEQ), len(starts)


def rsplit_tools():
    """(flat, offs, counts, V) of the 8K map of this tokenizer: token t's pieces are flat[offs[t]:offs[t + 1]]."""
    path = f"{PREFIX}/tokenizers/marin-small/v8000/" + (f"expand_v{K // 1000}k.npz" if K else "expand_full.npz")
    z = np.load(path)
    flat, offs = z["flat"].astype(np.int32), z["offs"].astype(np.int64)
    return flat, offs, np.diff(offs), len(offs) - 1


def rsplit(rows, e, a, tools):
    """The windows ``rows`` with each rare token replaced, with probability P, by its 8K pieces; one id stream.
    The draw is a fixed function of (pass, chunk start)."""
    flat, offs, counts, V = tools
    x = rows.reshape(-1).astype(np.int64)
    rare = (x >= 8000) & (x < V - 256)
    split = rare & (np.random.default_rng([30000 + e, a]).random(len(x)) < P)
    lengths = np.where(split, counts[x], 1)
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    tok = np.repeat(np.arange(len(x)), lengths)
    within = np.arange(int(lengths.sum())) - starts[tok]
    out = np.where(split[tok], flat[np.minimum(offs[x[tok]] + within, len(flat) - 1)], x[tok])
    return out.astype(np.int32), int(split.sum())


def bpd_tools():
    """(dropout tokenizer, id -> bytes, special ids) for the Marin tokenizer."""
    import glob

    from tokenizers import Tokenizer
    path = glob.glob("/scratch/gpfs/GROUP/USER/cache/huggingface/hub/models--marin-community--marin-tokenizer/snapshots/*/tokenizer.json")[0]
    j = json.load(open(path))
    j["model"]["dropout"], j["model"]["ignore_merges"] = P, False
    tok = Tokenizer.from_str(json.dumps(j))
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, m = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + m); m += 1
    u2b = {chr(c): b for b, c in zip(bs, cs)}
    special = {a["id"] for a in j["added_tokens"]}
    V = max(max(j["model"]["vocab"].values()), max(special)) + 1
    idb = [b""] * V
    for s_, i in j["model"]["vocab"].items():
        idb[i] = bytes(u2b[c] for c in s_)
    return tok, idb, special


def resegment(rows, tok, idb, special):
    """Re-encode each window's text with BPE-dropout, keeping special tokens; returns one id stream and the count of
    replacement characters (partial UTF-8 at window edges)."""
    texts, plan = [], []   # plan per window: list of ("t", text index) or ("s", special id)
    bad = 0
    for row in rows:
        parts, cur = [], []
        for t in row.tolist():
            if t in special:
                if cur:
                    s_ = b"".join(idb[x] for x in cur).decode("utf-8", errors="replace"); bad += s_.count("�")
                    texts.append(s_); parts.append(("t", len(texts) - 1)); cur = []
                parts.append(("s", t))
            else:
                cur.append(t)
        if cur:
            s_ = b"".join(idb[x] for x in cur).decode("utf-8", errors="replace"); bad += s_.count("�")
            texts.append(s_); parts.append(("t", len(texts) - 1))
        plan.append(parts)
    enc = tok.encode_batch(texts, add_special_tokens=False)
    stream = []
    for parts in plan:
        for kind, v in parts:
            stream.extend(enc[v].ids if kind == "t" else [v])
    return np.asarray(stream, np.int32), bad


def main():
    t0 = time.time()
    sub, tc = subset()
    if os.environ.get("SUBSET_ONLY"):   # a quick check of the subset read
        print("subset", sub.shape, sub[0, :8].tolist(), flush=True)
        return
    n = len(sub)
    print(f"subset: {n} sequences of {SEQ} ({n * SEQ / 1e9:.3f}B tokens), {time.time() - t0:.0f}s", flush=True)
    os.makedirs(OUT, exist_ok=True)
    meta = dict(size=SIZE, vocab_k=K, mode=MODE, p=P if MODE == "bpd" else None, subset_sequences=n, seq_len=SEQ, pass_sequences=[], replaced_chars=0)
    tools = bpd_tools() if MODE == "bpd" else rsplit_tools() if MODE == "rsplit" else None
    carry = np.zeros(0, np.int32)
    with SerialCacheWriter(f"{OUT}/train", EX, metadata=CacheMetadata.empty()) as w:
        for e in range(1, PASSES + 1):
            o = order(e, n)
            if MODE == "rs" or e == 1:
                for a in range(0, n, 8192):
                    w.write_batch([{"input_ids": r} for r in sub[o[a:a + 8192]]])
                meta["pass_sequences"].append(n)
                continue
            if MODE == "repack":
                rp, pieces = repack(sub, e)
                for a in range(0, n, 8192):
                    w.write_batch([{"input_ids": r} for r in rp[a:a + 8192]])
                meta["pass_sequences"].append(n)
                meta.setdefault("pieces", []).append(pieces)
                print(f"pass {e}: {pieces} pieces repacked, {time.time() - t0:.0f}s", flush=True)
                continue
            written = 0
            for a in range(0, n, 8192):   # windows of the re-segmented stream; a pass's tail carries into the next chunk
                if MODE == "rsplit":
                    stream, nsplit = rsplit(sub[o[a:a + 8192]], e, a, tools)
                    meta["split_tokens"] = meta.get("split_tokens", 0) + nsplit
                else:
                    stream, bad = resegment(sub[o[a:a + 8192]], *tools)
                    meta["replaced_chars"] += bad
                stream = np.concatenate([carry, stream])
                k = len(stream) // SEQ
                w.write_batch([{"input_ids": r} for r in stream[:k * SEQ].reshape(k, SEQ)])
                carry, written = stream[k * SEQ:], written + k
            meta["pass_sequences"].append(written)
            carry = np.zeros(0, np.int32)   # the last partial window of a pass is dropped
            print(f"pass {e}: {written} windows ({written / n:.4f} x the subset), {time.time() - t0:.0f}s", flush=True)
    meta["sequences"] = int(sum(meta["pass_sequences"]))
    meta["tokens"] = meta["sequences"] * SEQ
    # checks
    c = TreeCache.load(f"{OUT}/train", EX)
    assert len(c) == meta["sequences"], (len(c), meta["sequences"])
    idx = np.random.default_rng(0).choice(n, 64, replace=False)
    got = np.stack([np.asarray(x["input_ids"]) for x in c.get_batch_sync(idx.tolist())])
    assert np.array_equal(got, sub[idx]), "pass 1 differs from the subset"
    if MODE == "repack":
        assert int((sub == BOS).sum()) > n // 4, f"BOS={BOS} is rare in the subset: wrong document separator"
        for e in (2, PASSES):
            start = (e - 1) * n
            got = np.stack([np.asarray(x["input_ids"]) for x in c.get_batch_sync(list(range(start, start + n)))])
            assert np.array_equal(np.bincount(got.reshape(-1), minlength=int(sub.max()) + 1), np.bincount(sub.reshape(-1), minlength=int(sub.max()) + 1)), f"pass {e} has other tokens than the subset"
            assert not np.array_equal(got[:64], sub[:64]), f"pass {e} starts like the subset: no repacking"
        meta["check_token_counts_equal"] = True
    elif MODE == "rs":
        for e in (2, PASSES):
            o = order(e, n); start = (e - 1) * n
            got = np.stack([np.asarray(x["input_ids"]) for x in c.get_batch_sync([start + int(i) for i in idx])])
            assert np.array_equal(got, sub[o[idx]]), f"pass {e} is not the subset in its order"
    elif MODE == "rsplit":   # the first pass-2 windows equal the subset in rs order with the drawn rare tokens expanded
        o = order(2, n)
        want, _ = rsplit(sub[o[:8192]], 2, 0, tools)
        got = np.concatenate([np.asarray(x["input_ids"]) for x in c.get_batch_sync(list(range(n, n + 64)))])
        assert np.array_equal(got, want[:len(got)]), "pass 2 is not the expanded subset"
        flat, offs, counts, V = tools
        r0 = int(((sub >= 8000) & (sub < V - 256)).sum()); r2 = int(((want >= 8000) & (want < V - 256)).sum())
        meta["rare_share_pass1"], meta["check_expanded"] = r0 / sub.size, True
        print(f"rare tokens: pass 1 {r0 / sub.size:.4f} of positions; pass 2 first chunk keeps {r2 / max(1, int(((sub[o[:8192]] >= 8000) & (sub[o[:8192]] < V - 256)).sum())):.3f} of them", flush=True)
    else:
        tok, idb, special = tools
        o = order(2, n); start = n
        first = [np.asarray(x["input_ids"]) for x in c.get_batch_sync(list(range(start, start + 8)))]
        a_ = b"".join(idb[t] for t in np.concatenate(first).tolist() if t not in special)
        b_ = b"".join(idb[t] for t in sub[o[:16]].reshape(-1).tolist() if t not in special)
        same = a_[:4000].decode("utf-8", errors="replace").replace("�", "") == b_[:4000].decode("utf-8", errors="replace").replace("�", "")
        assert same, "re-segmented text differs from the original"
        changed = not np.array_equal(first[0], sub[o[0]])
        assert changed, "the first re-segmented window equals the original: dropout had no effect"
        meta["check_text_equal"] = True
    meta["seconds"] = round(time.time() - t0)
    json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=1)
    print("written", OUT, json.dumps({k: v for k, v in meta.items() if k != "pass_sequences"}), flush=True)


if __name__ == "__main__":
    main()
