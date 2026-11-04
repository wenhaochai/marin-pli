# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Builds the training caches and the Paloma caches of the runs with a vocabulary that shrinks as data repeats (owner 2026-11-04,
a friend's two ideas; experiments/references/vocab_shrink_qwen3.py is the model).

Every arm reads the same text: the Marin tokenizer's repeated eighth at 300m (183,040 windows of 4096 tokens, pass 1 of
materialized/300m-rep8-rs, in the order the plain 8-pass run reads them). For pass e of 8 each source window takes a view, a K,
from that pass's shares, and goes through the exact id-level expansion to K tokens (small_vocab_tokenizer.expansion; a truncation's
BPE on a Marin token's bytes), kept in Marin ids. Each view's stream is cut into windows of the arm's SEQ_LEN, a partial window
at the end of a pass is dropped, and one written window holds one view. A pass with one view keeps its order, so
equal views in later passes are equal windows, as in a plain 8-pass run; a pass with several views shuffles each chunk of
8192 source windows (seed 30000 + pass), so batches mix them.

Arms (ARMS below): vsfix<K> K on all 8 passes (the control, on the same text as the other arms); vscut<K> path 2, one view per
pass, cut when the epoch count doubles; vsmix<e> path 1, the friend's shifting shares (read off their figure) on all five
vocabularies, with the eighth pass at the figure's epoch <e>.

    python -m experiments.references.vocab_shrink_passes check                 (expansion against the K caches of Paloma)
    python -m experiments.references.vocab_shrink_passes paloma 64000 32000 16000 8000  (Paloma caches of K in Marin ids: tokenized/paloma-v<K>m)
    python -m experiments.references.vocab_shrink_passes <arm> [...]           (MARIN_PREFIX/materialized/300m-rep8-<arm>/{train,meta.json})
meta.json: schedule (shares per pass), seq_len, eval_k, pass_sequences, sequences, view_windows per pass. Checks while writing:
every window of a K view holds ids below K or specials only; a pass of one view holds the source text, byte for byte, on a
sample of windows.
"""

import json
import math
import os
import sys
import time

import numpy as np

from levanter.store.cache import CacheMetadata, SerialCacheWriter, TreeCache

from experiments.references.small_vocab_tokenizer import N_ORDINARY, convert, expansion

PREFIX = os.environ.get("MARIN_PREFIX", "/scratch/gpfs/GROUP/USER/marin_store_big")
SOURCE = f"{PREFIX}/materialized/300m-rep8-rs"
EX = {"input_ids": np.zeros((0,), dtype=np.int32)}
EQW = {128000: 4096, 64000: 4224, 32000: 4480, 16000: 4992, 8000: 5504}   # each K's equal-text window (the page's fig-eqwin)
VIEWS = (128000, 64000, 32000, 16000, 8000)
# The friend's Path 1 figure, shares of sequences at its epoch marks, read off the plot (rounded to sum to 1)
MARKS = {1: (0.60, 0.18, 0.10, 0.07, 0.05), 2: (0.35, 0.27, 0.19, 0.11, 0.08), 4: (0.13, 0.19, 0.25, 0.22, 0.21), 8: (0.00, 0.05, 0.16, 0.29, 0.50)}


def shares_at(epoch: float):
    """The figure's shares at an epoch, linear in log2(epoch) between its marks; epochs below 1 take the first mark's."""
    x = math.log2(max(epoch, 1.0))
    lo, hi = min(math.floor(x), 2), min(math.floor(x), 2) + 1
    f = min(x - lo, 1.0)
    a, b = MARKS[2 ** lo], MARKS[2 ** hi]
    s = [(1 - f) * u + f * v for u, v in zip(a, b)]
    t = sum(s)
    return {k: round(v / t, 4) for k, v in zip(VIEWS, s) if v > 0}


def _mix(last_epoch: float):
    """Pass e of 8 at the figure's epoch 1 + (e - 1) x (last_epoch - 1) / 7 on the log scale: e^(log(last_epoch)/log(8))."""
    return [shares_at(e ** (math.log(last_epoch) / math.log(8))) for e in range(1, 9)]


def _one(ks):
    return [{k: 1.0} for k in ks]


ARMS = {
    # name: (views per pass, SEQ_LEN, the K that evaluation in training uses)
    "vsfix32k": (_one([32000] * 8), 4480, 32000),
    "vsfix16k": (_one([16000] * 8), 4992, 16000),
    # the figure's rule on our 8 passes: cut at epochs 2 and 4 (8 ends the run)
    "vscut32k": (_one([128000, 64000, 64000, 32000, 32000, 32000, 32000, 32000]), 4480, 32000),
    # the same rule one doubling earlier, so that the last half runs at 16K, as in the figure's 16 epochs
    "vscut16k": (_one([128000, 64000, 32000, 32000, 16000, 16000, 16000, 16000]), 4992, 16000),
    "vsmix8": (_mix(8.0), 4992, 16000),    # the figure's real epoch count
    "vsmix4": (_mix(4.0), 4992, 16000),    # a slower shift: the eighth pass at the figure's epoch 4
}


def expansions(keys):
    """K -> (flat, offs), every Marin id to its K-token pieces, in Marin ids (K's specials back at N_ORDINARY + j)."""
    out = {}
    for k in keys:
        if k == N_ORDINARY:
            continue
        flat, offs = expansion(k, f"{PREFIX}/tokenizers/marin-small/v{k}")
        out[k] = (np.where(flat >= k, flat + (N_ORDINARY - k), flat).astype(np.int32), offs)
    return out


def expand(ids, k, ex):
    if k == N_ORDINARY:
        return np.asarray(ids, np.int32).ravel()
    return np.asarray(convert(np.asarray(ids, np.int64).ravel(), *ex[k]), np.int32).ravel()


def read_docs(path, a, b):
    c = TreeCache.load(path, EX)
    return [np.asarray(x["input_ids"], np.int64) for x in c.get_batch_sync(list(range(a, b)))]


def read_all(path):
    c = TreeCache.load(path, EX)
    return [np.asarray(x["input_ids"], np.int64) for a in range(0, len(c), 4096) for x in c.get_batch_sync(list(range(a, min(len(c), a + 4096))))]


def paloma_subsets():
    return sorted(d for d in os.listdir(f"{PREFIX}/tokenized/paloma-v8000") if os.path.isdir(f"{PREFIX}/tokenized/paloma-v8000/{d}/validation"))


def marin_paloma(sub):
    import glob  # noqa: PLC0415
    (path,) = glob.glob(f"{PREFIX}/paloma/{sub}-marin-tokenizer/*/validation")
    return path


def check_expansion(keys):
    """The expansion of the Marin Paloma docs against the K caches of Paloma, document by document, on 4 subsets."""
    ex = expansions(keys)
    for sub in ("ptb", "gab", "c4_en", "dolma_100_programing_languages"):
        src = read_all(marin_paloma(sub))
        for k in keys:
            got = read_all(f"{PREFIX}/tokenized/paloma-v{k}/{sub}/validation")
            assert len(got) == len(src), (sub, k, len(got), len(src))
            bad = 0
            for d, g in zip(src, got):
                e = expand(d, k, ex)
                g2 = np.where(g >= k, g + (N_ORDINARY - k), g)
                bad += not np.array_equal(e, g2)
            print(f"{sub} K={k}: {len(src)} documents, {bad} differ", flush=True)
            assert bad == 0, (sub, k)
    print("expansion check ok", flush=True)


def paloma(keys):
    """tokenized/paloma-v<K>m/<subset>/validation: the K caches of Paloma with their specials at the Marin ids."""
    for k in keys:
        for sub in paloma_subsets():
            docs = read_all(f"{PREFIX}/tokenized/paloma-v{k}/{sub}/validation")
            out = f"{PREFIX}/tokenized/paloma-v{k}m/{sub}/validation"
            os.makedirs(out, exist_ok=True)
            with SerialCacheWriter(out, EX, metadata=CacheMetadata.empty()) as w:
                for a in range(0, len(docs), 4096):
                    w.write_batch([{"input_ids": np.where(d >= k, d + (N_ORDINARY - k), d).astype(np.int32)} for d in docs[a:a + 4096]])
            print(f"K={k} {sub}: {len(docs)} documents", flush=True)


def source_subset():
    m = json.load(open(f"{SOURCE}/meta.json"))
    n = m["subset_sequences"]
    assert m["mode"] == "rs" and m["seq_len"] == 4096 and m["vocab_k"] == 0, m
    rows = np.zeros((n, 4096), np.int32)
    c = TreeCache.load(f"{SOURCE}/train", EX)
    for a in range(0, n, 8192):
        for i, x in enumerate(c.get_batch_sync(list(range(a, min(n, a + 8192))))):
            rows[a + i] = np.asarray(x["input_ids"], np.int32)
    return rows


def build(arm, sub, ex):
    t0 = time.time()
    views, seq, eval_k = ARMS[arm]
    n = len(sub)
    out = f"{PREFIX}/materialized/300m-rep8-{arm}"
    os.makedirs(out, exist_ok=True)
    meta = dict(arm=arm, source=SOURCE, subset_sequences=n, seq_len=seq, eval_k=eval_k, schedule=[{str(k): v for k, v in p.items()} for p in views], pass_sequences=[], view_windows=[])
    with SerialCacheWriter(f"{out}/train", EX, metadata=CacheMetadata.empty()) as w:
        for e, shares in enumerate(views, 1):
            ks = list(shares)
            rng = np.random.default_rng(30000 + e)
            draw = rng.choice(ks, size=n, p=np.asarray([shares[k] for k in ks]) / sum(shares.values())) if len(ks) > 1 else np.full(n, ks[0])
            carry = {k: np.zeros(0, np.int32) for k in ks}
            counts = {k: 0 for k in ks}
            for a in range(0, n, 8192):
                rows, d = sub[a:a + 8192], draw[a:a + 8192]
                parts = []
                for k in ks:
                    sel = d == k
                    if not sel.any():
                        continue
                    s = np.concatenate([carry[k], expand(rows[sel], k, ex)])
                    m = len(s) // seq
                    if m:
                        win = s[:m * seq].reshape(m, seq)
                        assert ((win < k) | (win >= N_ORDINARY)).all(), (arm, e, k)
                        parts.append(win)
                        counts[k] += m
                    carry[k] = s[m * seq:]
                if not parts:
                    continue
                win = np.concatenate(parts)
                if len(ks) > 1:
                    win = win[rng.permutation(len(win))]
                w.write_batch([{"input_ids": r} for r in win])
                if a == 0 and len(ks) == 1 and ks[0] != N_ORDINARY:
                    # the same text: expanding the first source windows again gives these windows
                    assert np.array_equal(win[0], expand(sub[:8192], ks[0], ex)[:seq]), (arm, e)
            meta["pass_sequences"].append(int(sum(counts.values())))
            meta["view_windows"].append({str(k): int(v) for k, v in counts.items()})
            print(f"{arm} pass {e}: {meta['view_windows'][-1]} ({time.time() - t0:.0f}s)", flush=True)
    meta["sequences"] = int(sum(meta["pass_sequences"]))
    meta["tokens"] = meta["sequences"] * seq
    meta["seconds"] = round(time.time() - t0)
    json.dump(meta, open(f"{out}/meta.json", "w"), indent=1)
    print(f"{arm}: {meta['sequences']} windows of {seq}, {meta['sequences'] // 128} steps at batch 128, {meta['seconds']} s", flush=True)


def main():
    a = sys.argv[1:]
    if a[0] == "check":
        check_expansion([64000, 32000, 16000, 8000])
    elif a[0] == "paloma":
        paloma([int(x) for x in a[1:]])
    elif a[0] == "show":
        for arm, (v, seq, ek) in ARMS.items():
            print(arm, seq, ek, [p for p in v])
    else:
        sub = source_subset()
        print("source", sub.shape, flush=True)
        ex = expansions([k for k in VIEWS if any(k in p for arm in a for p in ARMS[arm][0])])
        for arm in a:
            build(arm, sub, ex)


if __name__ == "__main__":
    main()
