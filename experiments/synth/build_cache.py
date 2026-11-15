# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generated text -> a Levanter training cache that SYNTH_MIX can read (della_muonh_qwen3_scaling.py).

    python -m experiments.synth.build_cache --raw RAW_DIR --name NAME [--max-tokens 65e6] [--min-chars 50]

RAW_DIR holds the generator's calls: call-<i>.txt (texts separated by lines holding only <|endofdocument|>),
call-<i>.done.json (finish reason) and call-<i>.meta.json. A call that did not end normally (no done.json, or a finish
other than stop) loses its last piece, which may be cut mid-text. Pieces under --min-chars, pieces that open like a
refusal and exact duplicates are dropped. Each kept text is one document in the Marin tokenizer: BOS + text + EOS, as in
the fineweb-edu-10B cache. Documents are written in call order until --max-tokens tokens (0: all).
Writes MARIN_PREFIX/synth/NAME/train (the cache) and MARIN_PREFIX/synth/NAME/stats.json (calls, texts, drops, tokens,
document-length quantiles). Run on a vis node with the marin venv and this worktree on PYTHONPATH.
"""

import argparse
import glob
import json
import os
import re

import numpy as np
from levanter.store.cache import CacheMetadata, SerialCacheWriter
from tokenizers import Tokenizer

PREFIX = os.environ.get("MARIN_PREFIX", "/scratch/gpfs/KARTHIKN/wc9403/marin_store_big")
TOKENIZER = glob.glob("/scratch/gpfs/KARTHIKN/wc9403/cache/huggingface/hub/models--marin-community--marin-tokenizer/snapshots/*/tokenizer.json")[0]
BOS, EOS = 128000, 128001
SEP = re.compile(r"^\s*(?:<\|endofdocument\|>|§§§)\s*$", re.M)   # cont, reph: <|endofdocument|>; scratch since 2026-11-14: §§§
REFUSAL = re.compile(r"^(I can't|I cannot|I can not|I'm sorry|I am sorry|Sorry,|As an AI|I'm unable|I am unable)", re.I)
EX = {"input_ids": np.zeros((0,), dtype=np.int32)}


DROPS = {"cut_last": 0, "short": 0, "refusal": 0, "duplicate": 0, "no_text": 0}


def pieces(raw):
    """(call id, text) for every kept piece, in call order; the drops are counted in DROPS."""
    drops = DROPS
    seen = set()
    for p in sorted(glob.glob(os.path.join(raw, "call-*.txt"))):
        text = open(p).read()
        if not text.strip():
            drops["no_text"] += 1
            continue
        done = p[:-4] + ".done.json"
        finish = json.load(open(done)).get("finish") if os.path.exists(done) else None
        parts = [x.strip() for x in SEP.split(text)]
        if finish != "stop" and parts:
            parts = parts[:-1]
            drops["cut_last"] += 1
        for x in parts:
            if len(x) < MIN_CHARS:
                drops["short"] += 1
            elif REFUSAL.match(x):
                drops["refusal"] += 1
            elif x in seen:
                drops["duplicate"] += 1
            else:
                seen.add(x)
                yield os.path.basename(p)[:-4], x


def main():
    global MIN_CHARS
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--max-tokens", type=float, default=0)
    ap.add_argument("--min-chars", type=int, default=50)
    a = ap.parse_args()
    MIN_CHARS = a.min_chars
    out = os.path.join(PREFIX, "synth", a.name)
    if os.path.exists(os.path.join(out, "train")):
        raise SystemExit(f"{out}/train exists; pick a new name")
    os.makedirs(out, exist_ok=True)
    tok = Tokenizer.from_file(TOKENIZER)
    total, n_docs, lens, calls = 0, 0, [], set()
    batch_texts, batch_calls = [], []

    def flush(w):
        nonlocal total, n_docs
        enc = tok.encode_batch(batch_texts, add_special_tokens=False)
        rows = []
        for e in enc:
            ids = np.asarray([BOS] + e.ids + [EOS], dtype=np.int32)
            rows.append({"input_ids": ids})
            total += len(ids)
            lens.append(len(ids))
            n_docs += 1
        w.write_batch(rows)
        calls.update(batch_calls)
        batch_texts.clear()
        batch_calls.clear()

    with SerialCacheWriter(os.path.join(out, "train"), EX, metadata=CacheMetadata.empty()) as w:
        for call, x in pieces(a.raw):
            batch_texts.append(x)
            batch_calls.append(call)
            if len(batch_texts) >= 2048:
                flush(w)
                if a.max_tokens and total >= a.max_tokens:
                    break
        if batch_texts and not (a.max_tokens and total >= a.max_tokens):
            flush(w)
    q = np.quantile(lens, [0.1, 0.5, 0.9, 0.99]).tolist() if lens else []
    stats = {"name": a.name, "raw": a.raw, "calls": len(calls), "documents": n_docs, "tokens": total,
             "doc_tokens_mean": total / max(1, n_docs), "doc_tokens_q10_50_90_99": q, "drops": DROPS}
    json.dump(stats, open(os.path.join(out, "stats.json"), "w"), indent=1)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
