# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Split a generated cache in two halves of equal tokens by the function-word rate of each document (synth-pretrain).

    python -m experiments.synth.build_fw_split --src r1contfull --hi r1contfwhi --lo r1contfwlo

The generated prose of synth-pretrain holds fewer function words than fineweb-edu: 17.5% of the words of continuation
prose are one of the 12 words in FW, 24.6% of fineweb-edu prose (2026-11-17, 4000 documents each), and the two
distributions of documents barely overlap. The halves test whether that density matters, with one generator, one mode
and the same tokens on both sides. A document's rate is the share of FW among its words (lowercased [A-Za-z']+ runs);
the documents with at least 30 words, ranked by rate, fill the high half up to half the tokens; the few shorter documents
go half to each side in a random order (numpy default_rng(6)). Each half keeps the source order and writes
MARIN_PREFIX/synth/NAME/train and MARIN_PREFIX/synth/NAME/stats.json, so SYNTH_MIX=NAME:1 reads it.
"""

import argparse
import glob
import json
import os
import re
import time

import numpy as np
from levanter.store.cache import CacheMetadata, SerialCacheWriter, TreeCache
from tokenizers import Tokenizer

PREFIX = os.environ.get("MARIN_PREFIX", "/scratch/gpfs/KARTHIKN/wc9403/marin_store_big")
TOKENIZER = glob.glob("/scratch/gpfs/KARTHIKN/wc9403/cache/huggingface/hub/models--marin-community--marin-tokenizer/snapshots/*/tokenizer.json")[0]
EX = {"input_ids": np.zeros((0,), dtype=np.int32)}
FW = {"the", "of", "is", "be", "was", "that", "are", "a", "an", "and", "to", "in"}
WORD = re.compile(r"[A-Za-z']+")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--hi", required=True)
    ap.add_argument("--lo", required=True)
    ap.add_argument("--batch", type=int, default=8192)
    a = ap.parse_args()
    for name in (a.hi, a.lo):
        if os.path.exists(os.path.join(PREFIX, "synth", name, "train")):
            raise SystemExit(f"{PREFIX}/synth/{name}/train exists; pick a new name")
    t0 = time.time()
    tok = Tokenizer.from_file(TOKENIZER)
    src = TreeCache.load(os.path.join(PREFIX, "synth", a.src, "train"), EX)
    docs = []
    for i0 in range(0, len(src), a.batch):
        docs += [np.asarray(r["input_ids"], dtype=np.int32) for r in src.get_batch_sync(slice(i0, min(len(src), i0 + a.batch)))]
    texts = tok.decode_batch([d.tolist() for d in docs], skip_special_tokens=True)
    rate = np.full(len(docs), np.nan)
    for i, t in enumerate(texts):
        ws = [w.lower() for w in WORD.findall(t)]
        if len(ws) >= 30:
            rate[i] = sum(w in FW for w in ws) / len(ws)
    lens = np.array([len(d) for d in docs])
    short = np.isnan(rate)
    rng = np.random.default_rng(6)
    short_idx = rng.permutation(np.nonzero(short)[0])
    ranked = np.argsort(-np.where(short, -1.0, rate), kind="stable")
    ranked = ranked[~short[ranked]]
    short_hi, short_lo = short_idx[: len(short_idx) // 2], short_idx[len(short_idx) // 2:]
    cut = np.searchsorted(np.cumsum(lens[ranked]), lens.sum() / 2 - lens[short_hi].sum())
    side = np.zeros(len(docs), dtype=np.int8)   # 1: high half, 0: low half
    side[ranked[:cut]] = 1
    side[short_hi] = 1
    print(f"{len(docs):,d} documents ranked, {time.time() - t0:.0f} s", flush=True)
    for name, s in ((a.hi, 1), (a.lo, 0)):
        sel = np.nonzero(side == s)[0]
        os.makedirs(os.path.join(PREFIX, "synth", name), exist_ok=True)
        with SerialCacheWriter(os.path.join(PREFIX, "synth", name, "train"), EX, metadata=CacheMetadata.empty()) as w:
            for j in range(0, len(sel), a.batch):
                w.write_batch([{"input_ids": docs[k]} for k in sel[j:j + a.batch]])
        r = rate[sel]
        st = dict(name=name, source=a.src, side="high" if s else "low", function_words=sorted(FW), documents=len(sel),
                  tokens=int(lens[sel].sum()), source_documents=len(docs), source_tokens=int(lens.sum()),
                  short_documents=int(short[sel].sum()), rate_mean=float(np.nanmean(r)), rate_min=float(np.nanmin(r)),
                  rate_max=float(np.nanmax(r)), seconds=round(time.time() - t0))
        json.dump(st, open(os.path.join(PREFIX, "synth", name, "stats.json"), "w"), indent=1)
        print(json.dumps(st), flush=True)


if __name__ == "__main__":
    main()
