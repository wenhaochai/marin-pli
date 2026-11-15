# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Statistics of generated corpora next to a fineweb-edu sample of the same size, for the question of which property of
the text sets its Paloma gap. Run on a vis node with the marin venv and this worktree on PYTHONPATH.

    python -m experiments.synth.corpus_stats --raw scratch=DIR --raw cont=DIR --raw reph=DIR --tokens 5e6 --out OUT.json

Per corpus, over its first --tokens Marin tokens (documents split as build_cache.py does): documents, mean and median
document length, unigram entropy (bits per token), distinct tokens, the share of distinct 4-grams, the share of
documents with Markdown markers (**, a line opening with #, ```), the share of documents with a line ending in a colon
followed by a list, and the ten most frequent capitalized two-word names with the share of documents that hold any of
them. fineweb-edu: random documents of the prebuilt cache (seed 301), decoded.
"""

import argparse
import collections
import glob
import json
import math
import os
import re

import numpy as np
from levanter.store.cache import TreeCache
from tokenizers import Tokenizer

from experiments.synth.build_cache import BOS, EOS, TOKENIZER, pieces  # noqa: F401

FWE = "/scratch/gpfs/KARTHIKN/wc9403/marin_store_big/fineweb-edu-10B/2026.06.28/train"
NAME = re.compile(r"\b([A-Z][a-z]{2,15}) ([A-Z][a-z]{2,15})\b")
MD = re.compile(r"\*\*|^#{1,6} |```", re.M)


def stats(texts, tok, budget):
    n_tok, docs, lens, uni, four = 0, 0, [], collections.Counter(), set()
    four_total, md, names, docs_with = 0, 0, collections.Counter(), []
    for x in texts:
        ids = tok.encode(x, add_special_tokens=False).ids
        if not ids:
            continue
        docs += 1
        lens.append(len(ids))
        n_tok += len(ids)
        uni.update(ids)
        for i in range(len(ids) - 3):
            four.add(tuple(ids[i:i + 4]))
        four_total += max(0, len(ids) - 3)
        md += bool(MD.search(x))
        ns = set(" ".join(m) for m in NAME.findall(x))
        names.update(ns)
        docs_with.append(ns)
        if n_tok >= budget:
            break
    total = sum(uni.values())
    H = -sum(c / total * math.log2(c / total) for c in uni.values())
    # names: the frequent capitalized pairs that are not sentence-initial phrases common to every corpus
    top = [n for n, _ in names.most_common(40) if n.split()[0] not in {"The", "This", "That", "In", "For", "If", "It", "We", "You", "When", "As"}][:10]
    share_top = sum(1 for ns in docs_with if ns & set(top)) / max(1, docs)
    return {"documents": docs, "tokens": n_tok, "doc_tokens_mean": round(float(np.mean(lens)), 1), "doc_tokens_median": float(np.median(lens)),
            "unigram_entropy_bits": round(H, 3), "distinct_tokens": len(uni), "distinct_4gram_share": round(len(four) / max(1, four_total), 4),
            "markdown_doc_share": round(md / max(1, docs), 4), "top_names": [(n, names[n]) for n in top], "docs_with_top_names": round(share_top, 4)}


def fwe_texts(tok, seed=301):
    cache = TreeCache.load(FWE, {"input_ids": np.zeros((0,), dtype=np.int32)})
    rng = np.random.default_rng(seed)
    for i in rng.choice(len(cache), size=200000, replace=False).tolist():
        ids = [t for t in np.asarray(cache[i]["input_ids"]).tolist() if t not in (BOS, EOS)]
        yield tok.decode(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", action="append", default=[], help="name=DIR")
    ap.add_argument("--tokens", type=float, default=5e6)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tok = Tokenizer.from_file(TOKENIZER)
    res = {"fineweb-edu": stats(fwe_texts(tok), tok, a.tokens)}
    print("fineweb-edu", json.dumps(res["fineweb-edu"]), flush=True)
    for spec in a.raw:
        name, d = spec.split("=", 1)
        import experiments.synth.build_cache as bc
        bc.MIN_CHARS = 50
        res[name] = stats((x for _, x in pieces(d)), tok, a.tokens)
        print(name, json.dumps(res[name]), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
