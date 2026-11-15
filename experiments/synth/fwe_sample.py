# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Random documents of the fineweb-edu-10B prebuilt cache as text, for generator prompts (rephrase, continue).

    python -m experiments.synth.fwe_sample --n 20000 --seed 1 --out DOCS.jsonl [--min-tokens 128] [--max-tokens 3000]

Documents are drawn uniformly by index (numpy default_rng(seed)), decoded with the Marin tokenizer (BOS and EOS
dropped), and written as {"id": "fwe-<index>", "text", "tokens"}. The owner's rule (2026-11-14): natural text comes only
from fineweb-edu-10B; nothing is downloaded. Run on a vis node with the marin venv and this worktree on PYTHONPATH.
"""

import argparse
import glob
import json

import numpy as np
from levanter.store.cache import TreeCache
from tokenizers import Tokenizer

CACHE = "/scratch/gpfs/KARTHIKN/wc9403/marin_store_big/fineweb-edu-10B/2026.06.28/train"
TOKENIZER = glob.glob("/scratch/gpfs/KARTHIKN/wc9403/cache/huggingface/hub/models--marin-community--marin-tokenizer/snapshots/*/tokenizer.json")[0]
BOS, EOS = 128000, 128001


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-tokens", type=int, default=128)
    ap.add_argument("--max-tokens", type=int, default=3000)
    a = ap.parse_args()
    cache = TreeCache.load(CACHE, {"input_ids": np.zeros((0,), dtype=np.int32)})
    tok = Tokenizer.from_file(TOKENIZER)
    rng = np.random.default_rng(a.seed)
    n_docs = len(cache)
    written, tried = 0, 0
    with open(a.out, "w") as f:
        while written < a.n:
            idx = rng.choice(n_docs, size=min(4 * (a.n - written) + 64, n_docs), replace=False)
            for i in idx.tolist():
                tried += 1
                ids = np.asarray(cache[i]["input_ids"]).tolist()
                ids = [t for t in ids if t not in (BOS, EOS)]
                if not (a.min_tokens <= len(ids) <= a.max_tokens):
                    continue
                f.write(json.dumps({"id": f"fwe-{i}", "text": tok.decode(ids).strip(), "tokens": len(ids)}) + "\n")
                written += 1
                if written >= a.n:
                    break
    print(f"wrote {written} documents of {tried} drawn ({n_docs} in the cache) to {a.out}")


if __name__ == "__main__":
    main()
