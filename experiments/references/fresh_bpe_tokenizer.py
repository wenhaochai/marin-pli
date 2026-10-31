# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A byte-level BPE tokenizer trained afresh on our training text: the control for the truncations (owner 2026-10-30,
"你这样取截断科学吗"; 2026-10-31, "检查有没有实验遗漏我们补一下").

small_vocab_tokenizer.py's v<K> keeps the K lowest-ranked Marin tokens: BPE stopped after K tokens, but trained on the
Llama 3 tokenizer's own data. bpe<K/1000>k is BPE trained to K ordinary tokens on a uniform sample of the training text
(documents of the Marin-tokenized fineweb-edu-10B train cache, decoded), so its merges fit fineweb-edu. Everything else is
copied from v<K>: the Marin pre-tokenizer (Llama 3 split regex, then byte level), the 256 byte symbols, the decoder, the
post-processor, ignore_merges, and the 256 special tokens at ids K..K+255. Only the vocabulary and the merges differ.

The sample is a uniform random set of documents of the whole cache (seed 0), not the repeated eighth, so the tokenizer
favours neither arm of a unique/8-pass pair. convert_small_vocab_cache.py (VOCAB_SET=bpe<K/1000>k) re-encodes the caches
through the text.

    python -m experiments.references.fresh_bpe_tokenizer train K N_DOCS OUT_DIR
"""

import json
import os
import shutil
import sys
import time

import numpy as np
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

from levanter.store.cache import TreeCache

from experiments.references.small_vocab_tokenizer import N_ORDINARY, SRC

PREFIX = os.environ.get("MARIN_PREFIX", "/scratch/gpfs/GROUP/USER/marin_store_big")
TRAIN = f"{PREFIX}/fineweb-edu-10B/2026.06.28/train"
EX = {"input_ids": np.zeros((0,), dtype=np.int32)}
BATCH = 8192


def sample_texts(n_docs: int, seed: int = 0, stats: dict | None = None):
    """Texts of n_docs documents drawn uniformly without replacement from the training cache (special tokens dropped)."""
    src = TreeCache.load(TRAIN, EX)
    idx = np.sort(np.random.default_rng(seed).choice(len(src), size=n_docs, replace=False))
    full = Tokenizer.from_file(os.path.join(SRC, "tokenizer.json"))
    for a in range(0, n_docs, BATCH):
        docs = [np.asarray(d["input_ids"]) for d in src.get_batch_sync(idx[a:a + BATCH].tolist())]
        docs = [d[d < N_ORDINARY].tolist() for d in docs]
        texts = full.decode_batch(docs, skip_special_tokens=False)
        if stats is not None:
            stats["docs"] = stats.get("docs", 0) + len(docs)
            stats["tokens"] = stats.get("tokens", 0) + sum(len(d) for d in docs)
            stats["chars"] = stats.get("chars", 0) + sum(len(t) for t in texts)
        yield from texts


def train(k: int, n_docs: int, out: str):
    ref_dir = f"{PREFIX}/tokenizers/marin-small/v{k}"
    ref = json.load(open(f"{ref_dir}/tokenizer.json"))
    t = Tokenizer(models.BPE())
    t.pre_tokenizer = Tokenizer.from_file(f"{ref_dir}/tokenizer.json").pre_tokenizer
    trainer = trainers.BpeTrainer(vocab_size=k, min_frequency=2, show_progress=False, special_tokens=[],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    stats, t0 = {}, time.time()
    t.train_from_iterator(sample_texts(n_docs, stats=stats), trainer=trainer, length=n_docs)
    m = json.loads(t.to_str())["model"]
    if len(m["vocab"]) != k or sorted(m["vocab"].values()) != list(range(k)):
        raise ValueError(f"trained vocabulary has {len(m['vocab'])} tokens, ids not 0..{k - 1}")
    missing = set(pre_tokenizers.ByteLevel.alphabet()) - set(m["vocab"])
    if missing:
        raise ValueError(f"byte symbols missing from the vocabulary: {sorted(missing)}")
    new = dict(ref, model=dict(ref["model"], vocab=m["vocab"], merges=m["merges"]))
    os.makedirs(out, exist_ok=False)
    json.dump(new, open(f"{out}/tokenizer.json", "w"), ensure_ascii=False)
    for f in ("tokenizer_config.json", "special_tokens_map.json"):
        shutil.copy(f"{ref_dir}/{f}", f"{out}/{f}")
    meta = dict(k=k, sample_docs=n_docs, seed=0, source=TRAIN, min_frequency=2, merges=len(m["merges"]),
                seconds=round(time.time() - t0), **{f"sample_{a}": b for a, b in stats.items()})
    json.dump(meta, open(f"{out}/fresh.json", "w"), indent=1)
    # the assembled tokenizer: specials where v<K> has them, and a lossless round trip on held-out documents
    tok, small = Tokenizer.from_file(f"{out}/tokenizer.json"), Tokenizer.from_file(f"{ref_dir}/tokenizer.json")
    assert tok.get_vocab_size(with_added_tokens=True) == k + 256, tok.get_vocab_size(with_added_tokens=True)
    for s in ("<|begin_of_text|>", "<|end_of_text|>"):
        assert tok.token_to_id(s) == small.token_to_id(s), s
    held = list(sample_texts(200, seed=1))
    for x in held:
        ids = tok.encode(x, add_special_tokens=False).ids
        assert max(ids) < k and tok.decode(ids) == x, "round trip failed"
    ratio = sum(len(tok.encode(x, add_special_tokens=False).ids) for x in held) / sum(len(small.encode(x, add_special_tokens=False).ids) for x in held)
    print(json.dumps(meta), f"\nheld-out 200 docs: tokens bpe / v{k} = {ratio:.4f}; round trip exact", flush=True)


if __name__ == "__main__":
    if sys.argv[1] != "train":
        raise SystemExit(__doc__)
    train(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
