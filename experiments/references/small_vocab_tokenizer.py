# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Truncated versions of the Marin tokenizer (Llama 3 BPE) for the small-vocabulary experiment.

The Marin tokenizer is byte-level BPE with 128,000 ordinary tokens whose ids are their BPE ranks (training order), plus
256 special tokens from id 128,000. A truncated tokenizer keeps the K lowest-ranked tokens (ids 0..K-1, unchanged) and the
merges whose two parts and result are all among them, and moves the 256 special tokens to ids K..K+255. That is the
tokenizer BPE would have produced had training stopped after K tokens, with the same pre-tokenizer.

Already tokenized data converts without the text: within one pre-token, BPE with the K-token vocabulary yields the full
encoding with every token of id >= K re-encoded on its own (the merges above rank K only join tokens and never enable a
merge below rank K). So each full token t maps to expand[t] = [t] if t < K else the truncated BPE of t's bytes; special
tokens map to their new ids. validate() checks this against encoding the decoded text with the truncated tokenizer.

    python -m experiments.references.small_vocab_tokenizer build K OUT_DIR
    python -m experiments.references.small_vocab_tokenizer validate K OUT_DIR N_DOCS
"""

import copy
import json
import os
import sys

import numpy as np
from tokenizers import Tokenizer

SRC = os.environ.get("MARIN_TOKENIZER_DIR", "/scratch/gpfs/GROUP/USER/cache/huggingface/hub/models--marin-community--marin-tokenizer/snapshots/a5ca45f2feb6c959bd87b81689aa7279b5bdcaa2")
N_ORDINARY = 128000


def build(k: int, out_dir: str) -> None:
    """Writes tokenizer.json, tokenizer_config.json and special_tokens_map.json of the K-token tokenizer to out_dir."""
    t = json.load(open(os.path.join(SRC, "tokenizer.json")))
    m = t["model"]
    assert m["type"] == "BPE" and len(m["vocab"]) == N_ORDINARY and sorted(m["vocab"].values()) == list(range(N_ORDINARY))
    vocab = {tok: i for tok, i in m["vocab"].items() if i < k}
    merges = []
    for mg in m["merges"]:
        a, b = mg if isinstance(mg, list) else mg.split(" ")
        if a in vocab and b in vocab and (a + b) in vocab:
            merges.append(mg)
    shift = N_ORDINARY - k   # special token ids move down by this much
    t2 = copy.deepcopy(t)
    t2["model"]["vocab"], t2["model"]["merges"] = vocab, merges
    for at in t2["added_tokens"]:
        assert at["id"] >= N_ORDINARY
        at["id"] -= shift

    def remap(node):
        if isinstance(node, dict):
            if "ids" in node and isinstance(node["ids"], list):
                node["ids"] = [i - shift if i >= N_ORDINARY else i for i in node["ids"]]
            for v in node.values():
                remap(v)
        elif isinstance(node, list):
            for v in node:
                remap(v)
    remap(t2["post_processor"])
    os.makedirs(out_dir, exist_ok=True)
    json.dump(t2, open(os.path.join(out_dir, "tokenizer.json"), "w"), ensure_ascii=False)
    for name in ("tokenizer_config.json", "special_tokens_map.json"):
        cfg = json.load(open(os.path.join(SRC, name)))
        if "added_tokens_decoder" in cfg:
            cfg["added_tokens_decoder"] = {str(int(i) - shift): v for i, v in cfg["added_tokens_decoder"].items()}
        json.dump(cfg, open(os.path.join(out_dir, name), "w"), ensure_ascii=False, indent=1)
    print(f"K={k}: {len(vocab)} ordinary tokens, {len(merges)} of {len(m['merges'])} merges, specials at {k}..{k + 255} -> {out_dir}")


def expansion(k: int, out_dir: str):
    """expand[t] for every full-tokenizer id t (ordinary and special), as a flat array plus offsets."""
    full = json.load(open(os.path.join(SRC, "tokenizer.json")))
    inv = {i: tok for tok, i in full["model"]["vocab"].items()}
    small = Tokenizer.from_file(os.path.join(out_dir, "tokenizer.json"))
    parts, offs = [], [0]
    for t in range(N_ORDINARY + 256):
        if t < k:
            e = [t]
        elif t < N_ORDINARY:
            e = [x.id for x in small.model.tokenize(inv[t])]
            assert e and all(i < k for i in e), (t, e)
        else:
            e = [t - (N_ORDINARY - k)]
        parts.append(e)
        offs.append(offs[-1] + len(e))
    return np.concatenate([np.asarray(p, np.int32) for p in parts]), np.asarray(offs, np.int64)


def convert(ids: np.ndarray, flat: np.ndarray, offs: np.ndarray) -> np.ndarray:
    """Full-tokenizer ids -> truncated ids, by the expansion table (vectorised)."""
    lens = offs[ids + 1] - offs[ids]
    starts = np.repeat(offs[ids], lens)
    within = np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens)
    return flat[starts + within]


def validate(k: int, out_dir: str, n_docs: int) -> None:
    """On n_docs fineweb-edu-10B documents: converting the stored ids equals encoding the decoded text with the truncated
    tokenizer; prints tokens per byte for both tokenizers."""
    from experiments.references import ov_memorization_eval as ome
    full = Tokenizer.from_file(os.path.join(SRC, "tokenizer.json"))
    small = Tokenizer.from_file(os.path.join(out_dir, "tokenizer.json"))
    flat, offs = expansion(k, out_dir)
    cache = ome.data_config().build_caches("train")[ome.TRAIN_NAME]
    rng = np.random.default_rng(0)
    idx = rng.choice(len(cache), size=n_docs, replace=False)
    docs = cache.get_batch_sync([int(i) for i in idx])
    same = n_full = n_small = n_bytes = 0
    bad = []
    for d in docs:
        ids = np.asarray(d["input_ids"], np.int64)
        body = ids[(ids < N_ORDINARY)]   # the text tokens; BOS/EOS handled by the id shift
        text = full.decode(body.tolist(), skip_special_tokens=False)
        want = small.encode(text, add_special_tokens=False).ids
        got = convert(body, flat, offs).tolist()
        same += got == want
        if got != want and len(bad) < 3:
            bad.append((text[:120], len(got), len(want)))
        n_full += len(body); n_small += len(got); n_bytes += len(text.encode("utf-8"))
    print(f"K={k}: {same}/{len(docs)} documents convert exactly; tokens per byte: full {n_full / n_bytes:.4f}, K {n_small / n_bytes:.4f} "
          f"({n_small / n_full:.3f}x the tokens)")
    for b in bad:
        print("  mismatch:", b)


if __name__ == "__main__":
    cmd, k, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    if cmd == "build":
        build(k, out)
    elif cmd == "validate":
        validate(k, out, int(sys.argv[4]))
