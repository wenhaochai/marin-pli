# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Copy a generated cache with levanter's document ending: BOS + text + " " + EOS in place of BOS + text + EOS (synth-pretrain).

    python -m experiments.synth.build_eos_space --src r1cont --dst r1contsp

Levanter's BatchTokenizer appends " " + eos_str to every text (lib/levanter/src/levanter/data/text/_batch_tokenizer.py),
so every fineweb-edu-10B document ends with the token " " and then EOS; the caches of build_cache.py before 2026-11-18
end with the text's last token and EOS. A model trained on them meets that ending only in natural text and pays about
12 nats there (held-out fineweb-edu, 2026-11-18). This copy keeps every document and its order and only changes the
ending, as levanter would have encoded the same text: the ids of text + " " and then EOS. When the text ends in a
non-space character, that is the old ids plus the token " " (checked against a fresh encoding for the first 2000
documents); otherwise the text is encoded again. Writes MARIN_PREFIX/synth/DST/train and DST/stats.json.
"""

import argparse
import glob
import json
import os
import time

import numpy as np
from levanter.store.cache import CacheMetadata, SerialCacheWriter, TreeCache
from tokenizers import Tokenizer

PREFIX = os.environ.get("MARIN_PREFIX", "/scratch/gpfs/KARTHIKN/wc9403/marin_store_big")
TOKENIZER = glob.glob("/scratch/gpfs/KARTHIKN/wc9403/cache/huggingface/hub/models--marin-community--marin-tokenizer/snapshots/*/tokenizer.json")[0]
BOS, EOS = 128000, 128001
EX = {"input_ids": np.zeros((0,), dtype=np.int32)}
CHECK = 2000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--batch", type=int, default=8192)
    a = ap.parse_args()
    out = os.path.join(PREFIX, "synth", a.dst)
    if os.path.exists(os.path.join(out, "train")):
        raise SystemExit(f"{out}/train exists; pick a new name")
    t0 = time.time()
    tok = Tokenizer.from_file(TOKENIZER)
    space = tok.encode(" ", add_special_tokens=False).ids
    assert len(space) == 1, space
    space = space[0]
    ends_space = np.zeros(tok.get_vocab_size() + 256, dtype=bool)
    for i in range(tok.get_vocab_size()):
        s = tok.decode([i], skip_special_tokens=False)
        ends_space[i] = bool(s) and s[-1].isspace()
    src = TreeCache.load(os.path.join(PREFIX, "synth", a.src, "train"), EX)
    n_docs = n_tok = src_tok = n_reenc = checked = already = 0
    os.makedirs(out, exist_ok=True)
    with SerialCacheWriter(os.path.join(out, "train"), EX, metadata=CacheMetadata.empty()) as w:
        for i0 in range(0, len(src), a.batch):
            rows = []
            for r in src.get_batch_sync(slice(i0, min(len(src), i0 + a.batch))):
                d = np.asarray(r["input_ids"], dtype=np.int32)
                assert d[0] == BOS and d[-1] == EOS, (n_docs, d[:2], d[-2:])
                body = d[1:-1]
                src_tok += len(d)
                if len(body) and body[-1] == space:
                    already += 1
                    new = d
                elif len(body) and not ends_space[body[-1]]:
                    new = np.concatenate([d[:-1], [space, EOS]]).astype(np.int32)
                    if checked < CHECK:
                        text = tok.decode(body.tolist(), skip_special_tokens=False)
                        ref = [BOS] + tok.encode(text + " ", add_special_tokens=False).ids + [EOS]
                        assert ref == new.tolist(), (n_docs, ref[-4:], new[-4:].tolist())
                        checked += 1
                else:
                    text = tok.decode(body.tolist(), skip_special_tokens=False)
                    new = np.asarray([BOS] + tok.encode(text + " ", add_special_tokens=False).ids + [EOS], dtype=np.int32)
                    n_reenc += 1
                rows.append({"input_ids": new})
                n_docs += 1
                n_tok += len(new)
            w.write_batch(rows)
    st = dict(name=a.dst, source=a.src, ending="space+eos", documents=n_docs, tokens=n_tok, source_tokens=src_tok,
              already_ending_in_space=already, encoded_again=n_reenc, checked_against_fresh_encoding=checked, seconds=round(time.time() - t0))
    json.dump(st, open(os.path.join(out, "stats.json"), "w"), indent=1)
    print(json.dumps(st), flush=True)


if __name__ == "__main__":
    main()
