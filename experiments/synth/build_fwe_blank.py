# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""fineweb-edu-10B with blank lines in a share of its documents: the format control of synth-pretrain.

    python -m experiments.synth.build_fwe_blank --out fweblank50:0.5 --out fweblank12:0.125 [--limit N]

fineweb-edu-10B has no blank line (0 blank-line tokens per 1000 tokens; the generated text has 8 to 12), and the mix of
fineweb-edu with generated text gains on the Paloma sources with many blank lines. The control keeps the documents and
their order and makes every line break of a share of the documents a blank line ("\\n" -> "\\n\\n"), with no generated text.
Document i is edited in each output whose share exceeds u_i, u = numpy default_rng(20261117).random(number of documents),
so the documents edited at 12.5% are a subset of those edited at 50%. An edited document is decoded and encoded again
(BOS + text + EOS, as in the cache); every other document is copied token for token. One pass writes every output:
MARIN_PREFIX/synth/NAME/train and MARIN_PREFIX/synth/NAME/stats.json, so SYNTH_MIX=NAME:1 reads it. Run on a pilot's spare
CPUs or a vis node, with the marin venv and this worktree on PYTHONPATH.
"""

import argparse
import contextlib
import glob
import json
import os
import time

import numpy as np
from levanter.store.cache import CacheMetadata, SerialCacheWriter, TreeCache
from tokenizers import Tokenizer

PREFIX = os.environ.get("MARIN_PREFIX", "/scratch/gpfs/KARTHIKN/wc9403/marin_store_big")
FWE = os.path.join(PREFIX, "fineweb-edu-10B", "2026.06.28", "train")
TOKENIZER = glob.glob("/scratch/gpfs/KARTHIKN/wc9403/cache/huggingface/hub/models--marin-community--marin-tokenizer/snapshots/*/tokenizer.json")[0]
BOS, EOS = 128000, 128001
SEED = 20261117
EX = {"input_ids": np.zeros((0,), dtype=np.int32)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", action="append", required=True, help="NAME:SHARE, e.g. fweblank50:0.5 (repeatable)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N documents (a speed test)")
    ap.add_argument("--batch", type=int, default=8192)
    a = ap.parse_args()
    outs = [(o.rsplit(":", 1)[0], float(o.rsplit(":", 1)[1])) for o in a.out]
    if not all(0 < s <= 1 for _, s in outs) or len({n for n, _ in outs}) != len(outs):
        raise SystemExit("--out NAME:SHARE with 0 < SHARE <= 1 and distinct names")
    for name, _ in outs:
        if os.path.exists(os.path.join(PREFIX, "synth", name, "train")):
            raise SystemExit(f"{PREFIX}/synth/{name}/train exists; pick a new name")
    tok = Tokenizer.from_file(TOKENIZER)
    vocab = tok.get_vocab_size()
    blank = np.zeros(vocab + 256, dtype=bool)
    blank[[i for i in range(vocab) if "\n\n" in tok.decode([i], skip_special_tokens=False)]] = True
    src = TreeCache.load(FWE, EX)
    u_all = np.random.default_rng(SEED).random(len(src))   # drawn for every document, so --limit sees the same u
    n = min(a.limit, len(src)) if a.limit else len(src)
    smax = max(s for _, s in outs)
    st = [dict(name=name, share=s, source=FWE, seed=SEED, documents=0, edited=0, changed=0, tokens=0, source_tokens=0,
               blank_tokens=0) for name, s in outs]
    roundtrip = {"checked": 0, "same": 0}   # selected documents without a line break: re-encoding must give the same ids
    t0 = time.time()
    with contextlib.ExitStack() as stack:
        writers = []
        for name, _ in outs:
            os.makedirs(os.path.join(PREFIX, "synth", name), exist_ok=True)
            writers.append(stack.enter_context(SerialCacheWriter(os.path.join(PREFIX, "synth", name, "train"), EX,
                                                                 metadata=CacheMetadata.empty())))
        for i0 in range(0, n, a.batch):
            i1 = min(n, i0 + a.batch)
            ids = [np.asarray(r["input_ids"], dtype=np.int32) for r in src.get_batch_sync(slice(i0, i1))]
            u = u_all[i0:i1]
            sel = np.nonzero(u < smax)[0]
            heads = [x[:1] if len(x) and x[0] == BOS else x[:0] for x in (ids[k] for k in sel)]
            tails = [x[-1:] if len(x) > 1 and x[-1] == EOS else x[:0] for x in (ids[k] for k in sel)]
            bodies = [ids[k][len(h):len(ids[k]) - len(t)].tolist() for k, h, t in zip(sel, heads, tails)]
            texts = tok.decode_batch(bodies, skip_special_tokens=False)
            enc = tok.encode_batch([x.replace("\n", "\n\n") for x in texts], add_special_tokens=False)
            new = {}
            for k, h, t, x, e in zip(sel, heads, tails, texts, enc):
                y = np.concatenate([h, np.asarray(e.ids, dtype=np.int32), t])
                new[k] = (y, "\n" in x)
                if "\n" not in x:
                    roundtrip["checked"] += 1
                    roundtrip["same"] += int(np.array_equal(y, ids[k]))
            for (name, s), w, d in zip(outs, writers, st):
                rows = []
                for k, x in enumerate(ids):
                    y = x
                    if u[k] < s:
                        y, changed = new[k]
                        d["edited"] += 1
                        d["changed"] += int(changed)
                    rows.append({"input_ids": y})
                    d["documents"] += 1
                    d["tokens"] += len(y)
                    d["source_tokens"] += len(x)
                    d["blank_tokens"] += int(blank[y].sum())
                w.write_batch(rows)
            if (i0 // a.batch) % 50 == 0:
                print(f"{i1:,d}/{n:,d} documents, {time.time() - t0:.0f} s", flush=True)
    for d in st:
        d["blank_tokens_per_1000"] = 1000 * d["blank_tokens"] / max(1, d["tokens"])
        d["roundtrip_same"] = roundtrip
        d["seconds"] = round(time.time() - t0)
        json.dump(d, open(os.path.join(PREFIX, "synth", d["name"], "stats.json"), "w"), indent=1)
        print(json.dumps(d))


if __name__ == "__main__":
    main()
