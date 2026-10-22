"""Cut the 8K composition map (marin-small/v8000/expand_full.npz, rows = the 128,256 ids of the Marin tokenizer) to a
K-token truncation (survey idea 1, owner popup 2026-10-21): the truncation's ordinary ids are the Marin ids 0..K-1 and its
256 specials follow at K..K+255 (checked against both tokenizer.json files), so its row t < K is the full map's row t and
its special j is the full map's row 128000 + j. Writes marin-small/v8000/expand_v<K/1000>k.npz with the same flat/offs
layout. Run on a vis node: SIZE-free, numpy only: python -m experiments.references.vocab_cut_expand_map 64000 32000 16000"""
import json
import sys

import numpy as np

R = "/scratch/gpfs/GROUP/USER/marin_store_big/tokenizers"
FULL = "/scratch/gpfs/GROUP/USER/cache/huggingface/hub/models--marin-community--marin-tokenizer/snapshots/a5ca45f2feb6c959bd87b81689aa7279b5bdcaa2/tokenizer.json"
z = np.load(f"{R}/marin-small/v8000/expand_full.npz")
flat, offs = z["flat"], z["offs"]
full = json.load(open(FULL))
fv, fadd = full["model"]["vocab"], {a["content"]: a["id"] for a in full["added_tokens"]}
for K in map(int, sys.argv[1:]):
    t = json.load(open(f"{R}/marin-small/v{K}/tokenizer.json"))
    v, add = t["model"]["vocab"], {a["content"]: a["id"] for a in t["added_tokens"]}
    assert len(v) == K and all(fv[tok] == i for tok, i in v.items()), f"v{K}: ordinary ids are not the Marin prefix"
    assert sorted(add.values()) == list(range(K, K + 256)) and all(fadd[c] - 128000 == i - K for c, i in add.items()), f"v{K}: specials not in Marin order after K"
    rows = list(range(K)) + list(range(128000, 128256))
    parts = [flat[offs[r]:offs[r + 1]] for r in rows]
    new_offs = np.concatenate([[0], np.cumsum([len(p) for p in parts])]).astype(offs.dtype)
    out = f"{R}/marin-small/v8000/expand_v{K // 1000}k.npz"
    np.savez(out, flat=np.concatenate(parts).astype(flat.dtype), offs=new_offs)
    print(out, "rows", len(rows), "pieces", int(new_offs[-1]))
