# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""route="bpe" end to end on a token-start cache (convert_token_start_cache.py, a LIMIT test written to TS_STORE): the
Levanter data pipeline (TextLmDatasetFormat, the v256 tokenizer's EOS, no packing, as the training runs read the caches)
reads windows of the flagged cache and of the v256 cache; checks, per window:
1. the loss weights and segment ids are the same for both caches (the flags never touch EOS, so documents are cut the same);
2. the stripped flagged ids equal the v256 ids;
3. the bpe model's per-byte loss (compute_next_token_loss) on the flagged window equals the byte model with the same
   weights on the v256 window with the token starts given by hand (route_p), and its targets are the v256 ids.
    TS_STORE=<OUT_PREFIX of the LIMIT conversion> JAX_PLATFORMS=cpu python tests/test_hnet_bpe_pipeline.py"""
import os

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

from haliax import Axis
from levanter.data.text.datasets import DatasetComponent, LmDataConfig, UrlDatasetSourceConfig
from levanter.data.text.formats import TextLmDatasetFormat
from levanter.layers.attention import AttentionBackend
from levanter.models.loss import maybe_fused_next_token_loss

from experiments.references.hnet_qwen3 import BPE_FLAG, HNetQwen3Config, HNetQwen3LMHeadModel, HNetRoute

TS = os.environ["TS_STORE"]
STORE = "/scratch/gpfs/GROUP/USER/marin_store_big"
TOK = f"{STORE}/tokenizers/marin-small/v256"
L, NW = 2048, 6
ok = True


def check(name, good, extra=""):
    global ok
    ok &= bool(good)
    print(("ok " if good else "BAD"), name, extra)


def windows(path):
    src = UrlDatasetSourceConfig(tags=[], train_urls=[], validation_urls=[], cache_dir=os.path.dirname(path), format=TextLmDatasetFormat())
    comp = DatasetComponent(source=src, cache_dir=src.cache_dir, format=src.format, tags=[])
    cfg = LmDataConfig(components={"c4_en": comp}, train_weights={"c4_en": 0.0}, tokenizer=TOK, cache_dir=None, shuffle=False)
    ds = cfg.validation_sets(Axis("position", L))["c4_en"].as_sync_dataset()
    return [ds[i] for i in range(NW)]


wt = windows(f"{TS}/tokenized/paloma-v256ts/c4_en/validation")
wv = windows(f"{STORE}/tokenized/paloma-v256/c4_en/validation")
same_w = all(np.array_equal(np.asarray(a.loss_weight.array), np.asarray(b.loss_weight.array)) for a, b in zip(wt, wv))
same_s = all(np.array_equal(np.asarray(a.attn_mask.segment_ids[0].array), np.asarray(b.attn_mask.segment_ids[0].array)) for a, b in zip(wt, wv))
check("pipeline: loss weights and segment ids equal for the flagged and the v256 cache", same_w and same_s,
      f"{NW} windows of {L} bytes, documents per window {[int(np.asarray(w.attn_mask.segment_ids[0].array).max()) + 1 for w in wv]}")
nflag = [int((np.asarray(w.tokens.array) >= BPE_FLAG).sum()) for w in wt]
strip = all(np.array_equal(np.where(np.asarray(a.tokens.array) >= BPE_FLAG, np.asarray(a.tokens.array) - BPE_FLAG, np.asarray(a.tokens.array)),
                           np.asarray(b.tokens.array)) for a, b in zip(wt, wv))
check("pipeline: stripped ids equal the v256 window", strip, f"token starts per window {nflag}, {L * NW / sum(nflag):.2f} bytes per start")

Vocab = Axis("vocab", 512)
kw = dict(max_seq_len=L, hidden_dim=64, intermediate_dim=128, num_layers=2, num_heads=4, num_kv_heads=4, hybrid_norm=True,
          attn_backend=AttentionBackend.VANILLA, enc_layers=1, dec_layers=1, chunk_ratio=4.74)
bm = HNetQwen3LMHeadModel.init(Vocab, HNetQwen3Config(**kw, route=HNetRoute.BPE), key=jrandom.PRNGKey(0))
hm = HNetQwen3LMHeadModel.init(Vocab, HNetQwen3Config(**kw), key=jrandom.PRNGKey(0))
check("the two models have the same weights", all(bool(jnp.array_equal(a, b)) for a, b in zip(
    jax.tree_util.tree_leaves(eqx.filter(bm, eqx.is_array)), jax.tree_util.tree_leaves(eqx.filter(hm, eqx.is_array)))))
max_d, overflow = 0.0, []
for a, b in zip(wt, wv):
    lb = bm.compute_next_token_loss(a, reduction=None, reduction_axis=()).array
    x = np.asarray(b.tokens.array)
    seg = np.asarray(b.attn_mask.segment_ids[0].array)
    hand = np.asarray(a.tokens.array) >= BPE_FLAG
    hand |= x >= 256
    hand[0] = True
    hand[1:] |= seg[1:] != seg[:-1]
    overflow.append(int(hand.sum()) - hm.config.slots)
    y = hm.activations(b.tokens, b.attn_mask, key=None, route_p=jnp.asarray(hand, jnp.float32))
    lh = maybe_fused_next_token_loss(hm.Pos, hm.Embed, hm.Vocab, y, hm.get_lm_head(), b.tokens, loss_weight=b.loss_weight,
                                     reduction=None, reduction_axis=(), dtype=jnp.float32).array
    max_d = max(max_d, float(jnp.abs(lb - lh).max()))
check("bpe model on the flagged pipeline windows: per-byte loss equals the byte model with the token starts by hand", max_d == 0,
      f"max |d loss| {max_d:.1e}; boundaries minus slots per window {overflow} ({hm.config.slots} slots)")
print("ALL OK" if ok else "FAILED")
