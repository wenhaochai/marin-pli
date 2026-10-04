"""CPU tests for OV with a real 2-gram output vocabulary (over_vocab_qwen3, od_mode="hashed"), on a fake 4-device mesh.

Run from the marin repo on a vis node: nice -n 19 .venv/bin/python scripts/della/odhash_cpu_test.py
1. hashed_candidates: every label is among the candidates at its returned position, candidates are unique and ascending,
   and there are as many candidates as labels.
2. With od_m equal to the positions per device (64), every device's candidate set is the whole hashed vocabulary, so the
   sampled softmax is the full softmax: the loss and every gradient equal a dense reference, namely
   CE(main) + 0.1 * CE over the 2-gram classes c_t = (x_{t+1} + x_{t+2} V) mod m with logits (W_2 h_t) . out(table[c]),
   masked past the window, across documents and by the example weights.
3. With od_m = 1000 > 64 the loss runs, is finite, and stays below the full-softmax loss (fewer competing classes).
4. Evaluation is the main head's plain cross-entropy.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4"
os.environ.setdefault("LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS", "0")

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

import haliax as hax
from haliax.partitioning import ResourceAxis
from levanter.layers.attention import AttentionBackend, AttentionMask
from levanter.models.lm_model import LmExample
from levanter.models.loss import next_token_loss_weight
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig

from experiments.references.over_vocab_qwen3 import ROWS, OverVocabQwen3Config, OverVocabQwen3LMHeadModel, hashed_candidates

jax.config.update("jax_threefry_partitionable", True)
assert jax.device_count() == 4
rng = np.random.default_rng(0)

# 1. candidate sets
for m, n in ((1000, 64), (64, 64), (5000, 300)):
    labels = jnp.asarray(rng.integers(0, m, size=n), jnp.int32)
    cand, where = hashed_candidates(labels, m, jnp.int32(rng.integers(0, m)))
    c = np.asarray(cand)
    assert len(c) == n and np.all(np.diff(c) > 0) and c.min() >= 0 and c.max() < m, (m, n)
    assert np.array_equal(c[np.asarray(where)], np.asarray(labels)), (m, n)
print("1. hashed_candidates: unique, ascending, n of them, every label at its returned position")

B, T, V = 8, 32, 500
Batch, Pos, Vocab = hax.Axis("batch", B), hax.Axis("position", T), hax.Axis("vocab", V)
EOS = 1
tok = rng.integers(2, V, size=(B, T))
tok[:, 10] = EOS; tok[3, 20] = EOS
eos_mask = np.roll(tok, 1, axis=1) == EOS; eos_mask[:, 0] = False
seg = np.cumsum(eos_mask, axis=1).astype(np.int32)
lw = (np.arange(T) < T - 1).astype(np.float32)[None].repeat(B, 0); lw[2, :5] = 0.0
tokens = hax.named(jnp.asarray(tok, jnp.int32), (Batch, Pos))
ex = LmExample(tokens=tokens, loss_weight=hax.named(jnp.asarray(lw), (Batch, Pos)),
               attn_mask=AttentionMask.causal().with_segment_ids(hax.named(jnp.asarray(seg), (Batch, Pos))))
common = dict(max_seq_len=T, hidden_dim=48, intermediate_dim=64, num_layers=2, num_heads=2, num_kv_heads=2, hybrid_norm=True,
              attn_backend=AttentionBackend.VANILLA)
key = jrandom.PRNGKey(7)
y1 = np.roll(tok, -1, 1)
w1 = next_token_loss_weight(Pos, ex.loss_weight).array
tt = np.arange(T)[None]
ok2 = (tt <= T - 3) & (np.roll(seg, -1, 1) == seg) & (np.roll(seg, -2, 1) == seg)
w2 = lw * np.roll(lw, -1, 1) * ok2


def scalar(x):
    return x.array if isinstance(x, hax.NamedArray) else x


def ce_mean(logits, y, w):
    ce = jax.nn.logsumexp(logits, -1) - jnp.take_along_axis(logits, jnp.asarray(y).reshape(-1)[:, None], 1)[:, 0]
    w = jnp.asarray(w).reshape(-1)
    return jnp.sum(ce * w) / jnp.sum(w)


def close(a, b, what, rtol=2e-4):
    for (p, u), v in zip(jax.tree_util.tree_leaves_with_path(eqx.filter(a, eqx.is_array)), jax.tree.leaves(eqx.filter(b, eqx.is_array))):
        np.testing.assert_allclose(np.asarray(u, np.float32), np.asarray(v, np.float32), rtol=rtol, atol=1e-6, err_msg=f"{what} {jax.tree_util.keystr(p)}")


def train_loss(m_, example, k):
    def lf(mm):
        loss, stats = mm.compute_next_token_loss(example, key=k, logsumexp_weight=0.0)
        return scalar(loss), stats
    (loss, stats), g = eqx.filter_value_and_grad(lf, has_aux=True)(m_)
    return loss, {k_: v.value() for k_, v in stats.items()}, g


tc = TrainerConfig(mesh=MeshConfig(axes={"data": -1, "replica": 1, "model": 1}, compute_mapping={
    "token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
    "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
    ROWS: ResourceAxis.DATA}, param_mapping={"embed": "data", ROWS: "data"}))
with tc.use_device_mesh(), hax.axis_mapping(tc.compute_axis_mapping):
    es = hax.shard(ex, tc.compute_axis_mapping)
    # 2. od_m = positions per device: the candidate set is the whole hashed vocabulary
    m = B * T // 4
    cfg = OverVocabQwen3Config(**common, oe_m=1000, oe_k=2, od_weight=0.1, od_mode="hashed", od_m=m)
    model = OverVocabQwen3LMHeadModel.init(Vocab, cfg, key=jrandom.PRNGKey(0))
    ms = hax.shard(model, tc.parameter_axis_mapping)
    loss, stats, grads = eqx.filter_jit(train_loss)(ms, es, key)

    def ref(mm):
        k_main, _, _ = jax.random.split(key, 3)
        h = mm.transformer(mm.embed(tokens, ex.attn_mask), attn_mask=ex.attn_mask, key=k_main)
        f1 = ce_mean(h.array.reshape(-1, h.array.shape[-1]) @ mm.get_lm_head().rearrange((cfg.Embed, Vocab)).array, y1, w1)
        h2 = (h.array @ mm.od_proj.weight.rearrange(("od_in", "embed")).array).reshape(-1, h.array.shape[-1])
        u = mm.od_out(mm.od_table).rearrange((ROWS, "embed")).array[:m]   # every class's output embedding [m, d]
        c = (np.roll(tok, -1, 1) + np.roll(tok, -2, 1) * V) % m
        f2 = ce_mean(h2 @ u.T, c, w2)
        return f1 + 0.1 * f2, (f1, f2)
    (ref_l, (f1, f2)), ref_g = eqx.filter_value_and_grad(ref, has_aux=True)(model)
    np.testing.assert_allclose(float(loss), float(ref_l), rtol=1e-5)
    np.testing.assert_allclose(float(stats["od_loss"]), float(f2), rtol=1e-5)
    close(grads, ref_g, "hashed OD, full candidate set")
    print(f"2. od_m = {m} (= positions per device): loss {float(loss):.6f} == CE(main) {float(f1):.6f} + 0.1 x full 2-gram CE "
          f"{float(f2):.6f}; every gradient matches")
    # 3. od_m larger than the candidate set
    cfg3 = OverVocabQwen3Config(**common, oe_m=1000, oe_k=2, od_weight=0.1, od_mode="hashed", od_m=1000)
    m3 = hax.shard(OverVocabQwen3LMHeadModel.init(Vocab, cfg3, key=jrandom.PRNGKey(0)), tc.parameter_axis_mapping)
    l3, s3, _ = eqx.filter_jit(train_loss)(m3, es, key)
    assert np.isfinite(float(l3)) and float(s3["od_loss"]) < np.log(1000), float(s3["od_loss"])
    print(f"3. od_m = 1000: od_loss {float(s3['od_loss']):.4f} (finite, below log 1000 = {np.log(1000):.4f})")
    # 4. evaluation
    ev = eqx.filter_jit(lambda mm, e_: mm.compute_next_token_loss(e_, key=None))(ms, es)
    np.testing.assert_allclose(float(scalar(ev)), float(f1), rtol=1e-5)
    print("4. evaluation == main-head CE")
print("ALL HASHED-OD CPU TESTS PASSED")
