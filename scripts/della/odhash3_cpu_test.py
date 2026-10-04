"""CPU tests for OV's real 3-gram output vocabulary (over_vocab_qwen3, od_mode="hashed", od_orders=(2, 3)), on a fake
4-device mesh.

Run from the marin repo on a vis node: nice -n 19 .venv/bin/python scripts/della/odhash3_cpu_test.py
1. hashed_od_loss(order=3) with m equal to the positions per device (64) scores every class, so it equals the dense
   3-gram softmax: CE over classes c_t = (x_{t+1} + x_{t+2} V + x_{t+3} V^2) mod m with logits (W_3 h_t) . out(table[c]),
   weighted by the 3-gram mask (window end, document boundaries, example weights); loss and every gradient match.
2. The model with od_orders (2, 3): loss = CE(main) + 0.1 od + 0.1 od3, od3 equals a direct hashed_od_loss call with the
   model's own mask and key, every 3-gram parameter gets a nonzero gradient, and all parameters outside the 3-gram head
   are initialised exactly as with od_orders (2,) (old checkpoints keep loading).
3. Evaluation is the main head's plain cross-entropy.
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

from experiments.references.over_vocab_qwen3 import ROWS, OverVocabQwen3Config, OverVocabQwen3LMHeadModel, hashed_od_loss

jax.config.update("jax_threefry_partitionable", True)
assert jax.device_count() == 4
rng = np.random.default_rng(0)
B, T, V = 8, 32, 500
Batch, Pos, Vocab = hax.Axis("batch", B), hax.Axis("position", T), hax.Axis("vocab", V)
EOS = 1
tok = rng.integers(2, V, size=(B, T)); tok[:, 10] = EOS; tok[3, 20] = EOS
eos_mask = np.roll(tok, 1, axis=1) == EOS; eos_mask[:, 0] = False
seg = np.cumsum(eos_mask, axis=1).astype(np.int32)
lw = (np.arange(T) < T - 1).astype(np.float32)[None].repeat(B, 0); lw[2, :5] = 0.0; lw[5, 17] = 0.0
tokens = hax.named(jnp.asarray(tok, jnp.int32), (Batch, Pos))
ex = LmExample(tokens=tokens, loss_weight=hax.named(jnp.asarray(lw), (Batch, Pos)),
               attn_mask=AttentionMask.causal().with_segment_ids(hax.named(jnp.asarray(seg), (Batch, Pos))))
common = dict(max_seq_len=T, hidden_dim=48, intermediate_dim=64, num_layers=2, num_heads=2, num_kv_heads=2, hybrid_norm=True,
              attn_backend=AttentionBackend.VANILLA, oe_m=1000, oe_k=2, od_weight=0.1, od_mode="hashed")
key = jrandom.PRNGKey(7)
tt = np.arange(T)[None]
same = lambda s: np.roll(seg, -s, 1) == seg
ok3 = (tt <= T - 4) & same(1) & same(2) & same(3)
w3 = lw * np.roll(lw, -1, 1) * np.roll(lw, -2, 1) * ok3
y1 = np.roll(tok, -1, 1)
w1 = next_token_loss_weight(Pos, ex.loss_weight).array


def ce_mean(logits, y, w):
    ce = jax.nn.logsumexp(logits, -1) - jnp.take_along_axis(logits, jnp.asarray(y).reshape(-1)[:, None], 1)[:, 0]
    w = jnp.asarray(w).reshape(-1)
    return jnp.sum(ce * w) / jnp.sum(w)


def scalar(x):
    return x.array if isinstance(x, hax.NamedArray) else x


def close(a, b, what, rtol=2e-4):
    for (p, u), v in zip(jax.tree_util.tree_leaves_with_path(eqx.filter(a, eqx.is_array)), jax.tree.leaves(eqx.filter(b, eqx.is_array))):
        np.testing.assert_allclose(np.asarray(u, np.float32), np.asarray(v, np.float32), rtol=rtol, atol=1e-6, err_msg=f"{what} {jax.tree_util.keystr(p)}")


tc = TrainerConfig(mesh=MeshConfig(axes={"data": -1, "replica": 1, "model": 1}, compute_mapping={
    "token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
    "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
    ROWS: ResourceAxis.DATA}, param_mapping={"embed": "data", ROWS: "data"}))
with tc.use_device_mesh(), hax.axis_mapping(tc.compute_axis_mapping):
    es = hax.shard(ex, tc.compute_axis_mapping)
    w3n = hax.named(jnp.asarray(w3, jnp.float32), (Batch, Pos))
    # 1. the order-3 loss with every class a candidate == dense 3-gram softmax
    m = B * T // 4
    cfg = OverVocabQwen3Config(**common, od_m=m - 4, od_orders=(2, 3))   # the 3-gram modulus is od_m + 4 = 64
    model = OverVocabQwen3LMHeadModel.init(Vocab, cfg, key=jrandom.PRNGKey(0))
    h = jrandom.normal(jrandom.PRNGKey(3), (B, T, cfg.hidden_dim)) * 0.5
    hN = hax.named(h, (Batch, Pos, cfg.Embed))

    def direct(parts):
        proj, table, out = parts
        h3 = proj(hN.rename({cfg.Embed.name: "od_in"}))
        loss, _ = hashed_od_loss(Pos, cfg.Embed, h3, tokens, w3n, table, out, m=m, vocab_size=V, key=jrandom.PRNGKey(5),
                                 dtype=jnp.float32, reduction=hax.mean, reduction_axis=None, order=3)
        return scalar(loss)

    def dense(parts):
        proj, table, out = parts
        h3 = (h @ proj.weight.rearrange(("od_in", "embed")).array).reshape(-1, cfg.hidden_dim)
        u = out(table).rearrange((ROWS, "embed")).array[:m]
        c = (np.roll(tok, -1, 1) + np.roll(tok, -2, 1) * V + np.roll(tok, -3, 1) * V * V) % m
        return ce_mean(h3 @ u.T, c, w3)

    parts = (model.od3_proj, model.od3_table, model.od3_out)
    ld, gd = eqx.filter_jit(eqx.filter_value_and_grad(direct))(parts)
    lr, gr = eqx.filter_value_and_grad(dense)(parts)
    np.testing.assert_allclose(float(ld), float(lr), rtol=1e-5)
    close(gd, gr, "order-3 head, full candidate set")
    print(f"1. order 3, m = {m} (= positions per device): loss {float(ld):.6f} == dense 3-gram CE {float(lr):.6f}; every gradient matches")

    # 2. the model with both heads
    cfg2 = OverVocabQwen3Config(**common, od_m=1000, od_orders=(2,))
    cfg23 = OverVocabQwen3Config(**common, od_m=1000, od_orders=(2, 3))
    m2 = OverVocabQwen3LMHeadModel.init(Vocab, cfg2, key=jrandom.PRNGKey(0))
    m23 = OverVocabQwen3LMHeadModel.init(Vocab, cfg23, key=jrandom.PRNGKey(0))
    assert m2.od3_proj is None and m2.od3_table is None and m2.od3_out is None
    for a, b in zip(jax.tree.leaves(eqx.filter(m2, eqx.is_array)), jax.tree.leaves(eqx.filter(eqx.tree_at(lambda q: (q.od3_proj, q.od3_table, q.od3_out), m23, (None, None, None), is_leaf=lambda x: x is None), eqx.is_array))):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    ms = hax.shard(m23, tc.parameter_axis_mapping)

    def lf(mm):
        loss, stats = mm.compute_next_token_loss(es, key=key, logsumexp_weight=0.0)
        return scalar(loss), {k: v.value() for k, v in stats.items()}
    (loss, stats), g = eqx.filter_jit(eqx.filter_value_and_grad(lf, has_aux=True))(ms)
    np.testing.assert_allclose(float(loss), float(stats["ntp_loss"]) + 0.1 * float(stats["od_loss"]) + 0.1 * float(stats["od3_loss"]), rtol=1e-5)
    k_main, _, k_s2 = jax.random.split(key, 3)
    hh = m23.transformer(m23.embed(tokens, ex.attn_mask), attn_mask=ex.attn_mask, key=k_main)
    h3 = m23.od3_proj(hh.rename({cfg23.Embed.name: "od_in"}))
    want, _ = eqx.filter_jit(lambda h3_: hashed_od_loss(Pos, cfg23.Embed, h3_, tokens, w3n, m23.od3_table, m23.od3_out, m=1004, vocab_size=V,
                                                       key=jrandom.fold_in(k_s2, 3), dtype=jnp.float32, reduction=hax.mean, reduction_axis=None, order=3))(h3)
    np.testing.assert_allclose(float(stats["od3_loss"]), float(scalar(want)), rtol=1e-5)
    for name in ("od3_proj", "od3_table", "od3_out"):
        leaves = jax.tree.leaves(eqx.filter(getattr(g, name), eqx.is_array))
        assert all(np.isfinite(np.asarray(l)).all() for l in leaves) and max(float(jnp.abs(l).max()) for l in leaves) > 0, name
    assert np.log(1004) > float(stats["od3_loss"]) > 0
    print(f"2. od_orders (2, 3): loss {float(loss):.6f} == ntp + 0.1 od + 0.1 od3 (od3 {float(stats['od3_loss']):.4f} == direct call with the "
          "model's mask and key); 3-gram parameters get nonzero finite gradients; other parameters initialise as with (2,)")

    # 3. evaluation
    ev = eqx.filter_jit(lambda mm, e_: mm.compute_next_token_loss(e_, key=None))(ms, es)
    hh_s = m23.transformer(m23.embed(tokens, ex.attn_mask), attn_mask=ex.attn_mask, key=None)
    f1 = ce_mean(hh_s.array.reshape(-1, cfg23.hidden_dim) @ m23.get_lm_head().rearrange((cfg23.Embed, Vocab)).array, y1, w1)
    np.testing.assert_allclose(float(scalar(ev)), float(f1), rtol=1e-5)
    print("3. evaluation == main-head CE")
print("ALL HASHED 3-GRAM CPU TESTS PASSED")
