"""CPU tests for experiments.references.focal_qwen3 (focal loss) and OV with focal on both heads, on a fake 4-device mesh.

Run from the marin repo on a vis node: nice -n 19 .venv/bin/python scripts/della/focal_cpu_test.py
1. focal_transform is (1 - exp(-l))^gamma * l, the identity at gamma 0, and has finite gradients at l = 0 (masked tokens).
2. Baseline + focal, sharded: at gamma 0 the loss and every gradient equal the baseline's cross-entropy; at gamma 0.5 and
   1 they equal an independent reference from dense logits (weighted mean of the per-token focal loss over the example
   weights); the logged ce_loss is the plain cross-entropy; evaluation is the plain cross-entropy.
3. OV + focal (both heads), sharded: loss = focal(main) + 0.1 focal(OD) with OD masked past the window, across documents
   and by the example weights, every gradient matches, ntp_ce / od_ce are the plain cross-entropies, evaluation is the
   main head's plain cross-entropy.
4. focal with the sampled softmax is refused.
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
from levanter.models.lm_model import LmExample, split_activations
from levanter.models.loss import next_token_loss_weight
from levanter.models.qwen import Qwen3LMHeadModel
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig

from experiments.references.focal_qwen3 import FocalQwen3Config, FocalQwen3LMHeadModel, focal_transform
from experiments.references.over_vocab_qwen3 import ROWS, OverVocabQwen3Config, OverVocabQwen3LMHeadModel

jax.config.update("jax_threefry_partitionable", True)
assert jax.device_count() == 4
rng = np.random.default_rng(0)

# 1. the transform
Tok = hax.Axis("tok", 64)
l = jnp.asarray(np.concatenate([[0.0, 1e-9, 1e-4], rng.uniform(0, 12, 61)]), jnp.float32)
for g in (0.0, 0.5, 1.0, 2.0):
    got = focal_transform(hax.named(l, Tok), g).array
    want = (1 - np.exp(-np.asarray(l, np.float64))) ** g * np.asarray(l, np.float64)
    np.testing.assert_allclose(np.asarray(got), want, rtol=1e-5, atol=1e-12)
    grad = jax.grad(lambda x: jnp.sum(focal_transform(hax.named(x, Tok), g).array))(l)
    assert bool(jnp.all(jnp.isfinite(grad))), (g, grad[:3])
assert np.array_equal(np.asarray(focal_transform(hax.named(l, Tok), 0.0).array), np.asarray(l))
print("1. focal_transform == (1 - e^-l)^gamma l, identity at gamma 0, finite gradients at l = 0")

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


def per_token_ce(h, W, y):
    logits = h @ W
    return jax.nn.logsumexp(logits, axis=-1) - jnp.take_along_axis(logits, y[:, None], axis=1)[:, 0]


def focal_mean(h, W, y, w, g):
    ce = per_token_ce(h.reshape(-1, h.shape[-1]), W, jnp.asarray(y).reshape(-1))
    f = (1 - jnp.exp(-ce)) ** g * ce if g > 0 else ce
    w = jnp.asarray(w).reshape(-1)
    return jnp.sum(f * w) / jnp.sum(w), jnp.sum(ce * w) / jnp.sum(w)


def close(a, b, what, rtol=2e-4):
    for (p, u), v in zip(jax.tree_util.tree_leaves_with_path(eqx.filter(a, eqx.is_array)), jax.tree.leaves(eqx.filter(b, eqx.is_array))):
        np.testing.assert_allclose(np.asarray(u, np.float32), np.asarray(v, np.float32), rtol=rtol, atol=1e-6, err_msg=f"{what} {jax.tree_util.keystr(p)}")


def scalar(x):
    return x.array if isinstance(x, hax.NamedArray) else x


def train_loss(m_, example, k, method=None):
    def lf(mm):
        out = (method or type(mm).compute_next_token_loss)(mm, example, key=k, logsumexp_weight=0.0)
        loss, stats = out if isinstance(out, tuple) else (out, {})
        return scalar(loss), stats
    (loss, stats), g = eqx.filter_value_and_grad(lf, has_aux=True)(m_)
    return loss, {k_: v.value() for k_, v in stats.items()}, g


key = jrandom.PRNGKey(7)
y1 = np.roll(tok, -1, 1)
w1 = next_token_loss_weight(Pos, ex.loss_weight).array

# 2. baseline + focal
tc = TrainerConfig(mesh=MeshConfig(axes={"data": -1, "replica": 1, "model": 1}, compute_mapping={
    "token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
    "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA)}))
with tc.use_device_mesh(), hax.axis_mapping(tc.compute_axis_mapping):
    es = hax.shard(ex, tc.compute_axis_mapping)
    for g in (0.0, 0.5, 1.0):
        cfg = FocalQwen3Config(**common, focal_gamma=g)
        model = FocalQwen3LMHeadModel.init(Vocab, cfg, key=jrandom.PRNGKey(0))
        ms = hax.shard(model, tc.parameter_axis_mapping)
        loss, stats, grads = eqx.filter_jit(train_loss)(ms, es, key)
        if g == 0.0:
            b_loss, _, b_grads = eqx.filter_jit(lambda m_, e_, k: train_loss(m_, e_, k, Qwen3LMHeadModel.compute_next_token_loss))(ms, es, key)
            np.testing.assert_allclose(float(loss), float(b_loss), rtol=1e-6)
            close(grads, b_grads, "gamma 0 vs baseline", rtol=1e-5)

        def ref(mm):
            h = split_activations(mm.activations(tokens, ex.attn_mask, key=key))[0].array
            return focal_mean(h, mm.get_lm_head().rearrange((cfg.Embed, Vocab)).array, y1, w1, g)
        (ref_l, ref_ce), ref_g = eqx.filter_value_and_grad(ref, has_aux=True)(model)
        np.testing.assert_allclose(float(loss), float(ref_l), rtol=1e-5)
        np.testing.assert_allclose(float(stats["ce_loss"]), float(ref_ce), rtol=1e-5)
        close(grads, ref_g, f"focal gamma {g}")
        ev = eqx.filter_jit(lambda mm, e_: mm.compute_next_token_loss(e_, key=None))(ms, es)
        np.testing.assert_allclose(float(scalar(ev)), float(ref_ce), rtol=1e-5)
        print(f"2. baseline + focal gamma {g}: loss {float(loss):.6f} == reference (ce {float(ref_ce):.6f}); all grads match; eval == CE"
              + ("; == baseline loss and grads" if g == 0.0 else ""))

# 3. OV + focal on both heads
tc = TrainerConfig(mesh=MeshConfig(axes={"data": -1, "replica": 1, "model": 1}, compute_mapping={
    "token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
    "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
    ROWS: ResourceAxis.DATA}, param_mapping={"embed": "data", ROWS: "data"}))
tt = np.arange(T)[None]
ok2 = (tt <= T - 3) & (np.roll(seg, -1, 1) == seg) & (np.roll(seg, -2, 1) == seg)
w2 = lw * np.roll(lw, -1, 1) * ok2
y2 = np.roll(tok, -2, 1)
with tc.use_device_mesh(), hax.axis_mapping(tc.compute_axis_mapping):
    es = hax.shard(ex, tc.compute_axis_mapping)
    for g in (0.5, 1.0):
        cfg = OverVocabQwen3Config(**common, oe_m=1000, oe_k=2, od_weight=0.1, focal_gamma=g)
        model = OverVocabQwen3LMHeadModel.init(Vocab, cfg, key=jrandom.PRNGKey(0))
        ms = hax.shard(model, tc.parameter_axis_mapping)
        loss, stats, grads = eqx.filter_jit(train_loss)(ms, es, key)

        def ref(mm):
            k_main, _, _ = jax.random.split(key, 3)
            h = mm.transformer(mm.embed(tokens, ex.attn_mask), attn_mask=ex.attn_mask, key=k_main).array
            h2 = h @ mm.od_proj.weight.rearrange(("od_in", "embed")).array
            f1, c1 = focal_mean(h, mm.get_lm_head().rearrange((cfg.Embed, Vocab)).array, y1, w1, g)
            f2, c2 = focal_mean(h2, mm.od_lm_head.weight.rearrange((cfg.Embed, Vocab)).array, y2, w2, g)
            return f1 + cfg.od_weight * f2, (f1, f2, c1, c2)
        (ref_l, (f1, f2, c1, c2)), ref_g = eqx.filter_value_and_grad(ref, has_aux=True)(model)
        np.testing.assert_allclose(float(loss), float(ref_l), rtol=1e-5)
        np.testing.assert_allclose(float(stats["ntp_loss"]), float(f1), rtol=1e-5)
        np.testing.assert_allclose(float(stats["od_loss"]), float(f2), rtol=1e-5)
        np.testing.assert_allclose(float(stats["ntp_ce"]), float(c1), rtol=1e-5)
        np.testing.assert_allclose(float(stats["od_ce"]), float(c2), rtol=1e-5)
        close(grads, ref_g, f"ov focal gamma {g}")
        ev = eqx.filter_jit(lambda mm, e_: mm.compute_next_token_loss(e_, key=None))(ms, es)
        np.testing.assert_allclose(float(scalar(ev)), float(c1), rtol=1e-5)
        print(f"3. OV + focal gamma {g}: loss {float(loss):.6f} == focal(main) {float(f1):.6f} + 0.1 focal(OD) {float(f2):.6f}; "
              f"ntp_ce / od_ce match; all grads match; eval == main-head CE")

# 4. guard
try:
    OverVocabQwen3Config(**common, focal_gamma=1.0, ss_candidates=(100,), ss_stage_ends=(10,))
    raise AssertionError("focal + sampled softmax should be refused")
except ValueError:
    print("4. focal + sampled softmax is refused")
print("ALL FOCAL CPU TESTS PASSED")
