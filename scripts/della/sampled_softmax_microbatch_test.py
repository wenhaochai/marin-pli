"""CPU test of the sampled softmax under levanter's microbatching (520m and 1_2b run two microbatches per step).

Run from the marin repo on a vis node: nice -n 19 .venv/bin/python scripts/della/sampled_softmax_microbatch_test.py
Wraps the loss exactly as Trainer._compute_gradients_microbatched does (eqx.filter_value_and_grad(has_aux=True) inside
levanter.grad_accum.microbatched, the traced step published through the trainer's context var) on a fake 4-device mesh,
and checks loss, every gradient and the metrics against an independent reference: each microbatch (rows m*mbs.. of the
batch, its own split of the step key) split into the 4 device blocks, candidate sets built in numpy, loss and
gradients averaged over microbatches. Stages 0, 1 and the full softmax.
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
from levanter.grad_accum import microbatched
from levanter.layers.attention import AttentionBackend, AttentionMask
from levanter.models.lm_model import LmExample
from levanter.models.loss import next_token_loss_weight
from levanter.trainer import _TRACED_TRAIN_STEP, TrainerConfig
from levanter.utils.mesh import MeshConfig

from experiments.references.sampled_softmax_qwen3 import _SWEEP_KEY_FOLD, SampledSoftmaxQwen3Config, SampledSoftmaxQwen3LMHeadModel, stride_sweep

jax.config.update("jax_threefry_partitionable", True)
assert jax.device_count() == 4
B, MBS, T, V, NDEV = 16, 8, 64, 1000, 4
CAND, ENDS = (300, 500), (10, 20)
Vocab, Batch, Pos = hax.Axis("vocab", V), hax.Axis("batch", B), hax.Axis("position", T)
common = dict(max_seq_len=T, hidden_dim=32, intermediate_dim=64, num_layers=2, num_heads=2, num_kv_heads=2, hybrid_norm=True, attn_backend=AttentionBackend.VANILLA)
k_model, k_step = jrandom.split(jrandom.PRNGKey(1))
model = SampledSoftmaxQwen3LMHeadModel.init(Vocab, SampledSoftmaxQwen3Config(**common, ss_candidates=CAND, ss_stage_ends=ENDS), key=k_model)
rng = np.random.default_rng(5)
tok = rng.zipf(1.2, size=(B, T)) % V
lw = np.ones((B, T), np.float32); lw[3, :20] = 0.0; lw[12, 30:] = 0.0
ex = LmExample(tokens=hax.named(jnp.asarray(tok, jnp.int32), (Batch, Pos)), loss_weight=hax.named(jnp.asarray(lw), (Batch, Pos)), attn_mask=AttentionMask.causal())


def np_candidates(labels, P, offset):
    present = np.zeros(V, bool); present[labels] = True
    order = stride_sweep(V)[(offset + np.arange(V)) % V]
    return np.sort(np.concatenate([np.flatnonzero(present), order[~present[order]][: P - present.sum()]]))


def reference(m, step, key):
    """Mean over microbatches of each microbatch's weighted-mean CE over per-device candidate sets."""
    keys = jax.random.split(key, B // MBS)
    total = 0.0
    for a in range(B // MBS):
        rows = slice(a * MBS, (a + 1) * MBS)
        toks = hax.named(jnp.asarray(tok[rows]), (Batch.resize(MBS), Pos))
        h = m.activations(toks, AttentionMask.causal(), key=keys[a]).array
        W = m.get_lm_head().rearrange((m.Embed, m.Vocab)).array
        y = np.roll(tok[rows], -1, axis=1)
        w = next_token_loss_weight(Pos, hax.named(jnp.asarray(lw[rows]), (Batch.resize(MBS), Pos))).array
        stage = sum(step >= e for e in ENDS)
        off0 = int(jrandom.randint(jrandom.fold_in(keys[a], _SWEEP_KEY_FOLD), (), 0, V, dtype=jnp.int32))
        num, per = 0.0, MBS // NDEV
        for d in range(NDEV):
            r = slice(d * per, (d + 1) * per)
            yd = y[r].reshape(-1)
            logits = h[r].reshape(-1, h.shape[-1]) @ W
            C = np_candidates(yd, CAND[stage], (off0 + d * (V // NDEV)) % V) if stage < len(CAND) and len(np.unique(yd)) <= CAND[stage] else np.arange(V)
            l = jax.nn.logsumexp(logits[:, C], axis=-1) - jnp.take_along_axis(logits, jnp.asarray(yd)[:, None], axis=1)[:, 0]
            num = num + jnp.sum(l * w[r].reshape(-1))
        total = total + num / jnp.sum(w)
    return total / (B // MBS)


tc = TrainerConfig(mesh=MeshConfig(axes={"data": -1, "replica": 1, "model": 1}, compute_mapping={
    "token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
    "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA)}))


def loss_fn(m, example, *, key):
    out = m.compute_next_token_loss(example, key=key, logsumexp_weight=0.0)
    loss, stats = out
    return loss.array, stats


with tc.use_device_mesh(), hax.axis_mapping(tc.compute_axis_mapping):
    grad_fn = microbatched(eqx.filter_value_and_grad(loss_fn, has_aux=True), Batch, MBS, tc.parameter_axis_mapping, tc.compute_axis_mapping)

    @eqx.filter_jit
    def step_fn(m, example, step, key):
        token = _TRACED_TRAIN_STEP.set(step)
        try:
            (loss, stats), grads = grad_fn(m, example, key=key)
        finally:
            _TRACED_TRAIN_STEP.reset(token)
        return loss, {k: v.value() for k, v in stats.items()}, grads

    ms, es = hax.shard(model, tc.parameter_axis_mapping), hax.shard(ex, tc.compute_axis_mapping)
    for step in (0, 15, 25):
        loss, stats, grads = step_fn(ms, es, jnp.int32(step), k_step)
        ref_l, ref_g = eqx.filter_value_and_grad(lambda m: reference(m, step, k_step))(model)
        np.testing.assert_allclose(float(loss), float(ref_l), rtol=1e-5)
        for (path, u), v in zip(jax.tree_util.tree_leaves_with_path(eqx.filter(grads, eqx.is_array)), jax.tree.leaves(eqx.filter(ref_g, eqx.is_array))):
            np.testing.assert_allclose(np.asarray(u, np.float32), np.asarray(v, np.float32), rtol=2e-4, atol=1e-6, err_msg=f"step {step} {jax.tree_util.keystr(path)}")
        stage = sum(step >= e for e in ENDS)
        assert float(stats["ss/candidates"]) == (CAND + (V,))[stage] and float(stats["ss/overflow_frac"]) == 0.0, stats
        print(f"step {step:3d} stage {stage}: microbatched loss {float(loss):.6f} == reference {float(ref_l):.6f}; all grads match; metrics {({k: round(float(v), 1) for k, v in stats.items()})}")
print("MICROBATCH TEST PASSED")
