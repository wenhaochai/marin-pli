"""CPU tests for experiments.references.sampled_softmax_qwen3 on a fake 4-device mesh (the trainer's MeshConfig).

Run from the marin repo on a vis node (a few minutes):
    nice -n 19 .venv/bin/python scripts/della/sampled_softmax_cpu_test.py

1. The stride sweep is a permutation of the vocabulary.
2. Candidate sets hold every target plus exactly P - n absent classes, taken in sweep order from the offset.
3. Per-device sampled cross-entropy and its gradients match a dense reference over the same candidate set.
4. Under the trainer's sharding, the model loss and every parameter gradient match an independent reference that
   splits the batch into the four device blocks and builds each block's candidate set in numpy; the stage follows the
   step at run time inside ONE compiled step; the metrics report the per-device target counts.
5. The full-softmax stage, the overflow fallback and evaluation (key=None) reproduce the baseline Qwen3 loss and gradients.
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
from levanter.models.qwen import Qwen3Config, Qwen3LMHeadModel
from levanter.trainer import _TRACED_TRAIN_STEP, TrainerConfig
from levanter.utils.mesh import MeshConfig

from experiments.references.sampled_softmax_qwen3 import (
    _SWEEP_KEY_FOLD,
    SampledSoftmaxQwen3Config,
    SampledSoftmaxQwen3LMHeadModel,
    build_candidates,
    sampled_cross_entropy,
    stride_sweep,
)

jax.config.update("jax_threefry_partitionable", True)  # levanter.trainer.DEFAULT_JAX_CONFIG
assert jax.device_count() == 4, jax.devices()
rng = np.random.default_rng(0)


def scalar(x):
    return x.array if isinstance(x, hax.NamedArray) else x


def np_candidates(labels: np.ndarray, P: int, offset: int, V: int) -> np.ndarray:
    """The specification: present classes plus the first P - n absent classes of the sweep read from offset."""
    present = np.zeros(V, bool)
    present[labels] = True
    order = stride_sweep(V)[(offset + np.arange(V)) % V]
    negatives = order[~present[order]][: P - present.sum()]
    return np.sort(np.concatenate([np.flatnonzero(present), negatives]))


# 1. stride sweep
for V in (1000, 50304, 128256):
    assert np.array_equal(np.sort(stride_sweep(V)), np.arange(V)), V
print("1. stride sweep is a permutation for V in (1000, 50304, 128256)")

# 2. candidate sets
V = 1000
sweep = jnp.asarray(stride_sweep(V))
for trial in range(20):
    n_labels = int(rng.integers(1, 400))
    labels = rng.zipf(1.3, size=n_labels) % V
    P = int(rng.integers(len(np.unique(labels)), V))
    offset = int(rng.integers(V))
    present = jnp.zeros((V,), jnp.bool_).at[labels].set(True)
    cand = np.asarray(build_candidates(present, jnp.sum(present, dtype=jnp.int32), P, jnp.int32(offset), sweep))
    ref = np_candidates(labels, P, offset, V)
    assert np.array_equal(cand, ref), (trial, P)
    assert len(np.unique(cand)) == P and np.all(np.diff(cand) > 0) and set(labels) <= set(cand.tolist())
print("2. candidate sets == spec on 20 random draws (sorted, unique, all targets, P - n sweep negatives)")

# 3. per-device sampled cross-entropy vs dense reference, loss and gradients (f32)
N, D, candidates = 256, 16, (300, 500)
x = jnp.asarray(rng.normal(size=(N, D)), jnp.float32)
w = jnp.asarray(rng.normal(size=(D, V)) * 0.3, jnp.float32)
labels = jnp.asarray(rng.zipf(1.3, size=N) % V, jnp.int32)
r = jnp.asarray(rng.normal(size=N), jnp.float32)
kernel_kw = dict(logsumexp_weight=None, block_size=None, dtype=jnp.float32, logit_soft_cap=None, precision=None)
offset = 123


def dense_ref(x, w, C):
    logits = x @ w
    return jax.nn.logsumexp(logits[:, C], axis=-1) - jnp.take_along_axis(logits, labels[:, None], axis=1)[:, 0]


for stage in (0, 1, 2):
    f = lambda x, w: sampled_cross_entropy(x, labels, w, stage=jnp.int32(stage), offset=jnp.int32(offset), candidates=candidates, sweep=sweep, **kernel_kw)
    loss, n_present, branch = f(x, w)
    C = np_candidates(np.asarray(labels), candidates[stage], offset, V) if stage < len(candidates) else np.arange(V)
    assert int(branch) == stage and int(n_present) == len(np.unique(np.asarray(labels)))
    np.testing.assert_allclose(loss, dense_ref(x, w, C), rtol=1e-5, atol=1e-5)
    g = jax.grad(lambda x, w: jnp.sum(f(x, w)[0] * r), argnums=(0, 1))(x, w)
    g_ref = jax.grad(lambda x, w: jnp.sum(dense_ref(x, w, C) * r), argnums=(0, 1))(x, w)
    for a, b in zip(g, g_ref):
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-5)
    if stage < len(candidates):  # the lm_head gradient lives on the candidate columns only
        outside = np.setdiff1d(np.arange(V), C)
        assert np.all(np.asarray(g[1])[:, outside] == 0)
# overflow: more distinct labels than P -> full softmax
wide = jnp.asarray(rng.permutation(V)[:N], jnp.int32)  # N distinct labels > 200
loss, _, branch = sampled_cross_entropy(x, wide, w, stage=jnp.int32(0), offset=jnp.int32(0), candidates=(200,), sweep=sweep, **kernel_kw)
full = jax.nn.logsumexp(x @ w, axis=-1) - jnp.take_along_axis(x @ w, wide[:, None], axis=1)[:, 0]
assert int(branch) == 1
np.testing.assert_allclose(loss, full, rtol=1e-5, atol=1e-5)
print("3. sampled CE == dense reference (loss, dx, dW) at stages 0/1/full; dW zero off the set; overflow -> full softmax")

# 4-5. model level, under the trainer's mesh and axis mappings
B, T = 8, 64  # 2 sequences per device
Vocab, Batch, Pos = hax.Axis("vocab", V), hax.Axis("batch", B), hax.Axis("position", T)
common = dict(max_seq_len=T, hidden_dim=32, intermediate_dim=64, num_layers=2, num_heads=2, num_kv_heads=2, hybrid_norm=True, attn_backend=AttentionBackend.VANILLA)
k_model, k_step = jrandom.split(jrandom.PRNGKey(0))
ends = (10, 20)
base = Qwen3LMHeadModel.init(Vocab, Qwen3Config(**common), key=k_model)
ssm = SampledSoftmaxQwen3LMHeadModel.init(Vocab, SampledSoftmaxQwen3Config(**common, ss_candidates=candidates, ss_stage_ends=ends), key=k_model)
for a, b in zip(jax.tree.leaves(eqx.filter(base, eqx.is_array)), jax.tree.leaves(eqx.filter(ssm, eqx.is_array))):
    assert np.array_equal(a, b)
print("4a. sampled-softmax model initialises to exactly the baseline parameters")


def make_example(tok):
    tokens = hax.named(jnp.asarray(tok, jnp.int32), (Batch, Pos))
    lw = np.ones((B, T), np.float32)
    lw[1, : T // 2] = 0.0  # a masked span, as a prompt or padding would be
    return LmExample(tokens=tokens, loss_weight=hax.named(jnp.asarray(lw), (Batch, Pos)), attn_mask=AttentionMask.causal())


ex = make_example(rng.zipf(1.2, size=(B, T)) % V)
ex_wide = make_example(np.stack([rng.permutation(V)[:T] for _ in range(B)]))  # ~all distinct: 128 per device


def reference_loss(model, example, step, key, P_by_stage, ends):
    """Independent reference: dense logits, the batch split into the 4 device blocks, numpy candidate sets."""
    h = model.activations(example.tokens, example.attn_mask, key=key).array
    W = model.get_lm_head().rearrange((model.Embed, model.Vocab)).array
    y = np.asarray(hax.roll(example.tokens, -1, Pos).array)
    wts = next_token_loss_weight(Pos, example.loss_weight).array
    stage = sum(step >= e for e in ends)
    off0 = int(jrandom.randint(jrandom.fold_in(key, _SWEEP_KEY_FOLD), (), 0, V, dtype=jnp.int32))
    total, per = 0.0, B // 4
    for d in range(4):
        rows = slice(d * per, (d + 1) * per)
        yd = y[rows].reshape(-1)
        logits = h[rows].reshape(-1, h.shape[-1]) @ W
        if stage < len(P_by_stage) and len(np.unique(yd)) <= P_by_stage[stage]:
            C = np_candidates(yd, P_by_stage[stage], (off0 + d * (V // 4)) % V, V)
        else:
            C = np.arange(V)
        l = jax.nn.logsumexp(logits[:, C], axis=-1) - jnp.take_along_axis(logits, jnp.asarray(yd)[:, None], axis=1)[:, 0]
        total = total + jnp.sum(l * wts[rows].reshape(-1))
    return total / jnp.sum(wts)


traces = []


def train_step_loss(model, example, step, key):
    """What Trainer._train_step does around the loss: publish the traced step, differentiate the loss function."""
    traces.append(1)
    token = _TRACED_TRAIN_STEP.set(step)
    try:
        def lf(m):
            out = m.compute_next_token_loss(example, key=key, logsumexp_weight=0.0)  # train_lm's call, z_loss_weight=0.0
            loss, stats = out if isinstance(out, tuple) else (out, {})
            return scalar(loss), stats

        (loss, stats), grads = eqx.filter_value_and_grad(lf, has_aux=True)(model)
    finally:
        _TRACED_TRAIN_STEP.reset(token)
    return loss, {k: v.value() for k, v in stats.items()}, grads


tc = TrainerConfig(
    mesh=MeshConfig(
        axes={"data": -1, "replica": 1, "model": 1},
        compute_mapping={
            "token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
            "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
        },
    )
)


def close_trees(a, b, rtol, atol, what):
    for (path, u), v in zip(jax.tree_util.tree_leaves_with_path(eqx.filter(a, eqx.is_array)), jax.tree.leaves(eqx.filter(b, eqx.is_array))):
        np.testing.assert_allclose(np.asarray(u, np.float32), np.asarray(v, np.float32), rtol=rtol, atol=atol, err_msg=f"{what} {jax.tree_util.keystr(path)}")


with tc.use_device_mesh(), hax.axis_mapping(tc.compute_axis_mapping):
    ssm_s, base_s = hax.shard(ssm, tc.parameter_axis_mapping), hax.shard(base, tc.parameter_axis_mapping)
    ex_s, exw_s = hax.shard(ex, tc.compute_axis_mapping), hax.shard(ex_wide, tc.compute_axis_mapping)
    step_fn = eqx.filter_jit(train_step_loss)
    base_fn = eqx.filter_jit(lambda m, e, k: eqx.filter_value_and_grad(lambda m: scalar(m.compute_next_token_loss(e, key=k, logsumexp_weight=0.0)))(m))
    base_loss, base_grads = base_fn(base_s, ex_s, k_step)

    y0 = np.asarray(hax.roll(ex.tokens, -1, Pos).array)
    distinct = [len(np.unique(y0[2 * d : 2 * d + 2])) for d in range(4)]
    for step in (0, 9, 10, 19, 20, 4000):
        loss, stats, grads = step_fn(ssm_s, ex_s, jnp.int32(step), k_step)
        stage = sum(step >= e for e in ends)
        ref_l, ref_g = eqx.filter_value_and_grad(lambda m: reference_loss(m, ex, step, k_step, candidates, ends))(ssm)
        np.testing.assert_allclose(float(loss), float(ref_l), rtol=1e-5, err_msg=f"step {step}")
        close_trees(grads, ref_g, 2e-4, 1e-6, f"step {step} grad")
        assert stats["ss/present_max"] == max(distinct) and abs(float(stats["ss/present_mean"]) - np.mean(distinct)) < 1e-3, (stats, distinct)
        assert float(stats["ss/overflow_frac"]) == 0.0 and float(stats["ss/candidates"]) == (candidates + (V,))[stage], stats
        if stage == len(candidates):  # full softmax: the baseline's loss and gradients
            np.testing.assert_allclose(float(loss), float(base_loss), rtol=1e-6)
            close_trees(grads, base_grads, 1e-5, 1e-7, f"step {step} vs baseline grad")
        print(f"4b. step {step:5d} stage {stage}: loss {float(loss):.6f} == reference {float(ref_l):.6f}; grads match; candidates {int(stats['ss/candidates'])}, targets/device {distinct}")
    assert len(traces) == 1, f"the stage must be a runtime switch inside one compiled step, traced {len(traces)} times"
    print("4c. one compilation served all six steps (stage picked from the traced step at run time)")

    # overflow: every device has ~128 distinct targets, above P = 300? no -- use the wide batch against P = 100
    ssm_small = SampledSoftmaxQwen3LMHeadModel.init(Vocab, SampledSoftmaxQwen3Config(**common, ss_candidates=(100,), ss_stage_ends=(50,)), key=k_model)
    loss, stats, grads = step_fn(hax.shard(ssm_small, tc.parameter_axis_mapping), exw_s, jnp.int32(0), k_step)
    wl, wg = base_fn(base_s, exw_s, k_step)
    assert float(stats["ss/overflow_frac"]) == 1.0, stats
    np.testing.assert_allclose(float(loss), float(wl), rtol=1e-6)
    close_trees(grads, wg, 1e-5, 1e-7, "overflow grad")
    print(f"5a. overflow (targets/device {int(stats['ss/present_max'])} > P=100): every device falls back to the full softmax; loss and grads == baseline")

    # mixed: device 0 fits P = 100, devices 1-3 overflow -> per-device choice, checked against the reference
    tok_mixed = np.asarray(ex_wide.tokens.array).copy()
    tok_mixed[0:2] = rng.integers(0, 40, size=(2, T))
    exm = make_example(tok_mixed)
    loss, stats, grads = step_fn(hax.shard(ssm_small, tc.parameter_axis_mapping), hax.shard(exm, tc.compute_axis_mapping), jnp.int32(0), k_step)
    ref_l, ref_g = eqx.filter_value_and_grad(lambda m: reference_loss(m, exm, 0, k_step, (100,), (50,)))(ssm_small)
    assert abs(float(stats["ss/overflow_frac"]) - 0.75) < 1e-6, stats
    np.testing.assert_allclose(float(loss), float(ref_l), rtol=1e-5)
    close_trees(grads, ref_g, 2e-4, 1e-6, "mixed overflow grad")
    print("5b. mixed batch: device 0 samples, devices 1-3 fall back (overflow_frac 0.75); loss and grads == reference")

    # evaluation: key=None is the baseline loss, bit for bit
    ev = eqx.filter_jit(lambda m, e: scalar(m.compute_next_token_loss(e, key=None)))
    assert float(ev(ssm_s, ex_s)) == float(ev(base_s, ex_s))
    print("5c. evaluation (key=None) == baseline loss exactly")

    try:
        ssm.compute_next_token_loss(ex, key=k_step)
        raise AssertionError("a training call outside the train step must fail loudly")
    except RuntimeError as e:
        assert "current_train_step" in str(e)
    print("5d. a keyed call with no train step raises instead of silently running the full softmax")

print("ALL SAMPLED-SOFTMAX CPU TESTS PASSED")
