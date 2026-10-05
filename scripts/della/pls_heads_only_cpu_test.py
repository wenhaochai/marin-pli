"""CPU tests for the head refit (per_layer_qwen3: pls_heads_only, HeadsOnlyMuonHConfig; launcher HEADS_FROM).

Run from the worktree on a vis node (fake 4-device mesh):
    nice -n 19 .venv/bin/python scripts/della/pls_heads_only_cpu_test.py
1. Gradients: with pls_heads_only every parameter outside the heads (aux_lm_heads, aux_norms, lm_head, final norm) gets
   exactly zero gradient; the heads get the same gradient as in separate-heads training, for both source setups
   (separate heads, and probes = pls_detach_backbone). The loss and per-layer metrics are unchanged.
2. Labels: HeadsOnlyMuonHConfig labels exactly the head leaves as MuonHConfig does and every other leaf "frozen".
3. One optimizer step on the separate-heads gradient: every non-head leaf is bit-identical; every head leaf equals the
   step MuonHConfig takes on the same gradient (clipping off, so the per-group global norms cannot differ).
4. Three steps of head-only training: the backbone stays bit-identical, the per-layer losses fall.
5. flops_per_token: the backbone counted forward only.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4"
os.environ.setdefault("LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS", "0")

import dataclasses  # noqa: E402

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jrandom  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402

import haliax as hax  # noqa: E402
from haliax import Axis  # noqa: E402
from haliax.partitioning import ResourceAxis  # noqa: E402
from levanter.layers.attention import AttentionBackend, AttentionMask  # noqa: E402
from levanter.models.lm_model import LmExample  # noqa: E402
from levanter.optim.muonh import MuonHConfig  # noqa: E402
from levanter.trainer import TrainerConfig  # noqa: E402
from levanter.utils.jax_utils import leaf_key_paths  # noqa: E402
from levanter.utils.mesh import MeshConfig  # noqa: E402

from experiments.references.per_layer_qwen3 import HeadsOnlyMuonHConfig, PerLayerQwen3Config, PerLayerQwen3LMHeadModel, _is_head_path  # noqa: E402

jax.config.update("jax_threefry_partitionable", True)
rng = np.random.default_rng(0)
B, T, V, L = 8, 32, 500, 4
Batch, Pos, Vocab = Axis("batch", B), Axis("position", T), Axis("vocab", V)


def batch(seed):
    r = np.random.default_rng(seed)
    tok = r.integers(2, V, size=(B, T))
    return LmExample(
        tokens=hax.named(jnp.asarray(tok, jnp.int32), (Batch, Pos)),
        loss_weight=hax.named(jnp.asarray((np.arange(T) < T - 1).astype(np.float32)[None].repeat(B, 0)), (Batch, Pos)),
        attn_mask=AttentionMask.causal(),
    )


common = dict(max_seq_len=T, hidden_dim=48, intermediate_dim=64, num_layers=L, num_heads=2, num_kv_heads=2, attn_backend=AttentionBackend.VANILLA, pls_weight=1.0, pls_separate_heads=True, hybrid_norm=True)
key0 = jrandom.PRNGKey(0)
opt_kw = dict(learning_rate=0.01, adam_lr=0.002, beta1=0.9, beta2=0.98, epsilon=1e-15, momentum=0.98, nesterov=True, backend_steps=5, muon_epsilon=1e-5, max_grad_norm=None, weight_decay=0.1, lr_schedule="constant", warmup=0, min_lr_ratio=0.0, coefficient_type="simple")


def model(**kw):
    return PerLayerQwen3LMHeadModel.init(Vocab, PerLayerQwen3Config(**common, **kw), key=key0)


def heads_only(m):
    """The same model and weights with pls_heads_only (the model's config lives on its transformer)."""
    return dataclasses.replace(m, transformer=dataclasses.replace(m.transformer, config=dataclasses.replace(m.config, pls_heads_only=True)))


def flat(tree):
    """(path string, leaf) for every array leaf."""
    return [(jax.tree_util.keystr(p), np.asarray(x, np.float32)) for p, x in jax.tree_util.tree_leaves_with_path(tree)]


def head_flags(params):
    """One bool per array leaf (tree_leaves order): is it a head parameter."""
    paths = leaf_key_paths(params)
    return [_is_head_path(p) for p in jax.tree.leaves(paths)]


tc = TrainerConfig(mesh=MeshConfig(axes={"data": -1, "replica": 1, "model": 1}, compute_mapping={
    "token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
    "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA)}))
key = jrandom.PRNGKey(7)


def loss_and_grad(m_, example, k):
    def lf(mm):
        loss, stats = mm.compute_next_token_loss(example, key=k, logsumexp_weight=0.0)
        return (loss.array if isinstance(loss, hax.NamedArray) else loss), stats

    (loss, stats), g = eqx.filter_value_and_grad(lf, has_aux=True)(m_)
    return loss, stats, g


with tc.use_device_mesh(), hax.axis_mapping(tc.compute_axis_mapping):
    es = hax.shard(batch(0), tc.compute_axis_mapping)
    step = eqx.filter_jit(loss_and_grad)

    # 1. gradients, for both source setups
    for src in (dict(), dict(pls_detach_backbone=True)):
        m_src = model(**src)
        m_ho = heads_only(m_src)
        l_src, s_src, g_src = step(m_src, es, key)
        l_ho, s_ho, g_ho = step(m_ho, es, key)
        np.testing.assert_allclose(float(l_ho), float(l_src), rtol=1e-6)
        for k_ in s_src:
            np.testing.assert_allclose(float(s_ho[k_].value()), float(s_src[k_].value()), rtol=1e-6, err_msg=k_)
        params = eqx.filter(m_ho, eqx.is_inexact_array)
        flags = head_flags(params)
        leaves_ho, leaves_src = flat(eqx.filter(g_ho, eqx.is_inexact_array)), flat(eqx.filter(g_src, eqx.is_inexact_array))
        assert len(flags) == len(leaves_ho) == len(leaves_src), (len(flags), len(leaves_ho), len(leaves_src))
        n_head = n_zero = 0
        for is_head, (p, gh), (_, gs) in zip(flags, leaves_ho, leaves_src):
            if is_head:
                n_head += 1
                np.testing.assert_allclose(gh, gs, rtol=1e-5, atol=1e-8, err_msg=f"head grad {p}")
                assert np.abs(gh).max() > 0, f"head {p} gets no gradient"
            else:
                n_zero += 1
                assert np.all(gh == 0), f"backbone leaf {p} gets a gradient: max {np.abs(gh).max()}"
        print(f"1. {src or 'separate heads'}: {n_head} head leaves match, {n_zero} backbone leaves exactly zero; loss {float(l_ho):.4f}")
        head_paths = [p for f, (p, _) in zip(flags, leaves_ho) if f]
        assert any("aux_lm_heads" in p for p in head_paths) and any("aux_norms" in p for p in head_paths), head_paths
        assert any(p.startswith(".lm_head") for p in head_paths) and any(p.startswith(".transformer.norm") for p in head_paths), head_paths
        assert not any("embeddings" in p or "layers" in p for p in head_paths), head_paths

    # 2. labels
    m0 = model()
    params = eqx.filter(m0, eqx.is_inexact_array)
    ho, mu = HeadsOnlyMuonHConfig(**opt_kw), MuonHConfig(**opt_kw)
    lab_ho = jax.tree.leaves(ho.create_mask(params))
    lab_mu = jax.tree.leaves(mu.create_mask(params))
    flags = head_flags(params)
    assert len(lab_ho) == len(lab_mu) == len(flags), (len(lab_ho), len(lab_mu), len(flags))
    for f, a, b_ in zip(flags, lab_ho, lab_mu):
        assert (a == b_) if f else (a == "frozen"), (f, a, b_)
    print("2. labels:", sorted(set(a for f, a in zip(flags, lab_ho) if f)), "on heads;", sum(not f for f in flags), "leaves frozen")

    # 3. one optimizer step on the same gradient
    _, _, g = step(m0, es, key)
    grads = eqx.filter(g, eqx.is_inexact_array)
    tx_ho, tx_mu = ho.build(10), mu.build(10)
    up_ho, _ = tx_ho.update(grads, tx_ho.init(params), params)
    up_mu, _ = tx_mu.update(grads, tx_mu.init(params), params)
    new_ho, new_mu = optax.apply_updates(params, up_ho), optax.apply_updates(params, up_mu)
    for f, (p, a), (_, b_), (_, c) in zip(flags, flat(new_ho), flat(new_mu), flat(params)):
        if f:
            np.testing.assert_allclose(a, b_, rtol=1e-6, atol=1e-9, err_msg=f"head step {p}")
            assert not np.array_equal(a, c), f"head {p} did not move"
        else:
            assert np.array_equal(a, c), f"frozen leaf {p} moved"
    print("3. one step: heads equal MuonHConfig's step, the rest bit-identical")

    # 4. three steps of head-only training on fresh batches
    m = heads_only(m0)
    p = eqx.filter(m, eqx.is_inexact_array)
    tx = HeadsOnlyMuonHConfig(**dict(opt_kw, max_grad_norm=1.0)).build(10)
    state = tx.init(p)
    first = None
    for i in range(3):
        e = hax.shard(batch(1), tc.compute_axis_mapping)  # the same batch: the loss must fall on it
        loss, stats, g = step(m, e, key)
        first = first if first is not None else {k_: float(v.value()) for k_, v in stats.items()}
        up, state = tx.update(eqx.filter(g, eqx.is_inexact_array), state, p)
        p = optax.apply_updates(p, up)
        m = eqx.combine(p, m)
    _, last, _ = step(m, e, key)
    for f, (q, a), (_, c) in zip(flags, flat(p), flat(params)):
        if not f:
            assert np.array_equal(a, c), f"frozen leaf {q} moved after 3 steps"
    for k_ in sorted(first):
        assert float(last[k_].value()) < first[k_], (k_, first[k_], float(last[k_].value()))
    print("4. three steps: backbone bit-identical;", {k_: f"{first[k_]:.3f}->{float(last[k_].value()):.3f}" for k_ in sorted(first)})

    # 5. flops
    c0, c1 = PerLayerQwen3Config(**common), PerLayerQwen3Config(**common, pls_heads_only=True)
    f0, f1 = c0.flops_per_token(V, T), c1.flops_per_token(V, T)
    heads = L * 2 * c0.hidden_dim * V
    np.testing.assert_allclose(f1, (f0 - heads) / 3 + heads, rtol=1e-12)
    print(f"5. flops/token {f0:.0f} -> {f1:.0f}")

    # the guard
    for bad in (dict(pls_separate_heads=False), dict(pls_local_heads=True), dict(pls_probe=True)):
        try:
            PerLayerQwen3Config(**dict(common, **bad), pls_heads_only=True)
        except ValueError:
            continue
        raise AssertionError(f"pls_heads_only accepted {bad}")
print("ALL HEADS-ONLY CPU TESTS PASSED")
