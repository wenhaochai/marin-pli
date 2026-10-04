"""CPU test for OV's oe_freeze_step (over_vocab_qwen3), on a fake 4-device mesh.

Run from the marin repo on a vis node: nice -n 19 .venv/bin/python scripts/della/oe_freeze_cpu_test.py
With oe_freeze_step = 5 and the same parameters as an unfrozen model: before step 5 the loss and every gradient equal the
unfrozen model's; from step 5 on the loss is unchanged, the n-gram tables' gradients are exactly zero and every other
gradient equals the unfrozen model's.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4"

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

import haliax as hax
from haliax.partitioning import ResourceAxis
from levanter.layers.attention import AttentionBackend, AttentionMask
from levanter.models.lm_model import LmExample
from levanter.trainer import _TRACED_TRAIN_STEP, TrainerConfig
from levanter.utils.mesh import MeshConfig

from experiments.references.over_vocab_qwen3 import ROWS, OverVocabQwen3Config, OverVocabQwen3LMHeadModel

jax.config.update("jax_threefry_partitionable", True)
rng = np.random.default_rng(0)
B, T, V = 8, 32, 500
Batch, Pos, Vocab = hax.Axis("batch", B), hax.Axis("position", T), hax.Axis("vocab", V)
tok = rng.integers(2, V, size=(B, T)); tok[:, 10] = 1
seg = np.cumsum(np.roll(tok, 1, axis=1) == 1, axis=1).astype(np.int32); seg[:, 0] = 0
lw = (np.arange(T) < T - 1).astype(np.float32)[None].repeat(B, 0)
ex = LmExample(tokens=hax.named(jnp.asarray(tok, jnp.int32), (Batch, Pos)), loss_weight=hax.named(jnp.asarray(lw), (Batch, Pos)),
               attn_mask=AttentionMask.causal().with_segment_ids(hax.named(jnp.asarray(seg), (Batch, Pos))))
common = dict(max_seq_len=T, hidden_dim=48, intermediate_dim=64, num_layers=2, num_heads=2, num_kv_heads=2, hybrid_norm=True,
              attn_backend=AttentionBackend.VANILLA, oe_m=1000, oe_k=2, od_weight=0.1)
key = jrandom.PRNGKey(7)


def step_grads(m_, step):
    token = _TRACED_TRAIN_STEP.set(step)
    try:
        def lf(mm):
            loss, _ = mm.compute_next_token_loss(ex, key=key, logsumexp_weight=0.0)
            return loss.array if isinstance(loss, hax.NamedArray) else loss
        return eqx.filter_value_and_grad(lf)(m_)
    finally:
        _TRACED_TRAIN_STEP.reset(token)


tc = TrainerConfig(mesh=MeshConfig(axes={"data": -1, "replica": 1, "model": 1}, compute_mapping={
    "token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
    "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA), ROWS: ResourceAxis.DATA},
    param_mapping={"embed": "data", ROWS: "data"}))
with tc.use_device_mesh(), hax.axis_mapping(tc.compute_axis_mapping):
    plain = OverVocabQwen3LMHeadModel.init(Vocab, OverVocabQwen3Config(**common), key=jrandom.PRNGKey(0))
    frozen = OverVocabQwen3LMHeadModel.init(Vocab, OverVocabQwen3Config(**common, oe_freeze_step=5), key=jrandom.PRNGKey(0))
    for a, b in zip(jax.tree.leaves(eqx.filter(plain, eqx.is_array)), jax.tree.leaves(eqx.filter(frozen, eqx.is_array))):
        assert np.array_equal(np.asarray(a), np.asarray(b))   # same parameters: the field does not change init
    for step in (3, 7):
        lp, gp = step_grads(plain, step)
        lf, gf = step_grads(frozen, step)
        np.testing.assert_allclose(float(lf), float(lp), rtol=1e-6)
        for (path, a), b in zip(jax.tree_util.tree_leaves_with_path(eqx.filter(gf, eqx.is_array)), jax.tree.leaves(eqx.filter(gp, eqx.is_array))):
            name = jax.tree_util.keystr(path)
            a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
            if "oe_tables" in name and step >= 5:
                assert np.all(a == 0), name
                assert np.abs(b).max() > 0, name
            else:
                np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-7, err_msg=f"step {step} {name}")
        print(f"step {step}: loss {float(lf):.6f} == unfrozen; " + ("table gradients 0, all others equal" if step >= 5 else "every gradient equal"))
print("ALL OE-FREEZE CPU TESTS PASSED")
