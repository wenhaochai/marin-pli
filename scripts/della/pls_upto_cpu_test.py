"""CPU test for pls_sep_upto (owner popup 2026-10-18, page Figure 22): separate heads on a 4-layer model where only the
readouts of layers 1..n (n = 1, 2) reach the backbone and the deeper ones are probes.
1. Loss and logged train/pls/L{k} equal plain separate heads (stop-gradient changes no value).
2. Every gradient equals a dense reference (a Python loop over the layers) whose readouts k >= n read a stop-gradient copy
   of layer k's output.
3. Every aux head and aux norm gets plain separate heads' gradient (each head sees the same input either way); the
   backbone gradient lies strictly between plain separate heads and the fully detached probes.
4. Bad combinations are rejected (no separate heads, with pls_detach_backbone, n out of range).
Run on a vis node: PYTHONPATH=<worktree libs> nice -n 19 <venv>/bin/python scripts/della/pls_upto_cpu_test.py"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4"
os.environ.setdefault("LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS", "0")

from functools import partial  # noqa: E402

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jrandom  # noqa: E402
import numpy as np  # noqa: E402

import haliax as hax  # noqa: E402
from haliax import Axis  # noqa: E402
from haliax.partitioning import ResourceAxis  # noqa: E402
from levanter.data.dataset import ListAsyncDataset  # noqa: E402
from levanter.data.text.examples import GrugLmExample  # noqa: E402
from levanter.eval import TaggedEvaluator, _default_lm_eval_loss_fn  # noqa: E402
from levanter.layers.attention import AttentionBackend, AttentionMask  # noqa: E402
from levanter.models.lm_model import LmExample  # noqa: E402
from levanter.models.qwen import Qwen3Config, Qwen3LMHeadModel  # noqa: E402
from levanter.optim.muonh import MuonHConfig  # noqa: E402
from levanter.testing.helpers import use_test_mesh  # noqa: E402
from levanter.trainer import _TRACED_TRAIN_STEP, TrainerConfig  # noqa: E402
from levanter.utils.mesh import MeshConfig  # noqa: E402

import experiments.references.per_layer_qwen3 as pls  # noqa: E402
from experiments.references.per_layer_qwen3 import PerLayerQwen3Config, PerLayerQwen3LMHeadModel, ReadoutTaggedEvaluator  # noqa: E402

jax.config.update("jax_threefry_partitionable", True)
assert jax.device_count() == 4
rng = np.random.default_rng(0)

B, T, V, L, S = 8, 32, 500, 4, 4
Batch, Pos, Vocab = Axis("batch", B), Axis("position", T), Axis("vocab", V)
EOS = 1
tok = rng.integers(2, V, size=(B, T))
tok[:, 10] = EOS
tok[3, 20] = EOS
eos_mask = np.roll(tok, 1, axis=1) == EOS
eos_mask[:, 0] = False
seg = np.cumsum(eos_mask, axis=1).astype(np.int32)
lw = (np.arange(T) < T - 1).astype(np.float32)[None].repeat(B, 0)
lw[2, :5] = 0.0
lw[5, 7:9] = 0.0
tokens = hax.named(jnp.asarray(tok, jnp.int32), (Batch, Pos))
ex = LmExample(
    tokens=tokens,
    loss_weight=hax.named(jnp.asarray(lw), (Batch, Pos)),
    attn_mask=AttentionMask.causal().with_segment_ids(hax.named(jnp.asarray(seg), (Batch, Pos))),
)
common = dict(max_seq_len=T, hidden_dim=48, intermediate_dim=64, num_layers=L, num_heads=2, num_kv_heads=2, hybrid_norm=True, attn_backend=AttentionBackend.VANILLA)
key0 = jrandom.PRNGKey(0)
base = Qwen3LMHeadModel.init(Vocab, Qwen3Config(**common), key=key0)


def close(a, b, what, rtol=2e-4, atol=1e-6):
    la = jax.tree_util.tree_leaves_with_path(eqx.filter(a, eqx.is_array))
    lb = jax.tree.leaves(eqx.filter(b, eqx.is_array))
    assert len(la) == len(lb), (what, len(la), len(lb))
    for (p, u), v in zip(la, lb):
        np.testing.assert_allclose(np.asarray(u, np.float32), np.asarray(v, np.float32), rtol=rtol, atol=atol, err_msg=f"{what} {jax.tree_util.keystr(p)}")


def scalar(x):
    return float(x.array if isinstance(x, hax.NamedArray) else x)


# independent reference: a Python loop over the layers, dense logits
y = np.roll(tok, -1, axis=1)
w_next = lw * (np.arange(T) < T - 1)[None]


def ref_raw(m):
    """Residual stream after every layer (pre-norm), by applying each layer on its own (no scan)."""
    tr = m.transformer
    Block = tr.layers.Block
    x = m.embeddings.embed(tokens)
    out = []
    for k in range(L):
        layer = hax.tree_util.tree_map(lambda a, k=k: a[Block.name, k], tr.layers.stacked)
        x = layer(x, mask=ex.attn_mask, key=None, pos_ids=None)
        out.append(x)
    return out


def ref_hidden(m):
    """Normed (final norm) residual stream after every layer."""
    return [m.transformer.norm(x).array for x in ref_raw(m)]


def ref_ce_per_pos(h, W):
    logits = jnp.einsum("btd,dv->btv", h.astype(jnp.float32), W.astype(jnp.float32))
    return jax.nn.logsumexp(logits, axis=-1) - jnp.take_along_axis(logits, jnp.asarray(y)[..., None], axis=-1)[..., 0]


def ref_losses(m, weights_mask=None):
    """Per-layer weighted-mean CE (weights_mask selects positions, e.g. the monitor's stride)."""
    W = m.get_lm_head().rearrange((m.Embed, Vocab)).array
    wt = jnp.asarray(w_next if weights_mask is None else w_next * weights_mask)
    return [jnp.sum(ref_ce_per_pos(h, W) * wt) / jnp.sum(wt) for h in ref_hidden(m)]


tc = TrainerConfig(mesh=MeshConfig(axes={"data": -1, "replica": 1, "model": 1}, compute_mapping={
    "token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
    "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA)}))
key = jrandom.PRNGKey(7)


def train_step(m_, example, k):
    def lf(mm):
        out = mm.compute_next_token_loss(example, key=k, logsumexp_weight=0.0)
        loss, stats = out if isinstance(out, tuple) else (out, {})
        return (loss.array if isinstance(loss, hax.NamedArray) else loss), stats

    (loss, stats), g = eqx.filter_value_and_grad(lf, has_aux=True)(m_)
    return loss, {k_: v.value() for k_, v in stats.items()}, g



def ref_upto(mm, n):
    out = []
    for k, x in enumerate(ref_raw(mm)):
        if k < L - 1:
            xk = x if k < n else jax.lax.stop_gradient(x)
            h, W = mm.aux_norms[k](xk).array, mm.aux_lm_heads[k].weight.rearrange((mm.Embed, Vocab)).array
        else:
            h, W = mm.transformer.norm(x).array, mm.get_lm_head().rearrange((mm.Embed, Vocab)).array
        wt = jnp.asarray(w_next)
        out.append(jnp.sum(ref_ce_per_pos(h, W) * wt) / jnp.sum(wt))
    return out[L - 1] + sum(out[: L - 1]), out


def gmax(a, b):
    return max(float(jnp.max(jnp.abs(u - v))) for u, v in zip(jax.tree.leaves(eqx.filter(a, eqx.is_array)), jax.tree.leaves(eqx.filter(b, eqx.is_array))))


def mk(**kw):
    return PerLayerQwen3LMHeadModel.init(Vocab, PerLayerQwen3Config(**common, pls_weight=1.0, pls_monitor_stride=S, pls_separate_heads=True, **kw), key=key0)


with tc.use_device_mesh(), hax.axis_mapping(tc.compute_axis_mapping):
    es = hax.shard(ex, tc.compute_axis_mapping)
    step_fn = eqx.filter_jit(train_step)
    loss_s, stats_s, g_s = step_fn(hax.shard(mk(), tc.parameter_axis_mapping), es, key)
    loss_b, stats_b, g_b = step_fn(hax.shard(mk(pls_detach_backbone=True), tc.parameter_axis_mapping), es, key)
    trunk = lambda g: eqx.tree_at(lambda m: (m.aux_norms, m.aux_lm_heads), g, replace=(None, None))
    for n in (1, 2):
        m_n = mk(pls_sep_upto=n)
        loss_n, stats_n, g_n = step_fn(hax.shard(m_n, tc.parameter_axis_mapping), es, key)
        np.testing.assert_allclose(float(loss_n), float(loss_s), rtol=1e-6)
        for k in range(L):
            np.testing.assert_allclose(float(stats_n[f"pls/L{k}"]), float(stats_s[f"pls/L{k}"]), rtol=1e-6)
        (rl, _), rg = eqx.filter_value_and_grad(lambda mm: ref_upto(mm, n), has_aux=True)(m_n)
        np.testing.assert_allclose(float(loss_n), float(rl), rtol=1e-5)
        close(g_n, rg, f"upto {n} grad vs reference")
        close((g_n.aux_norms, g_n.aux_lm_heads), (g_s.aux_norms, g_s.aux_lm_heads), f"upto {n} aux head grads vs separate heads", rtol=1e-5)
        d_s, d_b = gmax(trunk(g_n), trunk(g_s)), gmax(trunk(g_n), trunk(g_b))
        assert d_s > 1e-7 and d_b > 1e-7, (n, d_s, d_b)
        print(f"upto {n}: loss/stats == separate heads; all grads == stop-gradient reference; aux head grads == separate heads; "
              f"backbone differs from separate heads by {d_s:.2e} and from probes by {d_b:.2e}")
    for bad in (dict(pls_separate_heads=False, pls_sep_upto=1), dict(pls_separate_heads=True, pls_sep_upto=1, pls_detach_backbone=True),
                dict(pls_separate_heads=True, pls_sep_upto=0 + L - 1), dict(pls_separate_heads=True, pls_sep_upto=-1)):
        try:
            PerLayerQwen3Config(**common, pls_weight=1.0, **bad)
            raise AssertionError(f"{bad} must be rejected")
        except ValueError:
            pass
    print("bad combinations rejected\nALL PASS")
