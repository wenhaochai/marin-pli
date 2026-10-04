"""CPU tests for experiments.references.depth_arch_qwen3 (DepthBench residual designs under per-layer heads).

Run from the worktree on a vis node (fake 4-device mesh), after `source scripts/della/pls_env.sh`:
    nice -n 19 $PY scripts/della/depth_arch_cpu_test.py
1. sandwich (the loop) with the baseline's weights == the baseline's Stacked transformer (hybrid_norm): every layer's
   output, the separate-heads training loss and every shared gradient, and the eval loss.
2. preln with the weights of a hybrid_norm=False baseline == that baseline, the same way.
3. moda, layer 0 (no earlier layers yet) == preln's layer 0 with the same weights: the hand-written chunked attention,
   with the packed-document mask, equals the library's.
4. hc at init (dynamic weights 0) == m x preln with the same weights, every layer's read-out.
5. Every design: a sharded training step (separate heads, and probes) and eval are finite; the new parameters
   (pseudo-queries, hyper-connection weights, extra norms) get non-zero gradients; probes leave the backbone gradient
   equal to the baseline training step of that design.
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

import haliax as hax  # noqa: E402
from haliax import Axis  # noqa: E402
from haliax.partitioning import ResourceAxis  # noqa: E402
from levanter.layers.attention import AttentionBackend, AttentionMask  # noqa: E402
from levanter.models.lm_model import LmExample  # noqa: E402
from levanter.trainer import TrainerConfig  # noqa: E402
from levanter.utils.mesh import MeshConfig  # noqa: E402

from experiments.references.depth_arch_qwen3 import ARCHS, DepthArchTransformer  # noqa: E402
from experiments.references.per_layer_qwen3 import PerLayerQwen3Config, PerLayerQwen3LMHeadModel  # noqa: E402

jax.config.update("jax_threefry_partitionable", True)
rng = np.random.default_rng(0)
B, T, V, L = 8, 32, 500, 4
Batch, Pos, Vocab = Axis("batch", B), Axis("position", T), Axis("vocab", V)
tok = rng.integers(2, V, size=(B, T))
tok[:, 10] = 1
eos = np.roll(tok, 1, axis=1) == 1
eos[:, 0] = False
seg = np.cumsum(eos, axis=1).astype(np.int32)
ex = LmExample(
    tokens=hax.named(jnp.asarray(tok, jnp.int32), (Batch, Pos)),
    loss_weight=hax.named(jnp.asarray((np.arange(T) < T - 1).astype(np.float32)[None].repeat(B, 0)), (Batch, Pos)),
    attn_mask=AttentionMask.causal().with_segment_ids(hax.named(jnp.asarray(seg), (Batch, Pos))),
)
common = dict(max_seq_len=T, hidden_dim=48, intermediate_dim=64, num_layers=L, num_heads=2, num_kv_heads=2, attn_backend=AttentionBackend.VANILLA, pls_weight=1.0, pls_separate_heads=True, moda_chunk=12, attnres_blocks=2)
key0 = jrandom.PRNGKey(0)


def cfg(arch, hybrid=True, **kw):
    return PerLayerQwen3Config(**common, hybrid_norm=hybrid, depth_arch=arch, **({"scan_layers": False} if arch != "baseline" else {}), **kw)


def model(arch, hybrid=True, **kw):
    return PerLayerQwen3LMHeadModel.init(Vocab, cfg(arch, hybrid, **kw), key=key0)


def with_baseline_weights(m, base):
    """m (a loop design) with the baseline's attention, MLP and norms in every layer, and its embeddings and heads."""
    st = base.transformer.layers.stacked
    blocks = []
    for k, blk in enumerate(m.transformer.layers.blocks):
        lay = hax.tree_util.tree_map(lambda a, k=k: a["layer", k], st)
        rep = dict(self_attn=lay.self_attn, mlp=lay.mlp, ln_1=lay.input_layernorm, ln_2=lay.post_attention_layernorm)
        if blk.post_1 is not None:
            rep.update(post_1=lay.post_attn_layernorm, post_2=lay.post_mlp_layernorm)
        blocks.append(dataclasses.replace(blk, **rep))
    tr = dataclasses.replace(m.transformer, layers=dataclasses.replace(m.transformer.layers, blocks=blocks), norm=base.transformer.norm)
    return dataclasses.replace(m, transformer=tr, embeddings=base.embeddings, lm_head=base.lm_head, aux_norms=base.aux_norms, aux_lm_heads=base.aux_lm_heads)


def leaves_close(a, b, what, rtol=2e-5, atol=1e-6):
    for (p, u), v in zip(jax.tree_util.tree_leaves_with_path(a), jax.tree.leaves(b)):
        np.testing.assert_allclose(np.asarray(u, np.float32), np.asarray(v, np.float32), rtol=rtol, atol=atol, err_msg=f"{what} {jax.tree_util.keystr(p)}")


def scalar(x):
    return float(x.array if isinstance(x, hax.NamedArray) else x)


tc = TrainerConfig(mesh=MeshConfig(axes={"data": -1, "replica": 1, "model": 1}, compute_mapping={
    "token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
    "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA)}))
key = jrandom.PRNGKey(7)


def train_step(m_, example, k):
    def lf(mm):
        out = mm.compute_next_token_loss(example, key=k, logsumexp_weight=0.0)
        loss = out[0] if isinstance(out, tuple) else out
        return loss.array if isinstance(loss, hax.NamedArray) else loss

    return eqx.filter_value_and_grad(lf)(m_)


def shared_grads(g):
    """Gradients of the parts both models have: embeddings, heads, final norm, and every layer's attention/MLP."""
    tr = g.transformer
    if isinstance(tr, DepthArchTransformer):
        layers = [(b.self_attn, b.mlp) for b in tr.layers.blocks]
    else:
        layers = [(hax.tree_util.tree_map(lambda a, k=k: a["layer", k], tr.layers.stacked).self_attn, hax.tree_util.tree_map(lambda a, k=k: a["layer", k], tr.layers.stacked).mlp) for k in range(L)]
    return (g.embeddings, g.lm_head, g.aux_lm_heads, g.aux_norms, tr.norm, layers)


with tc.use_device_mesh(), hax.axis_mapping(tc.compute_axis_mapping):
    es = hax.shard(ex, tc.compute_axis_mapping)
    step = eqx.filter_jit(train_step)
    outs = eqx.filter_jit(lambda m: m.layer_outputs(es.tokens, es.attn_mask))

    # 1, 2. the loop reproduces the baseline (sandwich with hybrid_norm, preln without)
    for arch, hybrid in (("sandwich", True), ("preln", False)):
        base = model("baseline", hybrid)
        m = with_baseline_weights(model(arch, hybrid), base)
        leaves_close(outs(m).array, outs(base).array, f"{arch} layer outputs")
        lb, gb = step(base, es, key)
        lm, gm = step(m, es, key)
        np.testing.assert_allclose(float(lm), float(lb), rtol=1e-6)
        leaves_close(shared_grads(gm), shared_grads(gb), f"{arch} grads", rtol=1e-4, atol=1e-7)
        np.testing.assert_allclose(scalar(m.compute_next_token_loss(es, key=None)), scalar(base.compute_next_token_loss(es, key=None)), rtol=1e-6)
        print(f"{1 if arch == 'sandwich' else 2}. {arch} loop == baseline (hybrid_norm={hybrid}): outputs, loss {float(lm):.6f}, grads, eval")

    # 3. moda layer 0 == preln layer 0
    pre = model("preln", False)
    moda = dataclasses.replace(model("moda", False), transformer=dataclasses.replace(model("moda", False).transformer, layers=pre.transformer.layers, norm=pre.transformer.norm))
    moda = dataclasses.replace(moda, transformer=dataclasses.replace(moda.transformer, layers=dataclasses.replace(pre.transformer.layers, blocks=[dataclasses.replace(b, arch="moda") for b in pre.transformer.layers.blocks])))
    o_pre, o_moda = outs(pre), outs(moda)
    leaves_close(o_moda["layer", 0].array, o_pre["layer", 0].array, "moda layer 0", rtol=1e-4, atol=1e-5)
    assert float(jnp.abs(o_moda["layer", L - 1].array - o_pre["layer", L - 1].array).max()) > 1e-3  # depth keys change later layers
    print("3. moda layer 0 == preln layer 0 (chunked attention with the document mask); later layers differ")

    # 4. hc at init == 4 x preln (same weights)
    hc = model("hc", False)
    hc = dataclasses.replace(hc, transformer=dataclasses.replace(hc.transformer, layers=dataclasses.replace(hc.transformer.layers, blocks=[dataclasses.replace(h, self_attn=p.self_attn, mlp=p.mlp, ln_1=p.ln_1, ln_2=p.ln_2) for h, p in zip(hc.transformer.layers.blocks, pre.transformer.layers.blocks)])), embeddings=pre.embeddings)
    leaves_close(outs(hc).array / 4.0, o_pre.array, "hc / 4 vs preln", rtol=1e-4, atol=1e-5)
    print("4. hc at init: sum of the 4 streams == 4 x preln, every layer")

    # 5. every design trains and evaluates; new parameters learn; probes leave the backbone gradient alone
    for arch in ARCHS:
        m = model(arch)
        loss, g = step(m, es, key)
        ev = scalar(m.compute_next_token_loss(es, key=None))
        assert np.isfinite(float(loss)) and np.isfinite(ev), arch
        assert all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(eqx.filter(g, eqx.is_array))), arch
        b0 = g.transformer.layers.blocks[0]
        new = [b0.post_1, b0.hyper, b0.query, g.transformer.final_query]
        new = [x for x in new if x is not None]
        for x in new:
            assert sum(float(jnp.abs(a).sum()) for a in jax.tree.leaves(eqx.filter(x, eqx.is_array))) > 0, (arch, x)
        mp = model(arch, pls_detach_backbone=True)
        _, gp = step(mp, es, key)
        ge = eqx.filter_jit(eqx.filter_grad(lambda mm: scalar(mm.compute_next_token_loss(es, key=None)) if False else mm.compute_next_token_loss(es, key=None).array))(mp)
        leaves_close([(b.self_attn, b.mlp) for b in gp.transformer.layers.blocks] + [gp.embeddings], [(b.self_attn, b.mlp) for b in ge.transformer.layers.blocks] + [ge.embeddings], f"{arch} probes backbone grad", rtol=1e-4, atol=1e-7)
        print(f"5. {arch}: train loss {float(loss):.4f}, eval {ev:.4f}, finite grads; new parameters get gradient ({len(new)} groups)")
print("all depth_arch tests passed")
