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
from haliax.nn.scan import Stacked  # noqa: E402


def per_layer(layers):
    """The layers of a Stacked or BlockSeq stack, as a list."""
    if isinstance(layers, Stacked):
        return [hax.tree_util.tree_map(lambda a, k=k: a["layer", k], layers.stacked) for k in range(L)]
    raise TypeError(type(layers))
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
    st = base.transformer.layers.stacked  # both stacks are Stacked: copy the stacked weights across
    blk = m.transformer.layers.stacked
    rep = dict(self_attn=st.self_attn, mlp=st.mlp, ln_1=st.input_layernorm, ln_2=st.post_attention_layernorm)
    if blk.post_1 is not None:
        rep.update(post_1=st.post_attn_layernorm, post_2=st.post_mlp_layernorm)
    tr = dataclasses.replace(m.transformer, layers=dataclasses.replace(m.transformer.layers, stacked=dataclasses.replace(blk, **rep)), norm=base.transformer.norm)
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
    layers = [(b.self_attn, b.mlp) for b in per_layer(tr.layers)]
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
    moda = model("moda", False)
    ps, ms = pre.transformer.layers.stacked, moda.transformer.layers.stacked
    moda = dataclasses.replace(moda, embeddings=pre.embeddings, transformer=dataclasses.replace(moda.transformer, layers=dataclasses.replace(moda.transformer.layers, stacked=dataclasses.replace(ms, self_attn=ps.self_attn, mlp=ps.mlp, ln_1=ps.ln_1, ln_2=ps.ln_2)), norm=pre.transformer.norm))
    o_pre, o_moda = outs(pre), outs(moda)
    leaves_close(o_moda["layer", 0].array, o_pre["layer", 0].array, "moda layer 0", rtol=1e-4, atol=1e-5)
    assert float(jnp.abs(o_moda["layer", L - 1].array - o_pre["layer", L - 1].array).max()) > 1e-3  # depth keys change later layers
    print("3. moda layer 0 == preln layer 0 (chunked attention with the document mask); later layers differ")

    # 4. hc at init == 4 x preln (same weights)
    hc = model("hc", False)
    ps, hs = pre.transformer.layers.stacked, hc.transformer.layers.stacked
    hc = dataclasses.replace(hc, transformer=dataclasses.replace(hc.transformer, layers=dataclasses.replace(hc.transformer.layers, stacked=dataclasses.replace(hs, self_attn=ps.self_attn, mlp=ps.mlp, ln_1=ps.ln_1, ln_2=ps.ln_2))), embeddings=pre.embeddings)
    leaves_close(outs(hc).array / 4.0, o_pre.array, "hc / 4 vs preln", rtol=1e-4, atol=1e-5)
    print("4. hc at init: sum of the 4 streams == 4 x preln, every layer")

    # 5. every design trains and evaluates; new parameters learn; probes leave the backbone gradient alone
    for arch in ARCHS:
        m = model(arch)
        loss, g = step(m, es, key)
        ev = scalar(m.compute_next_token_loss(es, key=None))
        assert np.isfinite(float(loss)) and np.isfinite(ev), arch
        assert all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(eqx.filter(g, eqx.is_array))), arch
        b0 = per_layer(g.transformer.layers)[0]
        new = [b0.post_1, b0.hyper, b0.query, g.transformer.final_query]
        new = [x for x in new if x is not None]
        for x in new:
            assert sum(float(jnp.abs(a).sum()) for a in jax.tree.leaves(eqx.filter(x, eqx.is_array))) > 0, (arch, x)
        mp = model(arch, pls_detach_backbone=True)
        _, gp = step(mp, es, key)
        ge = eqx.filter_jit(eqx.filter_grad(lambda mm: scalar(mm.compute_next_token_loss(es, key=None)) if False else mm.compute_next_token_loss(es, key=None).array))(mp)
        leaves_close([(b.self_attn, b.mlp) for b in per_layer(gp.transformer.layers)] + [gp.embeddings], [(b.self_attn, b.mlp) for b in per_layer(ge.transformer.layers)] + [ge.embeddings], f"{arch} probes backbone grad", rtol=1e-4, atol=1e-7)
        print(f"5. {arch}: train loss {float(loss):.4f}, eval {ev:.4f}, finite grads; new parameters get gradient ({len(new)} groups)")

    # 6. hc / mhc mixing with random parameters == an independent einsum reference of the update rule
    import haliax.nn as hnn  # noqa: E402
    from experiments.references.depth_arch_qwen3 import HyperParams, _sinkhorn  # noqa: E402
    rng6 = np.random.default_rng(6)
    for arch in ("hc", "mhc"):
        mdl = model(arch, False)
        lay0 = per_layer(mdl.transformer.layers)[1]  # layer index 1
        def rnd(a, sc):
            return hax.named(jnp.asarray(rng6.normal(size=a.array.shape) * sc, jnp.float32), a.axes)
        hyp = tuple(HyperParams(rnd(h.read, .3), rnd(h.res, .3), rnd(h.write, .3), rnd(h.w_read, .5), rnd(h.w_res, .5), rnd(h.w_write, .5), hax.named(jnp.asarray([0.7, 0.4]), h.scale.axes)) for h in lay0.hyper)
        lay = dataclasses.replace(lay0, hyper=hyp)
        m = 4
        H0 = hax.named(jnp.asarray(rng6.normal(size=(m, B, T, 48)), jnp.float32), (Axis("stream", m), Batch, Pos, Axis("embed", 48)))
        got = lay.hyper_step(H0, ex.attn_mask, key=None, pos_ids=None, layer=1).rearrange(H0.axes).array
        # reference: plain einsum, sublayer index 2*1 + j
        Hr = H0.array
        for j in range(2):
            p = hyp[j]
            Hn = Hr / np.sqrt(np.mean(Hr ** 2, axis=-1, keepdims=True) + 1e-6)
            dr, dres, dw = np.einsum("sbtd,d->bts", Hn, p.w_read.array), np.einsum("sbtd,do->btso", Hn, p.w_res.array), np.einsum("sbtd,d->bts", Hn, p.w_write.array)
            hot = (np.arange(m) == (2 + j) % m).astype(np.float32)
            if arch == "hc":
                r = hot + p.read.array + 0.7 * np.tanh(dr); A = np.eye(m) + p.res.array + 0.7 * np.tanh(dres); w = 1 + p.write.array + 0.4 * np.tanh(dw)
            else:
                sig = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731
                r = sig(4 * hot - 2 + p.read.array + 0.7 * dr); w = 2 * sig(0 + p.write.array + 0.4 * dw)
                lg = 4 * np.eye(m) - 2 + p.res.array + 0.7 * dres
                for _ in range(20):
                    lg = lg - np.log(np.exp(lg).sum(-1, keepdims=True)); lg = lg - np.log(np.exp(lg).sum(-2, keepdims=True))
                A = np.exp(lg)
            x = np.einsum("bts,sbtd->btd", r, Hr)
            f = lay.branch(j, hax.named(jnp.asarray(x), (Batch, Pos, Axis("embed", 48))), ex.attn_mask, key=None, pos_ids=None).array
            Hr = np.einsum("btso,sbtd->obtd", A, Hr) + np.einsum("bto,btd->obtd", w, f)
        np.testing.assert_allclose(np.asarray(got), Hr, rtol=2e-4, atol=2e-4, err_msg=arch)
        print(f"6. {arch}: hyper_step with random parameters == independent einsum reference (layer 1, both sublayers)")

    # 7. Sinkhorn: doubly stochastic, finite on extreme logits (a row 200 below the rest), finite gradient
    S4, So4 = Axis("stream", 4), Axis("stream_out", 4)
    lg = hax.named(jnp.asarray(rng6.normal(size=(4, 4)) * 3).at[2].add(-200.0), (S4, So4))
    sk = _sinkhorn(lg)
    assert np.all(np.isfinite(sk.array)) and np.allclose(sk.array.sum(0), 1, atol=1e-4) and np.allclose(sk.array.sum(1), 1, atol=1e-4)
    gsk = jax.grad(lambda a: jnp.sum(_sinkhorn(hax.named(a, (S4, So4))).array * jnp.arange(16.0).reshape(4, 4)))(lg.array)
    assert np.all(np.isfinite(gsk))
    print("7. Sinkhorn (log domain): rows and columns sum to 1, finite value and gradient with a row 200 below the rest")

    # 8. optimizer labels: the designs' new parameters -> adam_new; the baseline's parameters keep their labels; every
    # layer weight is stacked with the layer axis first (MuonH's per-layer norm projection)
    from experiments.references.depth_arch_qwen3 import DepthArchMuonHConfig  # noqa: E402
    from levanter.optim.muonh import MuonHConfig  # noqa: E402
    for arch in ("hc", "attnres"):
        mdl = model(arch)
        lab = DepthArchMuonHConfig().create_mask(mdl)
        base_lab = MuonHConfig().create_mask(mdl)
        st, lb, bb = mdl.transformer.layers.stacked, lab.transformer.layers.stacked, base_lab.transformer.layers.stacked
        new_leaves = jax.tree.leaves(lb.hyper if arch == "hc" else lb.query) + ([lab.transformer.final_query] if arch == "attnres" else [])
        assert new_leaves and all(x == "adam_new" for x in new_leaves), new_leaves
        same = [(a, b) for a, b in zip(jax.tree.leaves((lab.embeddings, lab.lm_head, lab.aux_lm_heads, lab.aux_norms, lb.self_attn, lb.mlp, lb.ln_1, lb.ln_2)), jax.tree.leaves((base_lab.embeddings, base_lab.lm_head, base_lab.aux_lm_heads, base_lab.aux_norms, bb.self_attn, bb.mlp, bb.ln_1, bb.ln_2)))]
        assert all(a == b for a, b in same) and {a for a, _ in same} == {"adam", "adamh", "muonh"}, set(a for a, _ in same)
        assert st.self_attn.q_proj.weight.axes[0].name == "layer" and st.self_attn.o_proj.weight.axes[0].name == "layer"
        print(f"8. {arch}: new parameters -> adam_new (eps 1e-8); baseline parameters keep muonh / adamh / adam; layer weights stacked on the layer axis")

# 9. production regime: 48 layers, bf16 compute, the launcher's 130m optimizer (eps 1e-20, warmup 0), 8 steps
import jmp  # noqa: E402
from experiments.references.depth_arch_qwen3 import DepthArchMuonHConfig  # noqa: E402
from levanter.optim.muonh import MuonHConfig  # noqa: E402
pol = jmp.get_policy("p=f32,c=bfloat16")
L48, T9 = 48, 64
Pos9 = Axis("position", T9)
tok9 = np.random.default_rng(9).integers(2, V, size=(2, T9)); tok9[:, 20] = 1
seg9 = np.cumsum(np.roll(tok9, 1, axis=1) == 1, axis=1).astype(np.int32); seg9[:, 0] = 0
ex9 = LmExample(tokens=hax.named(jnp.asarray(tok9, jnp.int32), (Axis("batch", 2), Pos9)), loss_weight=hax.named(jnp.ones((2, T9)), (Axis("batch", 2), Pos9)),
                attn_mask=AttentionMask.causal().with_segment_ids(hax.named(jnp.asarray(seg9), (Axis("batch", 2), Pos9))))
for arch in ("hc", "mhc", "attnres_block", "moda"):
    cfg9 = PerLayerQwen3Config(max_seq_len=T9, hidden_dim=128, intermediate_dim=448, num_layers=L48, num_heads=2, num_kv_heads=2, hybrid_norm=True, attn_backend=AttentionBackend.VANILLA,
                               pls_weight=1.0, pls_separate_heads=True, depth_arch=arch, scan_layers=False, moda_chunk=32)
    m9 = PerLayerQwen3LMHeadModel.init(Vocab, cfg9, key=key0)
    Opt = DepthArchMuonHConfig if arch != "moda" else MuonHConfig
    opt = Opt(learning_rate=0.02, adam_lr=0.008, beta1=0.9, beta2=0.98, epsilon=1e-20, momentum=0.95, nesterov=True, backend_steps=5, muon_epsilon=1e-5, max_grad_norm=1.0,
              weight_decay=0.1, lr_schedule="linear", decay=0.8, warmup=0, min_lr_ratio=0.0, coefficient_type="simple").build(100)
    st9 = opt.init(eqx.filter(m9, eqx.is_inexact_array))

    def lf9(mm):
        out = pol.cast_to_compute(mm).compute_next_token_loss(ex9, key=jrandom.PRNGKey(3))
        out = out[0] if isinstance(out, tuple) else out
        return out.array if isinstance(out, hax.NamedArray) else out

    vg = eqx.filter_jit(eqx.filter_value_and_grad(lf9))
    worst = 0.0
    for t in range(8):
        loss9, g9 = vg(m9)
        assert np.isfinite(float(loss9)), (arch, t)
        upd, st9 = opt.update(g9, st9, eqx.filter(m9, eqx.is_inexact_array))
        new = upd.transformer.layers.stacked.hyper if arch in ("hc", "mhc") else (upd.transformer.layers.stacked.query if arch.startswith("attnres") else None)
        if new is not None:
            worst = max(worst, max(float(jnp.max(jnp.abs(x.array if isinstance(x, hax.NamedArray) else x))) for x in jax.tree.leaves(new, is_leaf=lambda z: isinstance(z, hax.NamedArray))))
        m9 = eqx.apply_updates(m9, upd)
    assert worst <= 2 * 0.008, (arch, worst)
    print(f"9. {arch}: 48 layers, bf16, launcher optimizer (eps 1e-20, warmup 0): 8 steps finite, loss {float(loss9):.3f}; largest new-parameter step {worst:.4f} <= 2 x adam_lr")
print("all depth_arch tests passed")
