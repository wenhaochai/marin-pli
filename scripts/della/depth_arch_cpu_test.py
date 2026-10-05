"""CPU tests for experiments.references.depth_arch_qwen3 (DepthBench residual designs under per-layer heads).

Run from the worktree on a vis node (fake 4-device mesh), with this worktree first on PYTHONPATH, in two processes (one
process compiling all of them runs out of executable memory maps):
    DEPTH_TEST=main nice -n 19 $PY -u scripts/della/depth_arch_cpu_test.py     # 1-8, 10, 11
    DEPTH_TEST=prod nice -n 19 $PY -u scripts/della/depth_arch_cpu_test.py     # 9
1. sandwich (the loop) with the baseline's weights == the baseline's Stacked transformer (hybrid_norm): every layer's
   output, the separate-heads training loss and every shared gradient, and the eval loss.
2. preln with the weights of a hybrid_norm=False baseline == that baseline, the same way.
3. moda, layer 0 (no earlier slot) == preln's layer 0 with the same weights; later layers differ.
4. Init: hc at init == m x preln with the same weights; hc and mhc scale o_proj and down_proj by 1/sqrt(m); deepnorm's
   post norms centre (LayerNorm without bias), and its v, o and MLP weights are scaled by (8L)^(-1/4).
5. Every design: a sharded training step (separate heads, and probes) and eval are finite; the new parameters get
   non-zero gradients; probes leave the backbone gradient equal to the baseline training step of that design.
6. hc and mhc with random parameters == independent einsum references (hc: dynamic HC; mhc: Liger Kernel 0.8.0's
   formula: one projection of the RMS-normalised concatenated streams, sigmoid / 2 sigmoid / Sinkhorn).
7. The Liger Sinkhorn == a numpy transcription; columns sum to 1; finite on extreme logits, with a finite gradient.
8. Optimizer labels: the new parameters -> adam_new; the baseline's keep their labels; moda's kv_proj -> muonh.
9. Production regime: 48 layers, bf16, the launcher's 130m optimizer, 8 steps finite (hc, mhc, attnres,
   attnres_block, moda; the two-level scans at 6 x 8).
10. attnres and moda: the two-level scan == the Python loop, every layer's output and every gradient (random queries,
    key gains and depth projections); moda does not depend on the chunk size.
11. Independent references: keel (first layer special), deepnorm (centring LayerNorm), moda (one softmax over sequence
    and depth, two depth entries per layer, depth keys before RoPE, GQA), attnres and attnres_block (DepthBench's
    AttnResTransformerBlock.forward, with key gains).
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4"
os.environ.setdefault("LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS", "0")

import dataclasses  # noqa: E402
import math  # noqa: E402

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

from experiments.references.depth_arch_qwen3 import ARCHS, DepthArchTransformer, MhcParams, _mhc_constants, _sinkhorn_liger  # noqa: E402
from experiments.references.per_layer_qwen3 import PerLayerQwen3Config, PerLayerQwen3LMHeadModel  # noqa: E402
from haliax.nn.scan import Stacked  # noqa: E402

jax.config.update("jax_threefry_partitionable", True)
PART = os.environ.get("DEPTH_TEST", "main")
assert PART in ("main", "prod"), PART
rng = np.random.default_rng(0)
B, T, V, L = 8, 32, 500, 4
Batch, Pos, Vocab = Axis("batch", B), Axis("position", T), Axis("vocab", V)
KPos = Pos.alias("key_position")
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


def per_layer(layers, n=None):
    """The layers of a Stacked stack, as a list."""
    if isinstance(layers, Stacked):
        return [hax.tree_util.tree_map(lambda a, k=k: a["layer", k], layers.stacked) for k in range(n or L)]
    raise TypeError(type(layers))


def cfg(arch, hybrid=True, **kw):
    return PerLayerQwen3Config(**{**common, **kw}, hybrid_norm=hybrid, depth_arch=arch, **({"scan_layers": False} if arch != "baseline" else {}))


def model(arch, hybrid=True, **kw):
    return PerLayerQwen3LMHeadModel.init(Vocab, cfg(arch, hybrid, **kw), key=key0)


def with_baseline_weights(m, base):
    """m (a loop design) with the baseline's attention, MLP and norms in every layer, and its embeddings and heads."""
    st = base.transformer.layers.stacked
    blk = m.transformer.layers.stacked
    rep = dict(self_attn=st.self_attn, mlp=st.mlp, ln_1=st.input_layernorm, ln_2=st.post_attention_layernorm)
    if blk.post_1 is not None:
        rep.update(post_1=st.post_attn_layernorm, post_2=st.post_mlp_layernorm)
    tr = dataclasses.replace(m.transformer, layers=dataclasses.replace(m.transformer.layers, stacked=dataclasses.replace(blk, **rep)), norm=base.transformer.norm)
    return dataclasses.replace(m, transformer=tr, embeddings=base.embeddings, lm_head=base.lm_head, aux_norms=base.aux_norms, aux_lm_heads=base.aux_lm_heads)


def leaves_close(a, b, what, rtol=2e-5, atol=1e-6):
    la, lb = jax.tree.leaves(a), jax.tree.leaves(b)
    assert len(la) == len(lb), (what, len(la), len(lb))
    for (p, u), v in zip(jax.tree_util.tree_leaves_with_path(a), lb):
        np.testing.assert_allclose(np.asarray(u, np.float32), np.asarray(v, np.float32), rtol=rtol, atol=atol, err_msg=f"{what} {jax.tree_util.keystr(p)}")


def rel_close(a, b, what, tol):
    """Every leaf of a equals b's to tol in relative L2 norm (float32 sums in a different order differ elementwise); returns the worst."""
    worst = 0.0
    for (p, u), v in zip(jax.tree_util.tree_leaves_with_path(a), jax.tree.leaves(b)):
        u, v = np.asarray(u, np.float64), np.asarray(v, np.float64)
        r = np.linalg.norm(u - v) / max(np.linalg.norm(v), 1e-30)
        assert r <= tol, f"{what} {jax.tree_util.keystr(p)}: relative L2 error {r:.2e}"
        worst = max(worst, r)
    return worst


def scalar(x):
    return float(x.array if isinstance(x, hax.NamedArray) else x)


def arr(x, axes):
    return np.asarray(x.rearrange(axes).array, np.float64)


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


def perturb(tree, path_fn, k, scale, base=0.0):
    """tree with the NamedArray at path_fn replaced by base + scale * normal noise of its shape."""
    a = path_fn(tree)
    return eqx.tree_at(path_fn, tree, hax.named(base + scale * jrandom.normal(k, a.array.shape), a.axes))


def np_sinkhorn_liger(lg, iters=20, eps=1e-6):
    """Liger Kernel 0.8.0 mhc Sinkhorn on lg[..., out, in], numpy."""
    mat = np.exp(lg - lg.max(-1, keepdims=True))
    mat = mat / mat.sum(-1, keepdims=True) + eps
    mat = mat / (mat.sum(-2, keepdims=True) + eps)
    for _ in range(iters - 1):
        mat = mat / (mat.sum(-1, keepdims=True) + eps)
        mat = mat / (mat.sum(-2, keepdims=True) + eps)
    return mat


if PART == "main":
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

        # 4. init: hc at init == 4 x preln (same weights); 1/sqrt(m) output scales; deepnorm's centring LayerNorm and beta
        hc = model("hc", False)
        hs = hc.transformer.layers.stacked
        hc_same = dataclasses.replace(hc, transformer=dataclasses.replace(hc.transformer, layers=dataclasses.replace(hc.transformer.layers, stacked=dataclasses.replace(hs, self_attn=ps.self_attn, mlp=ps.mlp, ln_1=ps.ln_1, ln_2=ps.ln_2))), embeddings=pre.embeddings)
        leaves_close(outs(hc_same).array / 4.0, o_pre.array, "hc / 4 vs preln", rtol=1e-4, atol=1e-5)
        for arch in ("hc", "mhc"):
            st = model(arch, False).transformer.layers.stacked
            np.testing.assert_allclose(np.asarray(st.self_attn.o_proj.weight.array), 0.5 * np.asarray(ps.self_attn.o_proj.weight.array), rtol=1e-6)
            np.testing.assert_allclose(np.asarray(st.mlp.down_proj.weight.array), 0.5 * np.asarray(ps.mlp.down_proj.weight.array), rtol=1e-6)
            np.testing.assert_array_equal(np.asarray(st.self_attn.q_proj.weight.array), np.asarray(ps.self_attn.q_proj.weight.array))
            np.testing.assert_array_equal(np.asarray(st.mlp.up_proj.weight.array), np.asarray(ps.mlp.up_proj.weight.array))
        dn = model("deepnorm", False).transformer.layers.stacked
        beta = (8 * L) ** -0.25
        for got, ref in ((dn.self_attn.v_proj, ps.self_attn.v_proj), (dn.self_attn.o_proj, ps.self_attn.o_proj), (dn.mlp.gate_proj, ps.mlp.gate_proj), (dn.mlp.up_proj, ps.mlp.up_proj), (dn.mlp.down_proj, ps.mlp.down_proj)):
            np.testing.assert_allclose(np.asarray(got.weight.array), beta * np.asarray(ref.weight.array), rtol=1e-6)
        np.testing.assert_array_equal(np.asarray(dn.self_attn.q_proj.weight.array), np.asarray(ps.self_attn.q_proj.weight.array))
        dn0 = per_layer(model("deepnorm", False).transformer.layers)[0]
        assert dn0.post_1.bias is None and dn0.post_1.weight is not None
        z = hax.named(jnp.asarray(rng.normal(size=(B, T, 48)) * 3 + 5, jnp.float32), (Batch, Pos, Axis("embed", 48)))
        zz = np.asarray(z.array, np.float64)
        ln_ref = (zz - zz.mean(-1, keepdims=True)) / np.sqrt(zz.var(-1, keepdims=True) + 1e-5)
        np.testing.assert_allclose(arr(dn0.post_1(z), (Batch, Pos, Axis("embed", 48))), ln_ref, rtol=1e-4, atol=1e-4)
        print("4. hc at init == 4 x preln; hc, mhc: o_proj and down_proj x 1/2; deepnorm: centring LayerNorm (no bias), v/o/MLP x (8L)^(-1/4)")

        # 5. every design trains and evaluates; new parameters learn; probes leave the backbone gradient alone
        for arch in ARCHS:
            m = model(arch)
            loss, g = step(m, es, key)
            ev = scalar(m.compute_next_token_loss(es, key=None))
            assert np.isfinite(float(loss)) and np.isfinite(ev), arch
            assert all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(eqx.filter(g, eqx.is_array))), arch
            b1 = per_layer(g.transformer.layers)[1]
            new = [b1.post_1, b1.post_2, b1.hyper, b1.query, b1.kv_proj, g.transformer.final_query]
            new = [x for x in new if x is not None]
            for x in new:
                assert sum(float(jnp.abs(a).sum()) for a in jax.tree.leaves(eqx.filter(x, eqx.is_array))) > 0, (arch, x)
            if arch.startswith("attnres"):  # the queries start at 0, so the key gains' gradient is exactly 0 at init (test 10 moves it)
                assert float(jnp.abs(g.transformer.layers.stacked.key_gain.array).max()) == 0 and float(jnp.abs(g.transformer.final_key_gain.array).max()) == 0
            mp = model(arch, pls_detach_backbone=True)
            _, gp = step(mp, es, key)
            ge = eqx.filter_jit(eqx.filter_grad(lambda mm: mm.compute_next_token_loss(es, key=None).array))(mp)
            leaves_close([(b.self_attn, b.mlp) for b in per_layer(gp.transformer.layers)] + [gp.embeddings], [(b.self_attn, b.mlp) for b in per_layer(ge.transformer.layers)] + [ge.embeddings], f"{arch} probes backbone grad", rtol=1e-4, atol=1e-7)
            print(f"5. {arch}: train loss {float(loss):.4f}, eval {ev:.4f}, finite grads; new parameters get gradient ({len(new)} groups)")

        # 6. hc / mhc with random parameters == independent einsum references
        from experiments.references.depth_arch_qwen3 import HyperParams  # noqa: E402
        rng6 = np.random.default_rng(6)
        m_ = 4
        E48 = Axis("embed", 48)

        def rnd(a, sc, base=0.0):
            return hax.named(jnp.asarray(base + rng6.normal(size=a.array.shape) * sc, jnp.float32), a.axes)

        H0 = hax.named(jnp.asarray(rng6.normal(size=(m_, B, T, 48)), jnp.float32), (Axis("stream", m_), Batch, Pos, E48))
        for arch in ("hc", "mhc"):
            lay0 = per_layer(model(arch, False).transformer.layers)[1]  # layer index 1: sublayers 2, 3
            if arch == "hc":
                hyp = tuple(HyperParams(rnd(h.read, .3), rnd(h.res, .3), rnd(h.write, .3), rnd(h.w_read, .5), rnd(h.w_res, .5), rnd(h.w_write, .5), hax.named(jnp.asarray([0.7, 0.4]), h.scale.axes)) for h in lay0.hyper)
            else:
                hyp = tuple(MhcParams(rnd(h.phi, .2), rnd(h.bias, 1.0), hax.named(jnp.asarray([0.7, 0.4, 0.9]), h.gain.axes)) for h in lay0.hyper)
            lay = dataclasses.replace(lay0, hyper=hyp)
            got = arr(lay.hyper_step(H0, ex.attn_mask, key=None, pos_ids=None, layer=1), (Axis("stream", m_), Batch, Pos, E48))
            Hr = np.asarray(H0.array, np.float64)
            for j in range(2):
                p = hyp[j]
                k_sub = 2 + j
                if arch == "hc":
                    Hn = Hr / np.sqrt(np.mean(Hr ** 2, axis=-1, keepdims=True) + 1e-6)
                    dr, dres, dw = np.einsum("sbtd,d->bts", Hn, p.w_read.array), np.einsum("sbtd,do->btso", Hn, p.w_res.array), np.einsum("sbtd,d->bts", Hn, p.w_write.array)
                    hot = (np.arange(m_) == k_sub % m_).astype(np.float64)
                    r = hot + p.read.array + 0.7 * np.tanh(dr)
                    A = np.eye(m_) + p.res.array + 0.7 * np.tanh(dres)  # (b, t, in, out)
                    w = 1 + p.write.array + 0.4 * np.tanh(dw)
                    x = np.einsum("bts,sbtd->btd", r, Hr)
                    f = arr(lay.branch(j, hax.named(jnp.asarray(x, jnp.float32), (Batch, Pos, E48)), ex.attn_mask, key=None, pos_ids=None), (Batch, Pos, E48))
                    Hr = np.einsum("btso,sbtd->obtd", A, Hr) + np.einsum("bto,btd->obtd", w, f)
                else:
                    sig = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731
                    phi = np.asarray(p.phi.rearrange(("stream", "embed", "mhc")).array, np.float64).reshape(m_ * 48, -1)
                    flat = np.moveaxis(Hr, 0, 2).reshape(B, T, m_ * 48)  # (b, t, stream-major concatenation)
                    mix = (flat @ phi) / np.sqrt(np.mean(flat ** 2, axis=-1, keepdims=True) + 1e-6)
                    hot = (np.arange(m_) == k_sub % m_).astype(np.float64)
                    c = np.concatenate([8.0 * (2 * hot - 1), np.zeros(m_), -8.0 * (1 - np.eye(m_)).reshape(-1)]) + np.asarray(p.bias.array, np.float64)
                    g = np.asarray(p.gain.array, np.float64)
                    r = sig(mix[..., :m_] * g[0] + c[:m_])
                    w = 2 * sig(mix[..., m_:2 * m_] * g[1] + c[m_:2 * m_])
                    A = np_sinkhorn_liger((mix[..., 2 * m_:] * g[2] + c[2 * m_:]).reshape(B, T, m_, m_))  # (b, t, out, in)
                    x = np.einsum("bts,sbtd->btd", r, Hr)
                    f = arr(lay.branch(j, hax.named(jnp.asarray(x, jnp.float32), (Batch, Pos, E48)), ex.attn_mask, key=None, pos_ids=None), (Batch, Pos, E48))
                    Hr = np.einsum("btoi,ibtd->obtd", A, Hr) + np.einsum("bto,btd->obtd", w, f)
            np.testing.assert_allclose(got, Hr, rtol=2e-4, atol=2e-4, err_msg=arch)
            print(f"6. {arch}: hyper_step with random parameters == independent einsum reference (layer 1, both sublayers)")

        # 7. Liger Sinkhorn
        S4, So4 = Axis("stream", 4), Axis("stream_out", 4)
        lg = rng6.normal(size=(4, 4)) * 3
        lg[2] -= 200.0
        sk = np.asarray(_sinkhorn_liger(hax.named(jnp.asarray(lg, jnp.float32), (So4, S4))).array, np.float64)
        np.testing.assert_allclose(sk, np_sinkhorn_liger(lg), rtol=1e-4, atol=1e-6)
        assert np.all(np.isfinite(sk)) and np.allclose(sk.sum(0), 1, atol=1e-4), sk.sum(0)
        gsk = jax.grad(lambda a: jnp.sum(_sinkhorn_liger(hax.named(a, (So4, S4))).array * jnp.arange(16.0).reshape(4, 4)))(jnp.asarray(lg, jnp.float32))
        assert np.all(np.isfinite(gsk))
        c0 = np.asarray(_mhc_constants(4, 5).array)
        assert c0[1] == 8.0 and c0[0] == -8.0 and np.all(c0[4:8] == 0) and c0[8] == 0 and c0[9] == -8.0, c0
        print("7. Liger Sinkhorn == numpy transcription; columns sum to 1; finite value and gradient with a row 200 below the rest; mhc bias constants")

        # 8. optimizer labels
        from experiments.references.depth_arch_qwen3 import DepthArchMuonHConfig  # noqa: E402
        from levanter.optim.muonh import MuonHConfig  # noqa: E402
        for arch in ("hc", "mhc", "attnres"):
            mdl = model(arch)
            lab = DepthArchMuonHConfig().create_mask(mdl)
            base_lab = MuonHConfig().create_mask(mdl)
            st, lb, bb = mdl.transformer.layers.stacked, lab.transformer.layers.stacked, base_lab.transformer.layers.stacked
            new_leaves = jax.tree.leaves(lb.hyper) if arch in ("hc", "mhc") else jax.tree.leaves((lb.query, lb.key_gain, lab.transformer.final_query, lab.transformer.final_key_gain))
            assert new_leaves and all(x == "adam_new" for x in new_leaves), new_leaves
            same = [(a, b) for a, b in zip(jax.tree.leaves((lab.embeddings, lab.lm_head, lab.aux_lm_heads, lab.aux_norms, lb.self_attn, lb.mlp, lb.ln_1, lb.ln_2)), jax.tree.leaves((base_lab.embeddings, base_lab.lm_head, base_lab.aux_lm_heads, base_lab.aux_norms, bb.self_attn, bb.mlp, bb.ln_1, bb.ln_2)))]
            assert all(a == b for a, b in same) and {a for a, _ in same} == {"adam", "adamh", "muonh"}, set(a for a, _ in same)
            assert "adam_new" not in jax.tree.leaves((lb.self_attn, lb.mlp, lb.ln_1, lb.ln_2, lab.embeddings, lab.lm_head))
            assert st.self_attn.q_proj.weight.axes[0].name == "layer" and st.self_attn.o_proj.weight.axes[0].name == "layer"
            print(f"8. {arch}: new parameters -> adam_new (eps 1e-8); baseline parameters keep muonh / adamh / adam; layer weights stacked on the layer axis")
        lab_moda = MuonHConfig().create_mask(model("moda")).transformer.layers.stacked.kv_proj
        assert jax.tree.leaves(lab_moda) == ["muonh"], jax.tree.leaves(lab_moda)
        print("8. moda: kv_proj -> muonh")

        # 10. two-level scan == Python loop (attnres, moda), outputs and gradients; moda chunk invariance
        Embed = Axis("embed", 48)
        xr = hax.named(jnp.asarray(rng.normal(size=(B, T, 48)), jnp.float32), (Batch, Pos, Embed))
        tgt = hax.named(jnp.asarray(rng.normal(size=(L, B, T, 48)), jnp.float32), (Axis("layer", L), Batch, Pos, Embed))
        for arch in ("attnres", "moda"):
            for n_layers in (L, 6, 5):
                c10 = cfg(arch, num_layers=n_layers)
                tr = DepthArchTransformer.init(c10, arch, key=jrandom.PRNGKey(11))
                ks = jrandom.split(jrandom.PRNGKey(12), 4)
                if arch == "attnres":
                    tr = perturb(tr, lambda t: t.layers.stacked.query, ks[0], 0.5)
                    tr = perturb(tr, lambda t: t.layers.stacked.key_gain, ks[1], 0.3, 1.0)
                    tr = perturb(tr, lambda t: t.final_query, ks[2], 0.5)
                    tr = perturb(tr, lambda t: t.final_key_gain, ks[3], 0.3, 1.0)
                tg = tgt if n_layers == L else hax.named(jnp.asarray(rng.normal(size=(n_layers, B, T, 48)), jnp.float32), (Axis("layer", n_layers), Batch, Pos, Embed))
                scan_f = lambda t: (t.outputs(xr, ex.attn_mask, key=jrandom.PRNGKey(13)) * tg).sum().array  # noqa: E731
                loop_f = lambda t: (hax.stack(c10.Layers, t._loop_outputs(xr, ex.attn_mask, key=jrandom.PRNGKey(13))) * tg).sum().array  # noqa: E731
                o_scan = tr.outputs(xr, ex.attn_mask, key=jrandom.PRNGKey(13))
                o_loop = hax.stack(c10.Layers, tr._loop_outputs(xr, ex.attn_mask, key=jrandom.PRNGKey(13)))
                wo = rel_close(o_scan.array, o_loop.array, f"{arch} L={n_layers} scan vs loop outputs", 1e-5)
                g_scan = eqx.filter_jit(eqx.filter_grad(scan_f))(tr)
                g_loop = eqx.filter_jit(eqx.filter_grad(loop_f))(tr)
                worst = rel_close(eqx.filter(g_scan, eqx.is_inexact_array), eqx.filter(g_loop, eqx.is_inexact_array), f"{arch} L={n_layers} scan vs loop grads", 1e-4)
                learn = (g_scan.layers.stacked.query, g_scan.layers.stacked.key_gain, g_scan.final_query, g_scan.final_key_gain) if arch == "attnres" else (g_scan.layers.stacked.kv_proj.weight,)
                assert all(float(jnp.abs(x.array).max()) > 0 for x in learn), arch
                print(f"10. {arch}, {n_layers} layers: two-level scan == Python loop (outputs to relative L2 {wo:.1e}, every gradient to {worst:.1e})")
        tr4 = DepthArchTransformer.init(cfg("moda"), "moda", key=jrandom.PRNGKey(11))
        tr32 = dataclasses.replace(tr4, config=cfg("moda", moda_chunk=32))
        wc = rel_close(tr4.outputs(xr, ex.attn_mask).array, tr32.outputs(xr, ex.attn_mask).array, "moda chunk 12 vs 32", 1e-6)
        print(f"10. moda: chunk 12 == chunk 32 (relative L2 {wc:.1e})")

        # 11. independent references
        def layers_of(tr, n):
            return [tr.layer(i) for i in range(n)]

        # keel: the first layer special
        c11 = cfg("keel")
        tr = DepthArchTransformer.init(c11, "keel", key=jrandom.PRNGKey(21))
        tr = perturb(perturb(tr, lambda t: t.layers.stacked.post_1.weight, jrandom.PRNGKey(22), 0.2, 1.0), lambda t: t.layers.stacked.post_2.weight, jrandom.PRNGKey(23), 0.2, 1.0)
        got = tr.outputs(xr, ex.attn_mask)
        h = xr
        for i, lay in enumerate(layers_of(tr, L)):
            f = lay.self_attn(x=lay.ln_1(h), mask=ex.attn_mask, key=None, pos_ids=None)
            h = h + f if i == 0 else lay.post_1(h * float(2 * L) + f)
            f = lay.mlp(lay.ln_2(h), key=None)
            h = lay.post_2(h * (1.0 if i == 0 else float(2 * L)) + f)
            leaves_close(got["layer", i].array, h.array, f"keel layer {i}", rtol=1e-4, atol=1e-5)
        print("11. keel == reference (first layer: plain Pre-LN attention, MLP without the gain; then RN_post(2L h + F(RN_pre h)))")

        # deepnorm: centring LayerNorm after alpha h + F(h), no pre-norm
        c11 = cfg("deepnorm")
        tr = DepthArchTransformer.init(c11, "deepnorm", key=jrandom.PRNGKey(24))
        tr = perturb(tr, lambda t: t.layers.stacked.post_1.weight, jrandom.PRNGKey(25), 0.2, 1.0)
        got = tr.outputs(xr, ex.attn_mask)
        hh = np.asarray(xr.rearrange((Batch, Pos, Embed)).array, np.float64)
        alpha = (2 * L) ** 0.25
        for i, lay in enumerate(layers_of(tr, L)):
            for j in range(2):
                hn = hax.named(jnp.asarray(hh, jnp.float32), (Batch, Pos, Embed))
                f = arr(lay.self_attn(x=hn, mask=ex.attn_mask, key=None, pos_ids=None) if j == 0 else lay.mlp(hn, key=None), (Batch, Pos, Embed))
                y = alpha * hh + f
                w = np.asarray((lay.post_1 if j == 0 else lay.post_2).weight.array, np.float64)
                hh = (y - y.mean(-1, keepdims=True)) / np.sqrt(y.var(-1, keepdims=True) + 1e-5) * w
            np.testing.assert_allclose(arr(got["layer", i], (Batch, Pos, Embed)), hh, rtol=2e-4, atol=2e-4, err_msg=f"deepnorm layer {i}")
        print("11. deepnorm == reference (LN(alpha h + F(h)), LN centring, no bias)")

        # moda: one softmax over sequence and depth; two depth entries per layer; depth keys before RoPE; GQA (4 heads, 2 kv)
        c11 = cfg("moda", num_heads=4, num_kv_heads=2)
        tr = DepthArchTransformer.init(c11, "moda", key=jrandom.PRNGKey(26))
        got = tr.outputs(xr, ex.attn_mask)
        mk = np.asarray(ex.attn_mask.materialize(Pos, KPos).rearrange((Batch, Pos, KPos)).array)
        h = xr
        depth = []
        QA = ("batch", "position", "kv_head", "q_heads_per_group", "head_size")
        KA = ("batch", "position", "kv_head", "head_size")
        for i, lay in enumerate(layers_of(tr, L)):
            att = lay.self_attn
            xin = lay.ln_1(h)
            q, k, v = att._compute_qkv(xin, key=None, pos_ids=None)  # the library's: q and k rotated
            k_pre = att.k_norm(att.k_proj(xin))  # QK-normed, not rotated
            qa, ka, va = (np.asarray(t.rearrange(ax).array, np.float64) for t, ax in ((q, QA), (k, KA), (v, KA)))
            D = qa.shape[-1]
            s = np.einsum("bthgd,bshd->bthgs", qa, ka) / math.sqrt(D)
            s = np.where(mk[:, :, None, None, :], s, -np.inf)
            if depth:
                dk = np.stack([d[0] for d in depth], axis=3)
                dv = np.stack([d[1] for d in depth], axis=3)
                sd = np.einsum("bthgd,bthjd->bthgj", qa, dk) / math.sqrt(D)
                z = np.concatenate([s, sd], axis=-1)
            else:
                z = s
            p = np.exp(z - z.max(-1, keepdims=True))
            p = p / p.sum(-1, keepdims=True)
            o = np.einsum("bthgs,bshd->bthgd", p[..., :T], va)
            if depth:
                o = o + np.einsum("bthgj,bthjd->bthgd", p[..., T:], dv)
            on = hax.named(jnp.asarray(o, jnp.float32), QA).flatten_axes(("kv_head", "q_heads_per_group"), "heads")
            h = h + att.o_proj(on)
            xf = lay.ln_2(h)
            kv = lay.kv_proj(xf)
            kf, vf = att.k_norm(kv["kv2", 0]), kv["kv2", 1]
            h = h + lay.mlp(xf, key=None)
            depth += [(np.asarray(k_pre.rearrange(KA).array, np.float64), va), (np.asarray(kf.rearrange(KA).array, np.float64), np.asarray(vf.rearrange(KA).array, np.float64))]
            leaves_close(got["layer", i].array, h.array, f"moda layer {i}", rtol=2e-4, atol=2e-5)
        print("11. moda == reference (GQA 4/2; sequence and depth in one softmax; per layer its attention's (k before RoPE, v) and its MLP input's kv_proj (k_norm(k), v))")

        # attnres, attnres_block: DepthBench's AttnResTransformerBlock.forward, transcribed, with key gains
        def mix_ref(q, g, srcs):
            Sv = np.stack([np.asarray(s.rearrange((Batch, Pos, Embed)).array, np.float64) for s in srcs])
            r = Sv / np.sqrt(np.mean(Sv ** 2, axis=-1, keepdims=True) + 1e-6)
            lgt = np.einsum("nbtd,d->nbt", r, np.asarray(q.array, np.float64) * np.asarray(g.array, np.float64))
            w = np.exp(lgt - lgt.max(0))
            w = w / w.sum(0)
            return hax.named(jnp.asarray(np.einsum("nbt,nbtd->btd", w, Sv), jnp.float32), (Batch, Pos, Embed))

        Sub = Axis("sub", 2)
        for arch, blocks in (("attnres", 2), ("attnres_block", 2), ("attnres_block", 3)):
            c11 = cfg(arch, attnres_blocks=blocks, num_layers=6)
            tr = DepthArchTransformer.init(c11, arch, key=jrandom.PRNGKey(31))
            ks = jrandom.split(jrandom.PRNGKey(32), 4)
            tr = perturb(tr, lambda t: t.layers.stacked.query, ks[0], 0.5)
            tr = perturb(tr, lambda t: t.layers.stacked.key_gain, ks[1], 0.3, 1.0)
            tr = perturb(tr, lambda t: t.final_query, ks[2], 0.5)
            tr = perturb(tr, lambda t: t.final_key_gain, ks[3], 0.3, 1.0)
            got = tr.outputs(xr, ex.attn_mask)
            b = 1 if arch == "attnres" else max(1, (2 * 6) // blocks)
            states, hcur = None, xr
            for i, lay in enumerate(layers_of(tr, 6)):
                prefix = hcur
                if states is None:
                    h_in, states, prefix = xr, [prefix], None
                else:
                    residuals = states + [prefix]
                    if (2 * i) % b == 0:
                        states, prefix = residuals, None
                    h_in = mix_ref(lay.query[Sub, 0], lay.key_gain[Sub, 0], residuals)
                a = lay.branch(0, h_in, ex.attn_mask, key=None, pos_ids=None)
                prefix = a if prefix is None else prefix + a
                mres = states + [prefix]
                if (2 * i + 1) % b == 0:
                    states, prefix = mres, None
                mm_ = lay.branch(1, mix_ref(lay.query[Sub, 1], lay.key_gain[Sub, 1], mres), ex.attn_mask, key=None, pos_ids=None)
                hcur = mm_ if prefix is None else prefix + mm_
                nxt = tr.layer(i + 1) if i + 1 < 6 else None
                q, g = (nxt.query[Sub, 0], nxt.key_gain[Sub, 0]) if nxt is not None else (tr.final_query, tr.final_key_gain)
                worst_i = rel_close(got["layer", i].array, mix_ref(q, g, states + [hcur]).array, f"{arch} blocks={blocks} layer {i}", 1e-5)
            print(f"11. {arch} (blocks {blocks}, {b} sublayers per source) == DepthBench AttnResTransformerBlock.forward with key gains (last layer relative L2 {worst_i:.1e})")

if PART == "prod":
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
    for arch in ("hc", "mhc", "attnres", "attnres_block", "moda"):
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
        first = None
        for t in range(8):
            loss9, g9 = vg(m9)
            assert np.isfinite(float(loss9)), (arch, t)
            first = float(loss9) if first is None else first
            upd, st9 = opt.update(g9, st9, eqx.filter(m9, eqx.is_inexact_array))
            stk = upd.transformer.layers.stacked
            new = stk.hyper if arch in ("hc", "mhc") else ((stk.query, stk.key_gain) if arch.startswith("attnres") else None)
            if new is not None:
                worst = max(worst, max(float(jnp.max(jnp.abs(x.array if isinstance(x, hax.NamedArray) else x))) for x in jax.tree.leaves(new, is_leaf=lambda z: isinstance(z, hax.NamedArray))))
            m9 = eqx.apply_updates(m9, upd)
        assert worst <= 2 * 0.008, (arch, worst)
        print(f"9. {arch}: 48 layers, bf16, launcher optimizer (eps 1e-20, warmup 0): 8 steps finite, loss {first:.3f} -> {float(loss9):.3f}; largest new-parameter step {worst:.4f} <= 2 x adam_lr")
print(f"all depth_arch {PART} tests passed")
