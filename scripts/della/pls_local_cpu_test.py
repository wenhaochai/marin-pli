"""CPU tests for pls_local_heads (each layer loss trains its own head and its own layer only).

Run on a vis node after `source scripts/della/pls_env.sh`:  nice -n 19 $PY scripts/della/pls_local_cpu_test.py
1. Values: the training loss and every train/pls/L{k} equal the separate-heads model's (same init).
2. Gradients, against an independent reference: the embeddings, the last layer, the final norm and the main head get
   exactly the probes model's gradient (no layer loss reaches them); each separate head and norm gets the
   separate-heads model's gradient; layer k gets the probes gradient plus d CE_k / d layer_k, computed here by
   differentiating CE_k with respect to a slice of layer k alone on the recorded input.
3. Bad combinations are rejected (with schedules too).
4. Production regime: 6 layers, bf16 compute, the launcher's 130m MuonH (eps 1e-20, warmup 0, lr 0.02, adam 0.008),
   8 steps: finite and decreasing, for local heads and for separate heads with layer weight 0.2 (the R1 arm).
5. Memory: a training step's temporary memory with local heads stays within 1.5x of separate heads (remat scan).
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4"
os.environ.setdefault("LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS", "0")

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jrandom  # noqa: E402
import jmp  # noqa: E402
import numpy as np  # noqa: E402

import haliax as hax  # noqa: E402
from haliax import Axis  # noqa: E402
from haliax.partitioning import ResourceAxis  # noqa: E402
from levanter.layers.attention import AttentionBackend, AttentionMask  # noqa: E402
from levanter.models.lm_model import LmExample  # noqa: E402
from levanter.models.loss import maybe_fused_next_token_loss  # noqa: E402
from levanter.optim.muonh import MuonHConfig  # noqa: E402
from levanter.trainer import TrainerConfig  # noqa: E402
from levanter.utils.mesh import MeshConfig  # noqa: E402

from experiments.references.per_layer_qwen3 import PerLayerQwen3Config, PerLayerQwen3LMHeadModel  # noqa: E402

jax.config.update("jax_threefry_partitionable", True)
rng = np.random.default_rng(0)
B, T, V, L = 8, 32, 500, 4
Batch, Pos, Vocab = Axis("batch", B), Axis("position", T), Axis("vocab", V)
tok = rng.integers(2, V, size=(B, T)); tok[:, 10] = 1
eos = np.roll(tok, 1, axis=1) == 1; eos[:, 0] = False
seg = np.cumsum(eos, axis=1).astype(np.int32)
ex = LmExample(tokens=hax.named(jnp.asarray(tok, jnp.int32), (Batch, Pos)), loss_weight=hax.named(jnp.asarray((np.arange(T) < T - 1).astype(np.float32)[None].repeat(B, 0)), (Batch, Pos)),
               attn_mask=AttentionMask.causal().with_segment_ids(hax.named(jnp.asarray(seg), (Batch, Pos))))
common = dict(max_seq_len=T, hidden_dim=48, intermediate_dim=64, num_layers=L, num_heads=2, num_kv_heads=2, hybrid_norm=True, attn_backend=AttentionBackend.VANILLA, pls_weight=1.0, pls_separate_heads=True)
key0, key = jrandom.PRNGKey(0), jrandom.PRNGKey(7)


def model(**kw):
    return PerLayerQwen3LMHeadModel.init(Vocab, PerLayerQwen3Config(**common, **kw), key=key0)


def close(a, b, what, rtol=1e-4, atol=1e-7):
    la, lb = jax.tree.leaves(a), jax.tree.leaves(b)
    assert len(la) == len(lb), (what, len(la), len(lb))
    for u, v in zip(la, lb):
        np.testing.assert_allclose(np.asarray(u, np.float32), np.asarray(v, np.float32), rtol=rtol, atol=atol, err_msg=what)


def layer_slice(tree, k):
    return hax.tree_util.tree_map(lambda a: a["layer", k], tree)


tc = TrainerConfig(mesh=MeshConfig(axes={"data": -1, "replica": 1, "model": 1}, compute_mapping={
    "token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA), "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA)}))


def step(m, e, k):
    def lf(mm):
        out = mm.compute_next_token_loss(e, key=k, logsumexp_weight=0.0)
        loss, stats = out
        return loss.array if isinstance(loss, hax.NamedArray) else loss, {s: v.value() for s, v in stats.items()}
    (l, st), g = eqx.filter_value_and_grad(lf, has_aux=True)(m)
    return l, st, g


with tc.use_device_mesh(), hax.axis_mapping(tc.compute_axis_mapping):
    es = hax.shard(ex, tc.compute_axis_mapping)
    jstep = eqx.filter_jit(step)
    sep, loc, prb = model(), model(pls_local_heads=True), model(pls_detach_backbone=True)
    close(loc, sep, "init", rtol=0, atol=0)
    ls, ss, gs = jstep(sep, es, key)
    ll, sl, gl = jstep(loc, es, key)
    lp, sp, gp = jstep(prb, es, key)
    # 1. values
    np.testing.assert_allclose(float(ll), float(ls), rtol=1e-6)
    for k in range(L):
        np.testing.assert_allclose(float(sl[f"pls/L{k}"]), float(ss[f"pls/L{k}"]), rtol=1e-6)
    print(f"1. local heads: loss {float(ll):.6f} and every train/pls/L{{k}} == separate heads")
    # 2. gradients
    close((gl.embeddings, layer_slice(gl.transformer.layers.stacked, L - 1), gl.transformer.norm, gl.lm_head),
          (gp.embeddings, layer_slice(gp.transformer.layers.stacked, L - 1), gp.transformer.norm, gp.lm_head), "embeddings / last layer / final norm / main head == probes")
    close((gl.aux_lm_heads, gl.aux_norms), (gs.aux_lm_heads, gs.aux_norms), "separate heads and norms == separate-heads model")
    outs = loc.layer_outputs(es.tokens, es.attn_mask, key=key)
    keys = jrandom.split(key, L)
    x0 = loc.embeddings.embed(es.tokens)
    for k in range(L - 1):
        h_in = x0 if k == 0 else outs["layer", k - 1]
        lay = layer_slice(loc.transformer.layers.stacked, k)

        def ce_k(layer, k=k, h_in=h_in):
            h = layer(h_in, mask=es.attn_mask, key=keys[k], pos_ids=None)
            ce = maybe_fused_next_token_loss(loc.Pos, loc.Embed, loc.Vocab, loc.aux_norms[k](h), loc.aux_lm_heads[k].weight, es.tokens, loss_weight=es.loss_weight, reduction=hax.mean, reduction_axis=None, logsumexp_weight=0.0, dtype=jnp.float32)
            return ce.array if isinstance(ce, hax.NamedArray) else ce

        gk = eqx.filter_grad(ce_k)(lay)
        lv = lambda t: [x.array if isinstance(x, hax.NamedArray) else x for x in jax.tree.leaves(t, is_leaf=lambda z: isinstance(z, hax.NamedArray))]  # noqa: E731
        want = [a + b for a, b in zip(lv(layer_slice(gp.transformer.layers.stacked, k)), lv(gk))]
        close(lv(layer_slice(gl.transformer.layers.stacked, k)), want, f"layer {k}: probes grad + d CE_{k} / d layer_{k}")
        diff = max(float(jnp.max(jnp.abs((a - b).array if isinstance(a, hax.NamedArray) else a - b))) for a, b in zip(jax.tree.leaves(layer_slice(gl.transformer.layers.stacked, k)), jax.tree.leaves(layer_slice(gs.transformer.layers.stacked, k))))
        if k < L - 2:  # below the top intermediate layer, separate heads also send later layer losses into layer k
            assert diff > 1e-6, k
        else:  # the top intermediate layer gets no other layer loss in either setup
            assert diff < 1e-5, (k, diff)
    print("2. gradients: embeddings, last layer, final norm, main head == probes; heads == separate heads; layer k == probes + dCE_k/dlayer_k (independent reference), != separate heads")
    # 3. bad combinations
    for bad in (dict(pls_local_heads=True, pls_separate_heads=False), dict(pls_local_heads=True, pls_detach_backbone=True), dict(pls_local_heads=True, pls_probe=True), dict(pls_local_heads=True, pls_off_start=10, pls_off_end=20)):
        kw = dict(common, **bad)
        try:
            PerLayerQwen3Config(**kw); raise SystemExit(f"accepted {bad}")
        except ValueError:
            pass
    print("3. bad combinations rejected")

# 4. production regime: 6 layers, bf16, the launcher's 130m MuonH, 8 steps
pol = jmp.get_policy("p=f32,c=bfloat16")
T4, L4 = 128, 6
Pos4 = Axis("position", T4)
tok4 = np.random.default_rng(4).integers(2, V, size=(4, T4))
ex4 = LmExample(tokens=hax.named(jnp.asarray(tok4, jnp.int32), (Axis("batch", 4), Pos4)), loss_weight=hax.named(jnp.ones((4, T4)), (Axis("batch", 4), Pos4)), attn_mask=AttentionMask.causal())
for name, kw in (("local heads", dict(pls_local_heads=True)), ("layer weight 0.2", dict(pls_weight=0.2))):
    cfg = PerLayerQwen3Config(**dict(common, max_seq_len=T4, hidden_dim=128, intermediate_dim=448, num_layers=L4, **kw))
    m = PerLayerQwen3LMHeadModel.init(Vocab, cfg, key=key0)
    opt = MuonHConfig(learning_rate=0.02, adam_lr=0.008, beta1=0.9, beta2=0.98, epsilon=1e-20, momentum=0.95, nesterov=True, backend_steps=5, muon_epsilon=1e-5, max_grad_norm=1.0,
                      weight_decay=0.1, lr_schedule="linear", decay=0.8, warmup=0, min_lr_ratio=0.0, coefficient_type="simple").build(100)
    st = opt.init(eqx.filter(m, eqx.is_inexact_array))

    def lf(mm):
        out = pol.cast_to_compute(mm).compute_next_token_loss(ex4, key=jrandom.PRNGKey(3))
        out = out[0]
        return out.array if isinstance(out, hax.NamedArray) else out

    vg = eqx.filter_jit(eqx.filter_value_and_grad(lf))
    losses = []
    for t in range(8):
        lv, g = vg(m)
        assert np.isfinite(float(lv)), (name, t)
        losses.append(float(lv))
        upd, st = opt.update(g, st, eqx.filter(m, eqx.is_inexact_array))
        m = eqx.apply_updates(m, upd)
    assert losses[-1] < losses[0], (name, losses)
    print(f"4. {name}: 6 layers, bf16, launcher 130m optimizer: 8 steps finite, loss {losses[0]:.3f} -> {losses[-1]:.3f}")
# 5. memory: the recompute runs under the scan's remat policy, so the step's temporary memory stays near separate heads'
T5, L5, B5 = 512, 6, 8
Pos5 = Axis("position", T5)
tok5 = np.random.default_rng(5).integers(2, V, size=(B5, T5))
ex5 = LmExample(tokens=hax.named(jnp.asarray(tok5, jnp.int32), (Axis("batch", B5), Pos5)), loss_weight=hax.named(jnp.ones((B5, T5)), (Axis("batch", B5), Pos5)), attn_mask=AttentionMask.causal())
temp = {}
for name, kw in (("separate heads", {}), ("local heads", dict(pls_local_heads=True))):
    cfg = PerLayerQwen3Config(**dict(common, max_seq_len=T5, hidden_dim=256, intermediate_dim=896, num_layers=L5, **kw))
    m5 = PerLayerQwen3LMHeadModel.init(Vocab, cfg, key=key0)
    def lf5(mm):
        out = pol.cast_to_compute(mm).compute_next_token_loss(ex5, key=jrandom.PRNGKey(3))[0]
        return out.array if isinstance(out, hax.NamedArray) else out
    params, static = eqx.partition(m5, eqx.is_array)
    comp = jax.jit(lambda p: eqx.filter_value_and_grad(lf5)(eqx.combine(p, static))).lower(params).compile()
    temp[name] = comp.memory_analysis().temp_size_in_bytes / 2**20
ratio = temp["local heads"] / temp["separate heads"]
assert ratio < 1.5, temp
print(f"5. temporary memory of one training step: local heads {temp['local heads']:.0f} MiB vs separate heads {temp['separate heads']:.0f} MiB (x{ratio:.2f})")
print("all pls_local tests passed")
