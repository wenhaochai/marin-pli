"""Audit A/B: pls training loss through the REAL trainer loss wrapper (bf16 compute policy) and levanter's
microbatched gradient accumulation, against a reference built ONLY from the baseline code path:

    CE_k = baseline Qwen3LMHeadModel.compute_next_token_loss on a model TRUNCATED to layers 0..k
           (same stacked weights sliced, same final norm, same lm_head, baseline fold + fused CE kernel).

This shares no code with per_layer_qwen3 (no scan_via, no _readout_ce, no _monitor_ce).
Checks:
  A1 f32 compute, one batch: pls w=1 loss/stats/grads == sum of truncated baseline losses.
  A2 bf16 compute (jmp p=f32,c=bfloat16 via levanter's WrappedLossFunction): same, bf16 tolerance; report max diffs.
  A3 w=0 under bf16: loss and grads vs the baseline model through the same wrapper, bitwise?
  B1 microbatched (2 microbatches, levanter.grad_accum.microbatched), bf16: loss, stats and grads == mean over the two
     microbatches of the reference (per microbatch).
  B2 microbatched w=0 bf16: loss/grads == baseline microbatched; monitor stats == mean over microbatches of the strided
     reference (stride 16 on T=64).
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
import jmp  # noqa: E402
import numpy as np  # noqa: E402

import haliax as hax  # noqa: E402
from haliax import Axis, NamedArray  # noqa: E402
from haliax.nn.scan import Stacked  # noqa: E402
from haliax.partitioning import ResourceAxis  # noqa: E402
from levanter.grad_accum import microbatched  # noqa: E402
from levanter.layers.attention import AttentionBackend, AttentionMask  # noqa: E402
from levanter.models.lm_model import LmExample  # noqa: E402
from levanter.models.llama import LlamaTransformer  # noqa: E402
from levanter.models.qwen import Qwen3Config, Qwen3LMHeadModel  # noqa: E402
from levanter.trainer import TrainerConfig, WrappedLossFunction  # noqa: E402
from levanter.utils.mesh import MeshConfig  # noqa: E402

from experiments.references.per_layer_qwen3 import PerLayerQwen3Config, PerLayerQwen3LMHeadModel  # noqa: E402

jax.config.update("jax_threefry_partitionable", True)
assert jax.device_count() == 4
rng = np.random.default_rng(1)

B, T, V, L, S = 8, 64, 384, 4, 16
Batch, Pos, Vocab = Axis("batch", B), Axis("position", T), Axis("vocab", V)
EOS = 1
tok = rng.integers(2, V, size=(B, T))
tok[:, 21] = EOS
tok[6, 40] = EOS
eos_mask = np.roll(tok, 1, axis=1) == EOS
eos_mask[:, 0] = False
seg = np.cumsum(eos_mask, axis=1).astype(np.int32)
lw = (np.arange(T) < T - 1).astype(np.float32)[None].repeat(B, 0)
lw[1, :9] = 0.0  # a masked prefix
lw[4, 30:50] = 0.0
lw[7, :] = 0.0  # an all-masked sequence in microbatch 2
lw[7, 5] = 1.0
ex = LmExample(
    tokens=hax.named(jnp.asarray(tok, jnp.int32), (Batch, Pos)),
    loss_weight=hax.named(jnp.asarray(lw), (Batch, Pos)),
    attn_mask=AttentionMask.causal().with_segment_ids(hax.named(jnp.asarray(seg), (Batch, Pos))),
)
common = dict(max_seq_len=T, hidden_dim=64, intermediate_dim=128, num_layers=L, num_heads=4, num_kv_heads=2, hybrid_norm=True,
              attn_backend=AttentionBackend.VANILLA)
key0 = jrandom.PRNGKey(3)
base = Qwen3LMHeadModel.init(Vocab, Qwen3Config(**common), key=key0)


def perturb(m, seed):
    """Make the layers different from init so every readout differs (random init has near-identical CE by layer)."""
    leaves, tdef = jax.tree_util.tree_flatten(m, is_leaf=lambda x: isinstance(x, NamedArray))
    ks = jrandom.split(jrandom.PRNGKey(seed), len(leaves))
    out = []
    for leaf, k in zip(leaves, ks):
        if isinstance(leaf, NamedArray) and jnp.issubdtype(leaf.dtype, jnp.floating):
            out.append(leaf + 0.05 * hax.random.normal(k, leaf.axes) * (1.0 + jnp.abs(leaf.array).mean()))
        else:
            out.append(leaf)
    return jax.tree_util.tree_unflatten(tdef, out)


base = perturb(base, 11)
models = {}
for w in (1.0, 0.0):
    cfg = PerLayerQwen3Config(**common, pls_weight=w, pls_monitor_stride=S)
    models[w] = PerLayerQwen3LMHeadModel(dataclasses.replace(base.transformer, config=cfg), base.embeddings, base.lm_head)

tc = TrainerConfig(mesh=MeshConfig(axes={"data": -1, "replica": 1, "model": 1}, compute_mapping={
    "token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
    "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA)}))
cmap, pmap = tc.compute_axis_mapping, tc.parameter_axis_mapping
key = jrandom.PRNGKey(7)
F32 = jmp.get_policy("p=f32,c=f32")
BF16 = jmp.get_policy("p=f32,c=bfloat16")


def truncated(m, k):
    """Baseline Qwen3 model made of m's layers 0..k (sliced weights), final norm and lm_head."""
    tr = m.transformer
    Blk = Axis(tr.layers.Block.name, k + 1)

    def sl(a):
        if isinstance(a, NamedArray) and tr.layers.Block.name in a.axis_names:
            return a[tr.layers.Block.name, slice(0, k + 1)]
        if isinstance(a, jax.Array) and a.ndim > 0 and a.shape[0] == tr.layers.Block.size:
            raise AssertionError("unexpected unnamed stacked leaf")
        return a

    stacked = jax.tree_util.tree_map(sl, tr.layers.stacked, is_leaf=lambda x: isinstance(x, NamedArray))
    cfg_k = Qwen3Config(**{**common, "num_layers": k + 1})
    tr_k = LlamaTransformer(cfg_k, Stacked(stacked, Blk, tr.layers.gradient_checkpointing), tr.norm)
    return Qwen3LMHeadModel(tr_k, m.embeddings, m.lm_head)


def ref_per_layer(m, example, *, per_pos_mask=None):
    """[CE_0, ..., CE_{L-1}] via the baseline loss path on truncated models (weighted mean like the trainer)."""
    out = []
    for k in range(L):
        mk = truncated(m, k)
        if per_pos_mask is None:
            ce = mk.compute_next_token_loss(example, key=None, logsumexp_weight=0.0)
        else:  # monitor reference: baseline per-position loss, weighted mean over the sampled positions only
            pp = mk.compute_next_token_loss(example, key=None, reduction=None, reduction_axis=())  # = loss * weight
            wnext = example.loss_weight * (1 - hax.nn.one_hot(-1, example.tokens.resolve_axis("position"), dtype=jnp.float32))
            msk = hax.named(jnp.asarray(per_pos_mask, jnp.float32), (example.tokens.resolve_axis("position"),))
            ce = hax.sum(pp * msk) / hax.sum(wnext * msk)
        out.append(ce.array if isinstance(ce, NamedArray) else ce)
    return out


def ref_raw(w):
    def raw(m, example, *, key=None):
        per = ref_per_layer(m, example)
        total = per[L - 1] + w * sum(per[: L - 1])
        return total, {f"pls/L{k}": per[k] for k in range(L)}

    return raw


def pls_raw(m, example, *, key=None):
    return m.compute_next_token_loss(example, key=key, logsumexp_weight=0.0)


def base_raw(m, example, *, key=None):
    return m.compute_next_token_loss(example, key=key, logsumexp_weight=0.0)


def grad_fn(raw, mp, mbs=None):
    f = eqx.filter_value_and_grad(WrappedLossFunction(raw, mp, cmap), has_aux=True)
    if mbs is not None:
        f = microbatched(f, Batch, mbs, pmap, cmap)
    return f


def run(raw, mp, m, example, mbs=None):
    f = grad_fn(raw, mp, mbs)

    @eqx.filter_jit
    def go(m_, e_, k_):
        with hax.axis_mapping(cmap):
            (loss, metrics), g = f(m_, e_, key=k_)
        return loss, {n: v.value() for n, v in metrics.items()}, g

    return go(m, example, key)


def maxrel(a, b):
    la = jax.tree.leaves(eqx.filter(a, eqx.is_array))
    lb = jax.tree.leaves(eqx.filter(b, eqx.is_array))
    assert len(la) == len(lb)
    worst = 0.0
    for u, v in zip(la, lb):
        u, v = np.asarray(u, np.float64), np.asarray(v, np.float64)
        worst = max(worst, float(np.max(np.abs(u - v)) / (np.max(np.abs(v)) + 1e-30)))
    return worst


def bitwise(a, b):
    la = jax.tree.leaves(eqx.filter(a, eqx.is_array))
    lb = jax.tree.leaves(eqx.filter(b, eqx.is_array))
    return all(np.array_equal(np.asarray(u), np.asarray(v)) for u, v in zip(la, lb))


results = {}
with tc.use_device_mesh(), hax.axis_mapping(cmap):
    es = hax.shard(ex, cmap)
    m1 = hax.shard(models[1.0], pmap)
    m0 = hax.shard(models[0.0], pmap)
    mb = hax.shard(base, pmap)

    # A1 / A2: one batch, f32 and bf16 compute
    for name, mp, tol in (("A1 f32", F32, 2e-5), ("A2 bf16", BF16, 2e-2)):
        loss, stats, g = run(pls_raw, mp, m1, es)
        rloss, rstats, rg = run(ref_raw(1.0), mp, m1, es)
        per = [float(stats[f"pls/L{k}"]) for k in range(L)]
        rper = [float(rstats[f"pls/L{k}"]) for k in range(L)]
        dl = abs(float(loss) - float(rloss)) / abs(float(rloss))
        dper = max(abs(a - b) / abs(b) for a, b in zip(per, rper))
        dg = maxrel(g, rg)
        ok = dl < tol and dper < tol and dg < 20 * tol
        results[name] = ok
        print(f"{name}: loss {float(loss):.6f} ref {float(rloss):.6f} (rel {dl:.2e}); per-layer {np.round(per, 4)} ref {np.round(rper, 4)} "
              f"(max rel {dper:.2e}); grads max rel {dg:.2e} -> {'OK' if ok else 'FAIL'}")
        np.testing.assert_allclose(float(loss), sum(per[: L - 1]) + per[L - 1], rtol=1e-5)

    # A3: w=0 in bf16 vs baseline through the same wrapper
    l0, s0, g0 = run(pls_raw, BF16, m0, es)
    lb_, _, gb = run(base_raw, BF16, mb, es)
    print(f"A3 bf16 w=0 vs baseline: loss {float(l0):.7f} vs {float(lb_):.7f} bitwise_loss={np.array_equal(np.asarray(l0), np.asarray(lb_))} "
          f"bitwise_grads={bitwise(g0, gb)} grads max rel {maxrel(g0, gb):.2e}")
    results["A3"] = maxrel(g0, gb) < 1e-2 and abs(float(l0) - float(lb_)) < 1e-3

    # B1: microbatched (2 microbatches of 4), bf16, w=1
    mbs = 4
    loss, stats, g = run(pls_raw, BF16, m1, es, mbs=mbs)
    halves = [jax.tree_util.tree_map(lambda a, i=i: a["batch", slice(i * mbs, (i + 1) * mbs)] if isinstance(a, NamedArray) else a, ex,
                                     is_leaf=lambda x: isinstance(x, NamedArray)) for i in range(B // mbs)]
    refs = [run(ref_raw(1.0), BF16, m1, hax.shard(h, cmap)) for h in halves]
    rloss = np.mean([float(r[0]) for r in refs])
    rper = [np.mean([float(r[1][f"pls/L{k}"]) for r in refs]) for k in range(L)]
    rg = jax.tree_util.tree_map(lambda *xs: sum(xs) / len(xs), *[r[2] for r in refs])
    per = [float(stats[f"pls/L{k}"]) for k in range(L)]
    dl = abs(float(loss) - rloss) / abs(rloss)
    dper = max(abs(a - b) / abs(b) for a, b in zip(per, rper))
    dg = maxrel(g, rg)
    ok = dl < 2e-2 and dper < 2e-2 and dg < 0.2
    results["B1"] = ok
    print(f"B1 microbatched bf16 w=1: loss {float(loss):.6f} ref(mean of microbatches) {rloss:.6f} (rel {dl:.2e}); per-layer {np.round(per, 4)} "
          f"ref {np.round(rper, 4)} (max rel {dper:.2e}); grads max rel {dg:.2e} -> {'OK' if ok else 'FAIL'}")
    # full-batch token-weighted mean would differ (microbatch means are averaged with equal weight), same as the baseline
    fb = run(ref_raw(1.0), BF16, m1, es)
    print(f"    (for reference: full-batch token-weighted per-layer {np.round([float(fb[1][f'pls/L{k}']) for k in range(L)], 4)})")

    # B2: microbatched w=0 bf16: loss/grads == baseline microbatched; monitor == mean over microbatches of strided reference
    l0, s0, g0 = run(pls_raw, BF16, m0, es, mbs=mbs)
    lb_, sb_, gb = run(base_raw, BF16, mb, es, mbs=mbs)
    stride_mask = (np.arange(T) % S == 0)
    mon_ref = []
    for h in halves:
        mon_ref.append([float(v) for v in eqx.filter_jit(lambda mm, e_: ref_per_layer(BF16.cast_to_compute(mm), e_, per_pos_mask=stride_mask))(m0, hax.shard(h, cmap))])
    mon_ref = np.mean(np.array(mon_ref), axis=0)
    mon = [float(s0[f"pls/L{k}"]) for k in range(L - 1)]
    dmon = max(abs(a - b) / abs(b) for a, b in zip(mon, mon_ref[: L - 1]))
    ok = bitwise(g0, gb) or maxrel(g0, gb) < 1e-2
    results["B2"] = ok and dmon < 2e-2
    print(f"B2 microbatched bf16 w=0: loss {float(l0):.7f} vs baseline {float(lb_):.7f}; bitwise grads={bitwise(g0, gb)} (max rel {maxrel(g0, gb):.2e}); "
          f"monitor {np.round(mon, 4)} vs strided ref {np.round(mon_ref[:L - 1], 4)} (max rel {dmon:.2e}); L{L-1} stat {float(s0[f'pls/L{L-1}']):.5f} "
          f"== train loss {float(l0):.5f}")

print("RESULTS", results)
print("AUDIT LOSS TESTS", "PASSED" if all(results.values()) else "FAILED")
