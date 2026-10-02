"""CPU tests for experiments.references.per_layer_qwen3 (pls: every layer's LM loss through the shared final norm + lm_head).

Run from the worktree on a vis node (fake 4-device mesh):
    PYTHONPATH=<worktree libs> nice -n 19 <venv>/bin/python scripts/della/pls_cpu_test.py
1. Initialisation equals the baseline Qwen3's for the same key, and evaluation (key=None) is the baseline's loss.
2. Under the launcher's sharding, the pls training loss (weights 1 and 0.3) equals sum_k w_k CE_k computed independently
   (a Python loop over the layers, dense logits of the final norm + lm_head on every layer's output), the logged
   train/pls/L{k} values equal the per-layer CEs, and every gradient matches.
3. pls_weight = 0: loss and every gradient equal the baseline's training step (the monitor leaks nothing), and the
   monitored per-layer CEs equal the reference CE on every stride-th position.
4. Per-layer eval readouts: per-position losses equal the reference; readout L-1 equals the baseline's per-position eval.
5. ReadoutTaggedEvaluator on a sharded 2-dataset eval set: every readout's EvalResult (micro/macro, per-tag loss and bpb)
   equals levanter's TaggedEvaluator run on that readout alone, readout L-1 equals the baseline evaluator, and the
   log keys are eval/L{k}/<main eval keys>.
6. flops_per_token adds one lm_head per intermediate layer (trained) or 1/(3 stride) of it (monitor).
7. Weight schedule (pls_off_start/end, traced train step): before the ramp the step equals the always-on step, inside
   it the loss and grads equal the reference at the ramp's weight, after it the loss and every grad equal the baseline's
   and train/pls/L{k} is the stride monitor (through each layer's own head with separate heads, whose grads are zero).
8. Probe: training is unchanged; cos_L{k}, gratio_L{k}, cos_aux equal cosines / norm ratios of per-loss gradients from
   the dense reference restricted to the trunk (embeddings + blocks 0..k), and icl_L{k} equals the reference late-minus-
   early CE, for the shared head (w=1 and w=0) and for separate heads.
"""
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

B, T, V, L, S = 8, 32, 500, 3, 4
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


# 1. init and eval
models = {w: PerLayerQwen3LMHeadModel.init(Vocab, PerLayerQwen3Config(**common, pls_weight=w, pls_monitor_stride=S), key=key0) for w in (1.0, 0.3, 0.0)}
for m in models.values():
    close(m, base, "init vs baseline", rtol=0, atol=0)
ev_base = scalar(base.compute_next_token_loss(ex, key=None))
for w, m in models.items():
    assert scalar(m.compute_next_token_loss(ex, key=None)) == ev_base, w
print(f"1. init == baseline Qwen3 (same key); eval (key=None) == baseline loss {ev_base:.6f}")

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


with tc.use_device_mesh(), hax.axis_mapping(tc.compute_axis_mapping):
    es = hax.shard(ex, tc.compute_axis_mapping)
    step_fn = eqx.filter_jit(train_step)

    # 2. trained readouts, weights 1 and 0.3
    trained = {}
    for w in (1.0, 0.3):
        ms = hax.shard(models[w], tc.parameter_axis_mapping)
        loss, stats, g = step_fn(ms, es, key)
        trained[w] = (loss, stats, g)

        def ref_total(mm, w=w):
            per = ref_losses(mm)
            return per[L - 1] + w * sum(per[: L - 1]), per

        (rl, rper), rg = eqx.filter_value_and_grad(ref_total, has_aux=True)(models[w])
        np.testing.assert_allclose(float(loss), float(rl), rtol=1e-5)
        assert sorted(stats) == [f"pls/L{k}" for k in range(L)], sorted(stats)
        for k in range(L):
            np.testing.assert_allclose(float(stats[f"pls/L{k}"]), float(rper[k]), rtol=1e-5)
        close(g, rg, f"pls w={w} grad")
        assert float(jnp.abs(g.lm_head.weight.array).sum()) > 0 and float(jnp.abs(g.transformer.norm.weight.array).sum()) > 0
        print(f"2. w={w}: loss {float(loss):.6f} == CE_final + w * sum CE_k (per layer {[round(float(r), 4) for r in rper]}); stats and all grads match")

    # 3. monitor only
    ms0 = hax.shard(models[0.0], tc.parameter_axis_mapping)
    loss0, stats0, g0 = step_fn(ms0, es, key)
    lossb, _, gb = step_fn(hax.shard(base, tc.parameter_axis_mapping), es, key)
    np.testing.assert_allclose(float(loss0), float(lossb), rtol=1e-6)
    close(g0, gb, "pls w=0 grad vs baseline", rtol=1e-5)
    stride_mask = (np.arange(T) % S == 0).astype(np.float32)[None]
    rmon = ref_losses(models[0.0], stride_mask)
    rfull = ref_losses(models[0.0])
    for k in range(L - 1):
        np.testing.assert_allclose(float(stats0[f"pls/L{k}"]), float(rmon[k]), rtol=1e-5)
    np.testing.assert_allclose(float(stats0[f"pls/L{L - 1}"]), float(rfull[L - 1]), rtol=1e-5)
    print(f"3. w=0: loss and all grads == baseline training step; monitor CEs == reference on every {S}th position {[round(float(r), 4) for r in rmon[:-1]]}")

    # 3b. pls_detach_head: same forward as w=1; the trunk's gradient is w=1's (stopping the head's gradient does not change
    # dCE_k/dh_k), and the final norm / lm_head get the final layer's gradient alone, i.e. the baseline's.
    m_dh = PerLayerQwen3LMHeadModel.init(Vocab, PerLayerQwen3Config(**common, pls_weight=1.0, pls_monitor_stride=S, pls_detach_head=True), key=key0)
    close(m_dh, base, "detach init vs baseline", rtol=0, atol=0)
    loss_dh, stats_dh, g_dh = step_fn(hax.shard(m_dh, tc.parameter_axis_mapping), es, key)
    loss1, stats1, g1 = trained[1.0]
    np.testing.assert_allclose(float(loss_dh), float(loss1), rtol=1e-6)
    for k in range(L):
        np.testing.assert_allclose(float(stats_dh[f"pls/L{k}"]), float(stats1[f"pls/L{k}"]), rtol=1e-6)

    def split(g_):
        return eqx.tree_at(lambda m: (m.lm_head, m.transformer.norm), g_, replace=(None, None)), (g_.lm_head, g_.transformer.norm)

    trunk_dh, head_dh = split(g_dh)
    trunk_1, head_1 = split(g1)
    trunk_b, head_b = split(gb)
    close(trunk_dh, trunk_1, "detach trunk grad vs w=1", rtol=1e-5)
    close(head_dh, head_b, "detach head/norm grad vs baseline", rtol=1e-5)
    dh_diff = float(jnp.max(jnp.abs(head_1[0].weight.array - head_b[0].weight.array)))
    assert dh_diff > 1e-6, "w=1 must differ from the baseline on the lm_head gradient, else this check is vacuous"
    print(f"3b. detach_head: loss/stats == w=1; trunk grads == w=1; final norm + lm_head grads == baseline (w=1 differs there by up to {dh_diff:.2e})")

    # 3c. pls_separate_heads: every intermediate layer reads out through its own norm + lm_head
    cfg_sep = PerLayerQwen3Config(**common, pls_weight=1.0, pls_monitor_stride=S, pls_separate_heads=True)
    m_sep = PerLayerQwen3LMHeadModel.init(Vocab, cfg_sep, key=key0)
    close(eqx.tree_at(lambda m: (m.aux_norms, m.aux_lm_heads), m_sep, replace=(None, None)), base, "sep init (baseline part) vs baseline", rtol=0, atol=0)
    assert len(m_sep.aux_lm_heads) == L - 1 and len(m_sep.aux_norms) == L - 1
    assert not np.allclose(np.asarray(m_sep.aux_lm_heads[0].weight.array), np.asarray(m_sep.lm_head.weight.array))

    def ref_sep(mm):
        out = []
        for k, x in enumerate(ref_raw(mm)):
            if k < L - 1:
                h, W = mm.aux_norms[k](x).array, mm.aux_lm_heads[k].weight.rearrange((mm.Embed, Vocab)).array
            else:
                h, W = mm.transformer.norm(x).array, mm.get_lm_head().rearrange((mm.Embed, Vocab)).array
            wt = jnp.asarray(w_next)
            out.append(jnp.sum(ref_ce_per_pos(h, W) * wt) / jnp.sum(wt))
        return out[L - 1] + sum(out[: L - 1]), out

    loss_s, stats_s, g_s = step_fn(hax.shard(m_sep, tc.parameter_axis_mapping), es, key)
    (rl_s, rper_s), rg_s = eqx.filter_value_and_grad(ref_sep, has_aux=True)(m_sep)
    np.testing.assert_allclose(float(loss_s), float(rl_s), rtol=1e-5)
    for k in range(L):
        np.testing.assert_allclose(float(stats_s[f"pls/L{k}"]), float(rper_s[k]), rtol=1e-5)
    close(g_s, rg_s, "sep grad vs reference")
    close((g_s.lm_head, g_s.transformer.norm), (gb.lm_head, gb.transformer.norm), "sep main head/norm grad vs baseline", rtol=1e-5)
    per_s = eqx.filter_jit(lambda mm, e_: mm.readout_losses(e_, tuple(range(L))))(hax.shard(m_sep, tc.parameter_axis_mapping), es)
    xs = ref_raw(m_sep)
    for k in range(L - 1):
        W = m_sep.aux_lm_heads[k].weight.rearrange((m_sep.Embed, Vocab)).array
        np.testing.assert_allclose(np.asarray(per_s.array[k]) * w_next, np.asarray(ref_ce_per_pos(m_sep.aux_norms[k](xs[k]).array, W)) * w_next, rtol=1e-5, atol=1e-5)
    labels_s = MuonHConfig().create_mask(m_sep)
    assert all(l == "adamh" for l in labels_s.aux_lm_heads), labels_s.aux_lm_heads
    assert all(n.weight == "adam" for n in labels_s.aux_norms), labels_s.aux_norms
    assert labels_s.lm_head == "adamh"
    for bad in (dict(pls_weight=0.0, pls_separate_heads=True), dict(pls_weight=1.0, pls_separate_heads=True, pls_detach_head=True)):
        try:
            PerLayerQwen3Config(**common, **bad)
            raise AssertionError(f"{bad} must be rejected")
        except ValueError:
            pass
    print(f"3c. separate heads: baseline part of init == baseline; loss/stats/all grads == per-layer-head reference {[round(float(r), 4) for r in rper_s]}; "
          "main head/norm grads == baseline; eval readouts use each layer's own head; MuonH: aux heads adamh, aux norms adam; bad combos rejected")

    # 3d. separate heads with the backbone detached: same forward and per-layer CEs as 3c; the backbone, embeddings, main
    # norm and lm_head get the baseline's gradient; each aux head gets 3c's gradient (it sees the same h_k)
    m_bb = PerLayerQwen3LMHeadModel.init(Vocab, PerLayerQwen3Config(**common, pls_weight=1.0, pls_monitor_stride=S, pls_separate_heads=True, pls_detach_backbone=True), key=key0)
    loss_bb, stats_bb, g_bb = step_fn(hax.shard(m_bb, tc.parameter_axis_mapping), es, key)
    np.testing.assert_allclose(float(loss_bb), float(loss_s), rtol=1e-6)
    for k in range(L):
        np.testing.assert_allclose(float(stats_bb[f"pls/L{k}"]), float(stats_s[f"pls/L{k}"]), rtol=1e-6)
    close(eqx.tree_at(lambda m: (m.aux_norms, m.aux_lm_heads), g_bb, replace=(None, None)), gb, "detach-backbone main grads vs baseline", rtol=1e-5)
    close((g_bb.aux_norms, g_bb.aux_lm_heads), (g_s.aux_norms, g_s.aux_lm_heads), "detach-backbone aux head grads vs separate heads", rtol=1e-5)
    bb_diff = float(jnp.max(jnp.abs(g_s.embeddings.token_embeddings.weight.array - gb.embeddings.token_embeddings.weight.array)))
    assert bb_diff > 1e-7, "separate heads must differ from the baseline on the backbone, else this check is vacuous"
    try:
        PerLayerQwen3Config(**common, pls_weight=1.0, pls_detach_backbone=True)
        raise AssertionError("pls_detach_backbone without separate heads must be rejected")
    except ValueError:
        pass
    print(f"3d. separate heads, backbone detached: loss/stats == 3c; backbone + main head grads == baseline (3c differs by up to {bb_diff:.2e}); aux head grads == 3c")

    # 7. weight schedule on the traced train step
    def train_step_at(m_, example, k, step):
        token = _TRACED_TRAIN_STEP.set(step)
        try:
            return train_step(m_, example, k)
        finally:
            _TRACED_TRAIN_STEP.reset(token)

    step_at = eqx.filter_jit(train_step_at)
    loss1, stats1, g1 = trained[1.0]
    m_sched = PerLayerQwen3LMHeadModel.init(Vocab, PerLayerQwen3Config(**common, pls_weight=1.0, pls_monitor_stride=S, pls_off_start=2, pls_off_end=4), key=key0)
    ms_sched = hax.shard(m_sched, tc.parameter_axis_mapping)
    for st_, w_ in ((0, 1.0), (2, 1.0), (3, 0.5)):
        l_, s_, g_ = step_at(ms_sched, es, key, jnp.int32(st_))
        assert float(s_["pls/weight"]) == w_, (st_, float(s_["pls/weight"]))

        def ref_w(mm, w_=w_):
            per_ = ref_losses(mm)
            return per_[L - 1] + w_ * sum(per_[: L - 1]), per_

        (rl, rper_w), rg = eqx.filter_value_and_grad(ref_w, has_aux=True)(m_sched)
        np.testing.assert_allclose(float(l_), float(rl), rtol=1e-5)
        close(g_, rg, f"schedule step {st_} grad vs reference at w={w_}")
        for k in range(L):
            np.testing.assert_allclose(float(s_[f"pls/L{k}"]), float(rper_w[k]), rtol=1e-5)
        if w_ == 1.0:
            np.testing.assert_allclose(float(l_), float(loss1), rtol=1e-6)
            close(g_, g1, f"schedule step {st_} grad vs always-on", rtol=1e-5)
    for st_ in (4, 9):
        l_, s_, g_ = step_at(ms_sched, es, key, jnp.int32(st_))
        assert float(s_["pls/weight"]) == 0.0
        np.testing.assert_allclose(float(l_), float(lossb), rtol=1e-6)
        close(g_, gb, f"schedule step {st_} grad vs baseline", rtol=1e-5)
        for k in range(L - 1):
            np.testing.assert_allclose(float(s_[f"pls/L{k}"]), float(rmon[k]), rtol=1e-5)
    m_sw = hax.shard(PerLayerQwen3LMHeadModel.init(Vocab, PerLayerQwen3Config(**common, pls_weight=1.0, pls_monitor_stride=S, pls_off_start=3, pls_off_end=3), key=key0), tc.parameter_axis_mapping)
    l_on, _, _ = step_at(m_sw, es, key, jnp.int32(2))
    l_off, _, _ = step_at(m_sw, es, key, jnp.int32(3))
    np.testing.assert_allclose(float(l_on), float(loss1), rtol=1e-6)
    np.testing.assert_allclose(float(l_off), float(lossb), rtol=1e-6)
    # separate heads switched off: baseline step for the main model, zero grads for the aux heads, monitor through own heads
    m_ss = PerLayerQwen3LMHeadModel.init(Vocab, PerLayerQwen3Config(**common, pls_weight=1.0, pls_monitor_stride=S, pls_separate_heads=True, pls_off_start=1, pls_off_end=1), key=key0)
    l_, s_, g_ = step_at(hax.shard(m_ss, tc.parameter_axis_mapping), es, key, jnp.int32(5))
    np.testing.assert_allclose(float(l_), float(lossb), rtol=1e-6)
    close(eqx.tree_at(lambda m: (m.aux_norms, m.aux_lm_heads), g_, replace=(None, None)), gb, "sep switched off grad vs baseline", rtol=1e-5)
    assert all(float(jnp.abs(x).max()) == 0.0 for x in jax.tree.leaves(eqx.filter((g_.aux_norms, g_.aux_lm_heads), eqx.is_array)))
    xs_ss = ref_raw(m_ss)
    for k in range(L - 1):
        W = m_ss.aux_lm_heads[k].weight.rearrange((m_ss.Embed, Vocab)).array
        wt = jnp.asarray(w_next * stride_mask)
        want = jnp.sum(ref_ce_per_pos(m_ss.aux_norms[k](xs_ss[k]).array, W) * wt) / jnp.sum(wt)
        np.testing.assert_allclose(float(s_[f"pls/L{k}"]), float(want), rtol=1e-5)
    for bad in (dict(pls_weight=0.0, pls_off_end=3), dict(pls_weight=1.0, pls_off_start=4, pls_off_end=3), dict(pls_weight=1.0, pls_off_start=2)):
        try:
            PerLayerQwen3Config(**common, **bad)
            raise AssertionError(f"{bad} must be rejected")
        except ValueError:
            pass
    print("7. schedule: steps before the ramp == always-on step; ramp step == reference at w=0.5; after it loss and all grads == baseline, "
          "train/pls/L{k} == stride monitor; a switch flips at its step; separate heads off: aux grads 0, monitor through own heads; bad schedules rejected")

    # 8. probe: per-loss gradient alignment on the trunk and in-context scores, against the dense reference
    def trunk_vec(g, upto):
        Bn = g.transformer.layers.Block.name
        lay = [a[Bn, : upto + 1].array.ravel() for a in jax.tree.leaves(g.transformer.layers, is_leaf=lambda x: isinstance(x, hax.NamedArray)) if isinstance(a, hax.NamedArray)]
        emb = [a.array.ravel() for a in jax.tree.leaves(g.embeddings, is_leaf=lambda x: isinstance(x, hax.NamedArray)) if isinstance(a, hax.NamedArray)]
        return jnp.concatenate([*emb, *lay]).astype(jnp.float32)

    late_mask = ((np.arange(T) >= T // 2) & ((np.arange(T) - T // 2) % S == 0)).astype(np.float32)[None]
    early_mask = ((np.arange(T) >= 2) & (np.arange(T) < 6)).astype(np.float32)[None]

    def probe_reference(mm, per_layer_hw):
        """per_layer_hw(mm) -> list of (normed hidden, head matrix) per readout, from the no-scan reference."""
        def ce_k(m_, k):
            h, W = per_layer_hw(m_)[k]
            wt = jnp.asarray(w_next)
            return jnp.sum(ref_ce_per_pos(h, W) * wt) / jnp.sum(wt)

        grads = [eqx.filter_grad(lambda m_, k=k: ce_k(m_, k))(mm) for k in range(L)]
        want = {}
        for k in range(L - 1):
            a, b = trunk_vec(grads[k], k), trunk_vec(grads[L - 1], k)
            want[f"pls_probe/cos_L{k}"] = float(a @ b / jnp.sqrt((a @ a) * (b @ b)))
            want[f"pls_probe/gratio_L{k}"] = float(jnp.sqrt((a @ a) / (b @ b)))
        a = sum(trunk_vec(grads[k], L - 1) for k in range(L - 1))
        b = trunk_vec(grads[L - 1], L - 1)
        want["pls_probe/cos_aux"] = float(a @ b / jnp.sqrt((a @ a) * (b @ b)))
        for k, (h, W) in enumerate(per_layer_hw(mm)):
            ce = ref_ce_per_pos(h, W)
            late = jnp.sum(ce * w_next * late_mask) / jnp.sum(w_next * late_mask)
            early = jnp.sum(ce * w_next * early_mask) / jnp.sum(w_next * early_mask)
            want[f"pls_probe/icl_L{k}"] = float(late - early)
        return want

    def shared_hw(m_):
        W = m_.get_lm_head().rearrange((m_.Embed, Vocab)).array
        return [(h, W) for h in ref_hidden(m_)]

    def sep_hw(m_):
        xs_ = ref_raw(m_)
        return [(m_.aux_norms[k](xs_[k]).array, m_.aux_lm_heads[k].weight.rearrange((m_.Embed, Vocab)).array) for k in range(L - 1)] + [
            (m_.transformer.norm(xs_[L - 1]).array, m_.get_lm_head().rearrange((m_.Embed, Vocab)).array)]

    for name, extra, hw, (want_loss, want_grad) in (
        ("w=1", dict(pls_weight=1.0), shared_hw, (loss1, g1)),
        ("w=0", dict(pls_weight=0.0), shared_hw, (lossb, gb)),
        ("sep", dict(pls_weight=1.0, pls_separate_heads=True), sep_hw, (loss_s, g_s)),
    ):
        m_p = PerLayerQwen3LMHeadModel.init(Vocab, PerLayerQwen3Config(**common, pls_monitor_stride=S, pls_probe=True, pls_probe_early=(2, 6), **extra), key=key0)
        l_, s_, g_ = step_fn(hax.shard(m_p, tc.parameter_axis_mapping), es, key)
        np.testing.assert_allclose(float(l_), float(want_loss), rtol=1e-6)
        close(g_, want_grad, f"probe {name}: grads vs the same run without the probe", rtol=1e-5)
        want = probe_reference(m_p, hw)
        got = {k_: float(v) for k_, v in s_.items() if k_.startswith("pls_probe/")}
        assert sorted(got) == sorted(want), (sorted(got), sorted(want))
        for k_ in want:
            np.testing.assert_allclose(got[k_], want[k_], rtol=2e-4, atol=2e-6, err_msg=f"probe {name} {k_}")
        print(f"8. probe {name}: training unchanged; " + " ".join(f"{k_.split('/')[1]}={got[k_]:+.3f}" for k_ in sorted(got)) + " == reference")
    try:
        PerLayerQwen3Config(**common, pls_probe=True, pls_probe_early=(2, 20))
        raise AssertionError("an early window past max_seq_len/2 must be rejected")
    except ValueError:
        pass

    # 4. readout losses per position
    m1 = models[1.0]
    per = eqx.filter_jit(lambda mm, e_: mm.readout_losses(e_, tuple(range(L))))(hax.shard(m1, tc.parameter_axis_mapping), es)
    assert per.axes == (Axis("readout", L), Batch, Pos), per.axes
    W1 = m1.get_lm_head().rearrange((m1.Embed, Vocab)).array
    for k, h in enumerate(ref_hidden(m1)):
        np.testing.assert_allclose(np.asarray(per.array[k]) * w_next, np.asarray(ref_ce_per_pos(h, W1)) * w_next, rtol=1e-5, atol=1e-5)
    ev_pp = eqx.filter_jit(lambda mm, e_: mm.compute_next_token_loss(e_, key=None, reduction=None, reduction_axis=()))(hax.shard(m1, tc.parameter_axis_mapping), es)
    np.testing.assert_allclose(np.asarray(per.array[L - 1]), np.asarray(ev_pp.array), rtol=1e-6, atol=1e-6)
    bitwise = bool(np.array_equal(np.asarray(per.array[L - 1]), np.asarray(ev_pp.array)))
    print(f"4. readout losses == reference per position; readout L-1 == baseline per-position eval loss (bitwise: {bitwise})")

# 5. the readout evaluator against levanter's TaggedEvaluator, one readout at a time
EvalBatch = Axis("batch", 8)
seqs_a = [np.mod(np.arange(T) * (i + 3) + 7 * i, V - 2) + 2 for i in range(12)]
seqs_b = [rng.integers(2, V, size=T) for _ in range(9)]
ds_a = ListAsyncDataset([GrugLmExample.causal(jnp.asarray(s, jnp.int32)) for s in seqs_a])
ds_b = ListAsyncDataset([GrugLmExample.causal(jnp.asarray(s, jnp.int32)) for s in seqs_b])
tagged = [(ds_a, ["paloma/a"]), (ds_b, ["paloma/b"])]
bytes_per_token = jnp.asarray(rng.integers(1, 6, V), jnp.int32)
readouts = tuple(range(L))
with use_test_mesh(tensor_parallelism=1) as mesh:
    amap = {EvalBatch.name: ResourceAxis.DATA}
    rev = ReadoutTaggedEvaluator(readouts, EvalBatch=EvalBatch, tagged_eval_sets=tagged, loss_fn=partial(pls._readout_eval_loss_fn, readouts=readouts, EvalBatch=EvalBatch, mp=None), tokenizer=None, device_mesh=mesh, axis_mapping=amap)
    rev.bytes_per_token = bytes_per_token
    rev.accum_for_batch = rev._make_accum_for_batch()
    results = rev.evaluate_readouts(m1)

    def single(r):
        def loss_fn(model, batch):
            if r == L - 1:
                return _default_lm_eval_loss_fn(model, batch, EvalBatch=EvalBatch, mp=None)  # the main eval's own loss_fn
            losses, w_, ids = pls._readout_eval_loss_fn(model, batch, readouts=(r,), EvalBatch=EvalBatch, mp=None)
            return losses[0], w_, ids

        ev = TaggedEvaluator(EvalBatch=EvalBatch, tagged_eval_sets=tagged, loss_fn=loss_fn, tokenizer=None, device_mesh=mesh, axis_mapping=amap)
        ev.bytes_per_token = bytes_per_token
        ev.accum_for_batch = ev._make_accum_for_batch()
        return ev.evaluate(m1)

    for r in readouts:
        want, got = single(r), results[r]
        for f in ("micro_avg_loss", "macro_avg_loss", "micro_bpb", "macro_bpb"):
            np.testing.assert_allclose(getattr(got, f), getattr(want, f), rtol=1e-6, err_msg=f"readout {r} {f}")
        for f in ("tag_macro_losses", "tag_micro_losses", "tag_macro_bpb", "tag_micro_bpb"):
            gd, wd = getattr(got, f), getattr(want, f)
            assert gd.keys() == wd.keys(), (r, f, gd.keys(), wd.keys())
            for t_ in wd:
                np.testing.assert_allclose(gd[t_], wd[t_], rtol=1e-6, err_msg=f"readout {r} {f} {t_}")
    rev.tokenizer = "stand-in"  # construct_log_dict writes the bpb keys only when a tokenizer is set (training always has one)
    logd = pls.readout_log_dict(rev, results, 1.0)
    for k in readouts:
        for suffix in ("loss", "macro_loss", "bpb", "macro_bpb", "paloma/a/loss", "paloma/b/bpb", "paloma/macro_loss"):
            assert f"eval/L{k}/{suffix}" in logd, (k, suffix, sorted(logd)[:12])
    print(f"5. readout evaluator == TaggedEvaluator per readout (loss, macro, per-tag loss/bpb); readout {L - 1} == main eval; "
          f"eval/L{{k}} keys ok. micro loss by layer {[round(r_.micro_avg_loss, 4) for r_ in results]}")

# 6. flops
c1, c0 = PerLayerQwen3Config(**common, pls_weight=1.0, pls_monitor_stride=S), PerLayerQwen3Config(**common, pls_weight=0.0, pls_monitor_stride=S)
fb = Qwen3Config(**common).flops_per_token(V, T)
assert abs(c1.flops_per_token(V, T) - (fb + (L - 1) * 2 * 48 * V)) < 1e-6
assert abs(c0.flops_per_token(V, T) - (fb + (L - 1) * 2 * 48 * V / (3 * S))) < 1e-6
c_sched = PerLayerQwen3Config(**common, pls_weight=1.0, pls_monitor_stride=S, pls_off_start=2, pls_off_end=4)
assert abs(c_sched.flops_per_token(V, T) - c0.flops_per_token(V, T)) < 1e-6
try:
    PerLayerQwen3Config(**common, scan_layers=False)
    raise AssertionError("scan_layers=False must be rejected")
except ValueError:
    pass
print("6. flops_per_token: + (L-1) lm_head (trained) / + (L-1) lm_head / (3 stride) (monitor); scan_layers=False rejected")
print("ALL PLS CPU TESTS PASSED")
