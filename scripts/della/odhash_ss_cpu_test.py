"""CPU tests for OT-2 + sampled softmax (launcher VARIANT=ovgramss): over_vocab_qwen3 with od_mode="hashed" and the
sampled-softmax schedule on the main head, on a fake 4-device mesh.

Run from the marin repo on a vis node: nice -n 19 .venv/bin/python scripts/della/odhash_ss_cpu_test.py
1. The schedule adds no parameters: the OT-2 + ss model initialises exactly as OT-2.
2. In the full-softmax stage (step past the last stage end) the loss and every gradient equal OT-2's.
3. In each sampled stage, with od_m equal to the positions per device (so the 2-gram head scores every hashed class), the
   loss and every gradient equal a dense reference: the main head's cross-entropy over its per-device candidate sets
   (batch targets plus stride-sweep negatives from the stage's offset, as in the OV + ss test) plus 0.1 x the full
   2-gram cross-entropy; the logged candidate count is the stage's, and the 2-gram loss is the same as OT-2's.
4. Evaluation is the main head's plain cross-entropy.
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
from levanter.trainer import _TRACED_TRAIN_STEP, TrainerConfig
from levanter.utils.mesh import MeshConfig

from experiments.references.over_vocab_qwen3 import ROWS, OverVocabQwen3Config, OverVocabQwen3LMHeadModel
from experiments.references.sampled_softmax_qwen3 import stride_sweep

jax.config.update("jax_threefry_partitionable", True)
assert jax.device_count() == 4
rng = np.random.default_rng(0)

B, T, V = 8, 32, 500
Batch, Pos, Vocab = hax.Axis("batch", B), hax.Axis("position", T), hax.Axis("vocab", V)
EOS = 1
tok = rng.integers(2, V, size=(B, T))
tok[:, 10] = EOS; tok[3, 20] = EOS
eos_mask = np.roll(tok, 1, axis=1) == EOS; eos_mask[:, 0] = False
seg = np.cumsum(eos_mask, axis=1).astype(np.int32)
lw = (np.arange(T) < T - 1).astype(np.float32)[None].repeat(B, 0); lw[2, :5] = 0.0
tokens = hax.named(jnp.asarray(tok, jnp.int32), (Batch, Pos))
ex = LmExample(tokens=tokens, loss_weight=hax.named(jnp.asarray(lw), (Batch, Pos)),
               attn_mask=AttentionMask.causal().with_segment_ids(hax.named(jnp.asarray(seg), (Batch, Pos))))
common = dict(max_seq_len=T, hidden_dim=48, intermediate_dim=64, num_layers=2, num_heads=2, num_kv_heads=2, hybrid_norm=True,
              attn_backend=AttentionBackend.VANILLA)
M = B * T // 4   # positions per device: the 2-gram head's candidate set is every hashed class
CANDS, ENDS = (120, 200), (10, 20)
key = jrandom.PRNGKey(7)
y1, y2 = np.roll(tok, -1, 1), np.roll(tok, -2, 1)
w1 = next_token_loss_weight(Pos, ex.loss_weight).array
tt = np.arange(T)[None]
ok2 = (tt <= T - 3) & (np.roll(seg, -1, 1) == seg) & (np.roll(seg, -2, 1) == seg)
w2 = jnp.asarray(lw * np.roll(lw, -1, 1) * ok2)


def ce_rows(h, W, y, C=None):
    logits = h @ W
    lse = jax.nn.logsumexp(logits if C is None else logits[:, C], axis=-1)
    return lse - jnp.take_along_axis(logits, y[:, None], axis=1)[:, 0]


def np_candidates(labels, P, offset, Vv):
    present = np.zeros(Vv, bool); present[labels] = True
    order = stride_sweep(Vv)[(offset + np.arange(Vv)) % Vv]
    return np.sort(np.concatenate([np.flatnonzero(present), order[~present[order]][: P - present.sum()]]))


def reference(m_, stage):
    """Main head over per-device candidate sets (stage < len(CANDS)) or the full vocabulary, plus 0.1 x full 2-gram CE."""
    k_main, k_s1, _ = jax.random.split(key, 3)
    h = m_.transformer(m_.embed(tokens, ex.attn_mask), attn_mask=ex.attn_mask, key=k_main)
    W1 = m_.get_lm_head().rearrange((m_.config.Embed, Vocab)).array
    off0 = int(jrandom.randint(k_s1, (), 0, V, dtype=jnp.int32))
    num = 0.0
    for d in range(4):   # the batch is split into 4 contiguous blocks, one per device
        r = slice(2 * d, 2 * d + 2)
        yd = y1[r].reshape(-1)
        C = None
        if stage < len(CANDS) and len(np.unique(yd)) <= CANDS[stage]:
            C = np_candidates(yd, CANDS[stage], (off0 + d * (V // 4)) % V, V)
        num = num + jnp.sum(ce_rows(h.array[r].reshape(-1, h.array.shape[-1]), W1, jnp.asarray(yd), C) * w1[r].reshape(-1))
    f1 = num / jnp.sum(w1)
    h2 = (h.array @ m_.od_proj.weight.rearrange(("od_in", "embed")).array).reshape(-1, h.array.shape[-1])
    u = m_.od_out(m_.od_table).rearrange((ROWS, "embed")).array[:M]
    c = (y1 + y2 * V) % M
    f2 = jnp.sum(ce_rows(h2, u.T, jnp.asarray(c.reshape(-1))) * w2.reshape(-1)) / jnp.sum(w2)
    return f1 + 0.1 * f2, (f1, f2)


def close(a, b, what, rtol=2e-4):
    for (p, u), v in zip(jax.tree_util.tree_leaves_with_path(eqx.filter(a, eqx.is_array)), jax.tree.leaves(eqx.filter(b, eqx.is_array))):
        np.testing.assert_allclose(np.asarray(u, np.float32), np.asarray(v, np.float32), rtol=rtol, atol=1e-6, err_msg=f"{what} {jax.tree_util.keystr(p)}")


def train_step(m_, example, step, k):
    token = _TRACED_TRAIN_STEP.set(step)
    try:
        def lf(mm):
            loss, stats = mm.compute_next_token_loss(example, key=k, logsumexp_weight=0.0)
            return (loss.array if isinstance(loss, hax.NamedArray) else loss), stats
        (loss, stats), g = eqx.filter_value_and_grad(lf, has_aux=True)(m_)
    finally:
        _TRACED_TRAIN_STEP.reset(token)
    return loss, {k_: v.value() for k_, v in stats.items()}, g


tc = TrainerConfig(mesh=MeshConfig(axes={"data": -1, "replica": 1, "model": 1}, compute_mapping={
    "token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
    "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
    ROWS: ResourceAxis.DATA}, param_mapping={"embed": "data", ROWS: "data"}))
cfg2 = OverVocabQwen3Config(**common, oe_m=1000, oe_k=2, od_weight=0.1, od_mode="hashed", od_m=M)
cfg2s = OverVocabQwen3Config(**common, oe_m=1000, oe_k=2, od_weight=0.1, od_mode="hashed", od_m=M, ss_candidates=CANDS, ss_stage_ends=ENDS)
m2 = OverVocabQwen3LMHeadModel.init(Vocab, cfg2, key=jrandom.PRNGKey(0))
m2s = OverVocabQwen3LMHeadModel.init(Vocab, cfg2s, key=jrandom.PRNGKey(0))
close(m2s, m2, "init", rtol=0)
print("1. OT-2 + ss initialises exactly as OT-2 (the schedule adds no parameters)")
with tc.use_device_mesh(), hax.axis_mapping(tc.compute_axis_mapping):
    es = hax.shard(ex, tc.compute_axis_mapping)
    s2, s2s = hax.shard(m2, tc.parameter_axis_mapping), hax.shard(m2s, tc.parameter_axis_mapping)
    step_fn = eqx.filter_jit(train_step)
    l_ot2, st_ot2, g_ot2 = step_fn(s2, es, jnp.int32(0), key)
    l_full, _, g_full = step_fn(s2s, es, jnp.int32(25), key)
    np.testing.assert_allclose(float(l_full), float(l_ot2), rtol=1e-6)
    close(g_full, g_ot2, "full-softmax stage vs OT-2", rtol=1e-5)
    (rl, (rf1, rf2)), rg = eqx.filter_value_and_grad(lambda mm: reference(mm, len(CANDS)), has_aux=True)(m2)
    np.testing.assert_allclose(float(l_ot2), float(rl), rtol=1e-5)
    close(g_ot2, rg, "OT-2 vs dense reference")
    print(f"2. step 25 (full softmax): loss {float(l_full):.6f} == OT-2's == dense reference; every gradient matches")
    for st in (0, 15):
        stage = sum(st >= e_ for e_ in ENDS)
        l_s, stats_s, g_s = step_fn(s2s, es, jnp.int32(st), key)
        (rl, (f1, f2)), rg = eqx.filter_value_and_grad(lambda mm: reference(mm, stage), has_aux=True)(m2s)
        np.testing.assert_allclose(float(l_s), float(rl), rtol=1e-5)
        np.testing.assert_allclose(float(stats_s["ntp_loss"]), float(f1), rtol=1e-5)
        np.testing.assert_allclose(float(stats_s["od_loss"]), float(f2), rtol=1e-5)
        np.testing.assert_allclose(float(stats_s["od_loss"]), float(st_ot2["od_loss"]), rtol=1e-6)
        assert float(stats_s["ss/candidates"]) == CANDS[stage], stats_s
        assert float(stats_s["ntp_loss"]) < float(st_ot2["ntp_loss"]), (stats_s["ntp_loss"], st_ot2["ntp_loss"])
        close(g_s, rg, f"stage {stage} grad")
        print(f"3. step {st} (stage {stage}, {CANDS[stage]} candidates): loss {float(l_s):.6f} == sampled main CE {float(f1):.6f} "
              f"+ 0.1 x full 2-gram CE {float(f2):.6f}; every gradient matches; 2-gram loss == OT-2's")
    ev = eqx.filter_jit(lambda mm, e_: mm.compute_next_token_loss(e_, key=None))(s2s, es)
    np.testing.assert_allclose(float(ev.array if isinstance(ev, hax.NamedArray) else ev), float(rf1), rtol=1e-5)
    print("4. evaluation == main-head CE over the full vocabulary")
print("ALL OT-2 + SAMPLED SOFTMAX CPU TESTS PASSED")
