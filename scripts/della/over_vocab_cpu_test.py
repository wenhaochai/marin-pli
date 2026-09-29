"""CPU tests for experiments.references.over_vocab_qwen3 (OV = over-encoding + MTP-DS) on a fake 4-device mesh.

Run from the marin repo on a vis node: nice -n 19 .venv/bin/python scripts/della/over_vocab_cpu_test.py
1. The n-gram index equals the int64 formula (x_t + x_{t-1} V + x_{t-2} V^2) mod m, for the real V and m.
2. Shifted tokens are 0 before the window and across a document boundary.
3. The over-encoded embedding equals a direct computation from the tables and projections.
4. Under the launcher's sharding (table rows over the 4 devices), the training loss equals NTP + w * MTP computed
   independently (dense logits; MTP targets x_{t+2}, masked past the window, across documents and by the example
   weights), every gradient matches, the logged ntp/mtp losses are right, and evaluation is the NTP loss alone.
5. OV + sampled softmax: in the full-softmax stage it equals OV exactly (loss and gradients); in a sampled stage it
   equals the reference with each head's per-device candidate sets.
6. MuonH labels the tables 'adam' (like the token embedding), the projections and the MTP layer 'muonh'.
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
from levanter.optim.muonh import MuonHConfig
from levanter.trainer import _TRACED_TRAIN_STEP, TrainerConfig
from levanter.utils.mesh import MeshConfig

from experiments.references.over_vocab_qwen3 import ROWS, OverVocabQwen3Config, OverVocabQwen3LMHeadModel, ngram_index, shifted_tokens
from experiments.references.sampled_softmax_qwen3 import stride_sweep

jax.config.update("jax_threefry_partitionable", True)
assert jax.device_count() == 4
rng = np.random.default_rng(0)

# 1. hash
for V, m in [(128256, 12_800_000), (128256, 12_800_004), (500, 1000), (500, 97)]:
    z = [rng.integers(0, V, 5000) for _ in range(3)]
    for order in (2, 3):
        want = sum(z[i].astype(np.int64) * V ** i for i in range(order)) % m
        got = np.asarray(ngram_index([jnp.asarray(a, jnp.int32) for a in z[:order]], V, m))
        assert np.array_equal(got, want), (V, m, order)
print("1. n-gram index == int64 formula (V = 128,256, m = 12.8M and 12.8M + 4; small cases)")

B, T, V, M = 8, 32, 500, 1000
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

# 2. shifted tokens
sh = [np.asarray(a.array) for a in shifted_tokens(tokens, hax.named(jnp.asarray(seg), (Batch, Pos)), Pos, 3)]
for s in (1, 2):
    for b in range(B):
        for t in range(T):
            want = tok[b, t - s] if t - s >= 0 and seg[b, t - s] == seg[b, t] else 0
            assert sh[s][b, t] == want
print("2. shifted tokens are 0 before the window and across document boundaries")

common = dict(max_seq_len=T, hidden_dim=48, intermediate_dim=64, num_layers=2, num_heads=2, num_kv_heads=2, hybrid_norm=True, attn_backend=AttentionBackend.VANILLA)
cfg = OverVocabQwen3Config(**common, oe_m=M, oe_k=2, mtp_weight=0.1)   # k = 2: two tables per order, 8 dims each
model = OverVocabQwen3LMHeadModel.init(Vocab, cfg, key=jrandom.PRNGKey(0))
assert len(model.oe_tables) == 4 and cfg.table_dim == 8 and [m for _, m in cfg.moduli()] == [M, M + 4, M + 8, M + 12]


def embed_ref(m_):
    x = np.asarray(m_.embeddings.token_embeddings.weight.array)[tok]
    for t_, ((order, mod), table, proj) in enumerate(zip(cfg.moduli(), m_.oe_tables, m_.oe_proj)):
        idx = sum(sh[i].astype(np.int64) * V ** i for i in range(order)) % mod
        x = x + np.asarray(table.array)[idx] @ np.asarray(proj.weight.rearrange(("oe_dim", "embed")).array)
    x = x / (1 + cfg.k * (cfg.oe_n - 1))
    if m_.embeddings.norm is not None:
        x = np.asarray(m_.embeddings.norm(hax.named(jnp.asarray(x), (Batch, Pos, cfg.Embed))).array)
    return x


e = np.asarray(model.embed(tokens, ex.attn_mask).array)
np.testing.assert_allclose(e, embed_ref(model), rtol=1e-5, atol=1e-6)
print("3. over-encoded embedding == direct computation from tables and projections")


def ce_rows(h, W, y, C=None):
    logits = h @ W
    lse = jax.nn.logsumexp(logits if C is None else logits[:, C], axis=-1)
    return lse - jnp.take_along_axis(logits, y[:, None], axis=1)[:, 0]


def np_candidates(labels, P, offset, Vv):
    present = np.zeros(Vv, bool); present[labels] = True
    order = stride_sweep(Vv)[(offset + np.arange(Vv)) % Vv]
    return np.sort(np.concatenate([np.flatnonzero(present), order[~present[order]][: P - present.sum()]]))


def reference(m_, key, cands=None, stage=None):
    """NTP + w * MTP from dense logits; cands/stage switch on per-device candidate sets (4 contiguous batch blocks)."""
    k_main, k_mtp, k_s1, k_s2 = jax.random.split(key, 4)
    em = m_.embed(tokens, ex.attn_mask)
    h = m_.transformer(em, attn_mask=ex.attn_mask, key=k_main)
    W = m_.get_lm_head().rearrange((cfg.Embed, Vocab)).array
    u = hax.concatenate("mtp_in", [m_.mtp_norm_h(h).rename({"embed": "mtp_in"}), m_.mtp_norm_e(hax.roll(em, -1, Pos)).rename({"embed": "mtp_in"})])
    h2 = m_.mtp_norm_out(m_.mtp_layer(m_.mtp_proj(u), ex.attn_mask, key=k_mtp)).array
    y1, y2 = np.roll(tok, -1, 1), np.roll(tok, -2, 1)
    w1 = next_token_loss_weight(Pos, ex.loss_weight).array
    tt = np.arange(T)[None]
    ok2 = (tt <= T - 3) & (np.roll(seg, -1, 1) == seg) & (np.roll(seg, -2, 1) == seg)
    w2 = jnp.asarray(lw * np.roll(lw, -1, 1) * ok2)
    out = []
    for hh, y, w, k_s in ((h.array, y1, w1, k_s1), (h2, y2, w2, k_s2)):
        num = 0.0
        off0 = int(jrandom.randint(k_s, (), 0, V, dtype=jnp.int32)) if cands else 0
        for d in range(4):
            r = slice(2 * d, 2 * d + 2)
            yd = y[r].reshape(-1)
            C = None
            if cands and stage < len(cands) and len(np.unique(yd)) <= cands[stage]:
                C = np_candidates(yd, cands[stage], (off0 + d * (V // 4)) % V, V)
            num = num + jnp.sum(ce_rows(hh[r].reshape(-1, hh.shape[-1]), W, jnp.asarray(yd), C) * w[r].reshape(-1))
        out.append(num / jnp.sum(w))
    return out[0] + cfg.mtp_weight * out[1], out


tc = TrainerConfig(mesh=MeshConfig(axes={"data": -1, "replica": 1, "model": 1}, compute_mapping={
    "token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
    "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
    ROWS: ResourceAxis.DATA}, param_mapping={"embed": "data", ROWS: "data"}))
key = jrandom.PRNGKey(7)


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


with tc.use_device_mesh(), hax.axis_mapping(tc.compute_axis_mapping):
    ms, es = hax.shard(model, tc.parameter_axis_mapping), hax.shard(ex, tc.compute_axis_mapping)
    assert ms.oe_tables[0].array.sharding.spec[0] == "data", ms.oe_tables[0].array.sharding
    step_fn = eqx.filter_jit(train_step)
    loss, stats, g = step_fn(ms, es, jnp.int32(0), key)
    (ref_l, (ref_ntp, ref_mtp)), ref_g = eqx.filter_value_and_grad(lambda mm: reference(mm, key), has_aux=True)(model)
    np.testing.assert_allclose(float(loss), float(ref_l), rtol=1e-5)
    np.testing.assert_allclose(float(stats["ntp_loss"]), float(ref_ntp), rtol=1e-5)
    np.testing.assert_allclose(float(stats["mtp_loss"]), float(ref_mtp), rtol=1e-5)
    close(g, ref_g, "ov grad")
    assert float(jnp.abs(g.oe_tables[0].array).sum()) > 0 and float(jnp.abs(g.mtp_proj.weight.array).sum()) > 0
    ev = eqx.filter_jit(lambda mm, e_: mm.compute_next_token_loss(e_, key=None))(ms, es)
    np.testing.assert_allclose(float(ev.array if isinstance(ev, hax.NamedArray) else ev), float(ref_ntp), rtol=1e-5)
    print(f"4. sharded OV training loss {float(loss):.6f} == NTP {float(ref_ntp):.6f} + 0.1 x MTP {float(ref_mtp):.6f}; all grads match; eval == NTP alone")

    cands, ends = (120, 200), (10, 20)
    cfg_ss = OverVocabQwen3Config(**common, oe_m=M, oe_k=2, mtp_weight=0.1, ss_candidates=cands, ss_stage_ends=ends)
    model_ss = OverVocabQwen3LMHeadModel.init(Vocab, cfg_ss, key=jrandom.PRNGKey(0))
    close(model_ss, model, "init", rtol=0)
    mss = hax.shard(model_ss, tc.parameter_axis_mapping)
    loss_f, stats_f, g_f = step_fn(mss, es, jnp.int32(25), key)
    np.testing.assert_allclose(float(loss_f), float(loss), rtol=1e-6)
    close(g_f, g, "ovss full-stage vs ov grad", rtol=1e-5)
    for st in (0, 15):
        stage = sum(st >= e_ for e_ in ends)
        loss_s, stats_s, g_s = step_fn(mss, es, jnp.int32(st), key)
        (rl, _), rg = eqx.filter_value_and_grad(lambda mm: reference(mm, key, cands, stage), has_aux=True)(model_ss)
        np.testing.assert_allclose(float(loss_s), float(rl), rtol=1e-5)
        close(g_s, rg, f"ovss stage {stage} grad")
        assert float(stats_s["ss/candidates"]) == cands[stage] and float(stats_s["mtp_ss/candidates"]) == cands[stage], stats_s
    print("5. OV+ss: full-softmax stage == OV (loss, grads); sampled stages == per-head candidate-set reference (loss, grads)")

labels = MuonHConfig().create_mask(model)
assert all(l == "adam" for l in labels.oe_tables)
assert all(l.weight == "muonh" for l in labels.oe_proj) and labels.mtp_proj.weight == "muonh"
assert labels.mtp_layer.self_attn.q_proj.weight == "muonh" and labels.mtp_layer.mlp.gate_proj.weight == "muonh"
assert labels.lm_head.weight == "adamh" if hasattr(labels.lm_head, "weight") else labels.lm_head == "adamh"
print("6. MuonH groups: tables adam (as the token embedding), projections and MTP layer muonh, lm_head adamh")
print("ALL OVER-VOCAB CPU TESTS PASSED")
