"""Replay (CPU, or GPU with JAX_PLATFORMS=cuda, where float32 dots at default precision use TF32) of the first steps of training, on a small model of the real stack, to see the order of events in w959 (hc_res="sum"), which
diverged at step 27 of the real run while its 30-step smoke passed, and as a preflight of hc_res="sumshrink". The model and optimizer are the
real ones: HC, 48 layers, 4 streams, probes run (pls_detach_backbone), DepthArchMuonHConfig with the 130m numbers (lr 0.02, Adam 0.008, no
warmup), compute in bfloat16, gradient clip 1. Smaller: hidden 96, batch 16, 64 tokens, vocabulary 2,048 (real tokens modulo 2,048, so the
uniform loss is 7.62 nats). Tokens: Paloma c4_en validation; every step takes the next windows.
Every EVERY steps, on one batch: CE at layers 2, 12, 24, 48; per layer the mean token norm of the stream sum and of the stream differences
(H_s - mean_s H_s), relative to the embedding; largest |eig| of the static mixing the mode applies.
A small model need not diverge at the same step as the real one. What it can show is whether the mode diverges at all, and what grows first.
    JAX_PLATFORMS=cpu nice -n 19 $PY replay_hc.py OUT.jsonl MODE [MODE ...]   (MODE: learned, identity, sum, scalar, sumshrink, sumshrink:0.5, move, preln, preln-leak)"""
import json
import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS", "0")
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jmp
import numpy as np
import tensorstore as ts

import haliax as hax
from haliax import Axis
from levanter.layers.attention import AttentionBackend, AttentionMask
from levanter.models.lm_model import LmExample

from experiments.references.depth_arch_qwen3 import HC_SUM_GAIN, DepthArchMuonHConfig, _hc_constants
import experiments.references.depth_arch_qwen3 as _dq
MOVE_SHIFT = float(os.environ.get("MOVE_SHIFT", getattr(_dq, "HC_MOVE_SHIFT", 0.0)))   # 0.0 on the code before the audit of 2026-11-03
MOVE_BETA = float(os.environ.get("MOVE_BETA", getattr(_dq, "HC_MOVE_BETA", 20.0)))
_dq.HC_MOVE_SHIFT, _dq.HC_MOVE_BETA = MOVE_SHIFT, MOVE_BETA   # read at trace time, so a sweep needs no code change
from experiments.references.per_layer_qwen3 import PerLayerQwen3Config, PerLayerQwen3LMHeadModel

STEPS, EVERY = int(os.environ.get("STEPS", 120)), int(os.environ.get("EVERY", 5))
B, T, V, D, L, M = 16, 64, 2048, 96, 48, 4
Batch, Pos, Vocab = Axis("batch", B), Axis("position", T), Axis("vocab", V)
VAL = "/scratch/gpfs/GROUP/USER/marin_store_big/tokenized/paloma/c4_en-cf1f79/validation/part-00000-of-00001/input_ids/data"
# 2026-11-03: the Paloma validation array holds 967,618 tokens and then reads as zeros, so a replay past step 944 trained on zeros. TOK_PATH takes a training shard.
ids = np.asarray(ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": os.environ.get("TOK_PATH", VAL)}}).result()[: B * T * (STEPS + 1)].read().result(), np.int64)
assert (ids == 0).mean() < 0.01, f"{(ids == 0).mean():.3f} of the tokens are zero: past the end of the array"
print("tokens", ids.size, flush=True)
mp = jmp.get_policy("p=f32,c=" + os.environ.get("REPLAY_DTYPE", "bfloat16"))   # float32 here: the whole forward and the stored state in float32
opt_cfg = DepthArchMuonHConfig(learning_rate=0.02, adam_lr=0.008, beta1=0.9, beta2=0.98, epsilon=1e-20, momentum=0.95, nesterov=True, backend_steps=5, muon_epsilon=1e-5, max_grad_norm=1.0,
                               weight_decay=0.1, lr_schedule="linear", decay=0.8, warmup=0, min_lr_ratio=0.0, coefficient_type="simple")
out_f = open(sys.argv[1], "a")


def batch(step):
    t = ids[(step * B * T) % max(ids.size - B * T, 1):][: B * T]
    tok = t.reshape(B, T)
    seg = hax.named(jnp.asarray(np.cumsum(np.roll(tok, 1, axis=1) == 128001, axis=1), jnp.int32), (Batch, Pos))
    mask = AttentionMask.causal().with_segment_ids(seg)
    ex = LmExample(tokens=hax.named(jnp.asarray(tok % V, jnp.int32), (Batch, Pos)), loss_weight=hax.named(jnp.ones((B, T), jnp.float32).at[:, -1].set(0), (Batch, Pos)), attn_mask=mask)
    return ex, mask


for mode in sys.argv[2:]:
    preln = mode.startswith("preln")
    state_f32 = mode == "sumf32"   # the streams in float32, the rest in the compute dtype, as in the real run (arch3-hcsumf32)
    hc_mode, _, power = ("sum", "", "") if state_f32 else mode.partition(":")
    extra = dict(depth_arch="preln", depth_preln_leak=mode == "preln-leak") if preln else dict(depth_arch="hc", hc_streams=M, depth_hc_res=hc_mode, depth_hc_state_f32=state_f32, **({"depth_shrink_power": float(power)} if power else {}))
    cfg = PerLayerQwen3Config(max_seq_len=T, hidden_dim=D, intermediate_dim=int(3.5 * D), num_layers=L, num_heads=4, num_kv_heads=4, attn_backend=AttentionBackend.VANILLA, pls_weight=1.0, pls_separate_heads=True,
                              pls_detach_backbone=True, pls_serial_readouts=True, pls_fold_norm=True, hybrid_norm=True, scan_layers=False, use_bias=False, gradient_checkpointing=False, **extra)
    model = PerLayerQwen3LMHeadModel.init(Vocab, cfg, key=jrandom.PRNGKey(0))
    opt = opt_cfg.build(4959)
    params = eqx.filter(model, eqx.is_inexact_array)
    state = opt.init(params)

    def loss_fn(m_, ex, key):
        loss, stats = m_.compute_next_token_loss(ex, key=key, logsumexp_weight=0.0)
        loss = loss.array if isinstance(loss, hax.NamedArray) else loss
        return loss, {k: v.value() if hasattr(v, "value") else v for k, v in stats.items()}

    @eqx.filter_jit
    def train_step(m_, st, ex, key):
        (loss, stats), g = eqx.filter_value_and_grad(loss_fn, has_aux=True)(mp.cast_to_compute(m_), ex, key)
        g = mp.cast_to_param(g)
        p = eqx.filter(m_, eqx.is_inexact_array)
        up, st = opt.update(g, st, params=p)
        return eqx.apply_updates(m_, up), st, loss, stats, jnp.sqrt(sum(jnp.sum(jnp.square(a.astype(jnp.float32))) for a in jax.tree_util.tree_leaves(g)))

    @eqx.filter_jit
    def probe(m_, ex, mask):
        mc = mp.cast_to_compute(m_)
        tr = mc.transformer
        x = mc.embeddings.embed(ex.tokens)
        H = x if preln else hax.stack(Axis("stream", M), [x] * M)
        if state_f32:
            H = H.astype(jnp.float32)
        e = jnp.sqrt(jnp.sum(jnp.square(x.array.astype(jnp.float32)), axis=-1))
        sums, difs = [], []
        for i in range(L):
            if preln:
                H = tr.layer(i).residual_step(H, mask, key=None, pos_ids=None, layer=i).rearrange(H.axes)
                a = H.array.astype(jnp.float32)
                sums.append(jnp.mean(jnp.sqrt(jnp.sum(jnp.square(a), axis=-1)) / e))
                difs.append(jnp.zeros(()))
                continue
            H = tr.layer(i).hyper_step(H, mask, key=None, pos_ids=None, layer=i).rearrange(H.axes)
            a = H.array.astype(jnp.float32)  # (stream, batch, position, embed)
            s = jnp.sum(a, axis=0)
            d = a - jnp.mean(a, axis=0, keepdims=True)
            sums.append(jnp.mean(jnp.sqrt(jnp.sum(jnp.square(s), axis=-1)) / (M * e)))
            difs.append(jnp.mean(jnp.sqrt(jnp.sum(jnp.square(d), axis=(0, -1))) / (M * e)))
        return jnp.stack(sums), jnp.stack(difs)

    def max_eig(m_):
        rows = []
        if preln:
            return [(0, 0, 1.0, 0.0, 0.0, 0.0)]
        for j in (0, 1):
            p = m_.transformer.layers.stacked.hyper[j]
            ax = [a.name for a in p.res.axes if a.name not in ("stream", "stream_out")][0]
            res = np.asarray(p.res.rearrange((ax, "stream", "stream_out")).array, np.float64)
            sc = np.asarray(p.scale.rearrange((ax, "hp_scale")).array, np.float64)
            for l in range(L):
                A = np.eye(M) + res[l]
                if hc_mode == "sum":
                    A = A - (A.sum(1, keepdims=True) - 1) / M
                elif mode == "identity":
                    A = np.eye(M)
                elif mode == "scalar":
                    A = np.eye(M) * (np.trace(A) / M)
                elif hc_mode == "sumshrink":
                    A = np.eye(M) - (1 - HC_SUM_GAIN[2 * l + j] ** (float(power) if power else 1.0)) / M
                elif mode == "move":
                    A = np.logaddexp(0.0, MOVE_BETA * (np.eye(M) + res[l] - MOVE_SHIFT)) / MOVE_BETA + 1e-6 * np.eye(M)
                    A = A / A.sum(1, keepdims=True)
                off = A - np.diag(np.diag(A))
                rows.append((l, j, float(np.abs(np.linalg.eigvals(A)).max()), float(np.abs(sc[l, 0])), float(np.maximum(off, 0).sum() / M), float(-np.minimum(off, 0).sum() / M)))
        return rows

    print(f"== {mode}", flush=True)
    t0 = time.time()
    key = jrandom.PRNGKey(1)
    for step in range(STEPS + 1):
        ex, mask = batch(step)
        key, k = jrandom.split(key)
        model, state, loss, stats, gn = train_step(model, state, ex, k)
        if step % EVERY == 0 or step == STEPS:
            s, d = probe(model, ex, mask)
            s, d = np.asarray(s, np.float64), np.asarray(d, np.float64)
            ce = {l: float(stats[f"pls/L{l - 1}"]) for l in (2, 12, 24, 48)}
            eig = max_eig(model)
            e = np.array([r[2] for r in eig])
            rr = np.array([(r[0], r[4], r[5]) for r in eig])
            grp = {f"{a}-{b}": (float(rr[(rr[:, 0] >= a - 1) & (rr[:, 0] <= b - 1), 1].mean()), float(rr[(rr[:, 0] >= a - 1) & (rr[:, 0] <= b - 1), 2].mean())) for a, b in ((1, 12), (13, 24), (25, 36), (37, 48))}
            raw = np.concatenate([np.asarray(model.transformer.layers.stacked.hyper[j].res.array, np.float64).reshape(-1, M, M) for j in (0, 1)])
            off_raw = raw[:, ~np.eye(M, dtype=bool)]
            rec = dict(mode=mode, beta=MOVE_BETA, shift=MOVE_SHIFT, off_groups=grp, res_off_mean=float(off_raw.mean()), res_off_std=float(off_raw.std()), res_off_lowfrac=float((off_raw < MOVE_SHIFT - 14.5 / MOVE_BETA).mean()), res_off_abovefrac=float((off_raw > MOVE_SHIFT).mean()), step=step, loss=float(loss), grad=float(gn), ce=ce, sum=s.tolist(), dif=d.tolist(), log10_prod_eig=float(np.log10(e).sum()), max_eig=float(e.max()), s_rr=float(np.mean([r[3] for r in eig])), offdiag=float(np.mean([r[4] for r in eig])), offneg=float(np.mean([r[5] for r in eig])))
            out_f.write(json.dumps(rec) + "\n")
            out_f.flush()
            print(f"{step:4d} {time.time() - t0:6.0f}s loss {float(loss):8.2f} grad {float(gn):8.2f} CE L2 {ce[2]:6.2f} L12 {ce[12]:6.2f} L24 {ce[24]:6.2f} L48 {ce[48]:6.2f}  |sum| L12 {s[11]:9.2e} L48 {s[47]:9.2e}  |dif| L12 {d[11]:9.2e} L48 {d[47]:9.2e}  log10 prod eig {rec['log10_prod_eig']:+6.2f} max {rec['max_eig']:.2f} |s_rr| {rec['s_rr']:.3f} offdiag + {rec['offdiag']:.3f} - {rec['offneg']:.3f}", flush=True)
