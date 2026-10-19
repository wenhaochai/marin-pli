"""Internal-medicine check-up of the 48-layer DepthBench checkpoints (owner popup 2026-10-19, "按方案做"): the 10 residual
designs (depth_arch_qwen3), each as an ordinary model ("ord": backbone trained as usual, detached probes) and as a
separate-heads model ("sep"). Read-only: loads a finished checkpoint, trains nothing. Float32, matmul precision highest.

Text: Paloma c4_en validation, the first B x T tokens (16 x 1,024; skip runs: the first 8 sequences), documents isolated
by segment ids (attnres_checkup.make_batch). CE and DLA: mean over valid next-token targets (the last position and
positions whose next token starts a new document are left out). Write ratios, block_imp: mean over all tokens.

"The state after layer k" (depth_arch_qwen3 docstring; what the per-layer heads and the final norm read): h for the
residual designs (sandwich = the baseline's Stacked transformer with hybrid_norm, preln, lns, deepnorm, keel) and moda;
the sum of the m streams for hc and mhc; for attnres and attnres_block the mix the next layer's attention reads (after the
last layer the head's mix). The state before layer 1: the embedding x (hc/mhc: m x, the sum of the copied streams).

Per layer k = 1..48:
 1 skip_ce   final CE (the model's own norm + lm_head) with layer k "adding nothing" (8 x 1,024 tokens; base_ce without):
             residual designs and moda: both branch outputs (the writes, below) are zero. For deepnorm and keel the
             post-norms stay, so the layer still applies h <- N(a h) twice (a re-gain; deepnorm's LN(a h) also re-centres);
             skip_ce_bypass (deepnorm, keel only) passes h through the layer unchanged instead.
    skip_ce_delete (--delete, added afterwards to the same JSONs): layer k deleted, the model as close to the 47-layer model as
             the trained weights allow; DELETE_NOTE says per design what that is and when it equals skip_ce (preln, sandwich,
             moda, attnres) or skip_ce_bypass (deepnorm, keel); hc/mhc also skip_ce_delete_reindex.
             moda: zero writes, and the layer's two depth key/value slots (its attention's and its MLP input's) are
             left out of every later layer's depth softmax (as if the layer did not exist; masked like unfilled slots).
             hc, mhc: the layer's two branch writes w (x) f are zero; its read, stream mixing A and write weights still run.
             attnres: the layer's two source slots are never filled, so every later mix (and the head's) leaves them out.
             attnres_block: the two outputs are not added to their block's running sum (if the layer opens its block,
             the block's slot opens at the next layer, as the model's code does when a block has no running sum yet).
 2 attn_write_ratio, mlp_write_ratio: per token RMS(f) / RMS(u), mean over tokens; u = the state the branch read before
             its own pre-norm (residual designs and moda: h; deepnorm's branch has no pre-norm and reads h; hc/mhc: the
             read x = sum_s r_s H_s; attnres*: the branch's input mix). f = the branch output as written to the stream:
             sandwich includes its output RMSNorm (RN_out(F)); deepnorm and keel: F before the gain a and the post-norm;
             hc/mhc: f before the write weights; attnres*: the source value.
    block_imp = 1 - cos(state before layer k, state after layer k), mean over tokens.
 3 attn_dla, mlp_dla: s * (g_f . f) . (W_U[y] - mean_v W_U[v]), s = 1 / RMS(final pre-norm state) (with the final norm's
             eps), mean over targets: f placed directly into the final pre-norm state, no weighting.
    attn_dla_w, mlp_dla_w: the exact direct (linear) path of f to the final pre-norm state, with every norm scale,
             mixing and write weight frozen at its actual per-token value: equal to *_dla for sandwich, preln, lns, moda;
             deepnorm/keel: through the later gains a and post-norms (each frozen as g . P z / sigma(z), P the mean-centring
             for deepnorm's LayerNorm, identity for keel's RMSNorm); hc/mhc: (sum_o w_o kappa_o) * f, kappa the per-stream
             coefficient carried back from the read-out sum through the later mixings A; attnres: the head's mix weight
             on f's slot (attnres_block: on its block's slot). Top level: emb_dla_w (the embedding's direct path) and
             total_logit (s (g_f . h_T) . (W_U[y] - mean W_U)); emb_dla_w + sum of all *_dla_w = total_logit exactly
             (closure_err reports the difference).
 4 lens_ce: CE of the state through the final norm and lm_head (logit lens); own_ce: through the layer's own aux_norms[k]
             and aux_lm_heads[k] (probe or separate head) as training reads it; layer 48: the model's own head (= lens_ce).
 5 Over the state of 2,048 tokens (a fixed random sample, seed 0, of the 16 x 1,024 positions): eff_rank = exp(entropy of
             the normalised singular values) of the uncentred 2,048 x 512 matrix; aniso = mean cosine over all pairs of the
             2,048 token states. Over all 16 x 1,024 tokens: massive = max |activation| / RMS of all activations;
             massive_dim = the dimension holding the largest |activation|.
Checks: ref_ce = the model's own forward (PerLayerQwen3LMHeadModel.layer_outputs + final norm + lm_head) on the skip
tokens; base_ce must match it (|diff| reported as ref_diff; > 2e-3 prints WARN).

    source scripts/della/pls_env.sh
    $PY scripts/della/depthmed.py OUT_DIR CFG_DIR NAME [NAME ...] [--skip-layers 1,48] [--b 16] [--b-skip 8] [--t 1024]
NAME: a checkpoint dir under marin_store_big/speedrun; CFG_DIR/NAME.json holds {"model": W&B model config}.
Writes OUT_DIR/<arch>-<ord|sep>.json (arch "sandwich" for the baseline).
"""
import argparse
import dataclasses
import json
import math
import os
import sys
import time
import traceback

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

import haliax as hax
from haliax import Axis
from haliax.partitioning import ResourceAxis
from levanter.checkpoint import load_checkpoint
from levanter.layers.attention import AttentionBackend, AttentionMask
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig

from experiments.references.depth_arch_qwen3 import _hc_constants, _mhc_constants, _moda_attention, _rms, _sinkhorn_liger
from experiments.references.per_layer_qwen3 import PerLayerQwen3Config, PerLayerQwen3LMHeadModel

jax.config.update("jax_default_matmul_precision", "highest")

ap = argparse.ArgumentParser()
ap.add_argument("out_dir")
ap.add_argument("cfg_dir")
ap.add_argument("names", nargs="+")
ap.add_argument("--skip-layers", default="", help="comma-separated 1-based layers for measure 1 (default: all)")
ap.add_argument("--b", type=int, default=16)
ap.add_argument("--b-skip", type=int, default=8)
ap.add_argument("--t", type=int, default=1024)
ap.add_argument("--n-sample", type=int, default=2048)
ap.add_argument("--delete", action="store_true", help="only add skip_ce_delete to the existing OUT_DIR JSONs (see DELETE_NOTE)")
args = ap.parse_args()

ROOT = "/scratch/gpfs/GROUP/USER/marin_store_big/speedrun"
VAL = "/scratch/gpfs/GROUP/USER/marin_store_big/tokenized/paloma/c4_en-cf1f79/validation/part-00000-of-00001/input_ids/data"
Vocab = Axis("vocab", 128256)
T, B, B_SKIP = args.t, args.b, args.b_skip
assert B_SKIP <= B

tc = TrainerConfig(mesh=MeshConfig(axes={"data": -1, "replica": 1, "model": 1}, compute_mapping={
    "token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
    "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA)}))
tc.use_device_mesh().__enter__()
hax.axis_mapping(tc.parameter_axis_mapping).__enter__()
print(f"devices: {jax.devices()}", flush=True)

import tensorstore as ts  # noqa: E402

TOK = np.asarray(ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": VAL}}).result()[: B * T].read().result(), np.int64).reshape(B, T)


def make_batch(tok):
    b = tok.shape[0]
    eos = np.roll(tok, 1, axis=1) == 128001
    eos[:, 0] = False
    seg = np.cumsum(eos, axis=1)
    Batch, Pos = Axis("batch", b), Axis("position", T)
    mask = AttentionMask.causal().with_segment_ids(hax.named(jnp.asarray(seg, jnp.int32), (Batch, Pos)))
    tgt = np.roll(tok, -1, axis=1)
    valid = np.ones((b, T), bool)
    valid[:, -1] = False
    valid &= np.roll(seg, -1, axis=1) == seg
    return dict(tok=tok, b=b, Batch=Batch, Pos=Pos, mask=mask, tgt=jnp.asarray(tgt, jnp.int32), valid=jnp.asarray(valid, jnp.float32),
                n_valid=int(valid.sum()), ids=hax.named(jnp.asarray(tok, jnp.int32), (Batch, Pos)))


BT_REC, BT_SKIP = make_batch(TOK), make_batch(TOK[:B_SKIP])
N_SAMPLE = min(args.n_sample, B * T)
SAMPLE = jnp.asarray(np.sort(np.random.default_rng(0).choice(B * T, N_SAMPLE, replace=False)))


# ---- shared jitted pieces ------------------------------------------------------------------------------------------
@eqx.filter_jit
def ce_of(h, norm, W, tgt, valid):
    """Mean next-token CE of h (b, T, E) through norm and the head W (V, E), one sequence at a time."""
    Embed = norm.axis
    Pos1 = Axis("position", h.shape[1])

    def one(a):
        hb, tb, vb = a
        z = norm(hax.named(hb, (Pos1, Embed))).array
        logits = z @ W.T
        lse = jax.nn.logsumexp(logits, -1)
        ll = jnp.take_along_axis(logits, tb[:, None], -1)[:, 0]
        return jnp.sum((lse - ll) * vb)

    return jnp.sum(jax.lax.map(one, (h, tgt, valid))) / jnp.sum(valid)


@jax.jit
def state_stats(prev, S, sample):
    """block_imp, eff_rank, aniso, massive, massive_dim of the state S (b, T, E); prev: the state before the layer."""
    E = S.shape[-1]
    cos = jnp.sum(prev * S, -1) / (jnp.linalg.norm(prev, axis=-1) * jnp.linalg.norm(S, axis=-1) + 1e-30)
    a = jnp.abs(S).reshape(-1, E)
    per_dim = jnp.max(a, 0)
    massive = jnp.max(per_dim) / jnp.sqrt(jnp.mean(S * S))
    X = S.reshape(-1, E)[sample]
    sv = jnp.linalg.svd(X, compute_uv=False)
    p = sv / jnp.sum(sv)
    erank = jnp.exp(-jnp.sum(jnp.where(p > 0, p * jnp.log(jnp.where(p > 0, p, 1.0)), 0.0)))
    U = X / (jnp.linalg.norm(X, axis=-1, keepdims=True) + 1e-30)
    n = U.shape[0]
    g = jnp.sum(U.sum(0) ** 2)  # sum of all pairwise cosines incl. the diagonal
    aniso = (g - jnp.sum(U * U)) / (n * (n - 1))
    return 1.0 - jnp.mean(cos), erank, aniso, massive, jnp.argmax(per_dim)


@jax.jit
def wratio(f, u):
    return jnp.mean(jnp.sqrt(jnp.mean(f * f, -1)) / (jnp.sqrt(jnp.mean(u * u, -1)) + 1e-30))


@jax.jit
def gdot(gv, f):
    return jnp.sum(gv * f, -1)


@jax.jit
def gdot_m(gv, f, M):
    return jnp.sum(gv * (f @ M.T), -1)


# ---- the designs ----------------------------------------------------------------------------------------------------
class Design:
    """Per design: init(x) -> carry, before(carry) -> state before layer 1, layer(i, carry, keep, rec, skip_mode) ->
    (carry, rec dict or None), state(carry, i) -> state after layer i, final(carry) -> final pre-norm state, and
    dla_w(...) for the weighted direct paths. Arrays are raw (b, T, E) float32 unless named."""

    def __init__(self, m, cfg, bt):
        self.m, self.cfg, self.bt = m, cfg, bt
        self.tr = m.transformer
        self.Embed = cfg.Embed
        self.L = cfg.num_layers
        self.arch = cfg.depth_arch

    def N(self, a):
        return hax.named(a, (self.bt["Batch"], self.bt["Pos"], self.Embed))

    def A(self, n):
        return n.rearrange((self.bt["Batch"], self.bt["Pos"], self.Embed)).array


class Residual(Design):
    """sandwich (baseline), preln, lns, deepnorm, keel: h_t = N_t(a_t h_{t-1} + W_t) per sublayer t."""

    def __init__(self, m, cfg, bt):
        super().__init__(m, cfg, bt)
        L, arch = self.L, self.arch
        if arch == "baseline":
            Blk = self.tr.layers.Block.name
            self.layers = [hax.tree_util.tree_map(lambda a, i=i: a[Blk, i], self.tr.layers.stacked) for i in range(L)]
        else:
            self.layers = [self.tr.layer(i) for i in range(L)]
        # per sublayer t = 1..2L: (a_t, norm kind) -- for the direct-path propagation
        self.sub = []
        for i in range(L):
            for j in range(2):
                if arch == "deepnorm":
                    self.sub.append(((2 * L) ** 0.25, "ln"))
                elif arch == "keel":
                    if i == 0 and j == 0:
                        self.sub.append((1.0, None))
                    else:
                        self.sub.append((1.0 if i == 0 else float(2 * L), "rn"))
                else:
                    self.sub.append((1.0, None))
        self.has_norm = arch in ("deepnorm", "keel")
        if self.has_norm:  # M_t = L_T ... L_t (token-independent part), t = 1..2L; M_0 = M_1
            d = self.Embed.size
            P = np.eye(d) - np.ones((d, d)) / d
            mats = [None] * (2 * L + 1)
            M = np.eye(d)
            for t in range(2 * L, 0, -1):
                i, j = (t - 1) // 2, (t - 1) % 2
                kind = self.sub[t - 1][1]
                if kind is not None:
                    post = self.layers[i].post_1 if j == 0 else self.layers[i].post_2
                    g = np.asarray(post.weight.array, np.float64)
                    Lt = np.diag(g) @ P if kind == "ln" else np.diag(g)
                    M = M @ Lt
                mats[t] = jnp.asarray(M, jnp.float32)
            mats[0] = mats[1]
            self.mats = mats
        bt_ = bt
        mask = bt_["mask"]
        N, A_ = self.N, self.A
        L2 = 2 * L

        @eqx.filter_jit
        def sublayer(layer, j, h, keep, i, first, rec, gv, M):
            if arch == "baseline":
                if j == 0:
                    F = layer.post_attn_layernorm(layer.self_attn(x=layer.input_layernorm(N(h)), mask=mask, key=None, pos_ids=None))
                else:
                    F = layer.post_mlp_layernorm(layer.mlp(layer.post_attention_layernorm(N(h)), key=None))
            else:
                F = layer.branch(j, N(h), mask, key=None, pos_ids=None, layer=i)
            W = A_(F)
            logsig = None
            if arch == "deepnorm" or (arch == "keel" and not (first and j == 0)):
                a = (L2 ** 0.25) if arch == "deepnorm" else (1.0 if first else float(L2))
                post = layer.post_1 if j == 0 else layer.post_2
                z = a * h + keep * W
                hn = A_(post(N(z)))
                if arch == "deepnorm":
                    var = jnp.var(z, -1)
                else:
                    var = jnp.mean(z * z, -1)
                logsig = 0.5 * jnp.log(var + post.eps)
            else:
                hn = h + keep * W
            if not rec:
                return hn, None
            r = jnp.sum(gv * W, -1)
            out = dict(ratio=wratio(W, h), r=r, rw=r if M is None else jnp.sum(gv * (W @ M.T), -1))
            if logsig is not None:
                out["logsig"] = logsig
            return hn, out

        self._sublayer = sublayer

    def init(self, x):
        return x

    def before(self, carry):
        return carry

    def layer(self, i, h, keep, rec, gv=None, pos=None):
        pos = i if pos is None else pos  # the layer's position (lns's 1 / sqrt(pos + 1)); the deletion runs renumber it
        recs = []
        for j in range(2):
            t = 2 * i + j + 1
            M = self.mats[t] if (rec and self.has_norm) else None
            h, r = self._sublayer(self.layers[i], j, h, jnp.float32(keep), jnp.int32(pos), self.arch == "keel" and i == 0, rec, gv, M)
            recs.append(r)
        return h, recs

    def state(self, h, i):
        return h

    def final(self, h):
        return h

    def bypass(self, i, h):
        return h

    def direct(self, recs_all, x, gv):
        """Per sublayer (b, T) direct-path dot products (before the factor s) and the embedding's."""
        L2 = 2 * self.L
        if not self.has_norm:
            return [r["rw"] for r in recs_all], gdot(gv, x)
        loga = np.array([0.0] + [math.log(a) for a, _ in self.sub])  # index t = 0..2L
        logsig = [jnp.zeros_like(recs_all[0]["r"])] + [r.get("logsig", jnp.zeros_like(r["r"])) for r in recs_all]
        # coeff_t = exp(sum_{t' > t} log a_t' - sum_{t' >= t} log sigma_t')
        out = [None] * (L2 + 1)
        acc = jnp.zeros_like(recs_all[0]["r"])
        suma = 0.0
        for t in range(L2, -1, -1):
            acc = acc + logsig[t]
            coeff = jnp.exp(suma - acc)
            if t >= 1:
                out[t] = coeff * recs_all[t - 1]["rw"]
            else:
                out[0] = coeff * gdot_m(gv, x, self.mats[0])
            suma += loga[t]
        return out[1:], out[0]


class Moda(Design):
    def __init__(self, m, cfg, bt):
        super().__init__(m, cfg, bt)
        L = self.L
        self.layers = [self.tr.layer(i) for i in range(L)]
        acfg = cfg.attention_config()
        self.Depth = Axis("depth", 2 * L)
        self.acfg = acfg
        mask = bt["mask"]
        N, A_ = self.N, self.A
        chunk = cfg.moda_chunk

        @eqx.filter_jit
        def layer_fn(layer, h, dk, dv, valid, n, keep, rec, gv):
            u = h
            o, ka, va = _moda_attention(layer.self_attn, layer.ln_1(N(h)), mask, (dk, dv, valid), pos_ids=None, chunk=chunk)
            Wa = A_(o)
            h1 = h + keep * Wa
            xf = layer.ln_2(N(h1))
            kf, vf = layer.ffn_depth_kv(xf)
            Wm = A_(layer.mlp(xf, key=None))
            h2 = h1 + keep * Wm
            dk = dk.at["depth", 2 * n].set(ka.astype(dk.dtype)).at["depth", 2 * n + 1].set(kf.astype(dk.dtype))
            dv = dv.at["depth", 2 * n].set(va.astype(dv.dtype)).at["depth", 2 * n + 1].set(vf.astype(dv.dtype))
            if not rec:
                return h2, dk, dv, None
            ra, rm = jnp.sum(gv * Wa, -1), jnp.sum(gv * Wm, -1)
            return h2, dk, dv, [dict(ratio=wratio(Wa, u), r=ra, rw=ra), dict(ratio=wratio(Wm, h1), r=rm, rw=rm)]

        self._layer = layer_fn

    def init(self, x):
        xn = self.N(x)
        buf = hax.zeros(tuple(a for a in xn.axes if a.name != self.Embed.name) + (self.acfg.KVHeads, self.Depth, self.acfg.HeadSize), dtype=jnp.float32)
        self.removed = np.zeros(2 * self.L, bool)
        return (x, buf, buf)

    def before(self, carry):
        return carry[0]

    def layer(self, i, carry, keep, rec, gv=None):
        h, dk, dv = carry
        valid = (np.arange(2 * self.L) < 2 * i) & ~self.removed
        h, dk, dv, r = self._layer(self.layers[i], h, dk, dv, hax.named(jnp.asarray(valid), self.Depth), jnp.int32(i), jnp.float32(keep), rec, gv)
        if keep == 0.0:
            self.removed[2 * i: 2 * i + 2] = True
        return (h, dk, dv), (r if r is not None else [None, None])

    def state(self, carry, i):
        return carry[0]

    def final(self, carry):
        return carry[0]

    def direct(self, recs_all, x, gv):
        return [r["rw"] for r in recs_all], gdot(gv, x)


class Hyper(Design):
    """hc, mhc: H (stream, batch, position, embed); the read-out state is sum_s H_s."""

    def __init__(self, m, cfg, bt):
        super().__init__(m, cfg, bt)
        L, arch = self.L, self.arch
        self.layers = [self.tr.layer(i) for i in range(L)]
        self.mm = cfg.hc_streams
        mask = bt["mask"]
        A_ = self.A
        Embed = self.Embed
        Batch, Pos = bt["Batch"], bt["Pos"]

        @eqx.filter_jit
        def sub(layer, j, H, li, keep, rec, gv):
            S = H.resolve_axis("stream")
            mm, So = S.size, Axis("stream_out", S.size)
            f32 = lambda a: a.astype(jnp.float32)  # noqa: E731
            p = layer.hyper[j]
            if arch == "hc":
                c_read, c_res, c_write = _hc_constants(mm, 2 * li + j)
                s_rr, s_w = f32(p.scale)["hp_scale", 0], f32(p.scale)["hp_scale", 1]
                Hn = f32(_rms(H, Embed))
                read = c_read + f32(p.read) + s_rr * hax.tanh(hax.dot(Hn, f32(p.w_read), axis=Embed))
                Amat = c_res + f32(p.res) + s_rr * hax.tanh(hax.dot(Hn, f32(p.w_res), axis=Embed))  # (in = stream, out = stream_out)
                write = c_write + f32(p.write) + s_w * hax.tanh(hax.dot(Hn, f32(p.w_write), axis=Embed).rename({"stream": "stream_out"}))
                x = hax.dot(read, H, axis="stream")
                f = layer.branch(j, x, mask, key=None, pos_ids=None)
                H2 = (hax.dot(Amat, H, axis="stream") + keep * write * f.broadcast_axis(So)).rename({"stream_out": "stream"})
            else:
                Hf = f32(H)
                inv = hax.rsqrt(hax.mean(Hf * Hf, axis=(S, Embed)) + 1e-6)
                mix = hax.dot(H, f32(p.phi), axis=(S, Embed)) * inv
                g = f32(p.gain)
                cc = _mhc_constants(mm, 2 * li + j) + f32(p.bias)
                pre = mix["mhc", 0:mm] * g["hp_scale", 0] + cc["mhc", 0:mm]
                post = mix["mhc", mm: 2 * mm] * g["hp_scale", 1] + cc["mhc", mm: 2 * mm]
                resl = mix["mhc", 2 * mm:] * g["hp_scale", 2] + cc["mhc", 2 * mm:]
                read = hax.nn.sigmoid(pre.rename({"mhc": "stream"}))
                write = 2.0 * hax.nn.sigmoid(post.rename({"mhc": "stream_out"}))
                Amat = _sinkhorn_liger(resl.unflatten_axis("mhc", (So, S)))  # (out, in)
                x = hax.dot(read, Hf, axis=S)
                f = layer.branch(j, x, mask, key=None, pos_ids=None)
                H2 = (hax.dot(Amat, Hf, axis=S) + keep * write * f32(f).broadcast_axis(So)).rename({"stream_out": "stream"})
            H2 = H2.rearrange(H.axes)
            if not rec:
                return H2, None
            fa, xa = A_(f), A_(x)
            r = jnp.sum(gv * fa, -1)
            return H2, dict(ratio=wratio(fa, xa), r=r, w=write.rearrange((Batch, Pos, "stream_out")).array,
                            A=Amat.rearrange((Batch, Pos, "stream", "stream_out")).array)

        self._sub = sub

    def init(self, x):
        xn = self.N(x)
        return hax.stack(Axis("stream", self.mm), [xn] * self.mm)

    def before(self, H):
        return self.A(hax.sum(H, axis="stream"))

    def layer(self, i, H, keep, rec, gv=None, pos=None):
        pos = i if pos is None else pos  # the sublayer constants' index 2 pos + j (read / bias one-hot on stream (2 pos + j) mod m)
        recs = []
        for j in range(2):
            H, r = self._sub(self.layers[i], j, H, jnp.int32(pos), jnp.float32(keep), rec, gv)
            recs.append(r)
        return H, recs

    def bypass(self, i, H):  # deletion: no write and identity mixing, the streams pass through
        return H

    def state(self, H, i):
        return self.A(hax.sum(H, axis="stream"))

    def final(self, H):
        return self.A(hax.sum(H, axis="stream"))

    def direct(self, recs_all, x, gv):
        kappa = jnp.ones(recs_all[0]["w"].shape, jnp.float32)  # (b, T, m): after the last sublayer (the read-out sum)
        out = [None] * len(recs_all)
        for t in range(len(recs_all) - 1, -1, -1):
            r = recs_all[t]
            out[t] = jnp.sum(r["w"] * kappa, -1) * r["r"]
            kappa = jnp.einsum("btio,bto->bti", r["A"], kappa)
        return out, jnp.sum(kappa, -1) * gdot(gv, x)


class AttnRes(Design):
    """attnres, attnres_block: raw buffers V (slots, b, T, E), R = 1 / RMS (attnres_checkup's form, _mix_buf's logits)."""

    def __init__(self, m, cfg, bt):
        super().__init__(m, cfg, bt)
        L = self.L
        self.block = self.arch == "attnres_block"
        self.group = (2 * L) // cfg.attnres_blocks if self.block else None
        self.S = 1 + cfg.attnres_blocks if self.block else 2 * L + 1
        self.layers = [self.tr.layer(i) for i in range(L)]
        Sub = Axis("sub", 2)
        Q = []
        for s in range(2 * L):
            lay = self.layers[s // 2]
            Q.append((lay.query[Sub, s % 2].astype(jnp.float32) * lay.key_gain[Sub, s % 2].astype(jnp.float32)).array)
        Q.append(self.tr._final_query().array.astype(jnp.float32))
        self.Q = Q
        mask = bt["mask"]
        N, A_ = self.N, self.A

        @eqx.filter_jit
        def branch(layer, j, h):
            return A_(layer.branch(j, N(h), mask, key=None, pos_ids=None))

        self._branch = branch

    @staticmethod
    @jax.jit
    def _put(V, R, i, f):
        return V.at[i].set(f), R.at[i].set(jax.lax.rsqrt(jnp.mean(f * f, -1) + 1e-6))

    @staticmethod
    @jax.jit
    def _add(V, R, i, f):
        v = V[i] + f
        return V.at[i].set(v), R.at[i].set(jax.lax.rsqrt(jnp.mean(v * v, -1) + 1e-6))

    @staticmethod
    @jax.jit
    def _mix(V, R, q, allowed):
        lg = jnp.einsum("sbpe,e->sbp", V, q) * R
        lg = jnp.where(allowed[:, None, None], lg, -jnp.inf)
        w = jax.nn.softmax(lg, axis=0)
        return jnp.einsum("sbp,sbpe->bpe", w, V), w

    def init(self, x):
        b = x.shape[0]
        V = jnp.zeros((self.S, b, T, x.shape[-1]), jnp.float32)
        R = jnp.zeros((self.S, b, T), jnp.float32)
        V, R = self._put(V, R, 0, x)
        self.avail = np.zeros(self.S, bool)
        self.avail[0] = True
        self.slot_of = []  # slot of each recorded sublayer
        return (V, R)

    def before(self, carry):
        V, R = carry
        return self._mix(V, R, self.Q[0], jnp.asarray(self.avail))[0]

    def _write(self, V, R, s, f):
        if self.block:
            slot = 1 + s // self.group
            V, R = (self._add if self.avail[slot] else self._put)(V, R, slot, f)
        else:
            slot = 1 + s
            V, R = self._put(V, R, slot, f)
        self.avail[slot] = True
        return V, R, slot

    def bypass(self, i, carry):
        return carry

    def layer(self, i, carry, keep, rec, gv=None, pos=None):
        pos = i if pos is None else pos  # the layer's position: attnres_block's slot is 1 + (2 pos + j) // 12
        V, R = carry
        if keep == 0.0:  # removed: no source written
            return (V, R), [None, None]
        recs = []
        for j in range(2):
            s = 2 * i + j
            h = self._mix(V, R, self.Q[s], jnp.asarray(self.avail))[0]
            f = self._branch(self.layers[i], j, h)
            V, R, slot = self._write(V, R, 2 * pos + j, f)
            if rec:
                recs.append(dict(ratio=wratio(f, h), r=gdot(gv, f), slot=slot))
            else:
                recs.append(None)
        return (V, R), recs

    def state(self, carry, i):
        V, R = carry
        q = self.Q[2 * i + 2] if i + 1 < self.L else self.Q[2 * self.L]
        return self._mix(V, R, q, jnp.asarray(self.avail))[0]

    def final(self, carry):
        V, R = carry
        return self._mix(V, R, self.Q[2 * self.L], jnp.asarray(self.avail))[0]

    def direct(self, recs_all, x, gv, carry=None):
        V, R = carry
        w = self._mix(V, R, self.Q[2 * self.L], jnp.asarray(self.avail))[1]  # (S, b, T)
        return [w[r["slot"]] * r["r"] for r in recs_all], w[0] * gdot(gv, x)


def design_for(m, cfg, bt):
    a = cfg.depth_arch
    if a in ("baseline", "preln", "lns", "deepnorm", "keel"):
        return Residual(m, cfg, bt)
    if a == "moda":
        return Moda(m, cfg, bt)
    if a in ("hc", "mhc"):
        return Hyper(m, cfg, bt)
    if a in ("attnres", "attnres_block"):
        return AttnRes(m, cfg, bt)
    raise ValueError(a)


# ---- deletion runs (--delete; coordinator follow-up 2026-10-19, after PracRemove.lean) --------------------------------
DELETE_NOTE = {
    "preln": "skip_ce is a true deletion (zero writes = the 47-layer model)",
    "sandwich": "skip_ce is a true deletion (RN_out(0) = 0: zero writes = the 47-layer model)",
    "moda": "skip_ce is a true deletion (zero writes and the layer's depth slots masked = the 47-layer model)",
    "attnres": "skip_ce is a true deletion (the layer's sources never enter a mix = the 47-layer model)",
    "deepnorm": "skip_ce_bypass copied (the layer skipped, h passes unchanged); skip_ce leaves h <- LN(a h) twice. The 47-layer model would also "
                "change a = (2L)^(1/4) and the init-scale constant with L; both kept at L = 48",
    "keel": "skip_ce_bypass copied (h passes unchanged); skip_ce leaves h <- RN(a h) twice. Kept: a = 2L with L = 48, and layer 1's special "
            "first-layer form stays on layer 1 (deleting layer 1 does not make layer 2 the special one)",
    "hc": "skip_ce_delete: the layer skipped entirely (identity mixing, no write; the streams pass unchanged), later layers keep their own "
          "sublayer constants (read one-hot on stream (2l + j) mod m, l the trained index); skip_ce_delete_reindex: later layers renumbered "
          "l -> l - 1 as in a 47-layer model (one-hot shifts by 2 streams); skip_ce keeps the layer's trained mixing A, read and write weights",
    "mhc": "as hc: skip_ce_delete keeps later layers' bias constants (read +8 on stream (2l + j) mod m), skip_ce_delete_reindex renumbers them",
    "lns": "skip_ce_delete: the layer skipped and the later layers renumbered, so layer l > k scales by 1 / sqrt(l) instead of 1 / sqrt(l + 1) "
           "(the 47-layer model); skip_ce (zero writes) keeps the later layers' trained scales",
    "attnres_block": "skip_ce_delete: the layer skipped and the later layers moved up one position with the block size kept at 12 sublayers "
                     "(6 layers): a later sublayer at position p joins block p // 12, so later block boundaries move by one layer; skip_ce keeps "
                     "each later layer's trained block slot. (The model's own code at L = 47 would use 94 // 8 = 11 sublayers per block.)",
}


def run_delete(arch, setup, m, cfg, tr, W_main, forward):
    dpath = os.path.join(args.out_dir, f"{arch}-{setup}.json")
    old = json.load(open(dpath))
    bt, L = BT_SKIP, cfg.num_layers
    D = design_for(m, cfg, bt)
    ce = lambda h: float(ce_of(h, tr.norm, W_main, bt["tgt"], bt["valid"]))  # noqa: E731
    base = ce(forward(D, bt)[0])
    print(f"[{arch}-{setup}] base CE {base:.5f} (stored {old['base_ce']:.5f}, diff {base - old['base_ce']:.1e})", flush=True)
    layers = [int(v) for v in args.skip_layers.split(",") if v] or list(range(1, L + 1))
    t0 = time.time()
    for k in range(1, L + 1):
        r = old["layers"][k - 1]
        r["skip_ce_delete"] = None
        if arch in ("hc", "mhc"):
            r["skip_ce_delete_reindex"] = None
        if k not in layers:
            continue
        if arch in ("hc", "mhc"):
            r["skip_ce_delete"] = ce(forward(D, bt, skip=k - 1, bypass=True)[0])
            r["skip_ce_delete_reindex"] = ce(forward(D, bt, skip=k - 1, bypass=True, reindex=True)[0])
        else:  # lns, attnres_block
            r["skip_ce_delete"] = ce(forward(D, bt, skip=k - 1, bypass=True, reindex=True)[0])
        print(f"[{arch}-{setup}] delete layer {k}: {r['skip_ce_delete'] - base:+.4f} (skip_ce {r['skip_ce'] - base:+.4f})"
              + (f" reindex {r['skip_ce_delete_reindex'] - base:+.4f}" if arch in ("hc", "mhc") else "") + f" | {time.time() - t0:.0f}s", flush=True)
    old["delete_note"] = DELETE_NOTE[arch]
    old["delete_base_ce"] = base
    json.dump(old, open(dpath, "w"), indent=1)
    print(f"[{arch}-{setup}] wrote {dpath}", flush=True)


# ---- one model ------------------------------------------------------------------------------------------------------
def run_model(name):
    t_start = time.time()
    wcfg = json.load(open(os.path.join(args.cfg_dir, f"{name}.json")))["model"]
    fields = {f.name for f in dataclasses.fields(PerLayerQwen3Config)}
    kw = {k: v for k, v in wcfg.items() if k in fields and not isinstance(v, (dict, list)) and v is not None and k not in ("attn_backend", "max_seq_len", "tokenizer", "reference_checkpoint")}
    cfg = PerLayerQwen3Config(max_seq_len=T, attn_backend=AttentionBackend.VANILLA, **kw)
    L, arch_raw = cfg.num_layers, cfg.depth_arch
    arch = "sandwich" if arch_raw == "baseline" else arch_raw
    setup = "ord" if "-sep-bbfrozen" in name else "sep"
    assert (setup == "ord") == bool(cfg.pls_detach_backbone), (name, cfg.pls_detach_backbone)
    assert L == 48 and cfg.pls_separate_heads and (arch_raw != "baseline" or cfg.hybrid_norm), (L, cfg.pls_separate_heads)
    ckpt = f"{ROOT}/{name}/2026.09.13/checkpoints/step-4958"
    if args.delete:  # copy-only designs need no model
        dpath = os.path.join(args.out_dir, f"{arch}-{setup}.json")
        old = json.load(open(dpath))
        src = {"preln": "skip_ce", "sandwich": "skip_ce", "moda": "skip_ce", "attnres": "skip_ce", "deepnorm": "skip_ce_bypass", "keel": "skip_ce_bypass"}.get(arch)
        if src is not None:
            for r in old["layers"]:
                r["skip_ce_delete"] = r[src]
            old["delete_note"] = DELETE_NOTE[arch]
            json.dump(old, open(dpath, "w"), indent=1)
            print(f"[{arch}-{setup}] skip_ce_delete = {src} (copied) -> {dpath}", flush=True)
            return
    template = eqx.filter_eval_shape(PerLayerQwen3LMHeadModel.init, Vocab, cfg, key=jrandom.PRNGKey(0))
    m = load_checkpoint(template, ckpt, subpath="model")
    m = jax.tree_util.tree_map(lambda a: a.astype(jnp.float32) if eqx.is_inexact_array(a) else a, m)
    assert m.aux_lm_heads is not None and len(m.aux_lm_heads) == L - 1 and m.aux_adapters is None and m.mtp is None
    print(f"[{arch}-{setup}] loaded {ckpt} in {time.time() - t_start:.0f}s", flush=True)
    tr = m.transformer
    Embed = cfg.Embed
    W_main = m.get_lm_head().rearrange((Vocab, Embed)).array
    W_mean = jnp.mean(W_main, 0)
    g_f = tr.norm.weight.rearrange((Embed,)).array
    eps_f = float(tr.norm.eps)
    aux_W = [h.weight.rearrange((Vocab, Embed)).array for h in m.aux_lm_heads]

    def embed(bt):
        return m.embeddings.embed(bt["ids"]).rearrange((bt["Batch"], bt["Pos"], Embed)).array

    def forward(D, bt, skip=None, rec=False, bypass=False, gv=None, reindex=False):
        x = embed(bt)
        carry = D.init(x)
        prev = D.before(carry) if rec else None
        recs_all, rows = [], []
        for i in range(L):
            if bypass and skip == i:
                carry = D.bypass(i, carry)
                continue
            t1 = time.time()
            if reindex and skip is not None and i > skip:  # the layers after a deleted one move up one position
                carry, recs = D.layer(i, carry, 1.0, rec, gv, pos=i - 1)
            else:
                carry, recs = D.layer(i, carry, 0.0 if skip == i else 1.0, rec, gv)
            if rec:
                recs_all += recs
                S = D.state(carry, i)
                bi, er, an, ms, md = state_stats(prev, S, SAMPLE)
                lens = ce_of(S, tr.norm, W_main, bt["tgt"], bt["valid"])
                own = ce_of(S, m.aux_norms[i], aux_W[i], bt["tgt"], bt["valid"]) if i < L - 1 else lens
                rows.append(dict(layer=i + 1, attn_write_ratio=float(recs[0]["ratio"]), mlp_write_ratio=float(recs[1]["ratio"]), block_imp=float(bi),
                                 lens_ce=float(lens), own_ce=float(own), eff_rank=float(er), aniso=float(an), massive=float(ms), massive_dim=int(md)))
                prev = S
                if i % 8 == 7 or i == 0:
                    print(f"[{arch}-{setup}] layer {i + 1}: lens {float(lens):.3f} own {float(own):.3f} imp {float(bi):.4f} erank {float(er):.1f} "
                          f"aniso {float(an):.3f} massive {float(ms):.1f}@{int(md)} | {time.time() - t1:.1f}s", flush=True)
        return D.final(carry), carry, recs_all, rows, x

    if args.delete:
        run_delete(arch, setup, m, cfg, tr, W_main, forward)
        return

    out = dict(arch=arch, setup=setup, model=name, ckpt=ckpt, text=f"paloma c4_en validation, first {B} x {T} tokens (skip: first {B_SKIP}), documents isolated by segment ids",
               tokens_skip=BT_SKIP["n_valid"], tokens=BT_REC["n_valid"], n_sample=N_SAMPLE)
    out_path = os.path.join(args.out_dir, f"{arch}-{setup}.json")

    # ---- measures 2-5: one recorded forward on the 16 sequences
    t0 = time.time()
    bt = BT_REC
    gv = g_f[None, None, :] * (W_main[bt["tgt"]] - W_mean[None, None, :])
    D = design_for(m, cfg, bt)
    hT, carry, recs_all, rows, x = forward(D, bt, rec=True, gv=gv)
    s_tok = jax.lax.rsqrt(jnp.mean(hT * hT, -1) + eps_f)
    val = bt["valid"]
    nv = jnp.sum(val)

    def tmean(a):
        return float(jnp.sum(a * s_tok * val) / nv)

    if isinstance(D, AttnRes):
        dw, demb = D.direct(recs_all, x, gv, carry=carry)
    else:
        dw, demb = D.direct(recs_all, x, gv)
    for i, row in enumerate(rows):
        row["attn_dla"], row["mlp_dla"] = tmean(recs_all[2 * i]["r"]), tmean(recs_all[2 * i + 1]["r"])
        row["attn_dla_w"], row["mlp_dla_w"] = tmean(dw[2 * i]), tmean(dw[2 * i + 1])
    total = tmean(gdot(gv, hT))
    out["emb_dla_w"], out["total_logit"] = tmean(demb), total
    out["closure_err"] = total - (out["emb_dla_w"] + sum(r["attn_dla_w"] + r["mlp_dla_w"] for r in rows))
    out["base_ce_16"] = rows[-1]["lens_ce"]
    bi, er, an, ms, md = state_stats(x, x, SAMPLE)
    out["embed_stats"] = dict(eff_rank=float(er), aniso=float(an), massive=float(ms), massive_dim=int(md))
    print(f"[{arch}-{setup}] records in {time.time() - t0:.0f}s; final CE (16) {out['base_ce_16']:.4f}; DLA closure: total {total:.4f} err {out['closure_err']:.2e}", flush=True)
    del recs_all, dw, demb, carry, D, gv
    out["layers"] = rows
    json.dump(out, open(out_path, "w"), indent=1)

    # ---- measure 1: skip runs on the first B_SKIP sequences; reference = the model's own forward
    bt = BT_SKIP
    Blk = tr.layers.Block.name

    @eqx.filter_jit
    def ref_final(m, ids):
        return m.layer_outputs(ids, bt["mask"])[Blk, -1].rearrange((bt["Batch"], bt["Pos"], Embed)).array

    ref = float(ce_of(ref_final(m, bt["ids"]), tr.norm, W_main, bt["tgt"], bt["valid"]))
    D = design_for(m, cfg, bt)  # one instance for every skip run: its jitted pieces compile once (init() resets its state)
    base = float(ce_of(forward(D, bt)[0], tr.norm, W_main, bt["tgt"], bt["valid"]))
    out["base_ce"], out["ref_ce"], out["ref_diff"] = base, ref, base - ref
    print(f"[{arch}-{setup}] base CE {base:.5f} model's own forward {ref:.5f} diff {base - ref:.2e}" + ("  WARN: > 2e-3" if abs(base - ref) > 2e-3 else ""), flush=True)
    layers = [int(v) for v in args.skip_layers.split(",") if v] or list(range(1, L + 1))
    t0 = time.time()
    for k in range(1, L + 1):
        rows[k - 1]["skip_ce"] = None
        if arch in ("deepnorm", "keel"):
            rows[k - 1]["skip_ce_bypass"] = None
        if k not in layers:
            continue
        rows[k - 1]["skip_ce"] = float(ce_of(forward(D, bt, skip=k - 1)[0], tr.norm, W_main, bt["tgt"], bt["valid"]))
        if arch in ("deepnorm", "keel"):
            rows[k - 1]["skip_ce_bypass"] = float(ce_of(forward(D, bt, skip=k - 1, bypass=True)[0], tr.norm, W_main, bt["tgt"], bt["valid"]))
        print(f"[{arch}-{setup}] skip layer {k}: CE {rows[k - 1]['skip_ce']:.4f} (+{rows[k - 1]['skip_ce'] - base:.4f})"
              + (f" bypass {rows[k - 1]['skip_ce_bypass']:.4f}" if arch in ("deepnorm", "keel") else "") + f" | {time.time() - t0:.0f}s", flush=True)
    out["layers"] = rows
    out["seconds"] = time.time() - t_start
    json.dump(out, open(out_path, "w"), indent=1)
    print(f"[{arch}-{setup}] wrote {out_path} in {out['seconds']:.0f}s", flush=True)


os.makedirs(args.out_dir, exist_ok=True)
failed = []
for nm in args.names:
    try:
        run_model(nm)
    except Exception:
        traceback.print_exc()
        failed.append(nm)
    jax.clear_caches()
print("FAILED: " + " ".join(failed) if failed else "all done", flush=True)
sys.exit(1 if failed else 0)
