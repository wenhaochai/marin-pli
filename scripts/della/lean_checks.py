"""Three read-only checks of the trained 48-layer per-layer-contribution checkpoints for the Lean collaborator
(repo wenhaochai/resnext-lean, branch arch-system; owner popup 2026-10-19). Loads a finished checkpoint, trains nothing.

Text: paloma c4_en validation, the first B x T tokens (default 8 x 1024), documents isolated by segment ids, float32 on CPU.

(a) hyperplane  every model: the final hidden state h = final_norm(h_L) (the vector the LM head reads), per token,
    s = |sum_i h_i / g_i| over ||h / g||_1 and over ||h / g||_2 sqrt(d), for two gain vectors g: g_f (the final RMSNorm
    gain) and, for deepnorm, g_f * g_post (g_post the last layer's MLP post-LayerNorm gain). In our deepnorm the last
    post-norm is hnn.LayerNorm(use_weight, no bias): h_L = g_post * z with sum_i z_i = 0, then the final RMSNorm gives
    h = g_f * h_L / rms(h_L), so sum_i h_i / (g_f g_post)_i = 0 exactly in real arithmetic; with g_f alone it is 0 only
    for a constant g_post. Non-deepnorm models are the control.
(b) kappa  hc and mhc: the embedding x is copied into all m streams; each sublayer maps streams H_o <- sum_s A[s, o] H_s
    + w_o f (hc: A = res (in, out); mhc: Sinkhorn A (out, in)); the head reads sum_o H_o through the final norm. The
    coefficient vector c of x in the streams (the linear path only, branch outputs f left out although they also depend
    on x) starts at c = 1 (m ones) and follows c_o <- sum_s A[s, o] c_s with A computed on the token's own streams;
    kappa = sum_o c_o after all 2L sublayers (m = 4 at init, where A = I). Also kappa ||x|| / ||sum_o H_o|| (2-norms),
    the share of the read-out stream that the linear x path carries.
(c) bound  preln and sandwich (the baseline): ArchLean/RayBound.lean lbBranchBound (Pre-LN Qwen3):
    B_l = sum_h lbOp(wo_h) lbOp(wv_grp(h)) R_A + lbOp(down) (lbOp(gate) R_M) (lbOp(up) R_M), R = ||g_pre||_inf sqrt(d),
    lbOp = largest absolute row sum. Sandwich adds RN_out on every branch output, so lbBranchBound is not an output
    bound there; its valid bound is the post-norm's sup radius, B_l = sqrt(d) (||g_postattn||_inf + ||g_postmlp||_inf)
    (|RN(u)_i| <= ||g||_inf sqrt(d) for eps >= 0). Readout r_k = g_f * nrm_eps_f(h_k) (logit lens through the final
    RMSNorm, h_0 = the embedding), r_L = the model's final hidden state. Measured ||r_L - r_k||_inf per token against
    lbQwen3_readout_rms: (1 + sqrt d) ||g_f||_inf sum_{l>=k} B_l / rms(h_k) and lbQwen3_readout:
    sqrt(d) (1 + sqrt d) ||g_f||_inf sum_{l>=k} B_l / ||h_k||_inf; trivial bound 2 ||g_f||_inf sqrt(d).

    source scripts/della/pls_env.sh
    JAX_PLATFORMS=cpu nice -n 19 $PY scripts/della/lean_checks.py CKPT CONFIG.json OUT.json NAME [B T]
"""
import dataclasses
import json
import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jrandom  # noqa: E402
import numpy as np  # noqa: E402

import haliax as hax  # noqa: E402
from haliax import Axis  # noqa: E402
from levanter.checkpoint import load_checkpoint  # noqa: E402
from levanter.layers.attention import AttentionBackend, AttentionMask  # noqa: E402

from experiments.references.depth_arch_qwen3 import _hc_constants, _mhc_constants, _rms, _sinkhorn_liger  # noqa: E402
from experiments.references.per_layer_qwen3 import PerLayerQwen3Config, PerLayerQwen3LMHeadModel  # noqa: E402

CKPT, CFG, OUT, NAME = sys.argv[1:5]
B = int(sys.argv[5]) if len(sys.argv) > 5 else 8
T = int(sys.argv[6]) if len(sys.argv) > 6 else 1024
Vocab = Axis("vocab", 128256)

wcfg = json.load(open(CFG))["model"]
fields = {f.name for f in dataclasses.fields(PerLayerQwen3Config)}
kw = {k: v for k, v in wcfg.items() if k in fields and not isinstance(v, (dict, list)) and v is not None and k not in ("attn_backend", "max_seq_len", "tokenizer", "reference_checkpoint")}
cfg = PerLayerQwen3Config(max_seq_len=T, attn_backend=AttentionBackend.VANILLA, **kw)
ARCH, L = cfg.depth_arch, cfg.num_layers
D = cfg.hidden_dim
assert L == 48 and ARCH in ("baseline", "preln", "deepnorm", "hc", "mhc"), (ARCH, L)
assert cfg.hybrid_norm and not cfg.use_bias and not cfg.input_embedding_norm and not cfg.tie_word_embeddings

from haliax.partitioning import ResourceAxis  # noqa: E402
from levanter.trainer import TrainerConfig  # noqa: E402
from levanter.utils.mesh import MeshConfig  # noqa: E402

tc = TrainerConfig(mesh=MeshConfig(axes={"data": -1, "replica": 1, "model": 1}, compute_mapping={
    "token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
    "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA)}))
tc.use_device_mesh().__enter__()
hax.axis_mapping(tc.parameter_axis_mapping).__enter__()
t0 = time.time()
template = eqx.filter_eval_shape(PerLayerQwen3LMHeadModel.init, Vocab, cfg, key=jrandom.PRNGKey(0))
m = load_checkpoint(template, CKPT, subpath="model")
m = jax.tree_util.tree_map(lambda a: a.astype(jnp.float32) if eqx.is_inexact_array(a) else a, m)
print(f"loaded {CKPT} ({ARCH}) in {time.time() - t0:.0f}s", flush=True)
tr = m.transformer
Embed = cfg.Embed

import tensorstore as ts  # noqa: E402

VAL = "/scratch/gpfs/GROUP/USER/marin_store_big/tokenized/paloma/c4_en-cf1f79/validation/part-00000-of-00001/input_ids/data"
tok = np.asarray(ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": VAL}}).result()[: B * T].read().result(), np.int64).reshape(B, T)
eos = np.roll(tok, 1, axis=1) == 128001
eos[:, 0] = False
seg = np.cumsum(eos, axis=1)
Batch, Pos = Axis("batch", B), Axis("position", T)
mask = AttentionMask.causal().with_segment_ids(hax.named(jnp.asarray(seg, jnp.int32), (Batch, Pos)))
valid = np.ones((B, T), bool)
valid[:, -1] = False
valid &= np.roll(seg, -1, axis=1) == seg
tgt = np.roll(tok, -1, axis=1)
ids = hax.named(jnp.asarray(tok, jnp.int32), (Batch, Pos))


def arr(x, axes):
    return np.asarray(x.rearrange(axes).array, np.float64)


def q(v):
    v = np.asarray(v, np.float64).ravel()
    return dict(min=float(v.min()), p01=float(np.quantile(v, 0.01)), p05=float(np.quantile(v, 0.05)), median=float(np.median(v)),
                mean=float(v.mean()), p95=float(np.quantile(v, 0.95)), p99=float(np.quantile(v, 0.99)), max=float(v.max()))


# ---- forward: the state after every layer, the embedding, the final hidden state ----------------------------------
@eqx.filter_jit
def forward(m, ids):
    x = m.embeddings.embed(ids)
    outs = m.layer_outputs(ids, mask)
    return x.rearrange((Batch, Pos, Embed)).array, outs.rearrange((m.transformer.layers.Block.name, Batch, Pos, Embed)).array, m.transformer.norm(outs[m.transformer.layers.Block.name, -1]).rearrange((Batch, Pos, Embed)).array


t0 = time.time()
x0, outs, hfin = forward(m, ids)
x0, outs, hfin = np.asarray(x0), np.asarray(outs), np.asarray(hfin)
print(f"forward in {time.time() - t0:.0f}s; outs {outs.shape}", flush=True)

W = np.asarray(m.get_lm_head().rearrange((Vocab, Embed)).array, np.float32)
ce = []
for b in range(B):
    z = hfin[b] @ W.T
    lse = np.log(np.exp(z - z.max(-1, keepdims=True)).sum(-1)) + z.max(-1)
    ce.append(((lse - z[np.arange(T), tgt[b]]) * valid[b]).sum())
out = dict(model=NAME, arch=ARCH, ckpt=CKPT, d=D, L=L, tokens=B * T, text=f"paloma c4_en validation, first {B} x {T} tokens, documents isolated by segment ids; float32 CPU",
           final_ce=float(np.sum(ce) / valid.sum()), ce_tokens=int(valid.sum()))
print(f"final CE {out['final_ce']:.4f} over {out['ce_tokens']} tokens", flush=True)
del W

# ---- (a) hyperplane ------------------------------------------------------------------------------------------------
g_f = arr(tr.norm.weight, (Embed,))
h = hfin.reshape(-1, D).astype(np.float64)
gains = {"g_f": g_f}
if ARCH == "deepnorm":
    last = tr.layer(L - 1)
    assert type(last.post_2).__name__ == "LayerNorm" and last.post_2.bias is None and last.post_2.weight is not None
    assert type(tr.norm).__name__ == "RmsNorm" and tr.norm.bias is None
    g_post = arr(last.post_2.weight, (Embed,))
    gains["g_f*g_post"] = g_f * g_post
    out["deepnorm_premises"] = dict(post_norm="hnn.LayerNorm (mean-centred), weight, no bias", final_norm="hnn.RmsNorm, weight, no bias",
                                    order="final RMSNorm reads the last layer's MLP post-LayerNorm output directly", eps_post=float(last.post_2.eps), eps_final=float(tr.norm.eps),
                                    g_post_min_abs=float(np.abs(g_post).min()), g_post_max_abs=float(np.abs(g_post).max()), g_post_cv=float(g_post.std() / np.abs(g_post.mean())))
hyp = {}
for gname, g in gains.items():
    v = h / g
    s = np.abs(v.sum(-1))
    hyp[gname] = dict(min_abs_gain=float(np.abs(g).min()), l1=q(s / np.abs(v).sum(-1)), l2sqrtd=q(s / (np.linalg.norm(v, axis=-1) * np.sqrt(D))))
    print(f"(a) {gname}: |sum h/g| / ||h/g||_1 max {hyp[gname]['l1']['max']:.3e} mean {hyp[gname]['l1']['mean']:.3e}; "
          f"/ (||h/g||_2 sqrt d) max {hyp[gname]['l2sqrtd']['max']:.3e} mean {hyp[gname]['l2sqrtd']['mean']:.3e}", flush=True)
if ARCH == "deepnorm":  # the same statistic one step earlier: the last post-LN output h_L itself, sum_i h_L,i / g_post,i
    hl = outs[-1].reshape(-1, D).astype(np.float64) / g_post
    s = np.abs(hl.sum(-1))
    hyp["h_L/g_post"] = dict(l1=q(s / np.abs(hl).sum(-1)), l2sqrtd=q(s / (np.linalg.norm(hl, axis=-1) * np.sqrt(D))))
out["hyperplane"] = hyp
json.dump(out, open(OUT, "w"), indent=1)


# ---- (b) kappa -----------------------------------------------------------------------------------------------------
@eqx.filter_jit
def hc_sub(layer, j, H, c, li):
    """One hc / mhc sublayer as hyper_step / _mhc_step compute it, also mapping the x-coefficients c (batch, pos, stream)."""
    S = H.resolve_axis("stream")
    mm, So = S.size, Axis("stream_out", S.size)
    f32 = lambda a: a.astype(jnp.float32)  # noqa: E731
    if ARCH == "hc":
        p = layer.hyper[j]
        c_read, c_res, c_write = _hc_constants(mm, 2 * li + j)
        s_rr, s_w = f32(p.scale)["hp_scale", 0], f32(p.scale)["hp_scale", 1]
        Hn = f32(_rms(H, Embed))
        read = c_read + f32(p.read) + s_rr * hax.tanh(hax.dot(Hn, f32(p.w_read), axis=Embed))
        res = c_res + f32(p.res) + s_rr * hax.tanh(hax.dot(Hn, f32(p.w_res), axis=Embed))  # (in = stream, out = stream_out)
        write = c_write + f32(p.write) + s_w * hax.tanh(hax.dot(Hn, f32(p.w_write), axis=Embed).rename({"stream": "stream_out"}))
        x = hax.dot(read, H, axis="stream")
        f = layer.branch(j, x, mask, key=None, pos_ids=None)
        Hn2 = (hax.dot(res, H, axis="stream") + write * f.broadcast_axis(So)).rename({"stream_out": "stream"})
        c2 = hax.dot(res, c, axis="stream").rename({"stream_out": "stream"})
        colsum = hax.sum(res, axis="stream_out")  # row sums in (in, out): what one input stream sends out in total
    else:
        p = layer.hyper[j]
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
        A = _sinkhorn_liger(resl.unflatten_axis("mhc", (So, S)))  # (out, in)
        x = hax.dot(read, Hf, axis=S)
        f = layer.branch(j, x, mask, key=None, pos_ids=None)
        Hn2 = (hax.dot(A, Hf, axis=S) + write * f32(f).broadcast_axis(So)).rename({"stream_out": "stream"})
        c2 = hax.dot(A, c, axis=S).rename({"stream_out": "stream"})
        colsum = hax.sum(A, axis="stream_out")
    return Hn2, c2, hax.sum(c2, axis="stream").rearrange((Batch, Pos)).array, colsum.rearrange((Batch, Pos, "stream")).array


if ARCH in ("hc", "mhc"):
    mstreams = cfg.hc_streams
    St = Axis("stream", mstreams)
    xn = hax.named(jnp.asarray(x0), (Batch, Pos, Embed))
    H = hax.stack(St, [xn] * mstreams)
    c = hax.named(jnp.ones((B, T, mstreams), jnp.float32), (Batch, Pos, St))
    traj, cs_min, cs_max = [], [], []
    t0 = time.time()
    for i in range(L):
        layer = tr.layer(i)
        for j in range(2):
            H, c, kap, colsum = hc_sub(layer, j, H, c, jnp.asarray(i))
            cs_min.append(float(np.min(colsum)))
            cs_max.append(float(np.max(colsum)))
        kap = np.asarray(kap)
        traj.append(dict(layer=i, median=float(np.median(kap)), min=float(kap.min()), max=float(kap.max())))
        if i % 8 == 7:
            print(f"(b) layer {i + 1}: kappa median {traj[-1]['median']:.4f} min {traj[-1]['min']:.4f} max {traj[-1]['max']:.4f} | {time.time() - t0:.0f}s", flush=True)
    Hsum = np.asarray(hax.sum(H, axis="stream").rearrange((Batch, Pos, Embed)).array)
    check = float(np.max(np.abs(Hsum - outs[-1])) / np.max(np.abs(outs[-1])))
    kap = np.asarray(kap, np.float64).ravel()
    share = np.abs(kap) * np.linalg.norm(x0.reshape(-1, D), axis=-1) / np.linalg.norm(Hsum.reshape(-1, D), axis=-1)
    cvec = np.asarray(c.rearrange((Batch, Pos, St)).array, np.float64).reshape(-1, mstreams)
    out["kappa"] = dict(m=mstreams, kappa_at_init=float(mstreams), kappa=q(kap), kappa_over_m=q(kap / mstreams), n_tokens=int(kap.size),
                        frac_below={str(t): float(np.mean(kap < t)) for t in (0.0, 0.01, 0.1, 0.5, 1.0)},
                        final_c_per_stream_median=np.median(cvec, 0).tolist(), share_x_in_stream_sum=q(share),
                        per_layer=traj, outsum_rel_err_vs_model=check, A_row_or_col_sum_min=min(cs_min), A_row_or_col_sum_max=max(cs_max),
                        x_rms=q(np.sqrt(np.mean(x0.reshape(-1, D) ** 2, -1))))
    low = np.flatnonzero(kap < 1.0)
    out["kappa"]["below_1_tokens"] = [dict(seq=int(t // T), pos=int(t % T), token=int(tok.ravel()[t]), kappa=float(kap[t]), doc_start=bool(t % T == 0 or eos.ravel()[t])) for t in low[:64]]
    out["kappa"]["kappa_excl_pos0"] = q(kap.reshape(B, T)[:, 1:])
    print(f"(b) kappa: {out['kappa']['kappa']}  share {out['kappa']['share_x_in_stream_sum']['median']:.4f}  recompute rel err {check:.2e}", flush=True)
    json.dump(out, open(OUT, "w"), indent=1)

# ---- (c) bound -----------------------------------------------------------------------------------------------------
if ARCH in ("preln", "baseline"):
    Blk = tr.layers.Block.name
    st = tr.layers.stacked
    sd = np.sqrt(D)

    def lbop_rows(Wm):  # (..., rows, cols) -> largest absolute row sum
        return np.abs(Wm).sum(-1).max(-1)

    at = st.self_attn
    wo = arr(at.o_proj.weight, (Blk, "heads", Embed.name, "head_size"))  # wo_h: (embed, head_size) rows = embed
    wv = arr(at.v_proj.weight, (Blk, "kv_head", "head_size", Embed.name))  # wv_g: (head_size, embed)
    nh, nkv = wo.shape[1], wv.shape[1]
    grp = np.arange(nh) // (nh // nkv)  # heads flattened from (kv_head, q_heads_per_group)
    op_wo, op_wv = lbop_rows(wo), lbop_rows(wv)  # (L, heads), (L, kv)
    gate = arr(st.mlp.gate_proj.weight, (Blk, "mlp", Embed.name))
    up = arr(st.mlp.up_proj.weight, (Blk, "mlp", Embed.name))
    down = arr(st.mlp.down_proj.weight, (Blk, Embed.name, "mlp"))
    if ARCH == "preln":
        g_a, g_m = arr(st.ln_1.weight, (Blk, Embed.name)), arr(st.ln_2.weight, (Blk, Embed.name))
    else:
        g_a, g_m = arr(st.input_layernorm.weight, (Blk, Embed.name)), arr(st.post_attention_layernorm.weight, (Blk, Embed.name))
    RA, RM = np.abs(g_a).max(-1) * sd, np.abs(g_m).max(-1) * sd
    attn_b = np.sum(op_wo * op_wv[:, grp], -1) * RA
    mlp_b = lbop_rows(down) * (lbop_rows(gate) * RM) * (lbop_rows(up) * RM)
    Bw = attn_b + mlp_b  # lbBranchBound per layer
    if ARCH == "baseline":
        gpa, gpm = arr(st.post_attn_layernorm.weight, (Blk, Embed.name)), arr(st.post_mlp_layernorm.weight, (Blk, Embed.name))
        Bl = sd * (np.abs(gpa).max(-1) + np.abs(gpm).max(-1))
        bound_kind = "sandwich: B_l = sqrt(d) (||g_postattn||_inf + ||g_postmlp||_inf) (post-norm radius); lbBranchBound of the inner branch reported but not a valid output bound"
    else:
        Bl = Bw
        bound_kind = "preln: B_l = lbBranchBound (RayBound.lean)"
    tail = np.concatenate([np.cumsum(Bl[::-1])[::-1], [0.0]])  # tail[k] = sum_{l >= k} B_l, k = 0..L
    eps_f = float(tr.norm.eps)
    gfi = float(np.abs(g_f).max())
    states = np.concatenate([x0[None], outs], 0).reshape(L + 1, -1, D).astype(np.float64)  # h_k, k = 0..L
    hL = states[-1]

    def readout(hk):
        return g_f * hk / np.sqrt(np.mean(hk * hk, -1, keepdims=True) + eps_f)

    rL = readout(hL)
    rL_model = hfin.reshape(-1, D)
    rows = []
    for k in range(L + 1):
        hk = states[k]
        meas = np.abs(rL - readout(hk)).max(-1)
        rms = np.sqrt(np.mean(hk * hk, -1))
        sup = np.abs(hk).max(-1)
        b_rms = (1 + sd) * gfi * tail[k] / rms
        b_sup = sd * (1 + sd) * gfi * tail[k] / sup
        disp = np.abs(hL - hk).max(-1)
        ratio = np.where(meas > 0, b_rms / np.maximum(meas, 1e-30), np.inf)
        rows.append(dict(k=k, sum_B=float(tail[k]), meas_max=float(meas.max()), meas_median=float(np.median(meas)),
                         bound_rms_min=float(b_rms.min()), bound_rms_median=float(np.median(b_rms)), bound_sup_min=float(b_sup.min()), bound_sup_median=float(np.median(b_sup)),
                         ratio_rms_min=float(ratio[np.isfinite(ratio)].min()) if np.isfinite(ratio).any() else None,
                         ratio_rms_median=float(np.median(ratio[np.isfinite(ratio)])) if np.isfinite(ratio).any() else None,
                         violations_rms=int(np.sum(meas > b_rms * (1 + 1e-6) + 1e-6)), violations_sup=int(np.sum(meas > b_sup * (1 + 1e-6) + 1e-6)),
                         disp_max=float(disp.max()), disp_violations=int(np.sum(disp > tail[k] * (1 + 1e-6) + 1e-6)),
                         hk_rms_median=float(np.median(rms)), hk_sup_median=float(np.median(sup))))
    out["bound"] = dict(kind=bound_kind, readout="r_k = g_f * h_k / sqrt(mean h_k^2 + eps_f): logit lens through the final RMSNorm (h_0 = embedding), not the per-layer heads",
                        eps_f=eps_f, gf_sup=gfi, trivial_bound=2 * gfi * float(sd), sqrt_d=float(sd),
                        r_L_vs_model_max_abs=float(np.abs(rL - rL_model).max()),
                        B_l=Bl.tolist(), lbBranchBound=Bw.tolist(), attn_part=attn_b.tolist(), mlp_part=mlp_b.tolist(),
                        R_A=RA.tolist(), R_M=RM.tolist(), per_k=rows,
                        layer_disp_max=[float(np.abs(states[l + 1] - states[l]).max()) for l in range(L)])  # measured max_t ||h_{l+1} - h_l||_inf against B_l
    for r in rows[:: 8] + [rows[-2]]:
        print(f"(c) k {r['k']}: sumB {r['sum_B']:.3e} meas max {r['meas_max']:.3e} bound_rms min {r['bound_rms_min']:.3e} median {r['bound_rms_median']:.3e} "
              f"ratio min {r['ratio_rms_min']} viol {r['violations_rms']} disp max {r['disp_max']:.3e}", flush=True)
    json.dump(out, open(OUT, "w"), indent=1)

print("wrote", OUT, flush=True)
