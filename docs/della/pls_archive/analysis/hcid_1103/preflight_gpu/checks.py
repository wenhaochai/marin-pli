"""Independent checks for the 2026-11-03 audit. CPU only."""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS", "0")
import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jmp
import numpy as np

import haliax as hax
from jax._src import core as jcore
from haliax import Axis
from levanter.layers.attention import AttentionBackend, AttentionMask
from levanter.models.lm_model import LmExample

from experiments.references.depth_arch_qwen3 import HC_MOVE_BETA, HC_SUM_GAIN, DepthArchMuonHConfig
from experiments.references.per_layer_qwen3 import PerLayerQwen3Config, PerLayerQwen3LMHeadModel

jax.config.update("jax_default_matmul_precision", "highest")
rng = np.random.default_rng(1)
B, T, V, L, D, M = 2, 16, 300, 6, 48, 4
Batch, Pos, Vocab = Axis("batch", B), Axis("position", T), Axis("vocab", V)
Stream, Embed = Axis("stream", M), Axis("embed", D)
common = dict(max_seq_len=T, hidden_dim=D, intermediate_dim=64, num_layers=L, num_heads=2, num_kv_heads=2, attn_backend=AttentionBackend.VANILLA, pls_weight=1.0, pls_separate_heads=True)
mask = AttentionMask.causal()
mp = jmp.get_policy("p=f32,c=bfloat16")
tok = rng.integers(2, V, size=(B, T))
ex = LmExample(tokens=hax.named(jnp.asarray(tok, jnp.int32), (Batch, Pos)), loss_weight=hax.named(jnp.ones((B, T), jnp.float32).at[:, -1].set(0), (Batch, Pos)), attn_mask=mask)


def model(arch, **kw):
    cfg = PerLayerQwen3Config(**{**common, **kw}, hybrid_norm=True, depth_arch=arch, scan_layers=False)
    return PerLayerQwen3LMHeadModel.init(Vocab, cfg, key=jrandom.PRNGKey(0))


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def rand_hyper(m_, seed):
    hy = m_.transformer.layers.stacked.hyper
    r = lambda a, s, k: hax.named(jnp.asarray(np.random.default_rng(k).standard_normal(a.array.shape) * s, a.array.dtype), a.axes)  # noqa: E731
    new = [dataclasses.replace(p, read=r(p.read, .3, seed + j), write=r(p.write, .3, seed + 10 + j), res=r(p.res, .3, seed + 20 + j), w_read=r(p.w_read, .3, seed + 30 + j),
                               w_write=r(p.w_write, .3, seed + 40 + j), w_res=r(p.w_res, .3, seed + 50 + j), scale=r(p.scale, 1., seed + 60 + j)) for j, p in enumerate(hy)]
    return eqx.tree_at(lambda mm: mm.transformer.layers.stacked.hyper, m_, tuple(new))


# A. the scan (traced layer index) equals a Python loop with int indices, for preln leak, sumshrink, sumshrink p=0.5, move, sum f32
print("== A. scan path vs Python-int loop")
SKIP_A = os.environ.get("SKIP_A") == "1"
for arch, kw, hyper in (("preln", dict(), False), ("hc", dict(), True), ("preln", dict(depth_preln_leak=True), False), ("preln", dict(depth_preln_leak=True, depth_shrink_power=0.5), False),
                        ("hc", dict(depth_hc_res="sumshrink"), True), ("hc", dict(depth_hc_res="sumshrink", depth_shrink_power=0.5), True),
                        ("hc", dict(depth_hc_res="move"), True), ("hc", dict(depth_hc_res="sum", depth_hc_state_f32=True), True)):
    if SKIP_A:
        break
    m_ = model(arch, **kw)
    if hyper:
        m_ = rand_hyper(m_, 7)
    for dt_name, cast in (("f32", lambda z: z), ("bf16", mp.cast_to_compute)):
        mc = cast(m_)
        tr = mc.transformer
        x = mc.embeddings.embed(ex.tokens)
        o_scan = tr.outputs(x, mask)
        outs = []
        if arch == "preln":
            h = x
            for i in range(L):
                h = tr.layer(i).residual_step(h, mask, key=None, pos_ids=None, layer=i)
                outs.append(h)
        else:
            H = hax.stack(Stream, [x] * M)
            f32s = bool(kw.get("depth_hc_state_f32"))
            if f32s:
                H = H.astype(jnp.float32)
            for i in range(L):
                H = tr.layer(i).hyper_step(H, mask, key=None, pos_ids=None, layer=i).rearrange(H.axes)
                outs.append(hax.sum(H, axis="stream").astype(x.dtype) if f32s else hax.sum(H, axis="stream"))
        o_loop = hax.stack(tr.config.Layers, outs)
        r = rel(o_scan.rearrange(o_loop.axes).array.astype(jnp.float32), o_loop.array.astype(jnp.float32))
        # wrong-index control: the same loop with layer index 0 everywhere must differ (checks the index matters)
        print(f"{arch} {kw} {dt_name}: scan vs loop rel {r:.1e}; outputs dtype {o_scan.array.dtype}")

# A2. the per-sublayer gain the leak applies: recover g from h_out - F on one layer with traced index
# A1. control: what a wrong layer index gives (sumshrink, f32): loop with layer i+1 gains; and dif/sum of the bf16 streams
print("== A1. wrong-index control and dif/sum in the random-parameter sumshrink test")
m_ = rand_hyper(model("hc", depth_hc_res="sumshrink"), 7)
for dt_name, cast in (("f32", lambda z: z), ("bf16", mp.cast_to_compute)):
    mc = cast(m_); tr = mc.transformer; x = mc.embeddings.embed(ex.tokens)
    o_scan = tr.outputs(x, mask)
    H = hax.stack(Stream, [x] * M); outs = []; rat = []
    for i in range(L):
        H = tr.layer(i).hyper_step(H, mask, key=None, pos_ids=None, layer=(i + 1) % L).rearrange(H.axes)
        outs.append(hax.sum(H, axis="stream"))
        a = np.asarray(H.array.astype(jnp.float32), np.float64); d = a - a.mean(0, keepdims=True)
        rat.append(np.linalg.norm(d) / np.linalg.norm(a.sum(0)))
    o_loop = hax.stack(tr.config.Layers, outs)
    print(f"{dt_name}: scan vs loop with a shifted layer index rel {rel(o_scan.rearrange(o_loop.axes).array.astype(jnp.float32), o_loop.array.astype(jnp.float32)):.1e}; |dif|/|sum| per layer {np.round(rat, 1)}")

print("== A2. leak gain recovered per sublayer equals HC_SUM_GAIN[2l+j] (traced index in a jitted scan-free call)")
m_ = model("preln", depth_preln_leak=True)
for i in (0, 3, 5):
    lay = m_.transformer.layer(i)
    h = hax.named(jnp.asarray(rng.standard_normal((B, T, D)), jnp.float32), (Batch, Pos, Embed))
    f0 = lay.branch(0, h, mask, key=None, pos_ids=None)
    h1 = jax.jit(lambda h_, li: lay.residual_step(h_, mask, key=None, pos_ids=None, layer=li))(h, jnp.asarray(i))
    # h1 = g1 (g0 h + f0) + f1(g0 h + f0); compute expected
    hm = HC_SUM_GAIN[2 * i] * h + f0
    hm = HC_SUM_GAIN[2 * i + 1] * hm + lay.branch(1, hm, mask, key=None, pos_ids=None)
    print(f"layer {i}: traced-index leak vs manual rel {rel(h1.array, hm.rearrange(h1.axes).array):.1e}")

# B. move at init: the mixing is not the identity
print("== B. move at init")
sp = lambda a: np.logaddexp(0.0, HC_MOVE_BETA * a) / HC_MOVE_BETA  # noqa: E731
A = np.eye(M)
Pm = sp(A) + 1e-6 * np.eye(M)
Pm = Pm / Pm.sum(1, keepdims=True)
ev = np.sort(np.abs(np.linalg.eigvals(Pm)))
print(f"P diag {Pm[0, 0]:.4f}, off {Pm[0, 1]:.4f}; eigenvalues {ev}; difference-mode gain per sublayer {ev[0]:.4f}, over 96 sublayers {ev[0] ** 96:.2e}")
# through the model: distinct streams, writes off, default (init) mixing params
m_mv, m_hc = model("hc", depth_hc_res="move"), model("hc")
def nowrite(m_):
    hy = m_.transformer.layers.stacked.hyper
    new = [dataclasses.replace(p, write=hax.named(jnp.full(p.write.array.shape, -1.0), p.write.axes)) for p in hy]
    return eqx.tree_at(lambda mm: mm.transformer.layers.stacked.hyper, m_, tuple(new))
H0 = hax.named(jnp.asarray(rng.standard_normal((M, B, T, D)), jnp.float32), (Stream, Batch, Pos, Embed))
d0 = np.asarray(H0.array) - np.asarray(H0.array).mean(0, keepdims=True)
for name, m_ in (("move", m_mv), ("hc learned", m_hc)):
    H1 = nowrite(m_).transformer.layer(0).hyper_step(H0, mask, key=None, pos_ids=None, layer=0).rearrange(H0.axes)
    a = np.asarray(H1.array)
    d1 = a - a.mean(0, keepdims=True)
    print(f"{name}: one layer (2 sublayers) at init params, writes off: stream sum rel {rel(a.sum(0), np.asarray(H0.array).sum(0)):.1e}; difference norm x {np.linalg.norm(d1) / np.linalg.norm(d0):.4f}")
# res offsets needed to bring the off-diagonal weight to 1e-3
for off in (0.0, -0.1, -0.2, -0.3):
    Pq = sp(np.eye(M) + off * (1 - np.eye(M))) + 1e-6 * np.eye(M)
    Pq = Pq / Pq.sum(1, keepdims=True)
    print(f"  off-diagonal res offset {off:+.1f}: P off {Pq[0, 1]:.2e}, difference gain/sublayer {1 - M * Pq[0, 1]:.4f}")

# C. move numerics: a far-negative row; gradient through a row whose sum is near the 1e-6 guard
print("== C. move numerics")
def P_of(res_row):
    pos = jax.nn.softplus(HC_MOVE_BETA * res_row) / HC_MOVE_BETA + 1e-6 * jnp.eye(M)[0]
    return pos / pos.sum()
for row in ([-0.5, -5, -5, -5], [-1.5, -0.6, -5, -5], [-3, -3, -3, -3]):
    r_ = jnp.asarray(row, jnp.float32)
    P = P_of(r_)
    J = jax.jacobian(P_of)(r_)
    print(f"row {row}: P {np.round(np.asarray(P), 4)}, finite {bool(jnp.all(jnp.isfinite(J)))}, max |dP/dres| {float(jnp.max(jnp.abs(J))):.2e}")

# D. hcsumf32 under the real policy: dtypes of the carry, the outputs, the gradients; dot_generals on float32 operands
print("== D. hcsumf32 dtypes")
m_f = rand_hyper(model("hc", depth_hc_res="sum", depth_hc_state_f32=True), 3)
mc = mp.cast_to_compute(m_f)
x = mc.embeddings.embed(ex.tokens)
print("embedding dtype", x.array.dtype)
H32 = hax.stack(Stream, [x] * M).astype(jnp.float32)
H1 = mc.transformer.layer(0).hyper_step(H32, mask, key=None, pos_ids=None, layer=0)
print("carry after one layer", H1.array.dtype, "outputs", mc.transformer.outputs(x, mask).array.dtype)
def lf(m_):
    loss, _ = m_.compute_next_token_loss(ex, key=jrandom.PRNGKey(3), logsumexp_weight=0.0)
    return loss.array if isinstance(loss, hax.NamedArray) else loss
jaxpr = jax.make_jaxpr(lambda mm: eqx.filter_value_and_grad(lf)(mm))(mc)
f32dots = []
def walk(jp):
    for eqn in jp.eqns:
        if eqn.primitive.name == "dot_general":
            dts = [v.aval.dtype for v in eqn.invars]
            if all(d == jnp.float32 for d in dts):
                f32dots.append((tuple(v.aval.shape for v in eqn.invars), eqn.params.get("precision")))
        for p in eqn.params.values():
            if isinstance(p, jcore.ClosedJaxpr):
                walk(p.jaxpr)
            elif isinstance(p, jcore.Jaxpr):
                walk(p)
            elif isinstance(p, (tuple, list)):
                for q in p:
                    if isinstance(q, jcore.ClosedJaxpr):
                        walk(q.jaxpr)
                    elif isinstance(q, jcore.Jaxpr):
                        walk(q)
walk(jaxpr.jaxpr)
shapes = {}
for s, prec in f32dots:
    shapes.setdefault((s, str(prec)), 0)
    shapes[(s, str(prec))] += 1
print(f"float32 x float32 dot_generals in forward+backward: {len(f32dots)}")
for (s, prec), n in sorted(shapes.items(), key=lambda kv: -kv[1])[:12]:
    print(f"  {n:3d} x shapes {s} precision {prec}")
loss, g = eqx.filter_value_and_grad(lf)(mc)
g = mp.cast_to_param(g)
dts = {str(a.dtype) for a in jax.tree_util.tree_leaves(eqx.filter(g, eqx.is_inexact_array))}
print("loss", float(loss), "grad dtypes after cast_to_param", dts)

# E. optimizer: labels of the HC params; sumshrink res / w_res with zero gradient over several Adam steps, eps 1e-8 and eps 1e-20
print("== E. optimizer")
for kw in (dict(depth_hc_res="sumshrink"), dict(depth_hc_res="move")):
    m_ = rand_hyper(model("hc", pls_detach_backbone=True, **kw), 5)
    opt_cfg = DepthArchMuonHConfig(learning_rate=0.02, adam_lr=0.008, beta1=0.9, beta2=0.98, epsilon=1e-20, momentum=0.95, nesterov=True, backend_steps=5, muon_epsilon=1e-5, max_grad_norm=1.0,
                                   weight_decay=0.1, lr_schedule="linear", decay=0.8, warmup=0, min_lr_ratio=0.0, coefficient_type="simple")
    params = eqx.filter(m_, eqx.is_inexact_array)
    labels = opt_cfg.create_mask(params)
    hl = labels.transformer.layers.stacked.hyper[0]
    print(kw, "labels res/w_res/read/scale:", hl.res, hl.w_res, hl.read, hl.scale)
    opt = opt_cfg.build(4959)
    st = opt.init(params)
    res0 = [np.asarray(p.res.array) for p in m_.transformer.layers.stacked.hyper]
    for s in range(3):
        loss, g = eqx.filter_value_and_grad(lf)(mp.cast_to_compute(m_))
        g = mp.cast_to_param(g)
        up, st = opt.update(g, st, params=eqx.filter(m_, eqx.is_inexact_array))
        m_ = eqx.apply_updates(m_, up)
    leaves = jax.tree_util.tree_leaves(eqx.filter(m_, eqx.is_inexact_array))
    fin = all(bool(jnp.all(jnp.isfinite(a))) for a in leaves)
    dres = max(float(np.abs(np.asarray(p.res.array) - r0).max()) for p, r0 in zip(m_.transformer.layers.stacked.hyper, res0))
    print(f"  after 3 steps: all finite {fin}; max |change of res| {dres:.2e}; loss {float(loss):.4f}")
print("DONE")
