"""Causal check of w810's blow-up at the layer-5 MLP, on the final checkpoint (read-only, CPU). Rerun layers 1-4 and the layer-5 attention as
trained; at the layer-5 MLP compute the branch output for the blown tokens and the median token (a) as trained; (b) with ln_2 coordinate 16 of
layer 5 set to the layer's median scale. Also the MLP's input peak at coordinate 16, and the gate and up outputs. Then the same layer with
that coordinate fixed, all the way to layer 48: CE at the head on the tokens that blew up at layer 5 and on the others, against the trained
model. Same text as pls_write_size.py."""
import dataclasses, json, os, sys, glob
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import equinox as eqx, jax, jax.numpy as jnp, jax.random as jrandom, numpy as np
import haliax as hax
from haliax import Axis
from haliax.partitioning import ResourceAxis
from levanter.checkpoint import load_checkpoint
from levanter.layers.attention import AttentionBackend, AttentionMask
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from experiments.references.per_layer_qwen3 import PerLayerQwen3Config, PerLayerQwen3LMHeadModel
from experiments.references.depth_arch_qwen3 import _hc_constants, _rms
import tensorstore as ts
B, T = 4, 1024
tag = "-sep-ahc-hcid-fa4thd-vs-d48"
S_ = "/scratch/gpfs/GROUP/USER/marin_store_big/speedrun/muonh-qwen3-130m-della4xh100-pls1"
VAL = "/scratch/gpfs/GROUP/USER/marin_store_big/tokenized/paloma/c4_en-cf1f79/validation/part-00000-of-00001/input_ids/data"
tok = np.asarray(ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": VAL}}).result()[: B * T].read().result(), np.int64).reshape(B, T)
eos = np.roll(tok, 1, axis=1) == 128001
eos[:, 0] = False
Batch, Pos = Axis("batch", B), Axis("position", T)
mask = AttentionMask.causal().with_segment_ids(hax.named(jnp.asarray(np.cumsum(eos, axis=1), jnp.int32), (Batch, Pos)))
tc = TrainerConfig(mesh=MeshConfig(axes={"data": -1, "replica": 1, "model": 1}, compute_mapping={
    "token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
    "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA)}))
tc.use_device_mesh().__enter__()
hax.axis_mapping(tc.parameter_axis_mapping).__enter__()
Vocab = Axis("vocab", 128256)
fields = {f.name for f in dataclasses.fields(PerLayerQwen3Config)}
ck = sorted(glob.glob(f"{S_}{tag}/*/checkpoints/step-4958"))[0]
wcfg = json.load(open(f"/scratch/gpfs/GROUP/USER/tmp/pls_paths/cfg{tag}.json"))["model"]
kw = {k: v for k, v in wcfg.items() if k in fields and not isinstance(v, (dict, list)) and v is not None and k not in ("attn_backend", "max_seq_len", "tokenizer", "reference_checkpoint")}
cfg = PerLayerQwen3Config(max_seq_len=T, attn_backend=AttentionBackend.VANILLA, **kw)
model = load_checkpoint(eqx.filter_eval_shape(PerLayerQwen3LMHeadModel.init, Vocab, cfg, key=jrandom.PRNGKey(0)), ck, subpath="model")
tr = model.transformer
Embed, m = cfg.Embed, cfg.hc_streams
x = model.embeddings.embed(hax.named(jnp.asarray(tok, jnp.int32), (Batch, Pos)))
tgt = jnp.asarray(tok[:, 1:])
def ce_of(mdl):
    outs = eqx.filter_jit(lambda tr_, x_: tr_.outputs(x_, mask))(mdl.transformer, x)
    s = outs[cfg.Layers.name, -1]
    lp = jax.nn.log_softmax(mdl.lm_head(mdl.transformer.norm(s)).array.astype(jnp.float32)[:, :-1], axis=-1)
    ce = -np.asarray(jnp.take_along_axis(lp, tgt[..., None], axis=-1)[..., 0], np.float64)
    n = np.linalg.norm(np.asarray(outs.rearrange((cfg.Layers.name, Batch, Pos, Embed)).array, np.float64), axis=-1)
    return ce, n
ce0, n0 = ce_of(model)
med = np.median(n0, axis=(1, 2))
blown5 = n0[4] > 100 * med[4]
blown48 = n0[-1] > 100 * med[-1]
print("trained: CE", round(ce0.mean(), 4), "tokens blown at L5", int(blown5.sum()), "at L48", int(blown48.sum()), flush=True)
A = lambda a: np.asarray(a.rearrange((Batch, Pos, Embed)).array, np.float64).reshape(B * T, -1)
H = hax.stack(Axis("stream", m), [x] * m)
f32 = lambda a: a.astype(jnp.float32)
for li in range(5):
    lay = tr.layer(li)
    for j in range(2):
        p = lay.hyper[j]
        c_read, c_res, c_write = _hc_constants(m, 2 * li + j)
        Hn = f32(_rms(H, Embed)); sc = f32(p.scale)
        read = c_read + f32(p.read) + sc["hp_scale", 0] * hax.tanh(hax.dot(Hn, f32(p.w_read), axis=Embed))
        write = c_write + f32(p.write) + sc["hp_scale", 1] * hax.tanh(hax.dot(Hn, f32(p.w_write), axis=Embed).rename({"stream": "stream_out"}))
        xr = hax.dot(read.astype(H.dtype), H, axis="stream")
        if li == 4 and j == 1:
            w = np.asarray(lay.ln_2.weight.array, np.float64)
            k = int(np.abs(w).argmax()); wm = float(np.median(np.abs(w)))
            top = np.nonzero(blown5.reshape(-1))[0]; rest = np.nonzero(~blown5.reshape(-1))[0]
            order = np.argsort(np.abs(w))[::-1][:5]
            print("layer 5 ln_2: largest coordinates", [(int(i), round(float(w[i]), 2)) for i in order], "median |.|", round(wm, 3), "rms", round(float(np.sqrt((w ** 2).mean())), 3))
            def report(name, layer):
                xn = layer.ln_2(xr)
                g = layer.mlp.act(layer.mlp.gate_proj(xn)); u = layer.mlp.up_proj(xn)
                f = layer.branch(1, xr, mask, key=None, pos_ids=None)
                Xn, Fn = A(xn), A(f)
                ax = [a for a in g.axes if a.name not in ("batch", "position")][0]
                Gn = np.asarray(g.rearrange((Batch, Pos, ax)).array, np.float64).reshape(B * T, -1); Un = np.asarray(u.rearrange((Batch, Pos, ax)).array, np.float64).reshape(B * T, -1)
                st = lambda v, i: float(np.median(np.linalg.norm(v[i], axis=1)))
                print(f"{name}: blown tokens n={len(top)}: |RN(x)| {st(Xn, top):.3g}, coordinate {k} {np.median(np.abs(Xn[top, k])):.3g}, |act(gate)| {st(Gn, top):.3g}, |up| {st(Un, top):.3g}, |F| {st(Fn, top):.3g} | other tokens: |RN(x)| {st(Xn, rest):.3g}, coordinate {k} {np.median(np.abs(Xn[rest, k])):.3g}, |act(gate)| {st(Gn, rest):.3g}, |up| {st(Un, rest):.3g}, |F| {st(Fn, rest):.3g}, max |F| {np.linalg.norm(Fn[rest], axis=1).max():.3g}", flush=True)
            report("as trained", lay)
            fixed = eqx.tree_at(lambda l: l.ln_2.weight, lay, hax.named(lay.ln_2.weight.array.at[k].set(wm * np.sign(w[k])), lay.ln_2.weight.axes))
            report(f"coordinate {k} set to the median scale", fixed)
        f = lay.branch(j, xr, mask, key=None, pos_ids=None)
        upd = write.astype(H.dtype) * f.broadcast_axis(write.resolve_axis("stream_out"))
        H = H + upd.rename({"stream_out": "stream"})
# the whole model with that one coordinate fixed
stack = tr.layers.stacked
W = stack.ln_2.weight
ax = (cfg.Layers.name, cfg.Embed.name)
arr = np.array(W.rearrange(ax).array)
k5 = int(np.abs(arr[4]).argmax())
arr[4, k5] = np.median(np.abs(arr[4])) * np.sign(arr[4, k5])
m2 = eqx.tree_at(lambda mm: mm.transformer.layers.stacked.ln_2.weight, model, hax.named(jnp.asarray(arr), ax).rearrange(W.axes))
ce1, n1 = ce_of(m2)
b = blown5[:, :-1]
print(f"layer-5 coordinate fixed, whole model: CE {ce1.mean():.4f} (trained {ce0.mean():.4f}); on the tokens blown at L5 {ce1[b].mean():.3f} (trained {ce0[b].mean():.3f}); on the others {ce1[~b].mean():.4f} (trained {ce0[~b].mean():.4f}); tokens over 100 x median at L5 {int((n1[4] > 100 * np.median(n1[4])).sum())}, at L48 {int((n1[-1] > 100 * np.median(n1[-1])).sum())}", flush=True)
