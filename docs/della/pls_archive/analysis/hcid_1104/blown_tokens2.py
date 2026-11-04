"""In w810's final checkpoint about 2-3% of the tokens have a state norm far above the rest from layer 5 on. Do they explain the
layer-48 gap to HC? Same text as pls_write_size.py (Paloma c4_en validation, 4 x 1024 tokens). Blown token: |s_48| > 100 x the median
|s_48| of that run. Per-token next-token CE at layer 48 (final norm + lm_head). Then the sublayer that writes the blow-up: read weights,
RN(x) peak, branch output norm and write weights at the first layer where the token's norm jumps.
    JAX_PLATFORMS=cpu nice -n 19 $PY blown_tokens.py OUT.json TAG_A [TAG_REF ...]   (TAG_A the run examined; the others give CE on the same tokens)"""
import dataclasses, json, os, sys, time, glob
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
import tensorstore as ts
outp = sys.argv[1]; tags = sys.argv[2:]
B, T = int(os.environ.get("WS_B", 4)), 1024
S_ = "/scratch/gpfs/GROUP/USER/marin_store_big/speedrun/muonh-qwen3-130m-della4xh100-pls1"
T_ = "/scratch/gpfs/GROUP/USER/marin_store_big/tmp/checkpoints-temp" + S_
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
tgt = jnp.asarray(tok[:, 1:])
def load(tag):
    ck = sorted(glob.glob(f"{S_}{tag}/*/checkpoints/step-*") + glob.glob(f"{T_}{tag}/*/checkpoints/step-*"), key=lambda p: int(p.rsplit("-", 1)[1]))[-1]; print("checkpoint", ck, flush=True)
    wcfg = json.load(open(f"/scratch/gpfs/GROUP/USER/tmp/pls_paths/cfg{tag}.json"))["model"]
    kw = {k: v for k, v in wcfg.items() if k in fields and not isinstance(v, (dict, list)) and v is not None and k not in ("attn_backend", "max_seq_len", "tokenizer", "reference_checkpoint")}
    cfg = PerLayerQwen3Config(max_seq_len=T, attn_backend=AttentionBackend.VANILLA, **kw)
    return cfg, load_checkpoint(eqx.filter_eval_shape(PerLayerQwen3LMHeadModel.init, Vocab, cfg, key=jrandom.PRNGKey(0)), ck, subpath="model")
def forward(model, cfg):
    tr = model.transformer
    x = model.embeddings.embed(hax.named(jnp.asarray(tok, jnp.int32), (Batch, Pos)))
    outs = eqx.filter_jit(lambda tr_, x_: tr_.outputs(x_, mask))(tr, x)
    s = np.asarray(outs.rearrange((cfg.Layers.name, ..., cfg.Embed.name)).array, np.float32).astype(np.float64)
    last = hax.named(jnp.asarray(s[-1], jnp.float32), (Batch, Pos, cfg.Embed))
    logits = model.lm_head(tr.norm(last)).array.astype(jnp.float32)
    lp = jax.nn.log_softmax(logits[:, :-1], axis=-1)
    ce = -np.asarray(jnp.take_along_axis(lp, tgt[..., None], axis=-1)[..., 0], np.float64)   # (B, T-1)
    e = np.asarray(x.array, np.float32).astype(np.float64)
    return s, e, ce, x
res = {}
cfg, model = load(tags[0])
s, e, ce, x = forward(model, cfg)
nL = np.linalg.norm(s, axis=-1)   # (L, B, T)
blown = nL[-1] > 100 * np.median(nL[-1])
first = np.full(blown.shape, -1)
med = np.median(nL, axis=(1, 2))
over = nL > 100 * med[:, None, None]
for b, t in zip(*np.nonzero(blown)):
    first[b, t] = int(np.argmax(over[:, b, t]))
bl = blown[:, :-1]
print(tags[0], f"blown {blown.sum()} of {blown.size} ({blown.mean():.2%}); CE all {ce.mean():.4f}, blown {ce[bl].mean():.3f}, rest {ce[~bl].mean():.4f}", flush=True)
print("first layer over 100 x median (1-based), counts:", {int(k) + 1: int(v) for k, v in zip(*np.unique(first[blown], return_counts=True))})
res[tags[0]] = dict(n=int(blown.sum()), ce_all=float(ce.mean()), ce_blown=float(ce[bl].mean()), ce_rest=float(ce[~bl].mean()))
ids, cnt = np.unique(tok[blown], return_counts=True)
o = np.argsort(-cnt)[:15]
allc = {int(i): int(c) for i, c in zip(*np.unique(tok, return_counts=True))}
print("input ids of blown tokens (id, blown count, count in the text):", [(int(ids[i]), int(cnt[i]), allc[int(ids[i])]) for i in o])
try:
    from transformers import AutoTokenizer
    tz = AutoTokenizer.from_pretrained("stanford-crfm/marin-tokenizer", local_files_only=True)
    print("decoded:", [tz.decode([int(ids[i])]) for i in o])
except Exception as ex:
    print("no tokenizer:", type(ex).__name__)
for ref in tags[1:]:
    c2, m2 = load(ref)
    s2, e2, ce2, _ = forward(m2, c2)
    n2 = np.linalg.norm(s2, axis=-1)
    b2 = n2[-1] > 100 * np.median(n2[-1])
    print(ref, f"own blown {b2.sum()}; CE all {ce2.mean():.4f}; on {tags[0]}'s blown tokens {ce2[bl].mean():.3f}, on its other tokens {ce2[~bl].mean():.4f}; gap from blown tokens {(ce[bl].sum() - ce2[bl].sum()) / ce.size:+.4f}, from the rest {(ce[~bl].sum() - ce2[~bl].sum()) / ce.size:+.4f}; L48 norm/e median {np.median(n2[-1] / np.linalg.norm(e2, axis=-1)):.1f}, max {(n2[-1] / np.linalg.norm(e2, axis=-1)).max():.1f}", flush=True)
    res[ref] = dict(n=int(b2.sum()), ce_all=float(ce2.mean()), ce_on_blown=float(ce2[bl].mean()), ce_on_rest=float(ce2[~bl].mean()))
    del m2, s2
# the sublayer behind the jump: rerun the first layers of the examined run, one sublayer at a time, as hyper_step does
tr = model.transformer
if cfg.depth_arch == "hc":
    H = hax.stack(Axis("stream", cfg.hc_streams), [x] * cfg.hc_streams)
    Embed = cfg.Embed; m = cfg.hc_streams
    from experiments.references.depth_arch_qwen3 import _hc_constants, _rms
    tops = np.argsort(-nL[-1].reshape(-1))[:5]
    typ = np.argsort(nL[-1].reshape(-1))[blown.size // 2]
    upto = int(np.bincount(first[blown]).argmax()) + 1
    f32 = lambda a: a.astype(jnp.float32)
    for li in range(upto):
        lay = tr.layer(li)
        for j in range(2):
            p = lay.hyper[j]
            c_read, c_res, c_write = _hc_constants(m, 2 * li + j)
            Hn = f32(_rms(H, Embed))
            sc = f32(p.scale)
            read = c_read + f32(p.read) + sc["hp_scale", 0] * hax.tanh(hax.dot(Hn, f32(p.w_read), axis=Embed))
            write = c_write + f32(p.write) + sc["hp_scale", 1] * hax.tanh(hax.dot(Hn, f32(p.w_write), axis=Embed).rename({"stream": "stream_out"}))
            xr = hax.dot(read.astype(H.dtype), H, axis="stream")
            f = lay.branch(j, xr, mask, key=None, pos_ids=None)
            upd = write.astype(H.dtype) * f.broadcast_axis(write.resolve_axis("stream_out"))
            H = H + upd.rename({"stream_out": "stream"})
            if li >= upto - 2:
                A = lambda a: np.asarray(a.rearrange((Batch, Pos, Embed)).array, np.float64).reshape(B * T, -1)
                xa, fa = A(xr), A(f)
                wa = np.asarray(write.rearrange((Batch, Pos, "stream_out")).array, np.float64).reshape(B * T, m)
                ra = np.asarray(read.rearrange((Batch, Pos, "stream")).array, np.float64).reshape(B * T, m)
                pr = lambda i: f"|x| {np.linalg.norm(xa[i]):.3g} peak/rms {np.abs(xa[i]).max() / (np.sqrt((xa[i] ** 2).mean()) + 1e-30):.1f}, |F| {np.linalg.norm(fa[i]):.3g} F peak/rms {np.abs(fa[i]).max() / (np.sqrt((fa[i] ** 2).mean()) + 1e-30):.1f} argmax {int(np.abs(fa[i]).argmax())}, sum w {wa[i].sum():.3g}, w {np.round(wa[i], 2).tolist()}, r {np.round(ra[i], 2).tolist()}"
                print(f"layer {li + 1} sublayer {j} ({'attention' if j == 0 else 'MLP'}): median token: {pr(typ)}")
                for i in tops[:3]:
                    print(f"    token {int(i)} id {int(tok.reshape(-1)[i])}: {pr(i)}", flush=True)
json.dump(res, open(outp, "w"), indent=1)
