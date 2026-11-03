"""Do w810's blown tokens (about 3% of the tokens, state norm > 100 x median) carry the gradient growth in the first layers? On the final
checkpoint (read-only, CPU): the training loss gradient (all heads, as trained) on 4 x 1024 tokens of Paloma c4_en validation, once with every
token and once with loss_weight 0 on the blown tokens. Per layer: gradient norm of the branch weights (down_proj, o_proj) with all tokens and
without the blown tokens, and the cosine between the two. Then the same for HC separate heads as a control (with the same tokens removed).
Masking removes these tokens as prediction targets only; they stay in the inputs. Output: JSON and a table.
    JAX_PLATFORMS=cpu nice -n 19 $PY blown_grad.py OUT.json"""
import dataclasses, json, os, sys, glob, time
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS", "0")
import equinox as eqx, jax, jax.numpy as jnp, jax.random as jrandom, numpy as np
import haliax as hax
from haliax import Axis
from haliax.partitioning import ResourceAxis
from levanter.checkpoint import load_checkpoint
from levanter.layers.attention import AttentionBackend, AttentionMask
from levanter.models.lm_model import LmExample
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from experiments.references.per_layer_qwen3 import PerLayerQwen3Config, PerLayerQwen3LMHeadModel
import tensorstore as ts
outp = sys.argv[1]
B, T = 4, 1024
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
def load(tag):
    ck = sorted(glob.glob(f"{S_}{tag}/*/checkpoints/step-4958"))[0]
    wcfg = json.load(open(f"/scratch/gpfs/GROUP/USER/tmp/pls_paths/cfg{tag}.json"))["model"]
    kw = {k: v for k, v in wcfg.items() if k in fields and not isinstance(v, (dict, list)) and v is not None and k not in ("attn_backend", "max_seq_len", "tokenizer", "reference_checkpoint")}
    cfg = PerLayerQwen3Config(max_seq_len=T, attn_backend=AttentionBackend.VANILLA, scan_layers=True, **{k: v for k, v in kw.items() if k != "scan_layers"})
    return cfg, load_checkpoint(eqx.filter_eval_shape(PerLayerQwen3LMHeadModel.init, Vocab, cfg, key=jrandom.PRNGKey(0)), ck, subpath="model")
def grads(model, lw):
    ex = LmExample(tokens=hax.named(jnp.asarray(tok, jnp.int32), (Batch, Pos)), loss_weight=hax.named(jnp.asarray(lw, jnp.float32), (Batch, Pos)), attn_mask=mask)
    def lf(m_):
        out = m_.compute_next_token_loss(ex, key=jrandom.PRNGKey(3), logsumexp_weight=0.0)   # a key selects the training loss, all heads
        loss = out[0] if isinstance(out, tuple) else out
        return loss.array if isinstance(loss, hax.NamedArray) else loss
    t0 = time.time()
    loss, g = eqx.filter_jit(eqx.filter_value_and_grad(lf))(model)
    print(f"  loss {float(loss):.3f}, {time.time() - t0:.0f} s", flush=True)
    st = g.transformer.layers.stacked
    out = {}
    for name, w in (("down_proj", st.mlp.down_proj.weight), ("o_proj", st.self_attn.o_proj.weight)):
        ax = [a.name for a in w.axes]
        a = np.asarray(w.rearrange((cfg.Layers.name, ...)).array, np.float64)
        out[name] = a.reshape(a.shape[0], -1)
    return float(loss), np.concatenate([out["down_proj"], out["o_proj"]], axis=1)
res = {}
tag = "-sep-ahc-hcid-fa4thd-vs-d48"
cfg, model = load(tag)
tr = model.transformer
x = model.embeddings.embed(hax.named(jnp.asarray(tok, jnp.int32), (Batch, Pos)))
outs = eqx.filter_jit(lambda tr_, x_: tr_.outputs(x_, mask))(tr, x)
n = np.linalg.norm(np.asarray(outs.rearrange((cfg.Layers.name, Batch, Pos, cfg.Embed)).array, np.float64), axis=-1)
blown = n[-1] > 100 * np.median(n[-1])
print("blown tokens", int(blown.sum()), flush=True)
# a token is the input of the next position's target: drop the targets that the blown states predict (the loss weight sits on the input position)
lw_all = np.ones((B, T), np.float32); lw_all[:, -1] = 0
lw_no = lw_all * (~blown)
LS = [0, 1, 2, 3, 4, 5, 7, 9, 10, 13, 15, 23, 35, 47]
def run(t, m_):
    print(t, flush=True)
    l0, g0 = grads(m_, lw_all)
    l1, g1 = grads(m_, lw_no)
    print(f"  loss all {l0:.3f}; without blown tokens {l1:.3f}")
    print("  layer  |g all|  |g without|  cos")
    rows = {}
    for l in LS:
        a, b = g0[l], g1[l]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        c = float(a @ b / (na * nb + 1e-30))
        rows[l + 1] = dict(all=float(na), without=float(nb), cos=c)
        print(f"  L{l+1:<4d} {na:9.3f} {nb:9.3f}  {c:+.3f}", flush=True)
    res[t] = dict(loss_all=l0, loss_without=l1, rows=rows)
    json.dump(res, open(outp, "w"), indent=1)
run(tag, model)
del model
cfg2, m2 = load("-sep-ahc-fa4thd-vs-d48")
run("-sep-ahc-fa4thd-vs-d48", m2)
