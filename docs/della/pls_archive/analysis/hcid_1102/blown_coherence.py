"""Is the gradient of w810's blown tokens coherent or noise? Final checkpoint (read-only, CPU), 4 x 1024 tokens of Paloma c4_en validation. The blown tokens
are split into two random halves A and B. Training-loss gradients (all heads) with loss_weight 1 only on A, only on B, and only on the other tokens. Per
layer on down_proj and o_proj: the norms and the cosines A-B, A-other, B-other. The same for random disjoint sets of the same size among the other tokens,
as the noise floor. Each gradient is the mean over its own weighted tokens. Output: JSON and a table.
    JAX_PLATFORMS=cpu nice -n 19 $PY blown_coherence.py OUT.json"""
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
valid = np.ones((B, T), bool); valid[:, -1] = False
blown = (n[-1] > 100 * np.median(n[-1])) & valid
rng = np.random.default_rng(0)
idx = np.argwhere(blown); rng.shuffle(idx)
def sets(ix):
    k = len(ix) // 2; m1 = np.zeros((B, T), np.float32); m2 = np.zeros((B, T), np.float32)
    m1[tuple(ix[:k].T)] = 1; m2[tuple(ix[k:2 * k].T)] = 1
    return m1, m2
A, Bb = sets(idx)
oth = np.argwhere(valid & ~blown); rng.shuffle(oth)
C, D = sets(oth[: len(idx)])
other = (valid & ~blown).astype(np.float32)
print("tokens: A", int(A.sum()), "B", int(Bb.sum()), "C", int(C.sum()), "D", int(D.sum()), "other", int(other.sum()), flush=True)
G = {k: grads(model, v)[1] for k, v in (("A", A), ("B", Bb), ("C", C), ("D", D), ("other", other))}
def cos(a, b): return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
print("layer  |A|  |B|  |C|  |D|  |other|  cos(A,B)  cos(C,D)  cos(A,other)  cos(B,other)")
rows = {}
for l in [0, 1, 2, 3, 4, 5, 7, 9, 15, 23]:
    r = dict(A=float(np.linalg.norm(G["A"][l])), B=float(np.linalg.norm(G["B"][l])), C=float(np.linalg.norm(G["C"][l])), D=float(np.linalg.norm(G["D"][l])), other=float(np.linalg.norm(G["other"][l])),
             cos_AB=cos(G["A"][l], G["B"][l]), cos_CD=cos(G["C"][l], G["D"][l]), cos_A_other=cos(G["A"][l], G["other"][l]), cos_B_other=cos(G["B"][l], G["other"][l]))
    rows[l + 1] = r
    print(f"L{l+1:<3d} {r['A']:8.1f} {r['B']:8.1f} {r['C']:8.1f} {r['D']:8.1f} {r['other']:8.2f}   {r['cos_AB']:+.3f}   {r['cos_CD']:+.3f}   {r['cos_A_other']:+.3f}   {r['cos_B_other']:+.3f}", flush=True)
json.dump(rows, open(outp, "w"), indent=1)
