"""About 80% of the layer-48 gap between HC mixing = I and HC, separate heads, sits on tokens below the blown line (|s_48| > 100 x median). Is the rest of the gap also in the tail of the state norm, or spread over all tokens?
Text: Paloma c4_en validation, WS_B x 1024 tokens (same start as blown_tokens.py). Per-token next-token CE at layer 48 (final norm + lm_head) on the same tokens in run A and in the reference.
Table 1: by |s_48| relative to run A's median: share of tokens, CE in A, CE in the reference, contribution to the gap. Table 2: by quartile of the reference's CE (easy to hard tokens).
    JAX_PLATFORMS=cpu nice -n 19 $PY gap_by_norm.py TAG_A TAG_REF"""
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
import tensorstore as ts
B, T = int(os.environ.get("WS_B", 16)), 1024
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
def run(tag):
    ck = sorted(glob.glob(f"{S_}{tag}/*/checkpoints/step-*") + glob.glob(f"{T_}{tag}/*/checkpoints/step-*"), key=lambda p: int(p.rsplit("-", 1)[1]))[-1]
    print("checkpoint", ck, flush=True)
    wcfg = json.load(open(f"/scratch/gpfs/GROUP/USER/tmp/pls_paths/cfg{tag}.json"))["model"]
    kw = {k: v for k, v in wcfg.items() if k in fields and not isinstance(v, (dict, list)) and v is not None and k not in ("attn_backend", "max_seq_len", "tokenizer", "reference_checkpoint")}
    cfg = PerLayerQwen3Config(max_seq_len=T, attn_backend=AttentionBackend.VANILLA, **kw)
    model = load_checkpoint(eqx.filter_eval_shape(PerLayerQwen3LMHeadModel.init, Vocab, cfg, key=jrandom.PRNGKey(0)), ck, subpath="model")
    tr = model.transformer
    x = model.embeddings.embed(hax.named(jnp.asarray(tok, jnp.int32), (Batch, Pos)))
    last, = (eqx.filter_jit(lambda tr_, x_: (tr_.outputs(x_, mask).rearrange((cfg.Layers.name, ..., cfg.Embed.name)).array[-1],))(tr, x))
    nrm = np.linalg.norm(np.asarray(last, np.float64), axis=-1)
    logits = model.lm_head(tr.norm(hax.named(jnp.asarray(last, jnp.float32), (Batch, Pos, cfg.Embed)))).array.astype(jnp.float32)
    lp = jax.nn.log_softmax(logits[:, :-1], axis=-1)
    ce = -np.asarray(jnp.take_along_axis(lp, tgt[..., None], axis=-1)[..., 0], np.float64)
    return nrm[:, :-1], ce
a, ref = sys.argv[1:3]
na, ca = run(a)
nb, cb = run(ref)
np.savez(f"/scratch/gpfs/GROUP/USER/tmp/pls_hcid/gap_by_norm{a}.npz", na=na, ca=ca, nb=nb, cb=cb)
N = ca.size
print(f"tokens {N}; CE {a} {ca.mean():.4f}, {ref} {cb.mean():.4f}, gap {ca.mean() - cb.mean():+.4f}; paired se {(ca - cb).std() / np.sqrt(N):.4f}")
r = na / np.median(na)
print("|s_48| / median of run A: share of tokens, CE A, CE ref, part of the gap; norm of the reference at those tokens / its median")
E = [0, 1.5, 3, 10, 100, np.inf]
for lo, hi in zip(E[:-1], E[1:]):
    m = (r >= lo) & (r < hi)
    if m.any():
        print(f"  {lo:>5}-{hi:<5}: {m.mean():7.4f}  {ca[m].mean():6.3f} {cb[m].mean():6.3f}  {(ca[m] - cb[m]).sum() / N:+.4f}  se {(ca[m] - cb[m]).std() / np.sqrt(m.sum()) * m.mean():.4f}  ref norm {np.median(nb[m]) / np.median(nb):6.2f}")
print("quartile of the reference's CE: mean CE ref, CE A, part of the gap")
q = np.quantile(cb, [0, .25, .5, .75, 1])
for i in range(4):
    m = (cb >= q[i]) & (cb <= q[i + 1])
    print(f"  Q{i + 1} ({q[i]:.3f}-{q[i + 1]:.3f}): {cb[m].mean():6.3f} {ca[m].mean():6.3f}  {(ca[m] - cb[m]).sum() / N:+.4f}")
print("share of the gap from the largest token differences: top 0.1%, 1%, 10% of |CE A - CE ref| (signed sum / N)")
d = ca - cb
o = np.argsort(-np.abs(d.ravel()))
for f in (0.001, 0.01, 0.1):
    k = int(f * N)
    print(f"  {f:.1%}: {d.ravel()[o[:k]].sum() / N:+.4f}")
