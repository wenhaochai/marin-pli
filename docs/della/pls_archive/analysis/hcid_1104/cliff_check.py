"""Is the same token's state the same in different numeric paths? w810, batch 11 and 53 of the FineWeb windows, state norm ratio to the median at L4, L5, L6, L48 of tokens 3291 (batch 11) and 1015 (batch 53): layer_outputs under jit in bfloat16 (as ce_by_target), jit with the gradient-free model cast to float32, and jit bfloat16 after adding relative noise 1e-3 to the embeddings' parameters of nothing else... (small noise on the input embedding only)."""
import dataclasses, glob, json, os, sys
os.environ.setdefault("LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS", "0")
import equinox as eqx, jax, jax.numpy as jnp, jax.random as jrandom, jmp, numpy as np
import haliax as hax
from haliax import Axis
from haliax.partitioning import ResourceAxis
from levanter.checkpoint import load_checkpoint
from levanter.layers.attention import AttentionBackend, AttentionMask
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from experiments.references.per_layer_qwen3 import PerLayerQwen3Config, PerLayerQwen3LMHeadModel
sys.path.insert(0, "/scratch/gpfs/GROUP/USER/tmp/pls_hcid"); from fw_tokens import load
tag = sys.argv[1]
B, T, N = 1, 4096, 128
ids = load(B, T, N)
Batch, Pos = Axis("batch", B), Axis("position", T)
S_ = "/scratch/gpfs/GROUP/USER/marin_store_big/speedrun/muonh-qwen3-130m-della4xh100-pls1"
T_ = "/scratch/gpfs/GROUP/USER/marin_store_big/tmp/checkpoints-temp" + S_
tc = TrainerConfig(mesh=MeshConfig(axes={"data": -1, "replica": 1, "model": 1}, compute_mapping={"token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA), "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA)}))
tc.use_device_mesh().__enter__(); hax.axis_mapping(tc.parameter_axis_mapping).__enter__()
Vocab = Axis("vocab", 128256)
fields = {f.name for f in dataclasses.fields(PerLayerQwen3Config)}
ck = sorted(glob.glob(f"{S_}{tag}/*/checkpoints/step-*") + glob.glob(f"{T_}{tag}/*/checkpoints/step-*"), key=lambda p: int(p.rsplit("-", 1)[1]))[-1]
wcfg = json.load(open(f"/scratch/gpfs/GROUP/USER/tmp/pls_paths/cfg{tag}.json"))["model"]
kw = {k: v for k, v in wcfg.items() if k in fields and not isinstance(v, (dict, list)) and v is not None and k not in ("attn_backend", "max_seq_len", "tokenizer", "reference_checkpoint", "scan_layers")}
cfg = PerLayerQwen3Config(max_seq_len=T, attn_backend=AttentionBackend.VANILLA, scan_layers=True, **kw)
model = load_checkpoint(eqx.filter_eval_shape(PerLayerQwen3LMHeadModel.init, Vocab, cfg, key=jrandom.PRNGKey(0)), ck, subpath="model")
Ln = cfg.Layers.name
def norms(m_, toks, attn, eps, key):
    x = m_.embeddings.embed(toks)
    if eps > 0:
        x = x * (1 + eps * hax.named(jrandom.normal(key, (B, T, cfg.Embed.size), jnp.float32), (Batch, Pos, cfg.Embed)).astype(x.dtype))
    outs = m_.transformer.outputs(x, attn)
    hf = outs.astype(jnp.float32)
    return hax.sqrt(hax.sum(hf * hf, axis=cfg.Embed)).rearrange((Ln, Batch, Pos)).array
jf = eqx.filter_jit(norms)
MP = {"bf16": jmp.get_policy("p=f32,c=bfloat16").cast_to_compute(model), "f32": model}
for b, p in ((11, 3291), (53, 1015), (11, 3319)):
    tok = ids[b * B * T:][: B * T].reshape(B, T)
    seg = hax.named(jnp.asarray(np.cumsum(np.roll(tok, 1, axis=1) == 128001, axis=1), jnp.int32), (Batch, Pos))
    attn = AttentionMask.causal().with_segment_ids(seg)
    toks = hax.named(jnp.asarray(tok, jnp.int32), (Batch, Pos))
    print(f"batch {b} position {p} token {tok[0, p]}")
    for name, m_ in MP.items():
        for eps, seed in ((0.0, 0), (1e-3, 1), (1e-3, 2), (1e-2, 3)):
            n = np.asarray(jf(m_, toks, attn, eps, jrandom.PRNGKey(seed)), np.float64)
            r = n[:, 0, p] / np.median(n, axis=(1, 2))
            print(f"  {name} noise {eps:g} seed {seed}: ratio at L4 {r[3]:.2f} L5 {r[4]:.2f} L6 {r[5]:.2f} L12 {r[11]:.2f} L48 {r[47]:.2f}; share of the sequence above 10 x at L5 {(n[4] > 10 * np.median(n[4])).mean():.4f}", flush=True)
    if b == 11:
        n0 = np.asarray(jf(MP["bf16"], toks, attn, 0.0, jrandom.PRNGKey(0)), np.float64)[4, 0]
        n1 = np.asarray(jf(MP["bf16"], toks, attn, 1e-3, jrandom.PRNGKey(1)), np.float64)[4, 0]
        n2 = np.asarray(jf(MP["f32"], toks, attn, 0.0, jrandom.PRNGKey(0)), np.float64)[4, 0]
        t = 10 * np.median(n0)
        print(f"  L5 tokens above 10 x in bf16: {(n0 > t).sum()}; also above with noise 1e-3: {((n0 > t) & (n1 > t)).sum()}; also above in f32: {((n0 > t) & (n2 > t)).sum()}; above only with noise {((n0 <= t) & (n1 > t)).sum()}; only in f32 {((n0 <= t) & (n2 > t)).sum()}", flush=True)
