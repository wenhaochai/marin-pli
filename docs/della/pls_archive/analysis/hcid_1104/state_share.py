"""What is the state a probe reads made of? For each layer of a final checkpoint, on FineWeb-Edu windows (bfloat16 as training, statistics in float64 on the host):
 cos_e: median cos(s_l, s_0), s_0 = the embedding as a readout would see it; share of the current token's identity in the state.
 rel: median |s_l - s_{l-1}| / |s_{l-1}|. keep: median cos(s_l, s_{l-1}). keep_final: median cos(s_l, s_48).
 mean_share: |mean over tokens of s_l|^2 / mean |s_l|^2 after per-token RMS normalisation, the common direction every token shares.
 type_share: the share of the variance of RN(s_l) that the token type alone explains (tokens of types with at least 8 occurrences; the rest of the variance is context).
Output: state_share<tag>.npz and a table.
    CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda PYTHONPATH=. nice -n 19 $PY state_share.py TAG"""
import dataclasses, glob, json, os, sys, time
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
B, T, N = 1, 4096, int(os.environ.get("WS_N", 4))
ids = load(B, T, N, seed=7)
Batch, Pos = Axis("batch", B), Axis("position", T)
S_ = "/scratch/gpfs/GROUP/USER/marin_store_big/speedrun/muonh-qwen3-130m-della4xh100-pls1"
T_ = "/scratch/gpfs/GROUP/USER/marin_store_big/tmp/checkpoints-temp" + S_
tc = TrainerConfig(mesh=MeshConfig(axes={"data": -1, "replica": 1, "model": 1}, compute_mapping={"token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA), "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA)}))
tc.use_device_mesh().__enter__(); hax.axis_mapping(tc.parameter_axis_mapping).__enter__()
Vocab = Axis("vocab", 128256)
fields = {f.name for f in dataclasses.fields(PerLayerQwen3Config)}
mp = jmp.get_policy("p=f32,c=bfloat16")
ck = sorted(glob.glob(f"{S_}{tag}/*/checkpoints/step-*") + glob.glob(f"{T_}{tag}/*/checkpoints/step-*"), key=lambda p: int(p.rsplit("-", 1)[1]))[-1]
wcfg = json.load(open(f"/scratch/gpfs/GROUP/USER/tmp/pls_paths/cfg{tag}.json"))["model"]
kw = {k: v for k, v in wcfg.items() if k in fields and not isinstance(v, (dict, list)) and v is not None and k not in ("attn_backend", "max_seq_len", "tokenizer", "reference_checkpoint", "scan_layers")}
cfg = PerLayerQwen3Config(max_seq_len=T, attn_backend=AttentionBackend.VANILLA, scan_layers=True, **kw)
model = mp.cast_to_compute(load_checkpoint(eqx.filter_eval_shape(PerLayerQwen3LMHeadModel.init, Vocab, cfg, key=jrandom.PRNGKey(0)), ck, subpath="model"))
@eqx.filter_jit
def fwd(m_, tok, attn):
    outs = m_.layer_outputs(tok, attn, key=None)
    Block = m_.transformer.layers.Block.name
    return outs.rearrange((Block, Batch, Pos, cfg.Embed)).array.astype(jnp.float32), m_.embeddings.embed(tok).rearrange((Batch, Pos, cfg.Embed)).array.astype(jnp.float32)
rows, E, TOK = [], [], []
for b in range(N):
    tok = ids[b * B * T:][: B * T].reshape(B, T)
    seg = hax.named(jnp.asarray(np.cumsum(np.roll(tok, 1, axis=1) == 128001, axis=1), jnp.int32), (Batch, Pos))
    s, e = fwd(model, hax.named(jnp.asarray(tok, jnp.int32), (Batch, Pos)), AttentionMask.causal().with_segment_ids(seg))
    rows.append(np.asarray(s, np.float32)[:, 0]); E.append(np.asarray(e, np.float32)[0]); TOK.append(tok[0])
S, E, TOK = np.concatenate(rows, 1).astype(np.float64), np.concatenate(E, 0).astype(np.float64), np.concatenate(TOK)   # S: (layers, tokens, embed)
nrm = lambda a: np.sqrt(((a / np.maximum(np.abs(a).max(-1, keepdims=True), 1e-300)) ** 2).sum(-1)) * np.abs(a).max(-1)
L = S.shape[0]
unit = lambda a: a / np.maximum(nrm(a), 1e-300)[..., None]
ue, us = unit(E), unit(S)
cos_e = np.median((us * ue[None]).sum(-1), 1)
prev = np.concatenate([E[None] * (S.shape[0] and 1), S[:-1]], 0)
rel = np.median(nrm(S - prev) / np.maximum(nrm(prev), 1e-300), 1)
keep = np.median((us * unit(prev)).sum(-1), 1)
keep_final = np.median((us * us[-1:]).sum(-1), 1)
mean_share = (us.mean(1) ** 2).sum(-1)
# token-type share of the variance of the unit state: between-type variance over total variance, on tokens whose type occurs at least 8 times
vals, inv, cnt = np.unique(TOK, return_inverse=True, return_counts=True)
keep_tok = cnt[inv] >= 8
type_share = []
for l in range(L):
    u = us[l][keep_tok]; idx = inv[keep_tok]
    mu = u.mean(0)
    tot = ((u - mu) ** 2).sum(-1).mean()
    sums = np.zeros((len(vals), u.shape[1])); np.add.at(sums, idx, u); n = np.bincount(idx, minlength=len(vals))
    means = sums / np.maximum(n, 1)[:, None]
    # unbiased: within-type variance from the residual, corrected for the finite sample of each type
    within = ((u - means[idx]) ** 2).sum(-1).sum() / max(len(u) - (n > 0).sum(), 1)
    type_share.append(1 - within / tot)
type_share = np.array(type_share)
np.savez(f"/scratch/gpfs/GROUP/USER/tmp/pls_hcid/state_share{tag}.npz", cos_e=cos_e, rel=rel, keep=keep, keep_final=keep_final, mean_share=mean_share, type_share=type_share, step=int(ck.rsplit("-", 1)[1]))
print(f"{tag} {ck.rsplit('/', 1)[1]} tokens {S.shape[1]} (typed {keep_tok.mean():.2f})")
print(" layer  cos_e    rel   keep keep_final mean_share type_share")
for l in (1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 20, 24, 30, 36, 42, 46, 47, 48):
    i = l - 1
    print(f"  L{l:<3d} {cos_e[i]:7.3f} {rel[i]:6.3f} {keep[i]:6.3f} {keep_final[i]:7.3f} {mean_share[i]:7.3f} {type_share[i]:7.3f}", flush=True)
