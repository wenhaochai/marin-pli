"""Forward only, same checkpoint, text and precision as grad_tail.py: per-token CE at chosen heads and the state norm at every layer, for N batches of 2 x 1024 tokens. The hot tokens of grad_hot2.py have CE 30-45 nats at the first heads (targets '.\n\n', '?\n\n', '\n\n'). Is that a property of the target, of the run, or of a few tokens? Output: ce_by_target<tag>.npz.
    CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda PYTHONPATH=. nice -n 19 $PY ce_by_target.py TAG"""
import dataclasses, glob, json, os, sys, time
os.environ.setdefault("LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS", "0")
import equinox as eqx, jax, jax.numpy as jnp, jax.random as jrandom, jmp, numpy as np
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
tag = sys.argv[1]
B, T, N = int(os.environ.get("WS_B", 2)), int(os.environ.get("WS_T", 1024)), int(os.environ.get("WS_N", 128))
HEADS = [0, 1, 2, 3, 4, 7, 11, 23, 47]
S_ = "/scratch/gpfs/GROUP/USER/marin_store_big/speedrun/muonh-qwen3-130m-della4xh100-pls1"
T_ = "/scratch/gpfs/GROUP/USER/marin_store_big/tmp/checkpoints-temp" + S_
TOK = "/scratch/gpfs/GROUP/USER/marin_store_big/tokenized/nemotron_cc/hq_actual-5af4cc/train/part-00000-of-01018/input_ids/data"
sys.path.insert(0, "/scratch/gpfs/GROUP/USER/tmp/pls_hcid"); from fw_tokens import load; ids = load(B, T, N)
Batch, Pos = Axis("batch", B), Axis("position", T)
tc = TrainerConfig(mesh=MeshConfig(axes={"data": -1, "replica": 1, "model": 1}, compute_mapping={
    "token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
    "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA)}))
tc.use_device_mesh().__enter__()
hax.axis_mapping(tc.parameter_axis_mapping).__enter__()
Vocab = Axis("vocab", 128256)
fields = {f.name for f in dataclasses.fields(PerLayerQwen3Config)}
mp = jmp.get_policy("p=f32,c=bfloat16")
ck = sorted(glob.glob(f"{S_}{tag}/*/checkpoints/step-*") + glob.glob(f"{T_}{tag}/*/checkpoints/step-*"), key=lambda p: int(p.rsplit("-", 1)[1]))[-1]
print("checkpoint", ck, flush=True)
wcfg = json.load(open(f"/scratch/gpfs/GROUP/USER/tmp/pls_paths/cfg{tag}.json"))["model"]
kw = {k: v for k, v in wcfg.items() if k in fields and not isinstance(v, (dict, list)) and v is not None and k not in ("attn_backend", "max_seq_len", "tokenizer", "reference_checkpoint", "scan_layers")}
if "BOUND" in os.environ:
    kw["depth_branch_bound"] = float(os.environ["BOUND"])
cfg = PerLayerQwen3Config(max_seq_len=T, attn_backend=AttentionBackend.VANILLA, scan_layers=True, **kw)
model = load_checkpoint(eqx.filter_eval_shape(PerLayerQwen3LMHeadModel.init, Vocab, cfg, key=jrandom.PRNGKey(0)), ck, subpath="model")
mc = mp.cast_to_compute(model)
@eqx.filter_jit
def fwd(m_, ex):
    outs = m_.layer_outputs(ex.tokens, ex.attn_mask, key=None)
    Block = m_.transformer.layers.Block.name
    ces = jnp.stack([m_._readout_ce(outs, k, ex, reduction=None, reduction_axis=(), dtype=jnp.float32, logsumexp_weight=None, logit_soft_cap=None).rearrange((Batch, Pos)).array.astype(jnp.float32) for k in HEADS])
    hf = outs.astype(jnp.float32)   # the norm scaled by the largest coordinate: a plain sum of squares overflows float32 above 1.8e19 (w890, 2026-11-04)
    big = hax.named(jnp.max(jnp.abs(hf.array), axis=hf.axes.index(cfg.Embed), keepdims=False), tuple(a for a in hf.axes if a != cfg.Embed))
    safe = hf / hax.where(big > 0, big, 1.0).broadcast_axis(cfg.Embed).rearrange(hf.axes)
    return ces, (big * hax.sqrt(hax.sum(safe * safe, axis=cfg.Embed))).rearrange((Block, Batch, Pos)).array
CE, NR = [], []
t0 = time.time()
for b in range(N):
    tok = ids[b * B * T:][: B * T].reshape(B, T)
    seg = hax.named(jnp.asarray(np.cumsum(np.roll(tok, 1, axis=1) == 128001, axis=1), jnp.int32), (Batch, Pos))
    ex = LmExample(tokens=hax.named(jnp.asarray(tok, jnp.int32), (Batch, Pos)), loss_weight=hax.named(jnp.ones((B, T), jnp.float32), (Batch, Pos)), attn_mask=AttentionMask.causal().with_segment_ids(seg))
    c, n = fwd(mc, ex)
    CE.append(np.asarray(c, np.float32)); NR.append(np.asarray(n, np.float32))
    if b < 2 or (b + 1) % 32 == 0:
        print(f"batch {b + 1}/{N} {time.time() - t0:.0f} s", flush=True)
np.savez(f"/scratch/gpfs/GROUP/USER/tmp/pls_hcid/fw_ce_by_target{tag}{os.environ.get('OUT_TAG', '')}.npz", ce=np.concatenate(CE, 1), nr=np.concatenate(NR, 1), tok=ids.reshape(N * B, T), heads=np.array(HEADS), ck=ck)
print("saved", time.time() - t0, flush=True)
