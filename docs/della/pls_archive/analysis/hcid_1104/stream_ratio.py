"""|stream differences| / |stream sum| per layer at the newest checkpoint, on real training windows, in the production regime (compute bfloat16; streams in float32 when depth_hc_state_f32). Preflight item for w1006, w1007 (docs pls.md item 17): near 1e7, float32 loses the sum as well.
Per token: dif = sqrt(sum_s |H_s - mean_s H|^2), sum = |sum_s H_s|. Prints by layer the median and the 99th percentile of dif / sum over the tokens, and the median |sum|.
    CUDA_VISIBLE_DEVICES=N XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda PYTHONPATH=. nice -n 19 $PY stream_ratio.py TAG [TAG ...]"""
import dataclasses, glob, json, os, sys, time
os.environ.setdefault("LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS", "0")
import equinox as eqx, jax, jax.numpy as jnp, jax.random as jrandom, jmp, numpy as np, wandb
import haliax as hax
from haliax import Axis
from haliax.partitioning import ResourceAxis
from levanter.checkpoint import load_checkpoint
from levanter.layers.attention import AttentionBackend, AttentionMask
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from experiments.references.per_layer_qwen3 import PerLayerQwen3Config, PerLayerQwen3LMHeadModel
sys.path.insert(0, "/scratch/gpfs/GROUP/USER/tmp/pls_hcid"); from fw_tokens import load
B, T, N = 1, 1024, 4
ids = load(B, T, N, seed=11)
Batch, Pos = Axis("batch", B), Axis("position", T)
S_ = "/scratch/gpfs/GROUP/USER/marin_store_big/speedrun/muonh-qwen3-130m-della4xh100-pls1"
T_ = "/scratch/gpfs/GROUP/USER/marin_store_big/tmp/checkpoints-temp" + S_
tc = TrainerConfig(mesh=MeshConfig(axes={"data": -1, "replica": 1, "model": 1}, compute_mapping={"token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA), "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA)}))
tc.use_device_mesh().__enter__(); hax.axis_mapping(tc.parameter_axis_mapping).__enter__()
Vocab = Axis("vocab", 128256)
fields = {f.name for f in dataclasses.fields(PerLayerQwen3Config)}
mp = jmp.get_policy("p=f32,c=bfloat16")
api = wandb.Api(timeout=300)
for tag in sys.argv[1:]:
    ck = sorted(glob.glob(f"{S_}{tag}/*/checkpoints/step-*") + glob.glob(f"{T_}{tag}/*/checkpoints/step-*"), key=lambda p: int(p.rsplit("-", 1)[1]))[-1]
    wcfg = dict(api.run("reself/marin-della/muonh-qwen3-130m-della4xh100-pls1" + tag).config)["model"]
    kw = {k: v for k, v in wcfg.items() if k in fields and not isinstance(v, (dict, list)) and v is not None and k not in ("attn_backend", "max_seq_len", "tokenizer", "reference_checkpoint", "scan_layers")}
    cfg = PerLayerQwen3Config(max_seq_len=T, attn_backend=AttentionBackend.VANILLA, scan_layers=True, **kw)
    m = mp.cast_to_compute(load_checkpoint(eqx.filter_eval_shape(PerLayerQwen3LMHeadModel.init, Vocab, cfg, key=jrandom.PRNGKey(0)), ck, subpath="model"))
    tr, M = m.transformer, cfg.hc_streams
    f32 = getattr(cfg, "depth_hc_state_f32", False)
    @eqx.filter_jit
    def run(m_, toks, mask):
        tr_ = m_.transformer
        x = m_.embeddings.embed(toks)
        def hstep(layer, H, i, *, key):
            H = layer.hyper_step(H, mask, key=key, pos_ids=None, layer=i).rearrange(H.axes)
            a = H.rearrange(("stream", Batch, Pos, cfg.Embed)).array.astype(jnp.float32)
            mean = a.mean(0, keepdims=True)
            dif = jnp.sqrt(((a - mean) ** 2).sum((0, 3)))
            s = jnp.sqrt(((a.sum(0)) ** 2).sum(-1))
            return H, hax.named(jnp.stack([dif, s]), (Axis("stat", 2), Batch, Pos))
        H0 = hax.stack(Axis("stream", M), [x] * M).astype(jnp.float32 if f32 else x.dtype)
        _, st = tr_.layers.scan_via(hstep)(H0, hax.arange(cfg.Layers), key=None)
        return st.rearrange((cfg.Layers.name, "stat", "batch", "position")).array
    R = []
    t0 = time.time()
    for b in range(N):
        tok = ids[b * B * T:][: B * T].reshape(B, T)
        seg = hax.named(jnp.asarray(np.cumsum(np.roll(tok, 1, axis=1) == 128001, axis=1), jnp.int32), (Batch, Pos))
        R.append(np.asarray(run(m, hax.named(jnp.asarray(tok, jnp.int32), (Batch, Pos)), AttentionMask.causal().with_segment_ids(seg)), np.float64))
    R = np.concatenate(R, 2)
    ratio = R[:, 0] / np.maximum(R[:, 1], 1e-30)
    print(f"{tag} {ck.rsplit('-', 1)[1]}: state float32 {f32}; {time.time() - t0:.0f} s; dif / sum over {R.shape[2] * R.shape[3]} tokens; non-finite tokens {int((~np.isfinite(R)).sum())}")
    for l in (0, 1, 3, 5, 7, 11, 15, 23, 35, 47):
        print(f"   L{l + 1:<3d} median {np.median(ratio[l]):10.3g}  p99 {np.percentile(ratio[l], 99):10.3g}  max {ratio[l].max():10.3g}  |sum| median {np.median(R[l, 1]):10.3g}  |dif| median {np.median(R[l, 0]):10.3g}", flush=True)
