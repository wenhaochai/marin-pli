"""Production-regime check of the one-stream arms before submission: real size (48 layers, width 512, 128,256 vocabulary, all 47 probes), fresh initialisation, real FineWeb-Edu windows, compute bfloat16, parameters float32, on an A40. For each arm: loss and every gradient finite at real size, compute bfloat16, on real tokens. Backbone only: the probes' parts are covered by the CPU test and the job's own smoke. Config: the W&B config of the finished HC 1 stream run, with only the arm's knobs changed.
    CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda PYTHONPATH=. nice -n 19 $PY real_step_s1.py ARM"""
import dataclasses, json, os, sys, time
os.environ.setdefault("LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS", "0")
import equinox as eqx, jax, jax.numpy as jnp, jax.random as jrandom, jmp, numpy as np
import haliax as hax
from haliax import Axis
from haliax.partitioning import ResourceAxis
from levanter.layers.attention import AttentionBackend, AttentionMask
from levanter.models.lm_model import LmExample
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from experiments.references.per_layer_qwen3 import PerLayerQwen3Config, PerLayerQwen3LMHeadModel
sys.path.insert(0, "/scratch/gpfs/GROUP/USER/tmp/pls_hcid"); from fw_tokens import load
arm = sys.argv[1]
ARMS = {"learned": {}, "identity": dict(depth_hc_res="identity"), "scalar": dict(depth_hc_res="scalar"), "static": dict(depth_route="static"), "leak": dict(depth_arch="preln", depth_preln_leak=True, depth_shrink_power=0.1)}
B, T = 1, 1024
ids = load(B, T, 1, seed=31)
Batch, Pos = Axis("batch", B), Axis("position", T)
tc = TrainerConfig(mesh=MeshConfig(axes={"data": -1, "replica": 1, "model": 1}, compute_mapping={"token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA), "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA)}))
tc.use_device_mesh().__enter__(); hax.axis_mapping(tc.parameter_axis_mapping).__enter__()
Vocab = Axis("vocab", 128256)
fields = {f.name for f in dataclasses.fields(PerLayerQwen3Config)}
mp = jmp.get_policy("p=f32,c=bfloat16")
wcfg = json.load(open("/scratch/gpfs/GROUP/USER/tmp/pls_paths/cfg-sep-bbfrozen-ahc-hcs1-vs-d48.json"))["model"]
kw = {k: v for k, v in wcfg.items() if k in fields and not isinstance(v, (dict, list)) and v is not None and k not in ("attn_backend", "max_seq_len", "tokenizer", "reference_checkpoint", "scan_layers")}
kw.update(ARMS[arm])
# The 47 probes' heads need about 13 GB of parameters and 13 GB of gradient, more than an A40 shared with other users has (first attempt: out of memory, 11.88 GiB, all arms). The probes never reach the backbone (the CPU test checks it), so here the backbone runs at real size with only the last layer's loss.
kw.update(pls_weight=0.0, pls_separate_heads=False, pls_detach_backbone=False)
cfg = PerLayerQwen3Config(max_seq_len=T, attn_backend=AttentionBackend.VANILLA, scan_layers=True, **kw)
print(arm, "arch", cfg.depth_arch, "streams", cfg.hc_streams, "res", cfg.depth_hc_res, "route", cfg.depth_route, "leak", getattr(cfg, "depth_preln_leak", False), "power", getattr(cfg, "depth_shrink_power", 1.0), "layers", cfg.num_layers, "width", cfg.hidden_dim, "detach", cfg.pls_detach_backbone, "sep", cfg.pls_separate_heads, flush=True)
t0 = time.time()
model = PerLayerQwen3LMHeadModel.init(Vocab, cfg, key=jrandom.PRNGKey(0))
tok = ids[:T].reshape(B, T)
seg = hax.named(jnp.asarray(np.cumsum(np.roll(tok, 1, axis=1) == 128001, axis=1), jnp.int32), (Batch, Pos))
ex = LmExample(tokens=hax.named(jnp.asarray(tok, jnp.int32), (Batch, Pos)), loss_weight=hax.named(jnp.ones((B, T), jnp.float32).at[:, -1].set(0), (Batch, Pos)), attn_mask=AttentionMask.causal().with_segment_ids(seg))
def lf(m_):
    loss, _ = m_.compute_next_token_loss(ex, key=jrandom.PRNGKey(1), logsumexp_weight=0.0)
    return loss.array if isinstance(loss, hax.NamedArray) else loss
@eqx.filter_jit
def vg(m_):
    return eqx.filter_value_and_grad(lf)(mp.cast_to_compute(m_))
@eqx.filter_jit
def per_layer(m_):
    mc = mp.cast_to_compute(m_)
    outs = mc.layer_outputs(ex.tokens, ex.attn_mask, key=None)
    return jnp.stack([mc._readout_ce(outs, k, ex, reduction=None, reduction_axis=(), dtype=jnp.float32, logsumexp_weight=None, logit_soft_cap=None).array[..., :-1].mean() for k in (47,)])
l0, g0 = vg(model)
leaves = jax.tree_util.tree_leaves(eqx.filter(g0, eqx.is_array))
bad = sum(int(not np.all(np.isfinite(np.asarray(a.astype(jnp.float32))))) for a in leaves)
tr_g = jax.tree_util.tree_leaves(eqx.filter(g0.transformer, eqx.is_array))
print(f"  loss {float(l0):.3f}, gradient leaves {len(leaves)}, non-finite {bad}, largest backbone gradient element {max(float(jnp.abs(a.astype(jnp.float32)).max()) for a in tr_g):.3e}; {time.time() - t0:.0f} s", flush=True)
print("  CE at layer 48 at initialisation:", np.round(np.asarray(per_layer(model)), 3).tolist(), "ln V", round(float(np.log(128256)), 3), flush=True)
assert bad == 0 and np.isfinite(float(l0))
print("  OK", flush=True)
