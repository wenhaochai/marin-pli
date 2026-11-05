"""Production-regime check before submitting HC at 12 and 24 layers (with and without mixing), ordinary model with probes: real width and vocabulary, all probes, fresh initialisation, real FineWeb-Edu windows, compute bfloat16, on an A40. Config: the W&B config of the finished run -sep-bbfrozen-ahc-vs-d{L}, with only depth_hc_res changed. Checks: the loss and every gradient are finite; the last layer's loss at initialisation is near ln V; the probes' gradient does not reach the backbone (backbone gradient of the full loss equals that of a model with the same weights and no probes, leaf by leaf); with mixing = I the leaves res and w_res have zero gradient, and with learned mixing they do not.
    CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda PYTHONPATH=. nice -n 19 $PY real_step_depth.py L identity|learned"""
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
L, res = int(sys.argv[1]), sys.argv[2]
B, T = 1, 1024
ids = load(B, T, 1, seed=31)
Batch, Pos = Axis("batch", B), Axis("position", T)
tc = TrainerConfig(mesh=MeshConfig(axes={"data": -1, "replica": 1, "model": 1}, compute_mapping={"token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA), "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA)}))
tc.use_device_mesh().__enter__(); hax.axis_mapping(tc.parameter_axis_mapping).__enter__()
Vocab = Axis("vocab", 128256)
fields = {f.name for f in dataclasses.fields(PerLayerQwen3Config)}
mp = jmp.get_policy("p=f32,c=bfloat16")
wcfg = json.load(open(f"/scratch/gpfs/GROUP/USER/tmp/pls_paths/cfg-sep-bbfrozen-ahc-vs-d{L}.json"))["model"]
kw = {k: v for k, v in wcfg.items() if k in fields and not isinstance(v, (dict, list)) and v is not None and k not in ("attn_backend", "max_seq_len", "tokenizer", "reference_checkpoint", "scan_layers")}
kw["depth_hc_res"] = res
kw["pls_fold_norm"] = True   # the production probes path (audit 2026-11-04): the old config has it off
cfg = PerLayerQwen3Config(max_seq_len=T, attn_backend=AttentionBackend.VANILLA, scan_layers=True, **kw)
print(f"L {L} res {cfg.depth_hc_res} arch {cfg.depth_arch} streams {cfg.hc_streams} layers {cfg.num_layers} width {cfg.hidden_dim} detach {cfg.pls_detach_backbone} sep {cfg.pls_separate_heads} weight {cfg.pls_weight}", flush=True)
assert cfg.num_layers == L and cfg.pls_detach_backbone and cfg.pls_separate_heads and cfg.hc_streams == 4
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
l0, g0 = vg(model)
bb0 = eqx.filter(g0.transformer, eqx.is_array)
leaves = jax.tree_util.tree_leaves(eqx.filter(g0, eqx.is_array))
bad = sum(int(not np.all(np.isfinite(np.asarray(a.astype(jnp.float32))))) for a in leaves)
print(f"  loss {float(l0):.3f}; gradient leaves {len(leaves)}, non-finite {bad}; {time.time() - t0:.0f} s", flush=True)
assert bad == 0
# the last layer alone: the same weights without probes; backbone gradients must match
cfg0 = dataclasses.replace(cfg, pls_weight=0.0, pls_separate_heads=False, pls_detach_backbone=False)
model0 = PerLayerQwen3LMHeadModel.init(Vocab, cfg0, key=jrandom.PRNGKey(0))
# self.config is the transformer's config, so only the weights move across: the layers, the final norm, the embedding and the head
model0 = eqx.tree_at(lambda m: (m.transformer.layers, m.transformer.norm, m.embeddings, m.lm_head), model0, (model.transformer.layers, model.transformer.norm, model.embeddings, model.lm_head))
assert model0.config.pls_weight == 0
l1, g1 = vg(model0)
bb1 = eqx.filter(g1.transformer, eqx.is_array)
d = max(float(jnp.abs(a.astype(jnp.float32) - b.astype(jnp.float32)).max()) for a, b in zip(jax.tree_util.tree_leaves(bb0), jax.tree_util.tree_leaves(bb1)))
print(f"  full loss {float(l0):.3f}, last layer alone {float(l1):.3f} (ln V {np.log(128256):.3f}); largest backbone gradient difference {d:.2e}", flush=True)
assert d < 1e-3, d
st = g0.transformer.layers.stacked.hyper
for name in ("res", "w_res", "write", "read"):
    v = max(float(jnp.abs(getattr(st[j], name).array.astype(jnp.float32)).max()) for j in (0, 1))
    print(f"  largest gradient of {name}: {v:.2e}", flush=True)
r = max(float(jnp.abs(getattr(st[j], "res").array.astype(jnp.float32)).max()) for j in (0, 1))
if res == "identity":
    assert r == 0.0, r
else:
    assert r > 0.0, r
print("  OK", flush=True)
