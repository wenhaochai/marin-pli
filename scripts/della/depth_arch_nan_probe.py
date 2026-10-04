"""CPU probe: forward loss of a 48-layer hc / mhc / preln model (tiny width) in float32 and under the training policy
(params f32, compute bf16), and the largest stream magnitude per layer, to find where a NaN appears."""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax, jax.numpy as jnp, jax.random as jrandom, jmp, numpy as np
import haliax as hax
from haliax import Axis
from levanter.layers.attention import AttentionBackend, AttentionMask
from levanter.models.lm_model import LmExample
from experiments.references.per_layer_qwen3 import PerLayerQwen3Config, PerLayerQwen3LMHeadModel
B, T, V, L = 2, 64, 500, 48
Batch, Pos, Vocab = Axis("batch", B), Axis("position", T), Axis("vocab", V)
tok = np.random.default_rng(0).integers(2, V, size=(B, T))
ex = LmExample(tokens=hax.named(jnp.asarray(tok, jnp.int32), (Batch, Pos)), loss_weight=hax.named(jnp.ones((B, T)), (Batch, Pos)), attn_mask=AttentionMask.causal())
for arch in ("preln", "hc", "mhc"):
    cfg = PerLayerQwen3Config(max_seq_len=T, hidden_dim=64, intermediate_dim=128, num_layers=L, num_heads=2, num_kv_heads=2, hybrid_norm=True, attn_backend=AttentionBackend.VANILLA, pls_weight=1.0, pls_separate_heads=True, depth_arch=arch, scan_layers=False)
    m = PerLayerQwen3LMHeadModel.init(Vocab, cfg, key=jrandom.PRNGKey(0))
    for name, pol in (("f32", None), ("bf16", jmp.get_policy("p=f32,c=bfloat16"))):
        mm = pol.cast_to_compute(m) if pol else m
        outs = mm.layer_outputs(ex.tokens, ex.attn_mask, key=jrandom.PRNGKey(1))
        mx = np.asarray(jnp.max(jnp.abs(outs.array.astype(jnp.float32)), axis=tuple(range(1, outs.array.ndim))))
        loss = mm.compute_next_token_loss(ex, key=jrandom.PRNGKey(1))
        loss = loss[0] if isinstance(loss, tuple) else loss
        print(arch, name, "loss", float(loss.array if hasattr(loss, "array") else loss), "max|out| first/mid/last", mx[0], mx[L // 2], mx[-1], "nonfinite layers", int((~np.isfinite(mx)).sum()), flush=True)

# gradient and a few real MuonH steps (the launcher's 130m optimizer values), for hc and mhc
import equinox as eqx, optax
from levanter.optim.muonh import MuonHConfig
opt_cfg = MuonHConfig(learning_rate=0.02, adam_lr=0.008, beta1=0.9, beta2=0.98, epsilon=1e-15, momentum=0.95, nesterov=True, backend_steps=5, muon_epsilon=1e-5, max_grad_norm=1.0, weight_decay=0.1, lr_schedule="linear", warmup=0.1, min_lr_ratio=0.0, coefficient_type="simple")
pol = jmp.get_policy("p=f32,c=bfloat16")
for arch in ("hc", "mhc"):
    cfg = PerLayerQwen3Config(max_seq_len=T, hidden_dim=64, intermediate_dim=128, num_layers=L, num_heads=2, num_kv_heads=2, hybrid_norm=True, attn_backend=AttentionBackend.VANILLA, pls_weight=1.0, pls_separate_heads=True, depth_arch=arch, scan_layers=False)
    m = PerLayerQwen3LMHeadModel.init(Vocab, cfg, key=jrandom.PRNGKey(0))
    opt = opt_cfg.build(100)
    state = opt.init(eqx.filter(m, eqx.is_inexact_array))
    def lf(mm):
        out = pol.cast_to_compute(mm).compute_next_token_loss(ex, key=jrandom.PRNGKey(1))
        out = out[0] if isinstance(out, tuple) else out
        return out.array if hasattr(out, "array") else out
    step = eqx.filter_jit(eqx.filter_value_and_grad(lf))
    for t in range(6):
        loss, g = step(m)
        bad = [jax.tree_util.keystr(p) for p, x in jax.tree_util.tree_leaves_with_path(eqx.filter(g, eqx.is_inexact_array)) if not bool(jnp.all(jnp.isfinite(x)))]
        upd, state = opt.update(g, state, eqx.filter(m, eqx.is_inexact_array))
        m = eqx.apply_updates(m, upd)
        badp = [jax.tree_util.keystr(p) for p, x in jax.tree_util.tree_leaves_with_path(eqx.filter(m, eqx.is_inexact_array)) if not bool(jnp.all(jnp.isfinite(x)))]
        print(arch, "step", t, "loss", float(loss), "nonfinite grads", bad[:4], "nonfinite params", badp[:4], flush=True)
        if bad or badp: break
