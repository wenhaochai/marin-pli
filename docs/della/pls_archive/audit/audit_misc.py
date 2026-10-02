"""Audit D: misc checks.
D1 scan_layers=False (BlockSeq): does PerLayerQwen3LMHeadModel.layer_outputs train correctly (BlockSeq.scan_via does not
   slice per-layer kwargs such as the per-layer key array)?
D2 draccus encode/decode round trip of PerLayerQwen3Config inside TrainLmConfig (config logging / YAML).
D3 FLOP accounting at the real 130m/300m shapes: flops_per_token(pls) / flops_per_token(baseline).
D4 pls_monitor_stride validation vs train_seq_len.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4"
os.environ.setdefault("LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS", "0")

import io  # noqa: E402

import draccus  # noqa: E402
import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jrandom  # noqa: E402
import numpy as np  # noqa: E402

import haliax as hax  # noqa: E402
from haliax import Axis  # noqa: E402
from levanter.layers.attention import AttentionBackend, AttentionMask  # noqa: E402
from levanter.models.lm_model import LmConfig, LmExample  # noqa: E402
from levanter.models.qwen import Qwen3Config, Qwen3LMHeadModel  # noqa: E402

from experiments.references.per_layer_qwen3 import PerLayerQwen3Config, PerLayerQwen3LMHeadModel  # noqa: E402

B, T, V, L = 2, 32, 200, 3
Batch, Pos, Vocab = Axis("batch", B), Axis("position", T), Axis("vocab", V)
rng = np.random.default_rng(0)
tok = hax.named(jnp.asarray(rng.integers(0, V, size=(B, T)), jnp.int32), (Batch, Pos))
lw = hax.named(jnp.ones((B, T), jnp.float32), (Batch, Pos))
ex = LmExample(tokens=tok, loss_weight=lw, attn_mask=AttentionMask.causal())
common = dict(max_seq_len=T, hidden_dim=32, intermediate_dim=64, num_layers=L, num_heads=2, num_kv_heads=2, hybrid_norm=True,
              attn_backend=AttentionBackend.VANILLA)

# D1
for scan in (True, False):
    cfg = PerLayerQwen3Config(**common, scan_layers=scan, pls_weight=1.0)
    m = PerLayerQwen3LMHeadModel.init(Vocab, cfg, key=jrandom.PRNGKey(0))
    try:
        out = m.compute_next_token_loss(ex, key=jrandom.PRNGKey(1))
        loss, stats = out
        # reference: truncated baseline models are awkward for BlockSeq; compare to the scan=True model with the same weights
        print(f"D1 scan_layers={scan}: loss {float(loss.array):.6f} per-layer {[round(float(v.value()), 5) for v in stats.values()]}")
    except Exception as e:  # noqa: BLE001
        print(f"D1 scan_layers={scan}: FAILED {type(e).__name__}: {str(e)[:300]}")
    try:
        ev = m.compute_next_token_loss(ex, key=None)
        per = m.readout_losses(ex, tuple(range(L)))
        print(f"D1 scan_layers={scan}: eval path ok {float(ev.array):.6f}; readout_losses axes {per.axes}")
    except Exception as e:  # noqa: BLE001
        print(f"D1 scan_layers={scan}: eval/readout FAILED {type(e).__name__}: {str(e)[:300]}")

# D2
cfg = PerLayerQwen3Config(**common, pls_weight=0.5, pls_monitor_stride=8, pls_eval=False)
buf = io.StringIO()
draccus.dump(cfg, buf)
text = buf.getvalue()
print("D2 encoded model config:\n" + "\n".join("    " + ln for ln in text.splitlines() if "pls" in ln or "type" in ln))
try:
    back = draccus.load(LmConfig, io.StringIO(text))
    print(f"D2 decode as LmConfig -> {type(back).__name__}; equal={back == cfg}")
except Exception as e:  # noqa: BLE001
    print(f"D2 decode FAILED {type(e).__name__}: {str(e)[:300]}")

# D3
V_REAL = 128256
for size, (h, inter, layers, heads) in {"130m": (512, 1792, 6, 8), "300m": (768, 2688, 12, 12)}.items():
    kw = dict(max_seq_len=4096, hidden_dim=h, intermediate_dim=inter, num_layers=layers, num_heads=heads, num_kv_heads=heads, hybrid_norm=True)
    fb = Qwen3Config(**kw).flops_per_token(V_REAL, 4096)
    f1 = PerLayerQwen3Config(**kw, pls_weight=1.0).flops_per_token(V_REAL, 4096)
    f0 = PerLayerQwen3Config(**kw, pls_weight=0.0).flops_per_token(V_REAL, 4096)
    head = 2 * h * V_REAL
    print(f"D3 {size}: baseline fwd FLOPs/token {fb / 1e6:.0f}M (lm_head {head / 1e6:.0f}M = {head / fb:.0%}); pls1 {f1 / 1e6:.0f}M = {f1 / fb:.2f}x; pls0 {f0 / 1e6:.0f}M = {f0 / fb:.3f}x")

# D4
try:
    c = PerLayerQwen3Config(**{**common, "max_seq_len": 4096}, pls_monitor_stride=16)
    print("D4 stride check passes for max_seq_len=4096 (the check does not see train_seq_len)")
except Exception as e:  # noqa: BLE001
    print("D4", e)
