"""Diagnostic (2026-10-07): does a probes-only model (separate heads, backbone detached) train its backbone exactly like
the baseline under the 130m recipe's MuonH, at 48 layers?

The backbone gradients are equal at every step for equal parameters (pls_cpu_test 3d), so any divergence comes from the
optimizer: MuonHConfig clips by global norm inside each of its three groups (muonh / adamh / adam), and the aux heads
sit in the adamh group (path contains "lm_head") and their norms in the adam group, beside lm_head, the embeddings and
the backbone's norm scales. Tiny widths, L layers, a learnable bigram task, the recipe's optimizer constants.
Arms: baseline vs probes, with the recipe's max_grad_norm (1.0) and without clipping. Prints the final-readout loss of
each arm along the way, the max |backbone param diff| (probes - baseline), and the per-group clip factors.

    source scripts/della/pls_env.sh; JAX_PLATFORMS=cpu nice -n 19 $PY scripts/della/pls_d48_optcoupling_cpu.py [L] [steps]
"""
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS", "0")

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jrandom  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402

import haliax as hax  # noqa: E402
from haliax import Axis  # noqa: E402
from levanter.layers.attention import AttentionBackend, AttentionMask  # noqa: E402
from levanter.models.lm_model import LmExample  # noqa: E402
from levanter.models.qwen import Qwen3Config, Qwen3LMHeadModel  # noqa: E402
from levanter.optim.muonh import MuonHConfig  # noqa: E402

from experiments.references.per_layer_qwen3 import PerLayerQwen3Config, PerLayerQwen3LMHeadModel  # noqa: E402

L = int(sys.argv[1]) if len(sys.argv) > 1 else 48
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 120
B, T, V = 16, 32, 256
Batch, Pos, Vocab = Axis("batch", B), Axis("position", T), Axis("vocab", V)
rng = np.random.default_rng(0)
perm = rng.permutation(V)


def batch(i):
    r = np.random.default_rng(1000 + i)
    tok = np.zeros((B, T), np.int64)
    tok[:, 0] = r.integers(0, V, B)
    for t in range(1, T):
        nxt = perm[tok[:, t - 1]]
        noise = r.random(B) < 0.2
        tok[:, t] = np.where(noise, r.integers(0, V, B), nxt)
    lw = (np.arange(T) < T - 1).astype(np.float32)[None].repeat(B, 0)
    return LmExample(tokens=hax.named(jnp.asarray(tok, jnp.int32), (Batch, Pos)), loss_weight=hax.named(jnp.asarray(lw), (Batch, Pos)), attn_mask=AttentionMask.causal())


common = dict(max_seq_len=T, hidden_dim=64, intermediate_dim=128, num_layers=L, num_heads=2, num_kv_heads=2, hybrid_norm=True, use_qk_norm=False, attn_backend=AttentionBackend.VANILLA)
key0 = jrandom.PRNGKey(0)


def make_opt(clip):
    return MuonHConfig(learning_rate=0.02, adam_lr=0.008, beta1=0.9, beta2=0.98, epsilon=1e-20, momentum=0.95, nesterov=True, backend_steps=5,
                       muon_epsilon=1e-5, max_grad_norm=clip, weight_decay=0.1, lr_schedule="linear", decay=0.8, warmup=0, min_lr_ratio=0.0,
                       coefficient_type="simple").build(STEPS)


PROJ = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def group_of(path):
    """MuonHConfig.create_mask's label for a parameter path (Embedding -> adam, *lm_head* -> adamh, Linear weight -> muonh)."""
    if "lm_head" in path:
        return "adamh"
    if "embeddings" in path:
        return "adam"
    if any(f"{q}.weight" in path for q in PROJ):
        return "muonh"
    return "adam"


def run(model, clip):
    opt = make_opt(clip)
    params = eqx.filter(model, eqx.is_array)
    state = opt.init(params)

    def lf(m, ex):
        out = m.compute_next_token_loss(ex, key=jrandom.PRNGKey(1), logsumexp_weight=0.0)
        loss, stats = out if isinstance(out, tuple) else (out, {})
        loss = loss.array if isinstance(loss, hax.NamedArray) else loss
        final = stats[f"pls/L{L - 1}"].value() if stats else loss
        return loss, final

    @eqx.filter_jit
    def step(m, s, ex):
        (_, final), g = eqx.filter_value_and_grad(lf, has_aux=True)(m, ex)
        groups = {"muonh": [], "adamh": [], "adam": []}
        for path, leaf in jax.tree_util.tree_leaves_with_path(eqx.filter(g, eqx.is_array)):
            groups[group_of(jax.tree_util.keystr(path))].append(leaf)
        norms = {lab: optax.global_norm(v) for lab, v in groups.items()}
        upd, s = opt.update(g, s, eqx.filter(m, eqx.is_array))
        return eqx.apply_updates(m, upd), s, final, norms

    traj, models = [], []
    for i in range(STEPS):
        model, state, final, norms = step(model, state, batch(i))
        traj.append((float(final), {k: float(v) for k, v in norms.items()}))
        if i in (0, 1, 2, 5, 10, 20, 40, 80, STEPS - 1):
            models.append((i, model))
    return traj, models


def backbone(m):
    return eqx.filter((m.embeddings, m.transformer, m.lm_head), eqx.is_array)


def maxdiff(a, b):
    return max(float(jnp.max(jnp.abs(x.astype(jnp.float32) - y.astype(jnp.float32)))) for x, y in zip(jax.tree.leaves(backbone(a)), jax.tree.leaves(backbone(b))))


base0 = Qwen3LMHeadModel.init(Vocab, Qwen3Config(**common), key=key0)
probe0 = PerLayerQwen3LMHeadModel.init(Vocab, PerLayerQwen3Config(**common, pls_weight=1.0, pls_separate_heads=True, pls_detach_backbone=True), key=key0)
assert maxdiff(base0, probe0) == 0.0, "backbone init differs"
print(f"L={L} steps={STEPS} width 64; backbone init identical")
for clip in (1.0, None):
    tb, mb = run(base0, clip)
    tp, mp = run(probe0, clip)
    print(f"\n== max_grad_norm={clip}")
    for (i, a), (_, b) in zip(mb, mp):
        print(f"step {i:4d}  final loss base {tb[i][0]:.4f} probes {tp[i][0]:.4f}  max|backbone diff| {maxdiff(a, b):.3e}"
              f"  group grad norms base {tb[i][1]['adamh']:.3f}/{tb[i][1]['adam']:.3f} probes {tp[i][1]['adamh']:.3f}/{tp[i][1]['adam']:.3f} (adamh/adam)")
    print(f"mean final loss, last 20 steps: base {np.mean([x[0] for x in tb[-20:]]):.4f} probes {np.mean([x[0] for x in tp[-20:]]):.4f}")
