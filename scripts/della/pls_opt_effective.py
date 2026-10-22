"""L6 (Lean follow-ups, 2026-10-21): print the effective optimizer settings of the 48-layer DepthBench runs, per group.
Run on a vis node from this worktree (CPU only):
    source scripts/della/pls_env.sh; nice -n 19 $PY -u scripts/della/pls_opt_effective.py
The optimizer constants are those W&B logged for the -vs-d48-g4 runs (config.yaml of the hc probes run). On a small model
of each kind (Pre-LN with MuonHConfig, HC and Full AttnRes with DepthArchMuonHConfig, separate heads, backbone detached =
the probes runs) it prints and checks:
1. Groups: which parameters each optimizer group holds (layer indices folded to *).
2. Learning rates: the two schedules inject_hyperparams holds (learning_rate, adam_lr) at the steps that matter, and which
   group reads which one: AdamH (lm_head and the probe heads) takes learning_rate, the MuonH peak 0.02, not adam_lr.
3. Weight decay: the config carries weight_decay=0.1, and one step with weight_decay 0.1 equals one step with 0.0 bit for
   bit, so no group decays.
4. Clipping: each group clips by its own global norm. Gradients of the probe heads change the update of lm_head (same
   AdamH group) and of the final norm and embeddings (same Adam group), and leave the MuonH update unchanged. With eps
   1e-20 a clip factor that stays the same over steps cancels in Adam, so the check grows the probe gradients in the
   second of two steps only.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS", "0")

import dataclasses  # noqa: E402
import re  # noqa: E402
from collections import defaultdict  # noqa: E402

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jrandom  # noqa: E402
import numpy as np  # noqa: E402

import haliax as hax  # noqa: E402
import haliax.nn as hnn  # noqa: E402
from haliax import Axis  # noqa: E402
from levanter.layers.attention import AttentionBackend  # noqa: E402
from levanter.optim.muonh import MuonHConfig  # noqa: E402
from levanter.utils.jax_utils import leaf_key_paths  # noqa: E402

from experiments.references.depth_arch_qwen3 import DepthArchMuonHConfig  # noqa: E402
from experiments.references.per_layer_qwen3 import PerLayerQwen3Config, PerLayerQwen3LMHeadModel  # noqa: E402

STEPS = 4959
# W&B config.yaml of muonh-qwen3-130m-della4xh100-pls1-sep-bbfrozen-ahc-vs-d48-g4 (every -vs-d48-g4 run has the same).
RUN = dict(learning_rate=0.02, adam_lr=0.008, beta1=0.9, beta2=0.98, epsilon=1e-20, momentum=0.95, nesterov=True,
           backend_steps=5, muon_epsilon=1e-5, max_grad_norm=1.0, weight_decay=0.1, lr_schedule="linear", decay=0.8,
           warmup=0, min_lr_ratio=0.0, coefficient_type="simple")
V, D, L = 200, 32, 4
Vocab = Axis("vocab", V)
FAIL = []


def check(ok, what):
    print(("ok   " if ok else "FAIL ") + what)
    if not ok:
        FAIL.append(what)


def fold(p):
    return re.sub(r"\.\d+(?=\.|$)", ".*", p)


def flat(tree):
    paths = leaf_key_paths(tree, is_leaf=lambda x: isinstance(x, hnn.Linear))
    out = {}
    for path, leaf in zip(jax.tree_util.tree_leaves(paths, is_leaf=lambda x: isinstance(x, (str, hnn.Linear))),
                          jax.tree_util.tree_leaves(tree, is_leaf=lambda x: isinstance(x, hnn.Linear))):
        out[path if isinstance(path, str) else str(path)] = leaf
    return out


def labelled(opt, params):
    """{folded path: label} for every array leaf, from the optimizer's own create_mask."""
    mask = opt.create_mask(params)
    lab = jax.tree_util.tree_leaves(mask)
    arrs = jax.tree_util.tree_leaves_with_path(eqx.filter(params, eqx.is_array))
    out = defaultdict(set)
    for (kp, _), l in zip(arrs, lab):
        out[l].add(fold(jax.tree_util.keystr(kp).replace("['", ".").replace("']", "").replace("[", ".").replace("]", "").lstrip(".")))
    return out


def model(arch):
    cfg = PerLayerQwen3Config(max_seq_len=16, hidden_dim=D, intermediate_dim=64, num_layers=L, num_heads=2, num_kv_heads=2,
                              attn_backend=AttentionBackend.VANILLA, pls_weight=1.0, pls_separate_heads=True,
                              pls_detach_backbone=True, hybrid_norm=True, depth_arch=arch, scan_layers=False)
    return PerLayerQwen3LMHeadModel.init(Vocab, cfg, key=jrandom.PRNGKey(0))


def rand_grads(params, seed, scale_aux=1.0):
    rng = np.random.default_rng(seed)

    def f(path, x):
        if not eqx.is_array(x):
            return x
        s = scale_aux if "aux_" in jax.tree_util.keystr(path) else 1.0
        return jnp.asarray(rng.standard_normal(x.shape) * 1e-2 * s, x.dtype)

    return jax.tree_util.tree_map_with_path(f, params)


def step(opt, params, grads):
    tx = opt.build(STEPS)
    st = tx.init(eqx.filter(params, eqx.is_array))
    up, st = tx.update(eqx.filter(grads, eqx.is_array), st, eqx.filter(params, eqx.is_array))
    return up, st


def two_steps(opt, params, g1, g2):
    """The second update after steps on g1 then g2. With eps 1e-20 Adam's first step is sign-like, so a clip factor that
    stays the same over steps changes nothing; one that changes between steps changes the update."""
    tx = opt.build(STEPS)
    p = eqx.filter(params, eqx.is_array)
    st = tx.init(p)
    _, st = tx.update(eqx.filter(g1, eqx.is_array), st, p)
    up, _ = tx.update(eqx.filter(g2, eqx.is_array), st, p)
    return up


def leaves_by(tree, frag):
    return [np.asarray(x) for kp, x in jax.tree_util.tree_leaves_with_path(tree) if frag in jax.tree_util.keystr(kp)]


for arch, cls in (("preln", MuonHConfig), ("hc", DepthArchMuonHConfig), ("attnres", DepthArchMuonHConfig)):
    print(f"\n=== {arch}: {cls.__name__} (probes: separate heads, backbone detached) ===")
    m = model(arch)
    opt = cls(**RUN)
    # The fields the build reads: everything but weight_decay (and the mask fields that only feed weight decay).
    print("weight_decay in the config:", opt.weight_decay)
    for lab, paths in sorted(labelled(opt, m).items()):
        print(f"  {lab:9s} {len(paths):3d} kinds: " + ", ".join(sorted(paths)[:12]) + (" ..." if len(paths) > 12 else ""))

    # 2. Schedules and which group reads which.
    tx = opt.build(STEPS)
    st = tx.init(eqx.filter(m, eqx.is_array))
    hp = st.hyperparams
    check(set(hp) == {"learning_rate", "adam_lr"}, f"{arch}: injected hyperparameters {sorted(hp)}")
    sched, asched = opt.lr_scheduler(STEPS), opt.lr_scheduler(STEPS, override_lr=opt.adam_lr)
    for s in (0, 991, 992, 2000, 4000, 4958):
        print(f"  step {s:4d}: learning_rate {float(sched(s)):.5f}  adam_lr {float(asched(s)):.5f}")
    g = rand_grads(m, 1)
    up0, _ = step(opt, m, g)
    p_head, u_head = leaves_by(eqx.filter(m, eqx.is_array), ".lm_head")[0], leaves_by(up0, ".lm_head")[0]
    # AdamH: a norm-preserving step of size learning_rate (ratio of the update to the weight, Frobenius).
    r = np.linalg.norm(u_head) / np.linalg.norm(p_head)
    print(f"  lm_head (AdamH) first step |du|/|w| = {r:.4f}  (learning_rate {RUN['learning_rate']}, adam_lr {RUN['adam_lr']})")
    check(abs(r - RUN["learning_rate"]) < 0.25 * RUN["learning_rate"], f"{arch}: lm_head steps at learning_rate, not adam_lr")

    # 3. Weight decay does nothing.
    up1, _ = step(dataclasses.replace(opt, weight_decay=0.0), m, g)
    same = all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(jax.tree_util.tree_leaves(up0), jax.tree_util.tree_leaves(up1)))
    check(same, f"{arch}: weight_decay 0.1 and 0.0 give the same update bit for bit (no decay anywhere)")

    # 4. Clip groups: in the second of two steps, scale only the probe heads' gradients (x1000) and see which updates move.
    up0 = two_steps(opt, m, g, rand_grads(m, 2))
    up2 = two_steps(opt, m, g, rand_grads(m, 2, scale_aux=1e3))
    moved = {}
    for frag in (".lm_head", "transformer.norm", "embeddings", "self_attn.q_proj", "mlp.down_proj", "hyper", "query", "src_logit"):
        a, b = leaves_by(up0, frag), leaves_by(up2, frag)
        if a:
            moved[frag] = max(float(np.linalg.norm(x - y) / max(np.linalg.norm(x), 1e-30)) for x, y in zip(a, b))
    print("  relative change of each update when only the probe heads' gradients grow x1000:")
    for k, v in moved.items():
        print(f"    {k:18s} {v:.3e}")
    check(moved[".lm_head"] > 0.5 and moved["transformer.norm"] > 0.5, f"{arch}: probe gradients rescale lm_head and the final norm (shared clip groups)")
    check(moved["self_attn.q_proj"] < 1e-6 and moved["mlp.down_proj"] < 1e-6, f"{arch}: MuonH layers do not see the probe gradients")

print("\nFAILED: " + "; ".join(FAIL) if FAIL else "\nall checks passed")
raise SystemExit(1 if FAIL else 0)
