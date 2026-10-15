"""CPU tests for post-hoc probes (launcher PROBE_FROM): per_layer_qwen3.load_without_probes, pls_probe_frozen_main and
ProbesOnlyMuonHConfig.

    WT=<worktree>; PYTHONPATH=$WT:$WT/lib/*/src JAX_PLATFORMS=cpu nice -n 19 <venv>/bin/python scripts/della/pls_probe_cpu_test.py
1. Load: checkpoints of an NTP source (pls_weight 0) and an MTP source (pls_weight 0, MTP module), saved with levanter's
   save_checkpoint, load into the probe model: embeddings, blocks, final norm and lm_head equal the source's bit for
   bit; the probes keep their fresh init.
2. Strict: a source without an lm_head (tied embeddings) fails to load; a model without probes is refused.
3. Readout: the probe model's last-layer next-token statistic equals the source model's own loss on the batch.
4. Gradients: zero into the embeddings, the blocks, the final norm and lm_head; nonzero into every intermediate head
   and every future probe.
5. Labels: aux/fut heads adamh, aux/fut norms adam, every other leaf (lm_head and the final norm included) frozen.
6. Training: 20 steps of ProbesOnlyMuonHConfig leave everything but the probes bit-identical, and every probe loss falls.
7. Config: pls_probe_frozen_main without pls_heads_only, and pls_heads_only with an MTP module, are rejected.
"""
import os
import tempfile

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS", "0")

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jrandom  # noqa: E402
import numpy as np  # noqa: E402

import haliax as hax  # noqa: E402
from haliax import Axis  # noqa: E402
from levanter.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from levanter.layers.attention import AttentionBackend, AttentionMask  # noqa: E402
from levanter.models.lm_model import LmExample  # noqa: E402

from experiments.references.per_layer_qwen3 import PerLayerQwen3Config, PerLayerQwen3LMHeadModel, ProbesOnlyMuonHConfig, load_without_probes  # noqa: E402

B, T, V, L = 8, 16, 128, 6
Batch, Pos, Vocab = Axis("batch", B), Axis("position", T), Axis("vocab", V)
perm = np.random.default_rng(0).permutation(V)
FAIL = []


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond:
        FAIL.append(msg)


def batch(i):
    r = np.random.default_rng(1000 + i)
    tok = np.zeros((B, T), np.int64)
    tok[:, 0] = r.integers(0, V, B)
    for t in range(1, T):
        tok[:, t] = np.where(r.random(B) < 0.2, r.integers(0, V, B), perm[tok[:, t - 1]])
    lw = (np.arange(T) < T - 1).astype(np.float32)[None].repeat(B, 0)
    return LmExample(tokens=hax.named(jnp.asarray(tok, jnp.int32), (Batch, Pos)), loss_weight=hax.named(jnp.asarray(lw), (Batch, Pos)), attn_mask=AttentionMask.causal())


def common(**kw):
    return dict(max_seq_len=T, hidden_dim=32, intermediate_dim=64, num_layers=L, num_heads=2, num_kv_heads=2, hybrid_norm=True, use_qk_norm=False, attn_backend=AttentionBackend.VANILLA, **kw)


PROBE = dict(pls_weight=1.0, pls_separate_heads=True, pls_heads_only=True, pls_probe_frozen_main=True, pls_future_probes=True)


def maxdiff(a, b):
    la, lb = jax.tree.leaves(a), jax.tree.leaves(b)
    assert len(la) == len(lb), (len(la), len(lb))
    return max(float(jnp.max(jnp.abs(x.astype(jnp.float32) - y.astype(jnp.float32)))) for x, y in zip(la, lb))


def trunk(m):
    return eqx.filter((m.embeddings, m.transformer, m.lm_head), eqx.is_array)


def probes(m):
    return eqx.filter((m.aux_norms, m.aux_lm_heads, m.fut_norms, m.fut_lm_heads), eqx.is_array)


def stat(v):
    for name in ("value", "item"):
        if hasattr(v, name) and callable(getattr(v, name)):
            try:
                return float(getattr(v, name)())
            except Exception:
                pass
    return float(v)


def scalar(x):
    x = x[0] if isinstance(x, tuple) else x
    return float((x.array if isinstance(x, hax.NamedArray) else x))


# Sources: trained a little so that they differ from any fresh init (random perturbation of every array).
def perturbed(m, seed):
    leaves, tdef = jax.tree.flatten(eqx.filter(m, eqx.is_array))
    keys = jrandom.split(jrandom.PRNGKey(seed), len(leaves))
    new = [x + 0.05 * jrandom.normal(k, x.shape, x.dtype) for x, k in zip(leaves, keys)]
    return eqx.combine(jax.tree.unflatten(tdef, new), m)


src_ntp = perturbed(PerLayerQwen3LMHeadModel.init(Vocab, PerLayerQwen3Config(**common(pls_weight=0.0)), key=jrandom.PRNGKey(7)), 1)
src_mtp = perturbed(PerLayerQwen3LMHeadModel.init(Vocab, PerLayerQwen3Config(**common(pls_weight=0.0, pls_mtp_weight=0.3)), key=jrandom.PRNGKey(8)), 2)
src_tied = perturbed(PerLayerQwen3LMHeadModel.init(Vocab, PerLayerQwen3Config(**common(pls_weight=0.0, tie_word_embeddings=True)), key=jrandom.PRNGKey(9)), 3)
probe_cfg = PerLayerQwen3Config(**common(**PROBE))
fresh = PerLayerQwen3LMHeadModel.init(Vocab, probe_cfg, key=jrandom.PRNGKey(0))
load = load_without_probes(load_checkpoint)
MESH = dict(mesh=jax.sharding.Mesh(np.array(jax.devices()[:1]), ("data",)), axis_mapping={})

with tempfile.TemporaryDirectory() as tmp:
    paths = {}
    for name, src in (("ntp", src_ntp), ("mtp", src_mtp), ("tied", src_tied)):
        paths[name] = os.path.join(tmp, name, "step-100")
        save_checkpoint({"model": src}, 100, paths[name], is_temporary=False)

    # 1. load
    got = {}
    for name, src in (("ntp", src_ntp), ("mtp", src_mtp)):
        got[name] = load(fresh, paths[name], subpath="model", **MESH)
        check(maxdiff(trunk(got[name]), trunk(src)) == 0.0, f"1. {name} source: embeddings, blocks, final norm, lm_head loaded bit for bit")
        check(maxdiff(probes(got[name]), probes(fresh)) == 0.0, f"1. {name} source: probes keep their fresh init")
        check(got[name].mtp is None, f"1. {name} source: the probe model carries no MTP module")
    check(maxdiff(trunk(fresh), trunk(src_ntp)) > 0.01, "1. the fresh model differs from the source (the load check is not vacuous)")

    # 2. strict
    try:
        load(fresh, paths["tied"], subpath="model", **MESH)
        check(False, "2. a source without an lm_head must fail to load")
    except Exception as e:
        check("lm_head" in str(e) or "Missing" in str(e) or "missing" in str(e), f"2. a source without an lm_head fails: {type(e).__name__}")
    try:
        load(src_ntp, paths["ntp"], subpath="model", **MESH)
        check(False, "2. a model without probes must be refused")
    except ValueError as e:
        check("probes" in str(e), "2. a model without probes is refused")

m = got["ntp"]
KW = dict(key=jrandom.PRNGKey(1), logsumexp_weight=0.0)

# 3. readout
ex = batch(0)
_, st = m.compute_next_token_loss(ex, **KW)
own = scalar(src_ntp.compute_next_token_loss(ex, key=None, logsumexp_weight=0.0))
check(abs(stat(st[f"pls/L{L - 1}"]) - own) < 1e-5, f"3. last-layer next-token statistic {stat(st[f'pls/L{L - 1}']):.6f} == the source's own loss {own:.6f}")
check(all(f"pls2/L{k}" in st for k in range(L)) and all(f"pls/L{k}" in st for k in range(L)), f"3. statistics for {L} next-token and {L} future readouts")


# 4. gradients
def loss_fn(mm, ex):
    out = mm.compute_next_token_loss(ex, **KW)[0]
    return out.array if isinstance(out, hax.NamedArray) else out


def norm(t):
    leaves = jax.tree.leaves(eqx.filter(t, eqx.is_array))
    return float(sum(jnp.sum(jnp.abs(x.astype(jnp.float32))) for x in leaves)) if leaves else 0.0


g = eqx.filter_grad(loss_fn)(m, ex)
check(norm((g.embeddings, g.transformer, g.lm_head)) == 0.0, "4. zero gradient into the embeddings, the blocks, the final norm and lm_head")
check(all(norm(h) > 0 for h in (*g.aux_lm_heads, *g.fut_lm_heads)) and all(norm(n) > 0 for n in (*g.aux_norms, *g.fut_norms)),
      f"4. nonzero gradient into each of {L - 1} intermediate heads and {L} future probes, and their norms")


# 5. labels
def labels(mask):
    flat = jax.tree_util.tree_leaves_with_path(mask, is_leaf=lambda x: isinstance(x, str))
    return {jax.tree_util.keystr(p): lab for p, lab in flat}


lab = labels(ProbesOnlyMuonHConfig().create_mask(eqx.filter(m, eqx.is_array)))
heads = {k: v for k, v in lab.items() if k.startswith((".aux_lm_heads", ".fut_lm_heads"))}
norms = {k: v for k, v in lab.items() if k.startswith((".aux_norms", ".fut_norms"))}
rest = {k: v for k, v in lab.items() if k not in heads and k not in norms}
check(len(heads) == 2 * L - 1 and set(heads.values()) == {"adamh"}, f"5. {len(heads)} probe heads adamh")
check(len(norms) == 2 * L - 1 and set(norms.values()) == {"adam"}, f"5. {len(norms)} probe norms adam")
check(len(rest) > 10 and set(rest.values()) == {"frozen"} and any("lm_head" in k for k in rest) and any(k.startswith(".transformer.norm") for k in rest),
      f"5. every other leaf frozen ({len(rest)}, lm_head and the final norm included)")

# 6. training
opt = ProbesOnlyMuonHConfig(learning_rate=0.02, adam_lr=0.008, beta1=0.9, beta2=0.98, epsilon=1e-20, momentum=0.95, nesterov=True, backend_steps=5,
                            muon_epsilon=1e-5, max_grad_norm=1.0, weight_decay=0.1, lr_schedule="linear", decay=0.8, warmup=2, min_lr_ratio=0.0,
                            coefficient_type="simple").build(20)
state = opt.init(eqx.filter(m, eqx.is_array))


@eqx.filter_jit
def step(mm, s, ex):
    gg = eqx.filter_grad(loss_fn)(mm, ex)
    upd, s = opt.update(gg, s, eqx.filter(mm, eqx.is_array))
    return eqx.apply_updates(mm, upd), s


first = {k: stat(v) for k, v in m.compute_next_token_loss(batch(100), **KW)[1].items()}
t = m
for i in range(20):
    t, state = step(t, state, batch(i))
last = {k: stat(v) for k, v in t.compute_next_token_loss(batch(100), **KW)[1].items()}
check(maxdiff(trunk(t), trunk(m)) == 0.0, "6. after 20 steps: embeddings, blocks, final norm and lm_head bit-identical")
fell = [k for k in first if k != f"pls/L{L - 1}" and last[k] < first[k] - 0.05]
check(len(fell) == 2 * L - 1, f"6. every probe loss fell on a held-out batch ({len(fell)} of {2 * L - 1}; e.g. pls2/L0 {first['pls2/L0']:.3f} -> {last['pls2/L0']:.3f})")
check(abs(last[f"pls/L{L - 1}"] - first[f"pls/L{L - 1}"]) < 1e-6, "6. the last-layer next-token readout is unchanged")

# 7. config
for kw, what in ((dict(pls_weight=1.0, pls_separate_heads=True, pls_probe_frozen_main=True), "frozen main without heads_only"),
                 (dict(pls_weight=1.0, pls_separate_heads=True, pls_heads_only=True, pls_mtp_weight=0.3), "heads_only with an MTP module")):
    try:
        PerLayerQwen3Config(**common(**kw))
        check(False, f"7. {what} must be rejected")
    except ValueError:
        check(True, f"7. {what} rejected")

print("ALL PROBE CPU TESTS PASSED" if not FAIL else f"{len(FAIL)} FAILED")
