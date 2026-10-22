"""CPU checks of LrMultMuonHConfig and DecayToPiecesMuonHConfig against MuonH on a tiny Qwen3.

LrMult: the main schedule equals the baseline's times 1 before lr_mult_from_step and times lr_mult from it on, the Adam
schedule is the baseline's, and every update before that step equals the baseline's bit for bit.
DecayToPieces: before dtp_from_step every update equals the baseline's; from it on the input-embedding update is the
baseline's minus adam_lr * dtp * (w_t - mean of t's pieces' rows) on the rare rows only (checked against a numpy
reference built from the map), every other update and the optimizer state stay the baseline's."""
import os
import tempfile

import equinox as eqx
import jax
import jax.random as jr
import numpy as np

import haliax as hax
from haliax import Axis
from levanter.layers.attention import AttentionMask
from levanter.models.qwen import Qwen3Config
from levanter.optim.muonh import MuonHConfig
from experiments.references.decay_to_pieces_muonh import DecayToPiecesMuonHConfig
from experiments.references.lr_mult_muonh import LrMultMuonHConfig

kw = dict(max_seq_len=16, hidden_dim=32, intermediate_dim=64, num_layers=2, num_heads=4, num_kv_heads=2, hybrid_norm=True, gradient_checkpointing=False)
NV, K = 300, 20                       # vocabulary: 20 small tokens, 24 rare (20..43), 256 specials (44..299)
V = Axis("vocab", NV); B = Axis("batch", 2); P = Axis("position", 16)
model = Qwen3Config(**kw).build(V, key=jr.PRNGKey(0))
ids = hax.named(jr.randint(jr.PRNGKey(1), (2, 16), 0, NV), (B, P))
params = eqx.filter(model, eqx.is_array)
grads = eqx.filter_grad(lambda m: (m.activations(ids, AttentionMask.causal()).array ** 2).mean())(model)
okw = dict(learning_rate=0.01, adam_lr=0.002, beta1=0.9, beta2=0.98, epsilon=1e-15, weight_decay=0.1, warmup=2, min_lr_ratio=0.0)
emb = lambda t: t.embeddings.token_embeddings.weight
leaves = lambda t: jax.tree_util.tree_leaves(eqx.filter(t, eqx.is_array))

# --- LrMult
base_cfg, lm_cfg = MuonHConfig(**okw), LrMultMuonHConfig(**okw, lr_mult=1.5, lr_mult_from_step=3)
fb, fl = base_cfg.lr_scheduler(10), lm_cfg.lr_scheduler(10)
ab, al = base_cfg.lr_scheduler(10, override_lr=0.002), lm_cfg.lr_scheduler(10, override_lr=0.002)
for c in range(10):
    assert np.isclose(float(fl(c)), float(fb(c)) * (1.5 if c >= 3 else 1.0), rtol=1e-6), c
    assert float(al(c)) == float(ab(c)), c
base, lm = base_cfg.build(10), lm_cfg.build(10)
sb, sl = base.init(params), lm.init(params)
for step in range(5):
    ub, sb = base.update(grads, sb, params); ul, sl = lm.update(grads, sl, params)
    same = all(np.array_equal(a, b) for a, b in zip(leaves(ub), leaves(ul)))
    assert same == (step < 3), f"step {step}: equal={same}"
print("LRMULT OK")

# --- DecayToPieces: a map whose rare token t (K..K+23) has the pieces (t % K, (3 t) % K) and the others their own row
offs, flat = [0], []
for t in range(NV):
    if K <= t < NV - 256:
        flat += [t % K, (3 * t) % K]
    else:
        flat += [t if t < K else K + t - (NV - 256)]
    offs.append(len(flat))
path = os.path.join(tempfile.mkdtemp(), "expand_full.npz")
np.savez(path, flat=np.array(flat, np.int32), offs=np.array(offs, np.int64))
base = MuonHConfig(**okw).build(10); dtp = DecayToPiecesMuonHConfig(**okw, dtp_map=path, dtp=0.35, dtp_from_step=2).build(10)
sb, sd = base.init(params), dtp.init(params)
w = np.asarray(emb(params).rearrange(("vocab", "embed")).array, np.float64)
ref = np.zeros_like(w)
for t in range(K, NV - 256):
    ref[t] = w[t] - (w[t % K] + w[(3 * t) % K]) / 2
for step in range(4):
    lr = float(sb.hyperparams["adam_lr"])
    ub, sb = base.update(grads, sb, params); ud, sd = dtp.update(grads, sd, params)
    rb, rd = leaves(eqx.tree_at(emb, ub, None)), leaves(eqx.tree_at(emb, ud, None))
    assert all(np.array_equal(a, b) for a, b in zip(rb, rd)), "non-embedding update differs"
    eb, ed = (np.asarray(emb(u).rearrange(("vocab", "embed")).array, np.float64) for u in (ub, ud))
    if step < 2:
        assert np.array_equal(eb, ed), f"step {step}: differs before the decay starts"
    else:
        assert np.allclose(ed, eb - lr * 0.35 * ref, atol=1e-7), f"step {step}: not baseline minus lr*dtp*(w - pieces)"
        assert np.array_equal(ed[:K], eb[:K]) and np.array_equal(ed[NV - 256:], eb[NV - 256:]), "a non-rare row moved"
        assert np.abs(ref[K:NV - 256]).max() > 0
    print("step", step, "ok, lr", lr)
assert all(np.array_equal(a, b) for a, b in zip(jax.tree_util.tree_leaves(sb), jax.tree_util.tree_leaves(sd))), "state differs"
print("DTP OK")
