"""CPU checks of experiments/references/hnet_qwen3.py (2026-10-28) on a tiny config, against a loop reference written
from goombalab/hnet (hnet/models/hnet.py, hnet/modules/dc.py) that calls the model's own encoder, main network and
decoder as black boxes:
1. activations equal the reference (routing, forced boundaries at document starts, chunking, EMA restarted per document,
   plug-back, STE forward value 1, zero residual), with ample slots and with an overflow of the slots;
2. causality: changing byte t leaves every output before t unchanged;
3. documents: changing a byte of document 1 leaves documents 0 and 2 unchanged;
4. the ratio loss matches the formula, and is absent at evaluation (no key);
5. gradients are finite, and reach q_proj, k_proj (through p) and residual_proj.
Run on a vis node: JAX_PLATFORMS=cpu python test_hnet.py"""
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrandom
import haliax as hax
from haliax import Axis
import equinox as eqx
from levanter.layers.attention import AttentionBackend, AttentionMask
from levanter.trainer import current_train_step
import levanter.trainer as lt
from experiments.references.hnet_qwen3 import HNetQwen3Config, HNetQwen3LMHeadModel

ok = True


def check(name, good, extra=""):
    global ok
    ok &= bool(good)
    print(("ok " if good else "BAD"), name, extra)


L, B = 256, 2
Vocab = Axis("vocab", 512)


def make(slots):
    cfg = HNetQwen3Config(max_seq_len=L, hidden_dim=64, intermediate_dim=128, num_layers=2, num_heads=4, num_kv_heads=4,
                          hybrid_norm=True, attn_backend=AttentionBackend.VANILLA, enc_layers=1, dec_layers=1,
                          chunk_ratio=4.0, chunk_slots=slots)
    model = HNetQwen3LMHeadModel.init(Vocab, cfg, key=jrandom.PRNGKey(0))
    # perturb q/k off the identity so that boundaries are not all-or-nothing
    q = model.q_proj.weight.array + 0.3 * jrandom.normal(jrandom.PRNGKey(1), model.q_proj.weight.array.shape)
    model = eqx.tree_at(lambda m: m.q_proj.weight, model, hax.named(q, model.q_proj.weight.axes))
    return cfg, model


Batch = Axis("batch", B)
Pos = Axis("position", L)
rng = np.random.default_rng(0)
ids_np = rng.integers(0, 256, size=(B, L))
seg_np = np.zeros((B, L), np.int32)
seg_np[0, 100:180] = 1
seg_np[0, 180:] = 2
seg_np[1, 60:] = 1
ids = hax.named(jnp.asarray(ids_np), (Batch, Pos))
mask = AttentionMask.causal().with_segment_ids(hax.named(jnp.asarray(seg_np), (Batch, Pos)))


def reference(model, cfg, ids, mask):
    """Loop reference of the H-Net forward, one row at a time."""
    E, M = cfg.Embed, cfg.slots
    x = model.embeddings.embed(ids)
    h = model.encoder(x, attn_mask=mask, key=None)
    H = np.asarray(h.rearrange((Batch, Pos, E)).array, np.float64)
    Wq = np.asarray(model.q_proj.weight.rearrange((model.q_proj.Out, E)).array, np.float64)
    Wk = np.asarray(model.k_proj.weight.rearrange((model.k_proj.Out, E)).array, np.float64)
    seg = seg_np
    hc = np.zeros((B, M, E.size))
    segc = np.zeros((B, M), np.int32)
    info = []
    for r in range(B):
        p = np.ones(L)
        for t in range(1, L):
            q, k = Wq @ H[r, t - 1], Wk @ H[r, t]
            p[t] = np.clip((1 - q @ k / np.linalg.norm(q) / np.linalg.norm(k)) / 2, 0, 1)
            if seg[r, t] != seg[r, t - 1]:
                p[t] = 1.0
        b = p > 0.5
        starts = [t for t in range(L) if b[t]][:M]
        bb = np.zeros(L, bool)
        bb[starts] = True
        for j, t in enumerate(starts):
            hc[r, j] = H[r, t]
            segc[r, j] = seg[r, t]
        segc[r, len(starts):] = seg.max() + 1
        info.append((p, b, bb, starts))
    Chunk = Pos.resize(M)
    zc = model.transformer(hax.named(jnp.asarray(hc, jnp.float32), (Batch, Chunk, E)),
                           attn_mask=AttentionMask.causal().with_segment_ids(hax.named(jnp.asarray(segc), (Batch, Chunk))), key=None)
    Z = np.asarray(zc.rearrange((Batch, Chunk, E)).array, np.float64)
    Y = np.zeros((B, L, E.size))
    for r in range(B):
        p, b, bb, starts = info[r]
        z = np.zeros((len(starts), E.size))
        for j, t in enumerate(starts):
            pj = np.clip(p[t], 1e-4, 1 - 1e-4)
            first = j == 0 or seg[r, t] != seg[r, starts[j - 1]]
            z[j] = pj * Z[r, j] + (0 if first else (1 - pj) * z[j - 1])
        j = -1
        for t in range(L):
            if bb[t]:
                j += 1
            Y[r, t] = z[j]   # STE forward value 1, zero residual at init
    y = model.decoder(hax.named(jnp.asarray(Y, jnp.float32), (Batch, Pos, E)), attn_mask=mask, key=None)
    return np.asarray(y.rearrange((Batch, Pos, E)).array), info


for slots in (128, 24):
    cfg, model = make(slots)
    got = np.asarray(model.activations(ids, mask, key=None).rearrange((Batch, Pos, cfg.Embed)).array)
    want, info = reference(model, cfg, ids, mask)
    nb = [int(i[1].sum()) for i in info]
    err = float(np.max(np.abs(got - want)))
    check(f"forward equals reference, {slots} slots", err < 2e-4, f"max err {err:.2e}, boundaries per row {nb}, kept {[len(i[3]) for i in info]}")
    if slots == 24:
        check("overflow exercised", max(nb) > slots, f"{nb} > {slots}")

cfg, model = make(128)
base = np.asarray(model.activations(ids, mask, key=None).array)
ids2_np = ids_np.copy()
ids2_np[0, 150] = (ids2_np[0, 150] + 7) % 256
out2 = np.asarray(model.activations(hax.named(jnp.asarray(ids2_np), (Batch, Pos)), mask, key=None).array)
d = np.abs(out2 - base).max(axis=-1)
check("causal: outputs before the changed byte unchanged", d[0, :150].max() < 1e-5, f"max {d[0, :150].max():.1e}; at/after {d[0, 150:180].max():.1e}")
check("documents: doc 2 of row 0 and row 1 unchanged", d[0, 180:].max() < 1e-5 and d[1].max() < 1e-5, f"{d[0, 180:].max():.1e}, {d[1].max():.1e}")
check("the change reaches its own document", d[0, 150:180].max() > 1e-3)

# training path: a traced train step is set (as Trainer._train_step does), so the ratio loss is returned
tok = lt._TRACED_TRAIN_STEP.set(jnp.asarray(5))
try:
    def loss_fn(m):
        y, aux = m.activations(ids, mask, key=jrandom.PRNGKey(7))
        w = jrandom.normal(jrandom.PRNGKey(3), y.array.shape)
        return jnp.sum(y.array.astype(jnp.float32) * w) * 1e-3 + aux, aux

    (val, aux), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
finally:
    lt._TRACED_TRAIN_STEP.reset(tok)
P = np.concatenate([i[0] for i in info]); Bm = np.concatenate([i[1] for i in info]).astype(float)
want_aux = 0.03 * 4.0 / 3.0 * (3.0 * Bm.mean() * P.mean() + (1 - Bm.mean()) * (1 - P.mean()))
check("ratio loss matches the formula", abs(float(aux) - want_aux) < 1e-4, f"{float(aux):.6f} vs {want_aux:.6f}")
check("no ratio loss at evaluation", not isinstance(model.activations(ids, mask, key=None), tuple))
leaves = jax.tree_util.tree_leaves_with_path(eqx.filter(grads, eqx.is_array))
check("all gradients finite", all(bool(jnp.all(jnp.isfinite(g))) for _, g in leaves))
for name, g in (("q_proj", grads.q_proj.weight.array), ("k_proj", grads.k_proj.weight.array), ("residual_proj", grads.residual_proj.array)):
    n = float(jnp.linalg.norm(g))
    check(f"gradient reaches {name}", n > 0, f"norm {n:.3e}")
x = jnp.asarray(0.3)
check("STE forward 1, backward identity", float(jax.value_and_grad(lambda v: __import__("experiments.references.hnet_qwen3", fromlist=["_ste"])._ste(v) * 2.0)(x)[0]) == 2.0
      and float(jax.grad(lambda v: __import__("experiments.references.hnet_qwen3", fromlist=["_ste"])._ste(v) * 2.0)(x)) == 2.0)
print("ALL OK" if ok else "FAILED")
