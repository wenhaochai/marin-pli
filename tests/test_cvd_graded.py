"""CPU check of the frequency-graded cvd rate (cvd_rate_count, survey idea 4): graded_rate gives p * min(1, C / count);
in a training step a rare token with count far above C is never replaced while one with count 1 is replaced at about
rate p; the default config (no graded rate) keeps the fixed rate at every rare token."""
import os
import tempfile

import jax.numpy as jnp
import jax.random as jr
import numpy as np

import haliax as hax
from haliax import Axis
from experiments.references.table_variants_qwen3 import ComposedDropoutQwen3Config, graded_rate

MAP = "/scratch/gpfs/GROUP/USER/marin_store_big/tokenizers/marin-small/v8000/expand_full.npz"
NV = 128256
kw = dict(max_seq_len=64, hidden_dim=32, intermediate_dim=64, num_layers=1, num_heads=4, num_kv_heads=2, hybrid_norm=True, gradient_checkpointing=False)
counts = np.where(np.arange(NV) % 2 == 0, 1, 10**9).astype(np.int64)   # even ids: count 1, odd ids: very common
path = os.path.join(tempfile.mkdtemp(), "counts.npy"); np.save(path, counts)
r = graded_rate(path, NV, 0.5, 10)
assert np.allclose(r[0::2], 0.5) and np.allclose(r[1::2], 0.5 * 10 / 10**9), "graded_rate"
V = Axis("vocab", NV); Bt, Ps = Axis("batch", 8), Axis("position", 64)
ids = hax.named(jr.randint(jr.PRNGKey(5), (8, 64), 8000, NV - 256), (Bt, Ps))   # all rare
own = ComposedDropoutQwen3Config(**kw, cvd_map=MAP, cvd_p=1.0).build(V, key=jr.PRNGKey(0))
x_own = own.embeddings.token_embeddings(ids).array
graded = ComposedDropoutQwen3Config(**kw, cvd_map=MAP, cvd_p=0.5, cvd_rate_count=10, cvd_counts=path).build(V, key=jr.PRNGKey(0))
x = graded._embed_train(ids, jnp.asarray(7)).array
if graded.embeddings.norm is not None:
    x_own = graded.embeddings.norm(hax.named(x_own, ids.axes + (graded.config.Embed,))).array
changed = ~np.all(np.isclose(x, x_own, atol=0), axis=-1)
even = np.asarray(ids.array) % 2 == 0
assert not changed[~even].any(), "a common rare token was replaced"
frac = changed[even].mean()
assert 0.35 < frac < 0.65, f"count-1 tokens replaced at {frac:.2f}, expected about 0.5"
print("graded ok: odd replaced", int(changed[~even].sum()), "even replaced fraction", round(float(frac), 3))
fixed = ComposedDropoutQwen3Config(**kw, cvd_map=MAP, cvd_p=0.5).build(V, key=jr.PRNGKey(0))
xf = fixed._embed_train(ids, jnp.asarray(7)).array
cf = ~np.all(np.isclose(xf, x_own, atol=0), axis=-1)
assert 0.35 < cf.mean() < 0.65 and cf[~even].any(), "fixed-rate draw"
print("GRADED OK")
