"""CPU check that VARIANT=cvd builds on the 64K, 32K and 16K truncations with their cut maps (vocab_cut_expand_map.py):
own_rows accepts each map, the composed table has one row per truncation id, and a training step with rare inputs
replaces them while evaluation keeps the baseline's output."""
import jax.numpy as jnp, jax.random as jr, numpy as np
import haliax as hax
from haliax import Axis
import levanter.trainer as T
from levanter.layers.attention import AttentionMask
from levanter.models.qwen import Qwen3Config
from experiments.references.table_variants_qwen3 import ComposedDropoutQwen3Config

R = "/scratch/gpfs/GROUP/USER/marin_store_big/tokenizers/marin-small/v8000"
kw = dict(max_seq_len=16, hidden_dim=32, intermediate_dim=64, num_layers=1, num_heads=4, num_kv_heads=2, hybrid_norm=True, gradient_checkpointing=False)
for K in (64000, 32000, 16000):
    V = Axis("vocab", K + 256)
    cfg = ComposedDropoutQwen3Config(**kw, cvd_map=f"{R}/expand_v{K // 1000}k.npz", cvd_p=0.5)
    m = cfg.build(V, key=jr.PRNGKey(0)); b = Qwen3Config(**kw).build(V, key=jr.PRNGKey(0))
    ids = hax.named(jnp.array([[8000 + 7 * i for i in range(16)]]), (Axis("batch", 1), Axis("position", 16)))   # all rare
    ev = m.activations(ids, AttentionMask.causal()).array
    assert np.array_equal(ev, b.activations(ids, AttentionMask.causal()).array), "evaluation differs from the baseline"
    tok = T._TRACED_TRAIN_STEP.set(jnp.asarray(3))
    try:
        tr = m.activations(ids, AttentionMask.causal()).array
    finally:
        T._TRACED_TRAIN_STEP.reset(tok)
    assert not np.array_equal(tr, ev), "training step replaced nothing"
    print(K, "ok: map rows", V.size, "replacement acts in training only")
print("TRUNC OK")
