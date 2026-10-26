"""Embedding-gradient scatter on one H100 (speed work 2026-10-26). In the 300m train step the input-embedding gradient
is a bf16 scatter-add into a buffer XLA lays out column-major ([768, V] row-major), because FSDP shards the table along
embed; each token's 768 values then sit V elements apart, and the scatter took 11.3 ms per 131072 tokens at 128K (w304)
and 7.9 ms at 8K (w305). Times the same scatter in other forms at both vocabularies, with Zipf-distributed token ids:
the column-major bf16 form, row-major bf16 and f32, row-major bf16 followed by a transpose into the column-major form,
and a sort plus sorted segment sum in f32; and the error of each bf16 form against the f32 sum on the 16 most frequent
rows. Prints one JSON line per form."""
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

N, E = 131072, 768


def zipf_ids(v, n, seed=0):
    rng = np.random.default_rng(seed)
    ids = np.minimum(rng.zipf(1.1, size=4 * n), v)[:n] - 1
    perm = rng.permutation(v)   # frequent ids spread over the table, as in a real vocabulary
    return jnp.asarray(perm[ids].astype(np.int32))


def timeit(fn, *args, reps=20):
    out = fn(*args)
    jax.block_until_ready(out)
    t0 = time.perf_counter()
    for _ in range(reps):
        out = fn(*args)
    jax.block_until_ready(out)
    return (time.perf_counter() - t0) / reps * 1e3, out


def make_forms(v):
    return {
        "colmajor_bf16": jax.jit(lambda g, ids: jnp.zeros((E, v), jnp.bfloat16).at[:, ids].add(g.T)),
        "rowmajor_bf16": jax.jit(lambda g, ids: jnp.zeros((v, E), jnp.bfloat16).at[ids].add(g)),
        "rowmajor_f32": jax.jit(lambda g, ids: jnp.zeros((v, E), jnp.float32).at[ids].add(g.astype(jnp.float32))),
        "rowmajor_bf16_then_T": jax.jit(lambda g, ids: jnp.zeros((v, E), jnp.bfloat16).at[ids].add(g).T),
        "rowmajor_f32_then_T_bf16": jax.jit(
            lambda g, ids: jnp.zeros((v, E), jnp.float32).at[ids].add(g.astype(jnp.float32)).T.astype(jnp.bfloat16)),
        "sorted_segsum_f32": jax.jit(lambda g, ids: jax.ops.segment_sum(
            g[jnp.argsort(ids)].astype(jnp.float32), jnp.sort(ids), num_segments=v, indices_are_sorted=True)),
    }


for v in (128256, 8000):
    ids = zipf_ids(v, N)
    g = jax.random.normal(jax.random.PRNGKey(0), (N, E), dtype=jnp.float32).astype(jnp.bfloat16)
    forms = make_forms(v)
    ref = np.asarray(forms["rowmajor_f32"](g, ids), dtype=np.float64)
    top = np.argsort(-np.bincount(np.asarray(ids), minlength=v))[:16]
    for name, fn in forms.items():
        ms, out = timeit(fn, g, ids)
        o = np.asarray(out.astype(jnp.float32), dtype=np.float64)
        if o.shape != ref.shape:
            o = o.T
        rel = float(np.linalg.norm(o[top] - ref[top]) / np.linalg.norm(ref[top]))
        print(json.dumps({"vocab": v, "form": name, "ms": round(ms, 3), "top16_rel_err": rel,
                          "top_count": int(np.bincount(np.asarray(ids), minlength=v).max())}), flush=True)
