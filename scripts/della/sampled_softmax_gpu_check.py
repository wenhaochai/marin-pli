"""H100 check of the sampled-softmax cross-entropy at the real per-device shape (32 x 4096 rows, 128,256 vocab).

Run on one GPU of a pli job (a few minutes): .venv/bin/python scripts/della/sampled_softmax_gpu_check.py
Labels are real fineweb-edu-10B windows, so the candidate sets have the training distribution of distinct targets.

1. Parity (8,192 rows): loss, dx and dW of every stage against a dense f32 reference over the same candidate set.
2. Timing (131,072 rows, D = 512 and 768): fwd+bwd of the full softmax (the baseline's kernel call) and of each stage,
   through levanter's dispatch (the full vocabulary takes the row-tiled path; P < 65,536 takes the vocab-streaming path)
   and through the row-tiled path forced at every P, plus the candidate build alone.
"""
import os
import time
from functools import partial

os.environ.setdefault("LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS", "0")

import jax
import jax.numpy as jnp
import numpy as np
import tensorstore as ts

from levanter.kernels.pallas.fused_cross_entropy_loss import fused_cross_entropy_loss_and_logsumexp_penalty as fused_ce
from levanter.kernels.pallas.fused_cross_entropy_loss.batched_xla import (
    _backward_b_tiled_from_lse,
    _linear_softmax_cross_entropy_loss_full_vocab_b_tiled,
)

from experiments.references.sampled_softmax_qwen3 import build_candidates, sampled_cross_entropy, stride_sweep

V, SEQ, PER_DEVICE = 128256, 4096, 32
STAGES = (24576, 36864, 65536)
ROOT = "/scratch/gpfs/GROUP/USER/marin_store_big/fineweb-edu-10B/2026.06.28/train/input_ids/data"
print("device", jax.devices()[0].device_kind, flush=True)

rng = np.random.default_rng(0)
data = ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": ROOT}}, read=True).result()
starts = rng.integers(0, 10_000_000_000 // SEQ - 1, size=PER_DEVICE) * SEQ
tokens = np.concatenate([np.asarray(data[int(s) : int(s) + SEQ + 1].read().result()) for s in starts[:1]] + [np.asarray(data[int(s) : int(s) + SEQ].read().result()) for s in starts[1:]])
labels_all = jnp.asarray(tokens[1 : 1 + PER_DEVICE * SEQ], jnp.int32)  # next-token targets of one device-batch
print(f"one device-batch: {labels_all.size} targets, {len(np.unique(np.asarray(labels_all)))} distinct", flush=True)
sweep = jnp.asarray(stride_sweep(V))
kernel_kw = dict(logsumexp_weight=0.0, block_size=None, dtype=jnp.float32, logit_soft_cap=None, precision=None)


# 1. parity on 8,192 rows (dense f32 logits fit)
N, D = 8192, 512
labels = labels_all[:N]
x = jnp.asarray(rng.normal(size=(N, D)), jnp.bfloat16)
w = jnp.asarray(rng.normal(size=(D, V)) * 0.05, jnp.bfloat16)
r = jnp.asarray(rng.normal(size=N) / N, jnp.float32)
offset = 777
present = jnp.zeros((V,), jnp.bool_).at[labels].set(True)
n_present = int(present.sum())


def dense(x, w, C):
    logits = x.astype(jnp.float32) @ w.astype(jnp.float32)
    return jax.nn.logsumexp(logits[:, C], axis=-1) - jnp.take_along_axis(logits, labels[:, None], axis=1)[:, 0]


for stage in range(len(STAGES) + 1):
    C = np.asarray(build_candidates(present, jnp.int32(n_present), STAGES[stage], jnp.int32(offset), sweep)) if stage < len(STAGES) else np.arange(V)
    f = jax.jit(lambda x, w, stage=stage: jnp.sum(sampled_cross_entropy(x, labels, w, stage=jnp.int32(stage), offset=jnp.int32(offset), candidates=STAGES, sweep=sweep, **kernel_kw)[0] * r))
    g = jax.jit(lambda x, w, C=C: jnp.sum(dense(x, w, C) * r))
    (lv, (gx, gw)), (lr, (rx, rw)) = jax.value_and_grad(f, argnums=(0, 1))(x, w), jax.value_and_grad(g, argnums=(0, 1))(x, w)
    rel = lambda a, b: float(jnp.linalg.norm(a.astype(jnp.float32) - b.astype(jnp.float32)) / jnp.linalg.norm(b.astype(jnp.float32)))
    per_loss = sampled_cross_entropy(x, labels, w, stage=jnp.int32(stage), offset=jnp.int32(offset), candidates=STAGES, sweep=sweep, **kernel_kw)[0]
    max_loss_err = float(jnp.max(jnp.abs(per_loss - dense(x, w, C))))
    print(f"parity stage {stage} (P={len(C)}): max |loss - dense| {max_loss_err:.2e}, rel err dx {rel(gx, rx):.2e}, dW {rel(gw, rw):.2e}", flush=True)
    assert max_loss_err < 2e-2 and rel(gx, rx) < 2e-2 and rel(gw, rw) < 2e-2, stage
    if stage < len(STAGES):
        assert float(jnp.abs(gw[:, np.setdiff1d(np.arange(V), C)]).max()) == 0.0


# 2. timing at the real device shape
@partial(jax.custom_vjp, nondiff_argnums=(3,))
def btiled_ce(x, labels, w, b_block):
    return _linear_softmax_cross_entropy_loss_full_vocab_b_tiled(x, labels, w, b_block_size=b_block, dtype=jnp.float32, logit_soft_cap=None, precision=None)[0]


def _btiled_fwd(x, labels, w, b_block):
    loss, lse = _linear_softmax_cross_entropy_loss_full_vocab_b_tiled(x, labels, w, b_block_size=b_block, dtype=jnp.float32, logit_soft_cap=None, precision=None)
    return loss, (x, labels, w, lse)


def _btiled_bwd(b_block, res, g):
    x, labels, w, lse = res
    gx, gw = _backward_b_tiled_from_lse(x, labels, w, lse, g, jnp.zeros_like(g), b_block_size=b_block, logit_soft_cap=None, precision=None)
    return gx, None, gw


btiled_ce.defvjp(_btiled_fwd, _btiled_bwd)


def bench(fn, *args, iters=10):
    out = fn(*args)
    jax.block_until_ready(out)
    out = fn(*args)
    jax.block_until_ready(out)
    t = time.perf_counter()
    for _ in range(iters):
        out = fn(*args)
    jax.block_until_ready(out)
    return (time.perf_counter() - t) / iters * 1e3


N = PER_DEVICE * SEQ
labels = labels_all
present = jnp.zeros((V,), jnp.bool_).at[labels].set(True)
n_present = jnp.sum(present, dtype=jnp.int32)
for D in (512, 768):
    x = jnp.asarray(rng.normal(size=(N, D)), jnp.bfloat16)
    w = jnp.asarray(rng.normal(size=(D, V)) * 0.05, jnp.bfloat16)
    r = jnp.full((N,), 1.0 / N, jnp.float32)
    grad = lambda loss_fn: jax.jit(jax.value_and_grad(lambda x, w: jnp.sum(loss_fn(x, w) * r), argnums=(0, 1)))
    t_full = bench(grad(lambda x, w: fused_ce(x, labels, w, reduction=None, weight=None, **kernel_kw)), x, w)
    t_build = bench(jax.jit(lambda o: build_candidates(present, n_present, STAGES[0], o, sweep)), jnp.int32(5))
    print(f"D={D}: full softmax fwd+bwd {t_full:.1f} ms (baseline kernel call); candidate build {t_build:.2f} ms", flush=True)
    for stage, P in enumerate(STAGES):
        t_model = bench(grad(lambda x, w, stage=stage: sampled_cross_entropy(x, labels, w, stage=jnp.int32(stage), offset=jnp.int32(5), candidates=STAGES, sweep=sweep, **kernel_kw)[0]), x, w)
        cand = build_candidates(present, n_present, P, jnp.int32(5), sweep)
        pos = jnp.zeros((V,), jnp.int32).at[cand].set(jnp.arange(P, dtype=jnp.int32))
        lab = pos[labels]
        t_dispatch = bench(grad(lambda x, w: fused_ce(x, lab, jnp.take(w, cand, axis=1), reduction=None, weight=None, **kernel_kw)), x, w)
        t_btiled = bench(grad(lambda x, w: btiled_ce(x, lab, jnp.take(w, cand, axis=1), 8192)), x, w)
        print(f"  P={P:6d} ({P / V:.0%} of V): model path {t_model:.1f} ms | dispatch kernel {t_dispatch:.1f} ms | row-tiled kernel {t_btiled:.1f} ms | vs full {t_full / t_model:.2f}x", flush=True)
print("GPU CHECK DONE", flush=True)
