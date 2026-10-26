"""CE gradient precision and speed by path on one H100 (speed test, 2026-10-26). For B rows, H 768 and each vocabulary V,
with labels drawn from the model's own softmax (Gumbel-max), the mean-loss gradients of x and w from:
  default  = the kernel API with its default implementations (loss.py's path when the GIF path is not taken: the
             XLA streaming path below 65536 entries, the b-tiled path from the logsumexp above),
  gif_kern = _full_vocab_b_tiled_forward_with_grads with the Triton lse kernel (bf16 logits; the 128K production path),
  gif_bf16 = the same without the kernel (f32 GEMM, logits rounded to bf16 for the backward),
  gif_f32  = the same with LEVANTER_CE_BTILED_F32_LOGITS=1 (the backward reads the f32 logits),
against an f32 reference (inputs upcast, HIGHEST precision). Prints one JSON line per (V, path): relative Frobenius
errors of grad_x and grad_w, the loss error, and the median ms of the jitted loss-and-gradients call."""
import json
import os
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np

from levanter.kernels.pallas.fused_cross_entropy_loss import batched_xla as bx
from levanter.kernels.pallas.fused_cross_entropy_loss import fused_cross_entropy_loss_and_logsumexp_penalty as api_ce

B, H, BLK = int(os.environ.get("CHECK_B", 65536)), 768, 8192


def bench(fn, *args, reps=10):
    jax.block_until_ready(fn(*args))
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e3


for V in (8256, 32256, 128256):
    kx, kw, kg = jax.random.split(jax.random.PRNGKey(V), 3)
    x = jax.random.normal(kx, (B, H), jnp.float32).astype(jnp.bfloat16)
    w = (jax.random.normal(kw, (H, V), jnp.float32) * (3.0 / np.sqrt(H))).astype(jnp.bfloat16)
    xf, wf = x.astype(jnp.float32), w.astype(jnp.float32)

    @jax.jit
    def ref_block(xb, gk):
        z = jnp.dot(xb, wf, precision=jax.lax.Precision.HIGHEST)
        lab = jnp.argmax(z + jax.random.gumbel(gk, z.shape), axis=1)
        lse = jax.nn.logsumexp(z, axis=1)
        loss = lse - z[jnp.arange(z.shape[0]), lab]
        d = (jnp.exp(z - lse[:, None]) - jax.nn.one_hot(lab, V, dtype=jnp.float32)) / B
        gx = jnp.dot(d, wf.T, precision=jax.lax.Precision.HIGHEST)
        gw = jnp.dot(xb.T, d, precision=jax.lax.Precision.HIGHEST)
        return lab, loss, gx, gw

    labs, losses, gxs, gw_ref = [], [], [], jnp.zeros((H, V), jnp.float32)
    for i in range(B // BLK):
        lab, loss, gx, gw = ref_block(xf[i * BLK:(i + 1) * BLK], jax.random.fold_in(kg, i))
        labs.append(lab); losses.append(loss); gxs.append(gx); gw_ref = gw_ref + gw
    labels = jnp.concatenate(labs).astype(jnp.int32)
    loss_ref = float(jnp.mean(jnp.concatenate(losses)))
    gx_ref = jnp.concatenate(gxs)
    g_loss = jnp.full((B,), 1.0 / B, jnp.float32)

    def gif(env):
        def run(x, w):
            old = {k: os.environ.get(k) for k in env}
            os.environ.update(env)
            try:
                loss, _, gx, gw = bx._full_vocab_b_tiled_forward_with_grads(
                    x, labels, w, g_loss, b_block_size=BLK, dtype=jnp.float32, logit_soft_cap=None,
                    precision=jax.lax.Precision.DEFAULT)
            finally:
                for k, v in old.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
            return jnp.mean(loss), gx, gw
        return jax.jit(run)

    def default_path(x, w):
        f = lambda x, w: api_ce(x, labels, w, reduction="mean", dtype=jnp.float32, precision=None)   # loss.py's fallback
        loss, (gx, gw) = jax.value_and_grad(f, argnums=(0, 1))(x, w)
        return loss, gx, gw

    def unnorm(x, w):
        """The CE backward from unnormalized probabilities P = bf16(exp(z)) (offset c = 0; speed doc idea, 2026-10-26):
        cuBLAS-shaped GEMMs on P with the row factor a = g exp(-lse) on the other side and the label terms apart."""
        losses, gxs, gw = [], [], jnp.zeros((H, V), jnp.float32)
        for i in range(B // BLK):
            xb, lab = x[i * BLK:(i + 1) * BLK], labels[i * BLK:(i + 1) * BLK]
            z = jnp.dot(xb, w, preferred_element_type=jnp.float32)
            p = jnp.exp(z).astype(jnp.bfloat16)
            lse = jnp.log(jnp.sum(jnp.exp(z), axis=1))
            losses.append(lse - z[jnp.arange(BLK), lab])
            g = jnp.full((BLK,), 1.0 / B, jnp.float32)
            a = g * jnp.exp(-lse)
            wl = jnp.take(w, lab, axis=1).T.astype(jnp.float32)   # [BLK, H]: each row's label column
            gx = a[:, None] * jnp.dot(p, w.T, preferred_element_type=jnp.float32) - g[:, None] * wl
            gw_b = jnp.dot((a[:, None] * xb.astype(jnp.float32)).astype(jnp.bfloat16).T, p, preferred_element_type=jnp.float32)
            gw_b = gw_b.at[:, lab].add(-(g[:, None] * xb.astype(jnp.float32)).T)
            gxs.append(gx); gw = gw + gw_b
        return jnp.mean(jnp.concatenate(losses)), jnp.concatenate(gxs), gw

    paths = {
        "default": jax.jit(default_path),
        "gif_kern": gif({"LEVANTER_CE_LSE_KERNEL": "1", "LEVANTER_CE_BTILED_ONEHOT": "1"}),
        "gif_bf16": gif({"LEVANTER_CE_LSE_KERNEL": "0", "LEVANTER_CE_BTILED_ONEHOT": "1"}),
        "gif_f32": gif({"LEVANTER_CE_LSE_KERNEL": "1", "LEVANTER_CE_BTILED_ONEHOT": "1", "LEVANTER_CE_BTILED_F32_LOGITS": "1"}),
        "unnorm": jax.jit(unnorm),
        "gif_unnorm": gif({"LEVANTER_CE_LSE_KERNEL": "1", "LEVANTER_CE_BTILED_ONEHOT": "1", "LEVANTER_CE_UNNORM": "1"}),
    }
    for name, fn in paths.items():
        try:
            jax.clear_caches()
            loss, gx, gw = fn(x, w)
            rel = lambda a, r: float(jnp.linalg.norm((a.astype(jnp.float32) - r).ravel()) / jnp.linalg.norm(r.ravel()))
            row = {"V": V, "path": name, "grad_x_rel": rel(gx, gx_ref), "grad_w_rel": rel(gw, gw_ref),
                   "loss_err": abs(float(loss) - loss_ref), "ms": round(bench(fn, x, w), 2)}
        except Exception as e:
            row = {"V": V, "path": name, "error": f"{type(e).__name__}: {str(e)[:300]}"}
        print(json.dumps(row), flush=True)
