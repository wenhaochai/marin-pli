"""LEVANTER_SCAN_EMPTY check on 4 GPUs (speed test, 2026-10-26): a 12-step lax.scan whose stacked output [12, B, D] is
sharded along B over the 4 devices, compiled with jax's zero fill and with levanter.utils.jax_utils.patch_scan_empty_outputs.
Prints, per form: AllocateBuffer custom calls in the lowered module, broadcast instructions of the stacked shape in the
compiled per-device HLO, the compiled temp bytes, the run time, and the largest difference of the two forms' outputs."""
import json
import os
import re
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

L, B, D = 12, int(os.environ.get("CHECK_B", 4 * 8192)), int(os.environ.get("CHECK_D", 2688))
mesh = Mesh(np.array(jax.devices()[:4]), ("data",))
xs = NamedSharding(mesh, P("data", None))
x = jax.device_put(jax.random.normal(jax.random.PRNGKey(0), (B, 768), jnp.bfloat16), xs)
w = jax.device_put(jax.random.normal(jax.random.PRNGKey(1), (L, 768, D), jnp.bfloat16) * 0.03, NamedSharding(mesh, P()))


def make():
    def f(x, w):
        def body(c, wl):
            g = jnp.dot(c, wl)   # [B, D], sharded along B
            return c + jnp.dot(g, wl.T) * 1e-3, g
        c, ys = jax.lax.scan(body, x, w)
        return c, ys
    return jax.jit(f, out_shardings=(xs, NamedSharding(mesh, P(None, "data", None))))


def run(name):
    jax.clear_caches()
    fn = make()
    lowered = fn.lower(x, w)
    n_alloc = lowered.as_text().count("AllocateBuffer")
    compiled = lowered.compile()
    hlo = compiled.as_text()
    big = len(re.findall(rf"= bf16\[12,{B // 4},{D}\]\{{[0-9,]*\}} broadcast\(", hlo))
    mem = compiled.memory_analysis()
    out = fn(x, w)
    jax.block_until_ready(out)
    t0 = time.perf_counter()
    for _ in range(10):
        out = fn(x, w)
    jax.block_until_ready(out)
    ms = (time.perf_counter() - t0) / 10 * 1e3
    print(json.dumps({"form": name, "allocate_buffer_calls": n_alloc, "stacked_broadcasts_per_device": big,
                      "temp_bytes": getattr(mem, "temp_size_in_bytes", None), "ms": round(ms, 3)}), flush=True)
    return [np.asarray(jax.device_get(o)).astype(np.float32) for o in out]


ref = run("zero_fill")
from levanter.utils.jax_utils import patch_scan_empty_outputs  # noqa: E402

patch_scan_empty_outputs()
new = run("allocate_buffer")
print(json.dumps({"max_abs_diff_carry": float(np.max(np.abs(ref[0] - new[0]))),
                  "max_abs_diff_stacked": float(np.max(np.abs(ref[1] - new[1])))}), flush=True)
