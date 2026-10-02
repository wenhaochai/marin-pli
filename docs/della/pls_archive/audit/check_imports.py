"""Audit: which checkout does every relevant module resolve to under scripts/della/pls_env.sh?"""
import importlib
import inspect
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
WT = "/scratch/gpfs/GROUP/USER/project/marin-pls"
OTHER = "/scratch/gpfs/GROUP/USER/project/marin/"

print("sys.path (first 20):")
for p in sys.path[:20]:
    print("   ", p)
print("meta_path:", [type(f).__name__ for f in sys.meta_path])

mods = [
    "levanter", "levanter.main.train_lm", "levanter.eval", "levanter.models.loss", "levanter.models.qwen",
    "levanter.trainer", "levanter.grad_accum", "levanter.kernels.pallas.fused_cross_entropy_loss.api",
    "haliax", "haliax.nn.scan", "marin", "marin.training.training", "marin.execution.lazy", "fray", "fray.local_backend",
    "rigging", "iris", "zephyr", "finelog", "finestore", "ducky", "dupekit", "dupekit_native", "iris_native",
    "tasktrove_verify", "experiments", "experiments.references.per_layer_qwen3",
    "experiments.references.della_muonh_qwen3_scaling", "experiments.datasets.paloma", "experiments.marin_tokenizer",
]
bad = []
for m in mods:
    try:
        mod = importlib.import_module(m)
    except Exception as e:  # noqa: BLE001
        print(f"{m:60s} IMPORT FAILED: {type(e).__name__}: {str(e)[:150]}")
        continue
    f = getattr(mod, "__file__", None)
    paths = list(getattr(mod, "__path__", []) or [])
    where = "WT" if (f or "").startswith(WT) else ("OTHER" if (f or "").startswith(OTHER) else "venv/other")
    if f is None and paths:
        where = "namespace:" + ",".join("WT" if p.startswith(WT) else ("OTHER" if p.startswith(OTHER) else p) for p in paths)
    print(f"{m:60s} {where:12s} {f}")
    if where == "OTHER" or "OTHER" in where:
        bad.append(m)

import levanter.main.train_lm as tl  # noqa: E402

src = inspect.getsource(tl.main)
print("train_lm.main has extra_eval_callbacks hook:", "extra_eval_callbacks" in src)
print("modules resolving into the OTHER checkout:", bad)
# every loaded module: anything from the other checkout?
leak = sorted(n for n, m in list(sys.modules.items()) if getattr(m, "__file__", None) and m.__file__.startswith(OTHER) and "/.venv/" not in m.__file__)
print("ALL loaded modules from OTHER checkout:", len(leak), leak[:30])
