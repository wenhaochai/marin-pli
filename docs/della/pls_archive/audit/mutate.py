"""Mutation testing of scripts/della/pls_cpu_test.py: shadow experiments.references.per_layer_qwen3 with a mutated copy
(in a scratch dir placed first on PYTHONPATH; the tracked file is untouched) and check the test fails."""
import os
import pathlib
import subprocess
import sys

WT = pathlib.Path("/scratch/gpfs/GROUP/USER/project/marin-pls")
SRC = (WT / "experiments/references/per_layer_qwen3.py").read_text()
ROOT = pathlib.Path("/scratch/gpfs/GROUP/USER/tmp/pls/audit/mut")

MUTANTS = {
    "M1 monitor target not shifted": ("target = hax.roll(example.tokens, -1, Pos)", "target = example.tokens"),
    "M2 monitor stride offset 1": ("[Stride.name, 0]", "[Stride.name, 1]"),
    "M3 readout skips final norm": ("return self.transformer.norm(outs[self.transformer.layers.Block.name, k])",
                                    "return outs[self.transformer.layers.Block.name, k]"),
    "M4 per-tag weights = total": ('this_weights_per_tag = jnp.einsum("bt,bk->k", weights, tags, out_sharding=tag_sharding)',
                                   'this_weights_per_tag = jnp.sum(weights) * jnp.ones((tags.shape[1],))'),
    "M5 readout k is the layer INPUT": ("        def step(layer, carry, **kw):\n            y = layer(carry, **kw)\n            return y, y",
                                        "        def step(layer, carry, **kw):\n            y = layer(carry, **kw)\n            return y, carry"),
    "M6 weight also on final layer": ("loss = self._readout_ce(outs, L - 1, example, **kw)",
                                      "loss = cfg.pls_weight * self._readout_ce(outs, L - 1, example, **kw)"),
    "M7 bpb per tag uses global bytes": ("bpb_per_tag = this_loss_per_tag / jnp.maximum(bytes_per_tag, 1.0) * log2e",
                                         "bpb_per_tag = this_loss_per_tag / jnp.maximum(this_bytes, 1.0) * log2e"),
    "M8 monitor drops loss weights": ("weight=strided(weight),", "weight=None,"),
    "M9 readouts reversed in eval": ("per = [self._readout_ce(outs, k, example, reduction=None, reduction_axis=()) for k in readouts]",
                                     "per = [self._readout_ce(outs, k, example, reduction=None, reduction_axis=()) for k in reversed(readouts)]"),
}
env = dict(os.environ)
results = {}
for name, (old, new) in MUTANTS.items():
    assert SRC.count(old) == 1, (name, SRC.count(old))
    d = ROOT / name.split()[0]
    (d / "experiments/references").mkdir(parents=True, exist_ok=True)
    (d / "experiments/__init__.py").write_text("")
    (d / "experiments/references/per_layer_qwen3.py").write_text(SRC.replace(old, new))
    e = dict(env, PYTHONPATH=f"{d}:{env['PYTHONPATH']}")
    p = subprocess.run([sys.executable, str(WT / "scripts/della/pls_cpu_test.py")], env=e, capture_output=True, text=True, timeout=600)
    out = p.stdout + p.stderr
    shadow_ok = str(d) in out or True
    killed = p.returncode != 0 and "ALL PLS CPU TESTS PASSED" not in out
    last = [ln for ln in out.splitlines() if ln.startswith(("1.", "2.", "3.", "4.", "5.", "6.", "AssertionError", "Mismatch", "Not equal"))]
    first_err = next((ln for ln in out.splitlines() if "Error" in ln and "Warning" not in ln), "")
    results[name] = killed
    print(f"{name:40s} {'KILLED' if killed else 'SURVIVED'} rc={p.returncode} passed_sections={[ln[:2] for ln in last if ln[:2] in ('1.', '2.', '3.', '4.', '5.', '6.')]} err={first_err[:140]!r}", flush=True)
print("mutation score", sum(results.values()), "/", len(results))
