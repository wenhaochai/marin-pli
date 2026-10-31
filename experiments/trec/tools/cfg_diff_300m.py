"""Config my branch builds for SIZE=300m SEED=1 KEEP_EVERY=572 against the W&B config of muonh-qwen3-300m-della4xh100-s1-rerun."""
import os, json, dataclasses, unittest.mock as mock
os.environ.update(SIZE="300m", SEED="1", KEEP_EVERY="572")
import draccus
import experiments.references.della_muonh_qwen3_scaling as m
tc = m.muonh_qwen3_run("300m").build_config(mock.MagicMock()).train_config
mine = json.loads(json.dumps(draccus.encode(tc), default=str))
ref = json.load(open("/scratch/gpfs/GROUP/USER/project/trec/cache/wandb/muonh-qwen3-300m-della4xh100-s1-rerun.json"))["config"]
def flat(d, p=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items(): out.update(flat(v, f"{p}{k}."))
    else:
        out[p[:-1]] = d
    return out
a, b = flat(mine), flat(ref)
for k in sorted(set(a) | set(b)):
    if k.split(".")[0] in ("optimizer", "model", "trainer", "data", "data_seed") and a.get(k) != b.get(k):
        if any(s in k for s in ("tracker", "checkpointer", "id", "cache_dir", "components", "load_checkpoint_path", "wandb", "output", "tags", "name", "ray")): continue
        print(f"{k}: mine={a.get(k)!r} ref={b.get(k)!r}")
