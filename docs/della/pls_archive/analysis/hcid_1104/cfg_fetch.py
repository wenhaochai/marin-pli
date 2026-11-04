"""Save the W&B config of a run to tmp/pls_paths/cfg<tag>.json, the form the analysis scripts read. Usage: cfg_fetch.py <tag after -pls1> ..."""
import json, sys, wandb
api = wandb.Api(timeout=300)
for tag in sys.argv[1:]:
    r = api.run("reself/marin-della/muonh-qwen3-130m-della4xh100-pls1" + tag)
    json.dump(dict(r.config), open(f"/scratch/gpfs/GROUP/USER/tmp/pls_paths/cfg{tag}.json", "w"))
    print(tag, "saved", list(dict(r.config))[:6], flush=True)
