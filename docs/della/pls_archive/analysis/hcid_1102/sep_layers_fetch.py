"""Raw W&B rows behind sep_layers.py: per-layer train CE, grad norm and routing-parameter norms of w810 (HC mixing = I,
separate heads) and its references, at the rows that carry grad norms (every 10 steps). W&B reself/marin-della. A cache
is re-read only when the run's last step moved, so a finished run is read once more after its last live read. Output: cache/<tag>.npz."""
import sys, numpy as np, wandb
from pathlib import Path
HERE = Path(__file__).parent; (HERE / "cache").mkdir(exist_ok=True)
P = "reself/marin-della/muonh-qwen3-130m-della4xh100-pls1-"
TAGS = ["sep-ahc-hcid-fa4thd-vs-d48", "sep-ahc-fa4thd-vs-d48", "sep-amhc-mhcid-mhcrwlin-fa4thd-vs-d48",
        "sep-bbfrozen-ahc-hcid-fa4thd-vs-d48", "sep-bbfrozen-ahc-fa4thd-vs-d48"]
api = wandb.Api(timeout=300)
for tag in TAGS:
    r = api.run(P + tag); last = int(r.summary.get("_step", -1)); f = HERE / "cache" / f"{tag}.npz"
    if f.exists() and int(np.load(f)["last"]) == last:   # a cache written while the run was live has a smaller last
        print(tag, "cached", last, flush=True); continue
    arch_mhc = "amhc" in tag
    hyper = ["phi", "bias", "gain"] if arch_mhc else ["read", "write", "scale", "w_read", "w_write", "res", "w_res"]
    K = ["train/loss", "grad/norm/total", "optim/learning_rate", "params/norm/total"] + [f"train/pls/L{k}" for k in range(48)]
    for k in range(48):
        for j in (0, 1):
            K += [f"{g}/norm/transformer.layers.{k}.hyper.{j}.{h}" for g in ("grad", "params") for h in hyper]
        K += [f"{g}/norm/transformer.layers.{k}.{w}.weight" for g in ("grad", "params") for w in ("mlp.down_proj", "self_attn.o_proj", "ln_1", "ln_2")]
    K += [f"grad/norm/aux_lm_heads.{k}.weight" for k in range(47)] + [f"grad/norm/aux_norms.{k}.weight" for k in range(47)]
    by = {}
    for lo in range(0, last + 1, 500):
        for x in r.scan_history(keys=["_step", *K], min_step=lo, max_step=lo + 500, page_size=200):
            by[x["_step"]] = [x.get(k, np.nan) for k in K]
        print(tag, lo, len(by), flush=True)
    st = np.array(sorted(by)); v = np.array([by[s] for s in st], float)
    np.savez(f, step=st, keys=np.array(K), v=v, last=last)
    print(tag, "wrote", v.shape, "state", r.state, flush=True)
