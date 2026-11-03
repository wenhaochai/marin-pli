"""W&B state, last step and depth_hc_res of the ten runs in training."""
import wandb
api = wandb.Api(timeout=300)
P = "reself/marin-della/muonh-qwen3-130m-della4xh100-pls1-"
T = {"w888": "sep-ahc-hcrwj-fa4thd-vs-d48", "w889": "sep-bbfrozen-ahc-hcrwj-fa4thd-vs-d48", "w890": "sep-bbfrozen-ahc-fa4thd-vs-d48-s1",
     "w891": "sep-bbfrozen-ahc-hcid-fa4thd-vs-d48-s1", "w892": "sep-bbfrozen-amhc-mhcid-mhcrwlin-fa4thd-vs-d48-s1", "w893": "sep-ahc-hcid-fa4thd-vs-d48-s1",
     "w959": "sep-bbfrozen-ahc-hcsum-fa4thd-vs-d48", "w960": "sep-bbfrozen-ahc-hcscalar-fa4thd-vs-d48", "w961": "sep-ahc-hcsum-fa4thd-vs-d48", "w962": "sep-ahc-hcscalar-fa4thd-vs-d48"}
def find(c, k):
    if isinstance(c, dict):
        for a, b in c.items():
            if a == k: return b
            v = find(b, k)
            if v is not None: return v
    return None
for w, t in T.items():
    try:
        r = api.run(P + t)
    except Exception as e:
        print(w, t, "no run:", str(e)[:70]); continue
    print(w, t, "state", r.state, "step", r.summary.get("_step"), "depth_hc_res", find(dict(r.config), "depth_hc_res"), "rwdyn", find(dict(r.config), "depth_hc_rwdyn"), "loss", r.summary.get("train/loss"), flush=True)
