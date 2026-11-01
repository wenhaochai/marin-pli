"""Early read of the w711-w714 causal test: the separate-heads cost (sep L48 train CE minus probes L48 train CE, each the
mean of a 50-step window) at matched steps, for the two running pairs and for finished reference pairs.
Finished pairs come from tmp/pls_blog/d48g4.npz (<tag>/step, <tag>/ce[:, 47]); running runs from W&B history of
train/pls/L47 (reself/marin-della). For each reference pair the cost is printed at windows ending at 3500, 4000, 4500,
4077 (where w713 stopped), the running runs' last common step, and the final step 4958, so the drift of the cost over
the last part of training can be read off. Read-only. Run on a vis node."""
import numpy as np, wandb
NPZ = "/scratch/gpfs/GROUP/USER/tmp/pls_blog/d48g4.npz"
P = "reself/marin-della/muonh-qwen3-130m-della4xh100-pls1"
d = dict(np.load(NPZ, allow_pickle=True))
api = wandb.Api(timeout=300)


def from_npz(tag):
    s, ce = np.asarray(d[tag + "/step"]), np.asarray(d[tag + "/ce"])[:, 47]
    return dict(zip(s.tolist(), ce.tolist()))


def from_wandb(tag):
    r = api.run(P + tag)
    last = int(r.summary.get("_step", -1))
    h = {}
    for lo in range(0, last + 1, 1000):
        for x in r.scan_history(keys=["_step", "train/pls/L47"], min_step=lo, max_step=lo + 1000, page_size=2000):
            h[int(x["_step"])] = float(x["train/pls/L47"])
    return h


def window(h, end, n=50):
    v = [h[s] for s in range(end - n + 1, end + 1) if s in h]
    return float(np.mean(v)) if len(v) >= 0.8 * n else float("nan")


NEW = {"mhcidlin": "-sep{}-amhc-mhcid-mhcrwlin-fa4thd-vs-d48", "hcwsh": "-sep{}-ahc-hcwsh-fa4thd-vs-d48"}
REF = {"HC": "-sep{}-ahc-vs-d48-g4", "HC fa4thd": "-sep{}-ahc-fa4thd-vs-d48", "mHC-id": "-sep{}-amhc-mhcid-vs-d48",
       "mHC-lin": "-sep{}-amhc-mhcrwlin-vs-d48", "mHC": "-sep{}-amhc-vs-d48-g4", "Pre-LN": "-sep{}-apreln-vs-d48-g4",
       "Pre-LN fa4thd": "-sep{}-apreln-fa4thd-vs-d48"}
hist = {}
for name, pat in NEW.items():
    for arm in ("", "-bbfrozen"):
        tag = pat.format(arm)
        hist[(name, arm)] = from_npz(tag) if tag + "/ce" in d else from_wandb(tag)
        print(name, arm or "-sep", "steps", min(hist[(name, arm)]), "-", max(hist[(name, arm)]))
for name, pat in REF.items():
    for arm in ("", "-bbfrozen"):
        hist[(name, arm)] = from_npz(pat.format(arm))
common = {name: min(max(hist[(name, "")]), max(hist[(name, "-bbfrozen")])) for name in NEW}
ends = sorted({3500, 4000, 4077, 4500, *common.values(), 4958})
print("cost = sep - probes, L48 train CE, 50-step window ending at step:")
print(f"{'pair':14s}" + "".join(f"{e:>8d}" for e in ends))
for name in [*NEW, *REF]:
    row = [window(hist[(name, "")], e) - window(hist[(name, "-bbfrozen")], e) for e in ends]
    print(f"{name:14s}" + "".join(f"{v:+8.3f}" if v == v else f"{'':>8s}" for v in row))
