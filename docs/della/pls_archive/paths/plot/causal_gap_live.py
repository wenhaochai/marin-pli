"""Separate-heads cost so far for the two causal-test pairs (wq w711-w714, still training), against finished references (HC: the fa4thd run on the same attention stack).

y: layer-48 train cross-entropy (W&B train/pls/L47) of the separate-heads run minus its probes run (PLS_DETACH_BB=1),
   each smoothed with a 250-step running mean, at the steps both runs have logged. Live runs: W&B reself/marin-della
   muonh-qwen3-130m-della4xh100-pls1-sep[-bbfrozen]-<tag>, read now; references from ../../pls_blog/d48g4.npz.
x: training step (4959 steps in all).
Output: causal_gap_live.{pdf,png}.
"""
import sys
from pathlib import Path
import numpy as np, wandb
from style import *

HERE = Path(__file__).parent
apply_style()
RED, GREEN = tone("red", 600), tone("green", 600)
P = "reself/marin-della/muonh-qwen3-130m-della4xh100-pls1"
LIVE = [("amhc-mhcid-mhcrwlin-fa4thd-vs-d48", "mHC, identity mixing + linear read/write", RED, "line"),
        ("ahc-hcwsh-fa4thd-vs-d48", "HC, one shared write", BLUE, "line")]
REF = [("ahc-fa4thd-vs-d48", "HC", BLUE, "dash"), ("amhc-mhcid-vs-d48", "mHC, identity mixing", RED, "dash"),
       ("ahc-hcs1-vs-d48", "HC, 1 stream", GREY, "dash")]
W = 250
api = wandb.Api(timeout=300)


def live(tag):
    r = api.run(f"{P}-{tag}")
    s, v = [], []
    for x in r.scan_history(keys=["_step", "train/pls/L47"], page_size=5000):
        s.append(x["_step"]); v.append(x["train/pls/L47"])
    return dict(zip(s, v))


def smooth(d):
    st = np.array(sorted(d)); v = np.array([d[k] for k in st])
    c = np.convolve(v, np.ones(W) / W, mode="valid")
    return dict(zip(st[W - 1:], c))


def gap(a, b):
    a, b = smooth(a), smooth(b)
    st = sorted(set(a) & set(b))
    return np.array(st), np.array([a[k] - b[k] for k in st])


npz = np.load(HERE.parents[1] / "pls_blog" / "d48g4.npz", allow_pickle=True)
fig, axes = canvas(1, 1, width=WIDTH_POST, panel_height=2.6, title=None,
                   legend=[(lab, col, kind) for _, lab, col, kind in LIVE + REF],
                   quantity="Separate-heads cost (nats)", xlabel="Training step", title_pt=9.2, tick_pt=TEXT_PT)
ax = axes[0][0] if hasattr(axes[0], "__len__") else axes[0]
for tag, lab, col, kind in REF:
    ce = {f"-{k}": npz[f"-{k}-{tag}/ce"][:, 47] for k in ("sep", "sep-bbfrozen")}
    st = npz[f"-sep-{tag}/step"]
    x, y = gap(dict(zip(st, ce["-sep"])), dict(zip(npz[f"-sep-bbfrozen-{tag}/step"], ce["-sep-bbfrozen"])))
    ax.plot(x, y, color=col, lw=1.1, ls=(0, (3, 2)), alpha=0.8)
for tag, lab, col, kind in LIVE:
    x, y = gap(live(f"sep-{tag}"), live(f"sep-bbfrozen-{tag}"))
    ax.plot(x, y, color=col, lw=1.8)
    print(f"{lab}: last common step {x[-1]}, smoothed gap {y[-1]:+.3f}", flush=True)
ax.set_xlim(0, 5000)
nice_y(ax, -0.01, 0.14, fmt="{:.2f}")
room(ax)
save(fig, HERE / "causal_gap_live")
