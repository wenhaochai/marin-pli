"""The HC identity-mixing pair (wq w810 separate heads, w811 probes; finished 2026-11-02), against finished
references on the same attention stack (fa4thd).

y: layer-48 train cross-entropy (W&B train/pls/L47) minus HC's (ahc-fa4thd-vs-d48) at the same step, each run smoothed
   with a 250-step running mean first. Left: the ordinary model (the probes run, PLS_DETACH_BB=1, whose layer-48 readout
   is the model's own head). Right: the separate-heads run.
   These runs: W&B reself/marin-della muonh-qwen3-130m-della4xh100-pls1-sep[-bbfrozen]-ahc-hcid-fa4thd-vs-d48, read now.
   References: ../../pls_blog/d48g4.npz (HC, mHC with identity mixing + linear read/write, Pre-LN; all fa4thd).
x: training step (4959 steps in all), from step 500.
Criterion (docs pls.md, 2026-11-01): the ordinary model reaching mHC's level (about 2.92, 0.04 below HC) means HC's
learned mixing is why HC's ordinary model is worse; staying at HC's level (2.94-2.98) means it is not.
Output: hcid_gap_live.{pdf,png}.
"""
from pathlib import Path
import numpy as np, wandb
from style import *

HERE = Path(__file__).parent
apply_style()
RED = tone("red", 600)
P = "reself/marin-della/muonh-qwen3-130m-della4xh100-pls1"
BASE = "ahc-fa4thd-vs-d48"
LIVE = ("ahc-hcid-fa4thd-vs-d48", "HC, identity mixing", BLUE, "line")
REF = [("amhc-mhcid-mhcrwlin-fa4thd-vs-d48", "mHC, identity mixing + linear read/write", RED, "dash"),
       ("apreln-fa4thd-vs-d48", "Pre-LN", GREY, "dash")]
W = 250
X0 = 500   # the first 500 steps are left out: the differences there are transients of 0.2-1 nats
api = wandb.Api(timeout=300)
npz = np.load(HERE.parents[1] / "pls_blog" / "d48g4.npz", allow_pickle=True)


def live(tag):
    r = api.run(f"{P}-{tag}")
    by = {}
    for x in r.scan_history(keys=["_step", "train/pls/L47"], page_size=5000):
        by[x["_step"]] = x["train/pls/L47"]   # a resumed run logs some steps twice; keep the last value
    return by


def ref(tag):
    return dict(zip(npz[f"-{tag}/step"].tolist(), npz[f"-{tag}/ce"][:, 47].tolist()))


def smooth(d):
    st = np.array(sorted(d)); v = np.array([d[k] for k in st])
    c = np.convolve(v, np.ones(W) / W, mode="valid")
    return dict(zip(st[W - 1:].tolist(), c))


def minus(a, b):
    a, b = smooth(a), smooth(b)
    st = sorted(set(a) & set(b))
    return np.array(st), np.array([a[k] - b[k] for k in st])


fig, axes = canvas(1, 2, width=WIDTH_POST, panel_height=2.0, title=None,
                   legend=[("HC", LIGHT, "line"), (LIVE[1], LIVE[2], LIVE[3])] + [(lab, col, kind) for _, lab, col, kind in REF],
                   quantity="Layer-48 loss minus HC's (nats)", xlabel="Training step", title_pt=9.2, tick_pt=TEXT_PT)
axs = np.ravel(axes)
curves = []
for ax, setup, name in ((axs[0], "sep-bbfrozen", "Ordinary model"), (axs[1], "sep", "Separate heads")):
    base = ref(f"{setup}-{BASE}")
    ax.axhline(0, color=LIGHT, lw=1.1)
    for tag, lab, col, kind in REF:
        x, y = minus(ref(f"{setup}-{tag}"), base)
        ax.plot(x[x >= X0], y[x >= X0], color=col, lw=1.1, ls=(0, (3, 2)))
        curves.append(y[x >= X0])
        print(f"{name} | {lab}: at step 1000 {y[x == 1000][0]:+.3f}, final (last 50 steps) "
              f"{npz[f'-{setup}-{tag}/final'][47] - npz[f'-{setup}-{BASE}/final'][47]:+.3f}")
    d = live(f"{setup}-{LIVE[0]}")
    x, y = minus(d, base)
    ax.plot(x[x >= X0], y[x >= X0], color=LIVE[2], lw=1.8)
    curves.append(y[x >= X0])
    xs = int(x[-1])
    refs_now = {}
    for tag, lab, _, _ in REF:
        xr, yr = minus(ref(f"{setup}-{tag}"), base)
        refs_now[lab] = float(yr[np.argmin(np.abs(xr - xs))])
    print(f"{name} | {LIVE[1]}: last logged step {max(d)}, smoothed difference at step {xs} {y[-1]:+.3f}; "
          f"references at that step {', '.join(f'{k} {v:+.3f}' for k, v in refs_now.items())}", flush=True)
    ax.set_xlim(0, 5000)
lo = min(float(c.min()) for c in curves if len(c)); hi = max(float(c.max()) for c in curves if len(c))
for ax, name in zip(axs, ("Ordinary model", "Separate heads")):
    nice_y(ax, min(lo, -0.01), max(hi, 0.01), fmt="{:.2f}")
    ax.set_xticks([0, 2500, 5000]); ax.set_xticklabels(["0", "2,500", "5,000"])
    room(ax)
    panel_label(ax, name, x=2500)
save(fig, HERE / "hcid_gap_live")
