"""Mean cosine between the states of different tokens in the same layer (Mufan Li's same-layer feature correlation,
rho_l^{ab}), by layer, for the ten 48-layer designs: the ordinary model (detached probes) against the separate-heads model.
Data: tmp/pls_blog/depthmed/<design>-<ord|sep>.json from depthmed.py (10-19 check-up, final checkpoints): `aniso` = mean
cosine over all pairs of a fixed sample of 2,048 of the 16 x 1,024 Paloma c4_en tokens, uncentred, of the state each
layer's readout reads (stream sum for HC/mHC, the next layer's mix for AttnRes); layer 0 = the embedding ([embed_stats]).
Also prints, over the ten designs, the Spearman correlation of the separate-heads cost (L48 final training loss, sep
minus probes, tmp/pls_blog/d48g4.npz) with each model's mean correlation over layers 1-48.
Output: tmp/pls_corr/token_corr.{pdf,png}. Scratch figure for the owner, 2026-11-01; contains no new runs."""
import json
from pathlib import Path
import numpy as np
from style import *

D = Path("/scratch/gpfs/GROUP/USER/tmp/pls_blog/depthmed")
OUT = Path("/scratch/gpfs/GROUP/USER/tmp/pls_corr")
DES = [("preln", "Pre-LN", "apreln-vs-d48-g4"), ("sandwich", "Sandwich", "vs-d48-g4"), ("lns", "LayerNorm Scaling", "alns-vs-d48-g4"),
       ("deepnorm", "DeepNorm", "adeepnorm-vs-d48-g4"), ("keel", "KEEL", "akeel-vs-d48-g4"), ("hc", "Hyper-Connections", "ahc-vs-d48-g4"),
       ("mhc", "mHC", "amhc-vs-d48-g4"), ("moda", "MoDA", "amoda-vs-d48-g4"), ("attnres_block", "AttnRes (Block)", "aattnres_block-vs-d48-g4"),
       ("attnres", "AttnRes (Full)", "aattnres-vs-d48-g4")]
npz = np.load("/scratch/gpfs/GROUP/USER/tmp/pls_blog/d48g4.npz", allow_pickle=True)
RED = tone("red", 600)


def curve(d, arm):
    j = json.load(open(D / f"{d}-{arm}.json"))
    return np.array([j["embed_stats"]["aniso"]] + [L["aniso"] for L in j["layers"]])


rows = []
for d, name, tag in DES:
    o, s = curve(d, "ord"), curve(d, "sep")
    cost = float(npz[f"-sep-{tag}/final"][47] - npz[f"-sep-bbfrozen-{tag}/final"][47])
    rows.append((name, cost, o[1:].mean(), s[1:].mean(), o[48], s[48], o, s))
    print(f"{name:18s} cost {cost:.3f}  mean corr L1-48 ord {o[1:].mean():.3f} sep {s[1:].mean():.3f}  L48 ord {o[48]:.2f} sep {s[48]:.2f}")


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


c = [r[1] for r in rows]
for i, lab in ((2, "mean ord"), (3, "mean sep"), (4, "L48 ord"), (5, "L48 sep")):
    print(f"Spearman(cost, {lab}) = {spearman(c, [r[i] for r in rows]):+.2f}")
print(f"Spearman(cost, mean sep - mean ord) = {spearman(c, [r[3] - r[2] for r in rows]):+.2f}")

apply_style()
fig, axes = canvas(5, 2, width=WIDTH_POST, panel_height=0.72, title=None,
                   legend=[("Ordinary model", RED, "line"), ("Separate heads", BLUE, "line")],
                   quantity="Mean cosine between token states", xlabel="Layer", title_pt=9.2, tick_pt=TEXT_PT,
                   ticks="left")
x = np.arange(49)
for k, (name, cost, mo, ms, lo, ls, o, s) in enumerate(rows):
    ax = axes[k // 2][k % 2]
    ax.plot(x, o, color=RED, lw=1.1)
    ax.plot(x, s, color=BLUE, lw=1.1)
    ax.set_ylim(0, 1.32); y_values(ax, [0, 0.5, 1.0], "{:.1f}")
    ax.set_xlim(0, 50); ax.set_xticks([0, 12, 24, 36, 48])
    panel_label(ax, name)
save(fig, OUT / "token_corr")
