"""300m separate heads (job 14940604, W&B reself/marin-della muonh-qwen3-300m-della4xh100-pls1-sep), in progress: per-layer
training loss (train/pls/L{k}, 50-step running mean) against training tokens, layers 1..12 light to dark; the 300m baseline's
train/loss in grey. Output fig_300m_sep_progress.{pdf,png}."""
from pathlib import Path
import numpy as np, wandb
from style import *
HERE = Path(__file__).resolve().parent
api = wandb.Api(timeout=300)
P = "reself/marin-della/muonh-qwen3-300m-della4xh100"
keys = [f"train/pls/L{k}" for k in range(12)]
rows = [x for x in api.run(P + "-pls1-sep").scan_history(keys=["_step", *keys], page_size=5000) if all(x.get(k) is not None for k in keys)]
s = np.array([x["_step"] for x in rows]); ce = np.array([[x[k] for k in keys] for x in rows])
b = [(x["_step"], x["train/loss"]) for x in api.run(P).scan_history(keys=["_step", "train/loss"], page_size=5000)]
bs, bl = map(np.array, zip(*b)); TOK = 128 * 4096
def sm(x, y, w=50): return x[w - 1:], np.convolve(y, np.ones(w) / w, mode="valid")
apply_style()
fam = [tone("blue", g) for g in (200, 300, 400, 500, 600, 700, 800, 900)]
cols = [fam[min(7, int(round(i * 7 / 11)))] for i in range(12)]
fig, axes = canvas(1, 1, width=WIDTH_POST, panel_height=2.3, title="Per-layer training loss, separate heads, 300m",
                   legend=[("Layers 1 to 12, light to dark", BLUE), ("Baseline", GREY)], quantity="Cross-entropy (nats) by training tokens (billions)", title_pt=9.2, tick_pt=7.35, side=0.10)
ax = axes[0][0] if hasattr(axes[0], "__len__") else axes[0]
m = s >= 200
for k in range(12):
    x, y = sm(s[m] * TOK / 1e9, ce[m, k]); ax.plot(x, y, color=cols[k], lw=1.1)
mb = (bs >= 200) & (bs <= s.max())
x, y = sm(bs[mb] * TOK / 1e9, bl[mb]); ax.plot(x, y, color=GREY, lw=1.4)

nice_y(ax, 3.0, 4.6, headroom=0.1, fmt='{:.1f}'); room(ax)
save(fig, HERE / "fig_300m_sep_progress")
print("step", s.max(), "of 11443; last L1..L12", np.round(ce[-50:].mean(0), 3), "baseline same steps", round(float(np.mean([v for t, v in b if s.max() - 50 < t <= s.max()])), 4))
