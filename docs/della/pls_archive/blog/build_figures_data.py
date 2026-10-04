"""Figures of wenhaochai.com/blogs/per-layer-contribution.html -> assets/data/plc-figures.js (generated; regenerate,
never hand-edit). Drawn on the page in the writing:plot (Epoch AI) style.

x axis everywhere: backbone compute 6ND. N = parameters of the transformer layers up to the one read out (the embedding
and the output heads not counted): per Qwen3 layer 4h^2 (q, k, v, o; 8 heads of 64 at 130m, so kv = h) + 3*h*inter
(SwiGLU) + 4h (four RMSNorms: pre and post of both sublayers, the baseline's hybrid norm) + 2*64 (q and k norms). D = training tokens = step * 128 * 4096.
  fig-modes    final-layer training loss over training, four modes (baseline, shared head, shared head with stop-grad
               into the head, separate heads); 50-step running means from step 100, faint, with the final points
  fig-traj     separate heads: training curves of layers 2..6 (faint, light to dark) and their final points, with the
               ordinary models of 2..6 layers (-d2..-d5 and the 6-layer baseline)
  fig-heads    final training loss by layer under the four setups (separate heads, shared head, shared head stop-grad,
               probes = separate heads on a detached backbone) and the ordinary models
Curves through final points: L = E + A exp(-(C/1e18)/u0), fitted freely where it holds (largest residual <= FIT_TOL = 0.005
nats), else a monotone PCHIP in log-log (the probes always). Of five forms tried on every fitted set, this one holds on all
with three parameters (largest residual 0.0039 nats, on 300m separate heads); the power law E + A (C/1e18)^-alpha left
0.021 there with an S-shaped residual pattern. The page does not label either curve as a fit. Final value = mean of the last 50 logged steps (fig-heads), last point of the running mean (fig-traj).
Placeholders (axes, legend and a status word, no data until the runs finish):
  fig-traj-300m                   fig-traj at 300m (12 layers, width 768, 6.0B tokens), with the 300m baseline's final loss
                                  as a horizontal line
  fig-d48                         48 layers of width 512 (the 130m layer): separate heads and probes, final loss by layer
  fig-d48-arch                    fig-d48 under the nine DepthBench architectures (arXiv 2609.32534) other than Sandwich-LN,
                                  which is the baseline's own design (hybrid norm) and so fig-d48 itself: separate heads against probes
Sources: flops.npz, layers.npz, bbfrozen.npz, depth.json, baselines.json (this directory)."""
import json, sys
from pathlib import Path
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import curve_fit
HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1])
TOK = 128 * 4096
D = dict(np.load(HERE / "flops.npz")); Z = np.load(HERE / "bbfrozen.npz")
B = json.load(open(HERE / "baselines.json")); DEPTH = json.load(open(HERE / "depth.json"))
_raw = np.load(HERE / "layers.npz", allow_pickle=True)
LY = {k: np.array([[np.nan if v is None else v for v in r] for r in _raw[k]], float) if _raw[k].ndim == 2 else np.array([np.nan if v is None else v for v in _raw[k]], float) for k in _raw.files}
h, inter = 512, 1792
NL = 4 * h * h + 3 * h * inter + 4 * h + 2 * 64          # parameters of one 130m layer
STEPS = 4959; DTOT = STEPS * TOK
BLUE7, RED, GREEN, PURPLE, GREY = "#1967D2", "#D93025", "#1E8E3E", "#9334E6", "#80868B"
def ramp(t):
    a = np.array([0xAE, 0xCB, 0xFA]); b = np.array([0x17, 0x4E, 0xA6]); return "#%02X%02X%02X" % tuple((a + (b - a) * t).round().astype(int))
SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
law = lambda c, E, A, u0: E + A * np.exp(-(c / 1e18) / u0)   # saturating in compute; holds on every fitted set (see the docstring)
def pts(x, y): return [[float(f"{a:.5g}"), round(float(b), 4)] for a, b in zip(x, y)]
FIT_TOL = 0.005   # nats: draw the fitted law only where it holds; above this largest residual, a monotone PCHIP instead
def fit(x, y):
    p, _ = curve_fit(law, x, y, p0=(y.min() - 0.02, 3 * (y.max() - y.min()), float(np.mean(x)) / 1e18), bounds=([0, 0, 1e-6], [y.min(), 100, 1e3]), maxfev=200000)
    if np.abs(y - law(x, *p)).max() > FIT_TOL:
        print(f"fit does not hold (max residual {np.abs(y - law(x, *p)).max():.4f} > {FIT_TOL}): PCHIP through the points")
        return pchip(x, y)
    g = np.geomspace(x.min(), x.max(), 60); return pts(g, law(g, *p))
def pchip(x, y):
    f = PchipInterpolator(np.log(x), np.log(y)); g = np.geomspace(x.min(), x.max(), 60); return pts(g, np.exp(f(np.log(g))))
def smooth(x, y, w=50): return x[w - 1:], np.convolve(y, np.ones(w) / w, mode="valid")
def thin(x, y, n=200):
    idx = np.unique(np.geomspace(1, len(x), n).astype(int) - 1); return pts(np.asarray(x)[idx], np.asarray(y)[idx])
def xticks(lo, hi):
    t = [[v, "10" + str(int(np.log10(v))).translate(SUP)] for v in (1e15, 1e16, 1e17, 1e18, 1e19) if lo <= v <= hi]
    if len(t) < 2:   # add 2x and 5x marks when the view spans less than a decade
        for e in (15, 16, 17, 18):
            for m in (2, 5):
                v = m * 10 ** e
                if lo <= v <= hi: t.append([v, f"{m}×10" + str(e).translate(SUP)])
    return sorted(t)
YOFF = 2.0
def view(ex, ey, x_lo_div=4, x_hi_mul=2, top=1.15, endpoints_only=False):
    lo, hi = min(ex) / x_lo_div, max(ex) * x_hi_mul; ylo, yhi = 0.97 * min(ey), max(ey) * top
    if endpoints_only:   # no trajectories to show: start at the power of ten just below the smallest point
        lo, hi = 10 ** np.floor(np.log10(min(ex))), max(ex) * 1.3
    lo, hi = float(lo), float(hi)
    # y: log of (loss - 2), so close losses spread apart; the axis keeps the raw loss values and does not reach 2
    return dict(xlog=True, ylog=True, yoff=YOFF, xlim=[lo, hi], ylim=[ylo, yhi], xticks=xticks(lo, hi),
                yticks=[[a, f"{a:g}"] for a in (3, 3.2, 3.4, 3.6, 3.8, 4, 4.5, 5) if ylo <= a <= yhi - 0.04 * (yhi - ylo)])
def line(p, c, w=1.6, a=1.0): return dict(type="line", color=c, width=w, alpha=a, pts=p)
def dots(p, c): return dict(type="dots", color=c, pts=p)
XL = {"en": "Backbone compute 6ND (FLOPs)", "zh": "骨干算力 6ND（FLOPs）"}
CE = {"en": "Cross-entropy (nats, log scale)", "zh": "交叉熵（nats，对数坐标）"}
figs = {}

# Figure 1: final-layer training loss over training, four modes; x = 6 * N(6 layers) * tokens so far
MODES = [("base", None, {"en": "Baseline", "zh": "基线"}, GREY), ("-pls1", 5, {"en": "Shared head", "zh": "共享头"}, GREEN),
         ("-pls1-dh", 5, {"en": "Shared head, stop-grad", "zh": "共享头 stop-grad"}, PURPLE), ("-pls1-sep", 5, {"en": "Separate heads", "zh": "独立头"}, BLUE7)]
marks, ends = [], []
for key, col, name, c in MODES:
    st, y = (LY["base/step"], LY["base/ce"]) if col is None else (LY[key + "/step"], LY[key + "/ce"][:, col])
    ok = np.isfinite(y) & (st >= 100); xs, ys = smooth(6 * NL * 6 * st[ok] * TOK, y[ok])
    marks.append(line(thin(xs, ys), c, 1.4, 0.85)); ends.append((xs[-1], ys[-1], c))
marks += [dots([[x, y]], c) for x, y, c in ends]
v = view([x for x, _, _ in ends], [y for _, y, _ in ends], x_hi_mul=1.25)
figs["fig-modes"] = dict(title={"en": "Final-layer training loss by gradient and parameter mode, 130m", "zh": "不同梯度与参数模式下的最终层训练损失，130m"},
                         legend=[[n, c, "line"] for _, _, n, c in MODES], quantity=CE, xlabel=XL, height=300, panels=[dict(**v, marks=marks)])
print("modes finals", [round(y, 4) for _, y, _ in ends])

# Figure 2: separate heads, training curves of layers 2..6 against 6 * N(k layers) * tokens, with the ordinary models
s, ce = D["130-own/step"], D["130-own/ce"]; m = s >= 100
marks, ends = [], []
for k in range(1, 6):                     # readout after layer k+1, numbered from 1
    xs, y = smooth((6 * NL * (k + 1) * s * TOK)[m], ce[m, k])
    marks.append(line(thin(xs, y), ramp((k - 1) / 4), 1.2, 0.45)); ends.append((xs[-1], y[-1]))
ex, ey = map(np.array, zip(*ends))
cx = np.array([6.0 * NL * d * DTOT for d in range(2, 7)])
cy = np.array([DEPTH[str(d)]["loss"] for d in range(2, 6)] + [B["130m"]["loss"]])
marks += [line(fit(ex, ey), BLUE7), dots(pts(ex, ey), BLUE7), line(fit(cx, cy), GREY), dots(pts(cx, cy), GREY)]
v = view(np.r_[ex, cx], np.r_[ey, cy])
figs["fig-traj"] = dict(title={"en": "Per-layer training loss, separate heads, 130m", "zh": "独立头的逐层训练损失，130m"},
                        legend=[[{"en": "Separate heads, final loss by layer", "zh": "独立头，各层最终损失"}, BLUE7, "line"], [{"en": "Baseline by depth", "zh": "各深度的基线"}, GREY, "line"]],
                        quantity=CE, xlabel=XL, height=300, panels=[dict(**v, label={"en": "Separate heads", "zh": "独立头"}, marks=marks)])

# Figure 3: final loss by layer, four setups and the ordinary models; x = 6 * N(k layers) * D
x6 = np.array([6.0 * NL * d * DTOT for d in range(2, 7)])
SETS = [(Z["-pls1-sep/final"][1:], {"en": "Separate heads", "zh": "独立头"}, BLUE7, fit), (Z["-pls1/final"][1:], {"en": "Shared head", "zh": "共享头"}, GREEN, fit),
        (Z["-pls1-dh/final"][1:], {"en": "Shared head, stop-grad", "zh": "共享头 stop-grad"}, PURPLE, fit),
        (Z["-pls1-sep-bbfrozen/final"][1:], {"en": "Probes only", "zh": "只加探针"}, RED, pchip), (cy, {"en": "Baseline by depth", "zh": "各深度的基线"}, GREY, fit)]
marks = sum(([line(f(x6, y), c), dots(pts(x6, y), c)] for y, _, c, f in SETS), [])
v = view(np.tile(x6, len(SETS)), np.concatenate([y for y, _, _, _ in SETS]), endpoints_only=True)
figs["fig-heads"] = dict(title={"en": "Final training loss by layer and setup, 130m", "zh": "按层和做法的最终训练损失，130m"},
                         legend=[[n, c, "line"] for _, n, c, _ in SETS], quantity=CE, xlabel=XL, height=320, panels=[dict(**v, marks=marks)])

# Placeholders for runs in progress or planned: axes from the planned shapes, no marks
def nl(h, inter): return 4 * h * h + 3 * h * inter + 4 * h + 2 * 64
def empty_view(xs, ylo, yhi, word, ys=(3, 3.2, 3.4, 3.6, 3.8, 4, 4.5, 5)):
    lo, hi = 10 ** np.floor(np.log10(min(xs))), max(xs) * 1.3
    return dict(xlog=True, ylog=True, yoff=YOFF, xlim=[float(lo), float(hi)], ylim=[ylo, yhi], xticks=xticks(lo, hi),
                yticks=[[a, f"{a:g}"] for a in ys if ylo <= a <= yhi - 0.04 * (yhi - ylo)], marks=[], empty=word)
RUNNING, QUEUED, PLANNED = {"en": "Running", "zh": "运行中"}, {"en": "Queued", "zh": "排队中"}, {"en": "Planned", "zh": "计划中"}
N300, D300 = nl(768, 2688), 11444 * TOK
x300 = [6.0 * N300 * d * D300 for d in range(2, 13)]
SEP300 = HERE / "sep300.npz"   # fetch_sep300.py, once job 14940604 has finished
legend300 = figs["fig-traj"]["legend"][:1]   # no 300m ordinary models by depth (owner)
if SEP300.exists() and str(np.load(SEP300)["state"]) == "finished":
    z = np.load(SEP300); s3, c3 = z["step"], z["ce"]; m3 = s3 >= 100
    marks, ends = [], []
    for k in range(1, 12):                 # readout after layer k+1: layers 2..12, as at 130m
        xs, y = smooth((6 * N300 * (k + 1) * s3 * TOK)[m3], c3[m3, k])
        marks.append(line(thin(xs, y), ramp((k - 1) / 10), 1.2, 0.45)); ends.append((xs[-1], y[-1]))
    ex3, ey3 = map(np.array, zip(*ends))
    marks += [line(fit(ex3, ey3), BLUE7), dots(pts(ex3, ey3), BLUE7)]
    b300 = B["300m"]["loss"]   # the 300m baseline's final training loss, a horizontal reference (owner)
    v3 = view(ex3, np.r_[ey3, b300])
    marks.insert(0, line([[v3["xlim"][0], b300], [v3["xlim"][1], b300]], GREY))
    p300 = dict(**v3, label={"en": "Separate heads", "zh": "独立头"}, marks=marks)
    legend300 = legend300 + [[{"en": "Baseline", "zh": "基线"}, GREY, "line"]]
else:
    p300 = dict(**empty_view([x / 4 for x in x300] + x300, 2.9, 4.4, RUNNING), label={"en": "Separate heads", "zh": "独立头"})
figs["fig-traj-300m"] = dict(title={"en": "Per-layer training loss, separate heads, 300m", "zh": "独立头的逐层训练损失，300m"},
                             legend=legend300, quantity=CE, xlabel=XL, height=300, panels=[p300])
x48 = [6.0 * NL * d * DTOT for d in range(2, 49)]
figs["fig-d48"] = dict(title={"en": "Final training loss by layer, 48 layers of width 512", "zh": "按层的最终训练损失，48 层、宽 512 的网络"},
                       legend=[[{"en": "Separate heads", "zh": "独立头"}, BLUE7, "line"], [{"en": "Probes only", "zh": "只加探针"}, RED, "line"]],
                       quantity=CE, xlabel=XL, height=320, panels=[empty_view(x48, 2.9, 5.0, QUEUED)])
ARCH = ["Pre-LN", "LayerNorm Scaling", "DeepNorm", "KEEL", "Hyper-Connections", "mHC", "AttnRes (Full)", "AttnRes (Block)", "MoDA"]
figs["fig-d48-arch"] = dict(title={"en": "Final training loss by layer and architecture, 48 layers of width 512", "zh": "按层和架构的最终训练损失，48 层、宽 512 的网络"},
                            legend=figs["fig-d48"]["legend"],
                            quantity=CE, xlabel=XL, height=150, cols=3,
                            panels=[dict(**empty_view(x48, 2.9, 5.0, PLANNED, ys=(3, 3.5, 4)), label={"en": a, "zh": a}) for a in ARCH])

# Figure 7 (placeholder until the runs finish): the 2 x 2 of layer-loss weight (1, 0.2) and gradient reach (all layers, own layer)
YELLOW9, CYAN9, PINK7 = "#E37400", "#007B83", "#C2185B"
figs["fig-fix"] = dict(title={"en": "Final training loss by layer, gradient routing and layer-loss weight, 130m", "zh": "按梯度去向和逐层损失权重的各层最终训练损失，130m"},
                       legend=[[{"en": "Separate heads", "zh": "独立头"}, BLUE7, "line"], [{"en": "Layer losses summing to 1", "zh": "各层损失权重合计为 1"}, YELLOW9, "line"],
                               [{"en": "Each head trains its own layer", "zh": "每个头只训练自己那一层"}, CYAN9, "line"], [{"en": "Both", "zh": "两者都用"}, PINK7, "line"],
                               [{"en": "Probes only", "zh": "只加探针"}, RED, "line"], [{"en": "Baseline by depth", "zh": "各深度的基线"}, GREY, "line"]],
                       quantity=CE, xlabel=XL, height=320, panels=[empty_view(list(x6), 2.9, 5.0, PLANNED)])

OUT.write_text("/* Generated by build_figures_data.py (pls branch, docs/della/pls_archive/blog); regenerate rather than hand-edit. */\nwindow.PLC_FIGS = " + json.dumps(figs, ensure_ascii=False, separators=(",", ":")) + ";\n")
print("N per layer", NL, "| x of layer 2..6 at the end", [f"{v:.3g}" for v in x6], "| wrote", OUT, OUT.stat().st_size, "bytes")
