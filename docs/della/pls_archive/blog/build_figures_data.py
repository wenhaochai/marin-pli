"""Figures of wenhaochai.com/blogs/per-layer-contribution.html -> assets/data/plc-figures.js (generated; regenerate,
never hand-edit). Drawn on the page in the writing:plot (Epoch AI) style.

x axis everywhere: backbone compute 6ND. N = parameters of the transformer layers up to the one read out (the embedding
and the output heads not counted): per Qwen3 layer 4h^2 (q, k, v, o; 8 heads of 64 at 130m, so kv = h) + 3*h*inter
(SwiGLU) + 4h (four RMSNorms: pre and post of both sublayers, the baseline's hybrid norm) + 2*64 (q and k norms). D = training tokens = step * 128 * 4096.
  fig-modes    final-layer training loss over training, four modes (baseline, shared head, shared head with stop-grad
               into the head, separate heads); 50-step running means from step 100, faint, with the final points
  fig-traj     separate heads: training curves of layers 1..6 (faint, light to dark) and their final points, with the
               ordinary models of 1..6 layers (-d1..-d5 and the 6-layer baseline)
  fig-heads    final training loss by layer under the four setups (separate heads, shared head, shared head stop-grad,
               probes = separate heads on a detached backbone) and the ordinary models
  fig-fix      separate heads at 130m, the 2 x 2 of layer-loss weight (1, 0.2) and gradient reach (every layer below, own
               layer only), final training loss by layer, with probes and the ordinary models (fix.npz)
Curves through final points: L = E + A exp(-((C/1e18)/u0)^b), a stretched exponential, fitted freely where it holds
(largest residual <= FIT_TOL = 0.015 nats; owner, 2026-10-04. The seed-to-seed spread of a final loss is smaller, sd 0.001 over nine 130m baseline runs, audit 2026-10-11), else a monotone PCHIP in log-log (the
probes always). With layer 1 on the figures, of five forms tried on every fitted set this one fits best (worst 0.010 nats,
300m separate heads); the 3-parameter exponential and power law leave 0.03-0.04 at 300m, since layer 1 (the embedding plus
one layer) sits well above the smooth run of the later layers. The page does not label any curve as a fit. Final value = mean of the last 50 logged steps (fig-heads), last point of the running mean (fig-traj).
Placeholders (axes, legend and a status word, no data until the runs finish):
  fig-heads-300m                  fig-heads at 300m: separate heads, shared head, probes; the 300m baseline as a line (s300.npz)
  fig-traj-300m                   fig-traj at 300m (12 layers, width 768, 6.0B tokens), with the 300m baseline's final loss
                                  as a horizontal line
  fig-d48                         48 layers of width 512 (the 130m layer): separate heads and probes, final loss by layer
  fig-mix                         130m separate heads: layer-loss weights uniform (0.2) or rising with depth (k/15), each with and
                                  without per-token gradient surgery at the layer outputs; separate heads at weight 1 and the
                                  ordinary models as references
  fig-headft (filled, headft.json) 300m heads retrained on the frozen backbone of each finished run (separate heads, probes):
                                  held-out loss by layer before and after
  fig-d48-arch                    fig-d48 under the nine DepthBench architectures (arXiv 2609.32534) other than Sandwich-LN,
                                  which is the baseline's own design (hybrid norm) and so fig-d48 itself: separate heads against probes
Sources: flops.npz, layers.npz, bbfrozen.npz, depth.json, baselines.json, fix.npz, headft.json (this directory)."""
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
law = lambda c, E, A, u0, b: E + A * np.exp(-((c / 1e18) / u0) ** b)   # stretched exponential: holds on every fitted set (see the docstring)
def pts(x, y): return [[float(f"{a:.5g}"), round(float(b), 4)] for a, b in zip(x, y)]
FIT_TOL = 0.015   # nats (owner, 2026-10-04; seed spread of a final loss is ~0.001): the fitted law where it holds, else a monotone PCHIP
def fit(x, y):
    best = None   # several starts; keep the smallest largest-residual fit
    for e0 in (0.02, 0.1, 0.3):
        for b0 in (0.5, 1.0, 2.0):
            try:
                q, _ = curve_fit(law, x, y, p0=(y.min() - e0, 3 * (y.max() - y.min()), float(np.mean(x)) / 1e18, b0), bounds=([0, 0, 1e-6, 0.05], [y.min(), 1e3, 1e4, 10]), maxfev=400000)
            except RuntimeError:
                continue
            r = np.abs(y - law(x, *q)).max()
            if best is None or r < best[0]:
                best = (r, q)
    p = best[1]
    if np.abs(y - law(x, *p)).max() > FIT_TOL:
        print(f"fit does not hold (max residual {np.abs(y - law(x, *p)).max():.4f} > {FIT_TOL}) on {len(x)} points, y {np.round(y[:3], 3)}...: PCHIP through the points")
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
        e = np.floor(np.log10(min(ex))); lo = max(m * 10 ** e for m in (1, 2, 5) if m * 10 ** e <= min(ex) / 1.1); hi = max(ex) * 1.3   # the 1-2-5 value just below the data
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

# Figure 2: separate heads, training curves of layers 1..6 against 6 * N(k layers) * tokens, with the ordinary models
s, ce = D["130-own/step"], D["130-own/ce"]; m = s >= 100
marks, ends = [], []
for k in range(0, 6):                     # readout after layer k+1, numbered from 1
    xs, y = smooth((6 * NL * (k + 1) * s * TOK)[m], ce[m, k])
    marks.append(line(thin(xs, y), ramp(k / 5), 1.2, 0.45)); ends.append((xs[-1], y[-1]))
ex, ey = map(np.array, zip(*ends))
cx = np.array([6.0 * NL * d * DTOT for d in range(1, 7)])
cy = np.array([DEPTH[str(d)]["loss"] for d in range(1, 6)] + [B["130m"]["loss"]])
marks += [line(fit(ex, ey), BLUE7), dots(pts(ex, ey), BLUE7), line(fit(cx, cy), GREY), dots(pts(cx, cy), GREY)]
v = view(np.r_[ex, cx], np.r_[ey, cy])
figs["fig-traj"] = dict(title={"en": "Per-layer training loss, separate heads, 130m", "zh": "独立头的逐层训练损失，130m"},
                        legend=[[{"en": "Separate heads, final loss by layer", "zh": "独立头，各层最终损失"}, BLUE7, "line"], [{"en": "Baseline by depth", "zh": "各深度的基线"}, GREY, "line"]],
                        quantity=CE, xlabel=XL, height=300, panels=[dict(**v, label={"en": "Separate heads", "zh": "独立头"}, marks=marks)])

# Figure 3: final loss by layer, four setups and the ordinary models; x = 6 * N(k layers) * D
x6 = np.array([6.0 * NL * d * DTOT for d in range(1, 7)])
SETS = [(Z["-pls1-sep/final"], {"en": "Separate heads", "zh": "独立头"}, BLUE7, fit), (Z["-pls1/final"], {"en": "Shared head", "zh": "共享头"}, GREEN, fit),
        (Z["-pls1-dh/final"], {"en": "Shared head, stop-grad", "zh": "共享头 stop-grad"}, PURPLE, fit),
        (Z["-pls1-sep-bbfrozen/final"], {"en": "Probes only", "zh": "只加探针"}, RED, pchip), (cy, {"en": "Baseline by depth", "zh": "各深度的基线"}, GREY, fit)]
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
x300 = [6.0 * N300 * d * D300 for d in range(1, 13)]
SEP300 = HERE / "sep300.npz"   # fetch_sep300.py, once job 14940604 has finished
legend300 = figs["fig-traj"]["legend"][:1]   # no 300m ordinary models by depth (owner)
if SEP300.exists() and str(np.load(SEP300)["state"]) == "finished":
    z = np.load(SEP300); s3, c3 = z["step"], z["ce"]; m3 = s3 >= 100
    marks, ends = [], []
    for k in range(0, 12):                 # readout after layer k+1: layers 1..12, as at 130m
        xs, y = smooth((6 * N300 * (k + 1) * s3 * TOK)[m3], c3[m3, k])
        marks.append(line(thin(xs, y), ramp(k / 11), 1.2, 0.45)); ends.append((xs[-1], y[-1]))
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
# Figure 5: final loss by layer and setup at 300m (separate heads, shared head, probes) with the 300m baseline as a line
S3 = np.load(HERE / "s300.npz")
x3 = np.array(x300)
sets3 = [(S3["sep"], {"en": "Separate heads", "zh": "独立头"}, BLUE7, fit), (S3["shared"], {"en": "Shared head", "zh": "共享头"}, GREEN, fit),
         (S3["probes"], {"en": "Probes only", "zh": "只加探针"}, RED, pchip)]
b3 = float(S3["baseline"])
v5 = view(np.tile(x3, len(sets3)), np.r_[np.concatenate([y for y, _, _, _ in sets3]), b3], endpoints_only=True)
marks5 = [line([[v5["xlim"][0], b3], [v5["xlim"][1], b3]], GREY)] + sum(([line(f(x3, y), c), dots(pts(x3, y), c)] for y, _, c, f in sets3), [])
figs["fig-heads-300m"] = dict(title={"en": "Final training loss by layer and setup, 300m", "zh": "按层和做法的最终训练损失，300m"},
                              legend=[[n, c, "line"] for _, n, c, _ in sets3] + [[{"en": "Baseline", "zh": "基线"}, GREY, "line"]], quantity=CE, xlabel=XL, height=320, panels=[dict(**v5, marks=marks5)])

# Figures 6 and 7: the -vs-d48-g4 runs (d48g4.npz, fetch_d48g4.py; finished runs only). A design is drawn once both its
# runs have finished: separate heads (fitted law, else PCHIP) and probes (PCHIP), final loss of readouts 1..48 at x = 6 * N(k) * D.
# One shared view over every drawn design, so the panels of Figure 7 compare by eye; designs still training keep their status word.
G4 = dict(np.load(HERE / "d48g4.npz", allow_pickle=True)) if (HERE / "d48g4.npz").exists() else {}
def g4(arch):
    a = f"-a{arch}" if arch else ""
    tags = [f"{s}{a}-vs-d48-g4" for s in ("-sep", "-sep-bbfrozen")]
    if any(G4.get(t + "/final") is None for t in tags):
        return None
    for t in tags:   # complete runs only (fetch_d48g4.py checks this too)
        if int(G4[t + "/step"][-1]) != STEPS - 1:
            raise RuntimeError(f"{t} ends at step {int(G4[t + '/step'][-1])}, not {STEPS - 1}")
    return [G4[t + "/final"] for t in tags]
# each design at its own layer size: NL plus its per-layer parameter difference from Sandwich-LN (W&B parameter_count of
# the separate-heads runs, which differ only in the backbone): MoDA +13.8%, mHC +2.6%, HC +0.1%, Pre-LN and LNS -0.2%
def x48(arch):
    pa, p0 = G4.get(f"-sep{'-a' + arch if arch else ''}-vs-d48-g4/params"), G4.get("-sep-vs-d48-g4/params")
    nl = NL + (int(pa) - int(p0)) / 48 if pa is not None and p0 is not None else NL
    return np.array([6.0 * nl * d * DTOT for d in range(1, 49)])
ARCH = [("preln", "Pre-LN"), ("lns", "LayerNorm Scaling"), ("deepnorm", "DeepNorm"), ("keel", "KEEL"), ("hc", "Hyper-Connections"),
        ("mhc", "mHC"), ("attnres", "AttnRes (Full)"), ("attnres_block", "AttnRes (Block)"), ("moda", "MoDA")]
# status words while a design trains (update with the queue): runs started 2026-10-09
STATUS48 = {"attnres": RUNNING, "moda": RUNNING}   # training under way (a chain job running or the next one queued); "" = Sandwich-LN; hc, mhc: queued
done48 = {a: g4(a) for a in [""] + [k for k, _ in ARCH] if g4(a) is not None}
for a, ys in done48.items():
    print("d48", a or "sandwich", "separate heads L1 L24 L48", np.round(ys[0][[0, 23, 47]], 3), "probes", np.round(ys[1][[0, 23, 47]], 3))
v48 = view(np.concatenate([x48(a) for a in done48 for _ in (0, 1)]), np.concatenate([y for ys in done48.values() for y in ys]), top=1.3, endpoints_only=True) if done48 else None   # headroom: the panel name sits above the first layers
def panel48(arch, ys=(3, 3.2, 3.4, 3.6, 3.8, 4, 4.5, 5)):
    if arch in done48:
        sep, prb = done48[arch]
        v = dict(v48, yticks=[t for t in v48["yticks"] if t[0] in ys])   # the small panels of Figure 7 keep fewer values
        xa = x48(arch)
        return dict(**v, marks=[line(fit(xa, sep), BLUE7), dots(pts(xa, sep), BLUE7), line(pchip(xa, prb), RED), dots(pts(xa, prb), RED)])
    if v48:   # the finished designs' axes, so every panel reads on one scale
        return dict(v48, yticks=[t for t in v48["yticks"] if t[0] in ys], marks=[], empty=STATUS48.get(arch, QUEUED))
    return empty_view(x48(arch), 2.9, 5.0, STATUS48.get(arch, QUEUED), ys=ys)
figs["fig-d48"] = dict(title={"en": "Final training loss by layer, 48 layers of width 512", "zh": "按层的最终训练损失，48 层、宽 512 的网络"},
                       legend=[[{"en": "Separate heads", "zh": "独立头"}, BLUE7, "line"], [{"en": "Probes only", "zh": "只加探针"}, RED, "line"]],
                       quantity=CE, xlabel=XL, height=320, panels=[panel48("")])
figs["fig-d48-arch"] = dict(title={"en": "Final training loss by layer and architecture, 48 layers of width 512", "zh": "按层和架构的最终训练损失，48 层、宽 512 的网络"},
                            legend=figs["fig-d48"]["legend"],
                            quantity=CE, xlabel=XL, height=200, cols=1,   # one column, each panel the full 640 canvas (owner, 2026-10-11: three columns too small; trying one)
                            panels=[dict(**panel48(k, ys=(3, 3.5, 4, 5)), label={"en": n, "zh": n}) for k, n in ARCH])

# Figures 8 and 9 (owner, 2026-10-11: "这两个图应该分别做单栏大图加入页面里"): the finished 48-layer designs side by side, x =
# layer index (not compute), one colour per design (writing:plot categories). Figure 8: separate heads minus probes,
# final training loss by layer. Figure 9: the two setups' absolute losses in two panels. Designs join as they finish.
DES = [("", "Sandwich-LN")] + ARCH
DCOL = {"": "#1A73E8", "preln": "#D93025", "lns": "#E37400", "deepnorm": "#1E8E3E", "keel": "#9334E6", "attnres_block": "#3C4043"}   # GM2 600 (yellow 900), then grey 800
drawn = [(a, n) for a, n in DES if a in done48 and a in DCOL]
LX = [[k, str(k)] for k in (1, 12, 24, 36, 48)]
def lin_ticks(lo, hi, step, fmt="{:.1f}"):
    v = np.arange(np.floor(lo / step) * step, hi + step / 2, step); return [[round(float(t), 6), fmt.format(t)] for t in v]
k48 = np.arange(1, 49)
gaps = {a: done48[a][0] - done48[a][1] for a, _ in drawn}
figs["fig-d48-gap"] = dict(title={"en": "Separate heads minus probes by layer and design, 48 layers", "zh": "按层和设计的独立头减只加探针，48 层"},
                           legend=[[{"en": n, "zh": n}, DCOL[a], "line"] for a, n in drawn],
                           quantity={"en": "Training loss difference (nats)", "zh": "训练损失之差（nats）"}, xlabel={"en": "Layer", "zh": "层"}, height=340,
                           panels=[dict(xlim=[1, 48], ylim=[-2.5, 0.5], xticks=LX, yticks=lin_ticks(-2.5, 0.5, 0.5),
                                        marks=[line([[0, 0], [49, 0]], GREY, 1.0)] + [line(pts(k48, gaps[a]), DCOL[a], 1.8) for a, _ in drawn])])
figs["fig-d48-abs"] = dict(title={"en": "Final training loss by layer and design, 48 layers", "zh": "按层和设计的最终训练损失，48 层"},
                           legend=figs["fig-d48-gap"]["legend"],
                           quantity={"en": "Training loss (nats)", "zh": "训练损失（nats）"}, xlabel={"en": "Layer", "zh": "层"}, height=300, cols=1,
                           panels=[dict(xlim=[1, 48], ylim=[2.75, 5.85], xticks=LX, yticks=lin_ticks(3.0, 5.5, 0.5), label=lab,
                                        marks=[line(pts(k48, done48[a][j]), DCOL[a], 1.8) for a, _ in drawn])
                                   for j, lab in ((0, {"en": "Separate heads", "zh": "独立头"}), (1, {"en": "Probes only", "zh": "只加探针"}))])
for a, n in drawn:
    print("fig-d48-gap", n, "min", round(float(gaps[a].min()), 3), "L48", round(float(gaps[a][-1]), 3))

# Figure 8: the 2 x 2 of layer-loss weight (1, 0.2) and gradient reach (all layers below, own layer only), separate
# heads at 130m (fix.npz, fetch_fix.py), with probes and the ordinary models as references
YELLOW9, CYAN9, PINK7 = "#E37400", "#007B83", "#C2185B"
FX = np.load(HERE / "fix.npz")
SETS8 = [(Z["-pls1-sep/final"], {"en": "Separate heads", "zh": "独立头"}, BLUE7, fit), (FX["-pls0.2-sep/final"], {"en": "Layer losses summing to 1", "zh": "各层损失权重合计为 1"}, YELLOW9, fit),
         (FX["-pls1-sep-local/final"], {"en": "Each head trains its own layer", "zh": "每个头只训练自己那一层"}, CYAN9, fit), (FX["-pls0.2-sep-local/final"], {"en": "Both", "zh": "两者都用"}, PINK7, fit),
         (Z["-pls1-sep-bbfrozen/final"], {"en": "Probes only", "zh": "只加探针"}, RED, pchip), (cy, {"en": "Baseline by depth", "zh": "各深度的基线"}, GREY, fit)]
# the weight sweep (owner, 2026-10-05): weights 0.1 and 0.05 join once finished, in lighter blues than separate heads'
# (weight 1, blue 700), as the sweep popup described; weight 0.2 keeps its yellow, shared with Figure 9
for tag, nm, col in (("-pls0.1-sep", {"en": "Layer-loss weight 0.1", "zh": "逐层损失权重 0.1"}, "#4285F4"), ("-pls0.05-sep", {"en": "Layer-loss weight 0.05", "zh": "逐层损失权重 0.05"}, "#8AB4F8")):
    if tag + "/final" in FX:
        SETS8.insert(2 + sum(1 for q in SETS8 if q[1]['en'].startswith('Layer-loss weight')), (FX[tag + "/final"], nm, col, fit))
marks8 = sum(([line(f(x6, y), c), dots(pts(x6, y), c)] for y, _, c, f in SETS8), [])
v8 = view(np.tile(x6, len(SETS8)), np.concatenate([y for y, _, _, _ in SETS8]), endpoints_only=True)
figs["fig-fix"] = dict(title={"en": "Final training loss by layer, gradient routing and layer-loss weight, 130m", "zh": "按梯度去向和逐层损失权重的各层最终训练损失，130m"},
                       legend=[[n, c, "line"] for _, n, c, _ in SETS8], quantity=CE, xlabel=XL, height=320, panels=[dict(**v8, marks=marks8)])
print("fig-fix last layer", {n["en"]: round(float(y[-1]), 3) for y, n, _, _ in SETS8})

# Figure 9 (placeholder until the runs finish): the shape of the layer-loss weights (uniform 0.2, or rising with depth,
# k/15; both sum to 1) crossed with per-token gradient surgery at every layer output, plus surgery at weight 1 (owner,
# 2026-10-05: every layer should sit low, not only the last); separate heads at weight 1 and the ordinary models as references (colours as approved in the owner's popup, 2026-10-05)
MIX = [(Z["-pls1-sep/final"], {"en": "Separate heads", "zh": "独立头"}, BLUE7), ("-pls0.2-sep", {"en": "Layer losses summing to 1", "zh": "各层损失权重合计为 1"}, YELLOW9),
       ("-pls0.2-sep-shdepth", {"en": "Weights rising with depth", "zh": "权重随深度增加"}, CYAN9), ("-pls0.2-sep-pcg", {"en": "Summing to 1, gradient surgery", "zh": "合计为 1，处理梯度冲突"}, PURPLE),
       ("-pls0.2-sep-shdepth-pcg", {"en": "Rising with depth, gradient surgery", "zh": "随深度增加，处理梯度冲突"}, PINK7), ("-pls1-sep-pcg", {"en": "Weight 1, gradient surgery", "zh": "权重 1，处理梯度冲突"}, "#0D652D")]
legend9 = [[n, c, "line"] for _, n, c in MIX] + [[{"en": "Baseline by depth", "zh": "各深度的基线"}, GREY, "line"]]
done9 = [(src if not isinstance(src, str) else FX[src + "/final"], c) for src, _, c in MIX if not isinstance(src, str) or src + "/final" in FX] + [(cy, GREY)]
if len(done9) > 3:   # some Figure 9 run has finished: draw the finished ones, the rest join as they finish
    marks9 = sum(([line(fit(x6, y), c), dots(pts(x6, y), c)] for y, c in done9), [])
    v9 = view(np.tile(x6, len(done9)), np.concatenate([y for y, _ in done9]), endpoints_only=True)
    p9 = dict(**v9, marks=marks9)
else:
    p9 = empty_view(list(x6), 2.9, 5.0, QUEUED)
figs["fig-mix"] = dict(title={"en": "Final training loss by layer, weight shape and gradient surgery, 130m", "zh": "按权重形状和梯度冲突处理的各层最终训练损失，130m"},
                       legend=legend9, quantity=CE, xlabel=XL, height=320, panels=[p9])

# Figure 11 (placeholder until the runs finish; owner-approved popup 2026-10-07): late start. Separate heads at weight 1 whose
# gradient into the backbone is 0 until step S and full from S + 100 (probes before, separate heads after), S = 300 or 1500;
# separate heads from step 0, probes and the ordinary models as references
LATE = [(Z["-pls1-sep/final"], {"en": "Separate heads", "zh": "独立头"}, BLUE7, fit), ("-pls1-sep-on300-400", {"en": "Separate heads from step 300", "zh": "第 300 步起用独立头"}, "#681DA8", fit),
        ("-pls1-sep-on1500-1600", {"en": "Separate heads from step 1,500", "zh": "第 1,500 步起用独立头"}, "#AF5CF7", fit), (Z["-pls1-sep-bbfrozen/final"], {"en": "Probes only", "zh": "只加探针"}, RED, pchip)]
legend11 = [[n, c, "line"] for _, n, c, _ in LATE] + [[{"en": "Baseline by depth", "zh": "各深度的基线"}, GREY, "line"]]
done11 = [(src if not isinstance(src, str) else FX[src + "/final"], c, f) for src, _, c, f in LATE if not isinstance(src, str) or src + "/final" in FX] + [(cy, GREY, fit)]
if len(done11) > 3:   # a late-start run has finished
    marks11 = sum(([line(f(x6, y), c), dots(pts(x6, y), c)] for y, c, f in done11), [])
    v11 = view(np.tile(x6, len(done11)), np.concatenate([y for y, _, _ in done11]), endpoints_only=True)
    p11 = dict(**v11, marks=marks11)
else:
    p11 = empty_view(list(x6), 2.9, 5.0, QUEUED)
figs["fig-late"] = dict(title={"en": "Final training loss by layer and the step separate heads start, 130m", "zh": "按独立头开始步数的各层最终训练损失，130m"},
                        legend=legend11, quantity=CE, xlabel=XL, height=320, panels=[p11])

# Figure 10: the 300m heads refit on a frozen backbone (launcher HEADS_FROM); held-out loss by layer before
# (the finished runs' final eval) and after; the earlier values as context in the light tone of each setup's colour
BLUE3, RED3 = "#8AB4F8", "#F28B82"
CE_EVAL = {"en": "Held-out cross-entropy (nats, log scale)", "zh": "留出数据上的交叉熵（nats，对数坐标）"}
HF = json.load(open(HERE / "headft.json"))   # fetch_headft.py: eval/L{k}/loss before and after the 2,000-step head refit
SETS10 = [(HF["-pls1-sep"]["eval"], BLUE3), (HF["-pls1-sep-headft2000"]["eval"], BLUE7), (HF["-pls1-sep-bbfrozen"]["eval"], RED3), (HF["-pls1-sep-bbfrozen-headft2000"]["eval"], RED)]
marks10 = sum(([line(pchip(x3, np.array(y, float)), c), dots(pts(x3, y), c)] for y, c in SETS10), [])
v10 = view(np.tile(x3, len(SETS10)), np.concatenate([np.array(y, float) for y, _ in SETS10]), endpoints_only=True)
v10["yticks"] = [[a, f"{a:g}"] for a in (3.5, 4, 4.5, 5, 6, 7) if v10["ylim"][0] <= a <= v10["ylim"][1]]
figs["fig-headft"] = dict(title={"en": "Per-layer loss before and after retraining the heads, 300m", "zh": "重训读出头前后的逐层损失，300m"},
                          legend=[[{"en": "Separate heads", "zh": "独立头"}, BLUE3, "line"], [{"en": "Separate heads, heads retrained", "zh": "独立头，重训读出头"}, BLUE7, "line"],
                                  [{"en": "Probes only", "zh": "只加探针"}, RED3, "line"], [{"en": "Probes only, heads retrained", "zh": "只加探针，重训读出头"}, RED, "line"]],
                          quantity=CE_EVAL, xlabel=XL, height=320, panels=[dict(**v10, marks=marks10)])

OUT.write_text("/* Generated by build_figures_data.py (pls branch, docs/della/pls_archive/blog); regenerate rather than hand-edit. */\nwindow.PLC_FIGS = " + json.dumps(figs, ensure_ascii=False, separators=(",", ":")) + ";\n")
print("N per layer", NL, "| x of layer 2..6 at the end", [f"{v:.3g}" for v in x6], "| wrote", OUT, OUT.stat().st_size, "bytes")
