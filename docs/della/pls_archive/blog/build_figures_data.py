"""Figures of wenhaochai.com/exps/per-layer-contribution.html -> assets/data/plc-figures.js (generated; regenerate,
never hand-edit). Drawn on the page in the writing:plot (Epoch AI) style.

x axis: the layer number on every by-layer figure (owner, 2026-10-17; to_layers at the end); backbone compute 6ND on
the figures that follow training (fig-modes, fig-traj, fig-traj-300m), and the fits below are made in compute. N = parameters of the transformer layers up to the one read out (the embedding
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
    # the data x too, so the line passes through every point (60 even samples alone skipped deep layers, 2026-10-13)
    f = PchipInterpolator(np.log(x), np.log(y)); g = np.union1d(np.geomspace(x.min(), x.max(), 60), x); return pts(g, np.exp(f(np.log(g))))
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
STATUS48 = {}   # training under way (a chain job running or the next one queued); "" = Sandwich-LN (all ten finished 2026-10-15)
done48 = {a: g4(a) for a in [""] + [k for k, _ in ARCH] if g4(a) is not None}
for a, ys in done48.items():
    print("d48", a or "sandwich", "separate heads L1 L24 L48", np.round(ys[0][[0, 23, 47]], 3), "probes", np.round(ys[1][[0, 23, 47]], 3))
# AttnRes (Full): readout k is the depth-attention mix that layer k+1 reads, not a residual stream, and with probes many of
# these mixes carry little next-token information (layers 17-47 jump between 3.3 and 7.8 nats; layer 48, the output, is
# 3.05). Its probes stay out of the shared view; its panel takes its own y range, said in the caption.
OWN_RANGE = {"attnres"}
v48 = view(np.concatenate([x48(a) for a in done48 for j in (0, 1) if not (a in OWN_RANGE and j)]),
           np.concatenate([y for a, ys in done48.items() for j, y in enumerate(ys) if not (a in OWN_RANGE and j)]), top=1.3, endpoints_only=True) if done48 else None   # headroom: the panel name sits above the first layers
def panel48(arch, ys=(3, 3.2, 3.4, 3.6, 3.8, 4, 4.5, 5)):
    if arch in done48:
        sep, prb = done48[arch]
        v = dict(v48, yticks=[t for t in v48["yticks"] if t[0] in ys])   # the small panels of Figure 7 keep fewer values
        if arch in OWN_RANGE:   # the shared x range, a y range up to the highest probe readout
            yhi = 2 + (max(sep.max(), prb.max()) - 2) * 2.2   # room for the panel name above the 8 value
            v = dict(v48, ylim=[v48["ylim"][0], float(yhi)], yticks=[[a, f"{a:g}"] for a in (3, 4, 5, 6, 8) if a <= yhi - 0.04 * (yhi - v48["ylim"][0])])
        xa = x48(arch)
        prb_line = line(pts(xa, prb), RED) if arch in OWN_RANGE else line(pchip(xa, prb), RED)   # AttnRes (Full) probes jump from layer to layer: straight segments through the points
        return dict(**v, marks=[line(fit(xa, sep), BLUE7), dots(pts(xa, sep), BLUE7), prb_line, dots(pts(xa, prb), RED)])
    if v48:   # the finished designs' axes, so every panel reads on one scale
        return dict(v48, yticks=[t for t in v48["yticks"] if t[0] in ys], marks=[], empty=STATUS48.get(arch, QUEUED))
    return empty_view(x48(arch), 2.9, 5.0, STATUS48.get(arch, QUEUED), ys=ys)
# Ordinary models of 12, 24 and 36 layers (same recipe and tokens; owner popup 2026-10-11): grey dots at the backbone
# compute of that depth, final training loss (d48ref.json, fetch_d48ref.py; finished runs only)
LEG48 = [[{"en": "Separate heads", "zh": "独立头"}, BLUE7, "line"], [{"en": "Probes only", "zh": "只加探针"}, RED, "line"]]
REF48 = {int(k): v for k, v in json.loads((HERE / "d48ref.json").read_text()).items()} if (HERE / "d48ref.json").exists() else {}
p48 = panel48("")
if REF48 and p48.get("marks") is not None and not p48.get("empty"):
    xr = x48("")
    p48 = dict(p48, marks=p48["marks"] + [dots(pts([xr[d - 1] for d in sorted(REF48)], [REF48[d]["train"] for d in sorted(REF48)]), GREY)])
figs["fig-d48"] = dict(title={"en": "Final training loss by layer, 48 layers of width 512", "zh": "按层的最终训练损失，48 层、宽 512 的网络"},
                       legend=LEG48 + ([[{"en": "Ordinary models by depth", "zh": "各深度的普通模型"}, GREY, "dot"]] if REF48 else []),
                       quantity=CE, xlabel=XL, height=320, panels=[p48])
figs["fig-d48-arch"] = dict(title={"en": "Final training loss by layer and architecture, 48 layers of width 512", "zh": "按层和架构的最终训练损失，48 层、宽 512 的网络"},
                            legend=LEG48,
                            quantity=CE, xlabel=XL, height=200, cols=1,   # one column, each panel the full 640 canvas (owner, 2026-10-11: three columns too small; trying one)
                            panels=[dict(**panel48(k, ys=(3, 3.5, 4, 5)), label={"en": n, "zh": n}) for k, n in ARCH])

# Figures 8 and 9 (owner, 2026-10-11: "这两个图应该分别做单栏大图加入页面里"): the finished 48-layer designs side by side, x =
# layer index (not compute), one colour per design (writing:plot categories). Figure 8: separate heads minus probes,
# final training loss by layer. Figure 9: the two setups' absolute losses in two panels. Designs join as they finish.
DES = [("", "Sandwich-LN")] + ARCH
DCOL = {"": "#1A73E8", "preln": "#D93025", "lns": "#E37400", "deepnorm": "#1E8E3E", "keel": "#9334E6", "attnres_block": "#3C4043", "hc": "#007B83", "mhc": "#C2185B", "moda": "#B06000"}   # GM2 600 (yellow 900), then grey 800, cyan 900, pink 700, orange 900
drawn = [(a, n) for a, n in DES if a in done48 and a in DCOL]   # no colour for AttnRes (Full): its probe readouts below layer 48 are not comparable (OWN_RANGE)
LX = [[k, str(k)] for k in (1, 12, 24, 36, 48)]
def lin_ticks(lo, hi, step, fmt="{:.1f}"):
    v = np.arange(np.floor(lo / step) * step, hi + step / 2, step); return [[round(float(t), 6), fmt.format(t)] for t in v]
k48 = np.arange(1, 49)
gaps = {a: done48[a][0] - done48[a][1] for a, _ in drawn}
figs["fig-d48-gap"] = dict(title={"en": "Separate heads minus probes by layer and design, 48 layers", "zh": "按层和设计的独立头减只加探针，48 层"},
                           legend=[[{"en": n, "zh": n}, DCOL[a], "line"] for a, n in drawn],
                           quantity={"en": "Training loss difference (nats)", "zh": "训练损失之差（nats）"}, xlabel={"en": "Layer", "zh": "层"}, height=340,
                           panels=[dict(xlim=[-3, 49], ylim=[-2.5, 0.5], xticks=LX, yticks=lin_ticks(-2.5, 0.5, 0.5),
                                        marks=[line([[0, 0], [49, 0]], GREY, 1.0)] + [line(pts(k48, gaps[a]), DCOL[a], 1.8) for a, _ in drawn])])
figs["fig-d48-abs"] = dict(title={"en": "Final training loss by layer and design, 48 layers", "zh": "按层和设计的最终训练损失，48 层"},
                           legend=figs["fig-d48-gap"]["legend"],
                           quantity={"en": "Training loss (nats)", "zh": "训练损失（nats）"}, xlabel={"en": "Layer", "zh": "层"}, height=300, cols=1,
                           panels=[dict(xlim=[-3, 49], ylim=[2.75, 5.85], xticks=LX, yticks=lin_ticks(3.0, 5.5, 0.5), label=lab,
                                        marks=[line(pts(k48, done48[a][j]), DCOL[a], 1.8) for a, _ in drawn])
                                   for j, lab in ((0, {"en": "Separate heads", "zh": "独立头"}), (1, {"en": "Probes only", "zh": "只加探针"}))])
# Figure 11 (placeholder; owner popup 2026-10-14): why HC pays about half. Layer-48 final training loss, separate heads minus
# probes, for the five single-stream designs, HC, mHC and HC with writes kept positive (ARCH=hc_pos, branch pls-hcpos):
# the causal test of the hidden-write account (docs/della/pls.md). Bars on a categorical axis; the hc_pos bar joins when
# its pair finishes (its category label says Queued / Running until then).
HCPOS_STATUS = QUEUED
CYAN3 = "#78D9EC"   # GM2 cyan 300: HC with positive writes, beside HC's cyan 900
WSET = [("", "Sandwich-LN", GREY), ("preln", "Pre-LN", GREY), ("lns", "LNS", GREY), ("deepnorm", "DeepNorm", GREY), ("keel", "KEEL", GREY), ("moda", "MoDA", GREY),
        ("hc", "HC", DCOL["hc"]), ("mhc", "mHC", DCOL["mhc"]), ("hc_pos", None, CYAN3)]
w48 = dict(done48)   # Figure 11 alone also reads HC with positive writes (the other figures keep their ten designs)
if g4("hc_pos") is not None:
    w48["hc_pos"] = g4("hc_pos")
wval = [round(float(w48[a][0][-1] - w48[a][1][-1]), 4) if a in w48 else None for a, _, _ in WSET]
wcats = [{"en": n, "zh": n} if n else ({"en": "HC, w > 0", "zh": "HC，w > 0"} if a in w48 else HCPOS_STATUS) for a, n, _ in WSET]   # the legend names the colour
figs["fig-hc-write"] = dict(title={"en": "Last-layer cost of separate heads by design, 48 layers", "zh": "各设计下独立头在最后一层的代价，48 层"},
                            legend=[[{"en": "One residual stream", "zh": "单个残差流"}, GREY, "box"], [{"en": "Hyper-Connections", "zh": "Hyper-Connections"}, DCOL["hc"], "box"],
                                    [{"en": "mHC", "zh": "mHC"}, DCOL["mhc"], "box"], [{"en": "HC with positive writes", "zh": "写入为正的 HC"}, CYAN3, "box"]],
                            quantity={"en": "Layer-48 training loss, separate heads minus probes (nats)", "zh": "第 48 层训练损失，独立头减只加探针（nats）"}, height=240,
                            panels=[dict(xlim=[-0.6, len(WSET) - 0.4], ylim=[0, 0.15], yticks=lin_ticks(0, 0.15, 0.05, "{:.2f}"), cats=wcats,
                                         marks=[dict(type="bars", color=c, vals=[v if i == j else None for j, v in enumerate(wval)]) for i, (_, _, c) in enumerate(WSET)])])
print("fig-hc-write", dict(zip([n or a for a, n, _ in WSET], wval)))
# Figure 8 (placeholder for seed 1; owner popup 2026-10-13): AttnRes (Full), layer-48 training loss by step, separate heads
# and probes, one panel per init seed. With seed 0 the probes run's last readout jumped from 3.41 to 8.32 at step 2106
# (the head's depth softmax collapsed: grad/norm/transformer.final_query 0.12 -> 3.8e-07) and came back over ~1000 steps;
# seed 1 (SEED=1, data order unchanged; jobs 15520124-26) tests whether the design's missing last-layer cost comes from it.
# Mean of 20 steps, from step 100. Seed 1 finished 2026-10-17 (15538869): final layer 48, separate heads 3.016, probes
# 2.955 (+0.061); seed 0: 3.023, 3.045 (-0.021). The seed-1 probes run also spikes (626, 818, 1024, 2578), earlier.
SEED1_STATUS = None
def seed_panel(sfx, word, label):
    tags = [f"-sep{b}-aattnres-vs-d48{sfx}-g4" for b in ("", "-bbfrozen")]
    if not all(t + "/ce" in G4 for t in tags):
        return dict(SV, marks=[], empty=word, label=label)
    marks = []
    for t, c in zip(tags, (BLUE7, RED)):
        st, ce = G4[t + "/step"], G4[t + "/ce"][:, -1]; m = st >= 100
        xs, y = smooth(st[m], ce[m], 20); idx = np.unique(np.linspace(0, len(xs) - 1, 400).astype(int))
        marks.append(line(pts(xs[idx], y[idx]), c, 1.4))
    return dict(SV, marks=marks, label=label)
SV = dict(ylog=True, yoff=YOFF, xlim=[0, 5100], ylim=[2.9, 16], xticks=[[v, f"{v:,}"] for v in range(0, 5001, 1000)],
          yticks=[[a, f"{a:g}"] for a in (3, 3.5, 4, 5, 6, 8)])
figs["fig-attnres-seed"] = dict(title={"en": "AttnRes (Full) layer-48 training loss by init seed, 48 layers", "zh": "AttnRes (Full) 第 48 层的训练损失，按初始化种子，48 层"},
                                legend=figs["fig-d48"]["legend"], quantity={"en": "Training loss (nats, log scale)", "zh": "训练损失（nats，对数坐标）"},
                                xlabel={"en": "Step", "zh": "步数"}, height=200, cols=1,
                                panels=[seed_panel("", None, {"en": "Seed 0", "zh": "种子 0"}), seed_panel("-s1", SEED1_STATUS, {"en": "Seed 1", "zh": "种子 1"})])

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

# Figure after fig-late (placeholder; owner popup 2026-10-16): ten more attempts at the last-layer cost, 130m, branch
# pls-auto10 (pls_auto10_pair.sbatch). Per attempt, the final training loss (mean of the last 50 logged steps, as in
# Figure 3) of layer 1 and of layer 6 minus the ordinary model of the same depth (grey dots of Figure 3), beside separate
# heads at weight 1. auto10.json (fetch_auto10.py, once runs finish): {key: [final loss of layers 1..6]}; a key without
# data shows its status word under its bar.
AUTO10 = [("sep", {"en": "Separate heads", "zh": "独立头"}, None), ("os", {"en": "1 Own clip", "zh": "1 单独裁剪"}, None),
          ("mix", {"en": "2 Mix 0.2", "zh": "2 混合 0.2"}, QUEUED), ("kd", {"en": "3 Distill", "zh": "3 自蒸馏"}, QUEUED),
          ("conf", {"en": "4 Confidence", "zh": "4 置信度"}, None), ("ad", {"en": "5 Adapter", "zh": "5 适配器"}, QUEUED),
          ("mlpo", {"en": "6 MLP only", "zh": "6 只更新 MLP"}, QUEUED), ("first", {"en": "7 First 256", "zh": "7 前 256"}, None),
          ("randl", {"en": "8 One layer", "zh": "8 随机单层"}, None), ("hc", {"en": "9 HC", "zh": "9 HC"}, QUEUED),
          ("attnres", {"en": "10 AttnRes", "zh": "10 AttnRes"}, None)]
A10 = json.load(open(HERE / "auto10.json")) if (HERE / "auto10.json").exists() else {}
A10["sep"] = [float(v) for v in Z["-pls1-sep/final"]]
gap10 = {k: [float(A10[k][j] - cy[j]) for j in range(6)] for k in A10}
cats10 = [lab if k in gap10 else {"en": lab["en"] + " · " + st["en"], "zh": lab["zh"] + " · " + st["zh"]} for k, lab, st in AUTO10]


def auto10_panel(j, label, lo, hi, step):
    vals = [round(gap10[k][j], 4) if k in gap10 else None for k, _, _ in AUTO10]
    return dict(xlim=[-1.3, len(AUTO10) - 0.4], ylim=[lo, hi], yticks=lin_ticks(lo, hi, step, "{:.2f}"), cats=cats10, label=label,
                marks=[dict(type="bars", color=BLUE7 if k == "sep" else "#8AB4F8", vals=[v if i == n else None for n, v in enumerate(vals)]) for i, (k, _, _) in enumerate(AUTO10)])


figs["fig-auto10"] = dict(title={"en": "Ten more attempts at the last-layer cost, 130m", "zh": "降低最后一层代价的另外十种尝试，130m"},
                          legend=[[{"en": "Separate heads, weight 1", "zh": "独立头，权重 1"}, BLUE7, "box"], [{"en": "One attempt", "zh": "一种尝试"}, "#8AB4F8", "box"]],
                          quantity={"en": "Final training loss minus the ordinary model of the same depth (nats)", "zh": "最终训练损失减去同深度普通模型（nats）"},
                          height=200, cols=1, panels=[auto10_panel(0, {"en": "Layer 1", "zh": "第 1 层"}, 0, 0.5, 0.1), auto10_panel(5, {"en": "Layer 6, the output", "zh": "第 6 层，即输出"}, 0, 0.15, 0.05)])
print("fig-auto10", {k: [round(g, 3) for g in (v[0], v[5])] for k, v in gap10.items()})

# Figure after fig-auto10 (placeholder; owner popups 2026-10-17): the logit lens of every finished model. Per checkpoint,
# held-out (Paloma) loss by layer through the layer's own readout (its separate head or probe: eval/L{k}/loss of the source
# run) and through the model's final norm and lm_head (the logit lens: launcher LENS_FROM, an eval of the frozen
# checkpoint). Panels: 130m (6 layers) and 48 layers of width 512 (Sandwich-LN); own readouts in the dark tone, the
# lens in the light tone of the same colour. lens.json (fetch_lens.py) fills it.
LENS_STATUS = QUEUED
figs["fig-lens"] = dict(title={"en": "Held-out loss by layer through each layer's own readout and through the final head", "zh": "按层的留出损失：经各层自己的读出头与经最终输出头"},
                        legend=[[{"en": "Ordinary model, probes", "zh": "普通模型，探针"}, RED, "line"], [{"en": "Ordinary model, logit lens", "zh": "普通模型，logit lens"}, "#F28B82", "line"],
                                [{"en": "Separate heads, own heads", "zh": "独立头，自己的头"}, BLUE7, "line"], [{"en": "Separate heads, logit lens", "zh": "独立头，logit lens"}, "#8AB4F8", "line"]],
                        quantity={"en": "Held-out cross-entropy (nats, log scale)", "zh": "留出数据上的交叉熵（nats，对数坐标）"}, xlabel=XL, height=220, cols=1,
                        panels=[dict(**empty_view(list(x6), 3.6, 9, LENS_STATUS, ys=(4, 5, 6, 7, 8)), label={"en": "6 layers, 130m", "zh": "6 层，130m"}),
                                dict(**empty_view(list(x48("")), 3.6, 9, LENS_STATUS, ys=(4, 5, 6, 7, 8)), label={"en": "48 layers, Sandwich-LN", "zh": "48 层，Sandwich-LN"})])

# Figure 10: the 300m heads refit on a frozen backbone (launcher HEADS_FROM); held-out loss by layer before
# (the finished runs' final eval) and after; the earlier values as context in the light tone of each setup's colour
BLUE3, RED3 = "#8AB4F8", "#F28B82"
CE_EVAL = {"en": "Held-out cross-entropy (nats, log scale)", "zh": "留出数据上的交叉熵（nats，对数坐标）"}
HF = json.load(open(HERE / "headft.json"))   # fetch_headft.py: eval/L{k}/loss before and after the 2,000-step head refit
SETS10 = [(HF["-pls1-sep"]["eval"], BLUE3), (HF["-pls1-sep-headft2000"]["eval"], BLUE7), (HF["-pls1-sep-bbfrozen"]["eval"], RED3), (HF["-pls1-sep-bbfrozen-headft2000"]["eval"], RED)]
marks10 = sum(([line(pchip(x3, np.array(y, float)), c), dots(pts(x3, y), c)] for y, c in SETS10), [])
v10 = view(np.tile(x3, len(SETS10)), np.concatenate([np.array(y, float) for y, _ in SETS10]), endpoints_only=True)
v10["yticks"] = [[a, f"{a:g}"] for a in (3.5, 4, 4.5, 5, 6, 7) if v10["ylim"][0] <= a <= v10["ylim"][1]]
# Last figure (placeholder; owner popup 2026-10-13, DeepSeek-V3 form): MTP and every layer. 130m, 6 layers, one pair job
# (15523669, branch pls-mtp): NTP and NTP + one DeepSeek-V3 MTP module (weight 0.3), both with detached probes on every
# layer for x_{t+1} and for x_{t+2}. Panels: final probe loss by layer for each target; the ordinary models by depth as
# grey dots on the x_{t+1} panel. Filled once the two runs finish.
MTP_STATUS = QUEUED
MTPC = "#1E8E3E"   # GM2 green 600: NTP + MTP (red stays the probes of the ordinary NTP model, as on the rest of the page)
figs["fig-mtp"] = dict(title={"en": "Final probe loss by layer, NTP and MTP, 130m", "zh": "按层的探针最终损失，NTP 与 MTP，130m"},
                       legend=[[{"en": "NTP", "zh": "NTP"}, RED, "line"], [{"en": "NTP + MTP", "zh": "NTP + MTP"}, MTPC, "line"],
                               [{"en": "Ordinary models by depth", "zh": "各深度的普通模型"}, GREY, "dot"]],
                       quantity=CE, xlabel=XL, height=220, cols=1,
                       panels=[dict(**empty_view(x6, 2.9, 10, MTP_STATUS, ys=(3, 4, 5, 6, 8)), label=lab)
                               for lab in ({"en": "Probes for the next token", "zh": "预测下一个词的探针"}, {"en": "Probes for the token after next", "zh": "预测下下个词的探针"})])
MT = json.loads((HERE / "mtp.json").read_text()) if (HERE / "mtp.json").exists() else {}
if "ntp" in MT and "mtp" in MT:   # both runs finished (fetch_mtp.py): fill the panels; each panel its own y range
    for j, (key, lab) in enumerate((("next", {"en": "Probes for the next token", "zh": "预测下一个词的探针"}), ("after", {"en": "Probes for the token after next", "zh": "预测下下个词的探针"}))):
        ys = [np.array(MT[r][key]) for r in ("ntp", "mtp")] + ([cy] if key == "next" else [])
        v = view(np.tile(x6, len(ys)), np.concatenate(ys), endpoints_only=True)
        v["yticks"] = [[a, f"{a:g}"] for a in (3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 8) if v["ylim"][0] <= a <= v["ylim"][1] - 0.04 * (v["ylim"][1] - v["ylim"][0])]
        mk = sum(([line(pchip(x6, y), c), dots(pts(x6, y), c)] for y, c in zip(ys[:2], (RED, MTPC))), []) + ([dots(pts(x6, cy), GREY)] if key == "next" else [])
        figs["fig-mtp"]["panels"][j] = dict(**v, marks=mk, label=lab)
        print("fig-mtp", key, "NTP", np.round(ys[0], 3).tolist(), "NTP + MTP", np.round(ys[1], 3).tolist())

# Figure after fig-mtp (placeholder; owner popup 2026-10-14): MTP across the ladder. NTP and NTP + MTP (weight 0.3) at
# 130m (the fig-mtp runs), 300m, 520m and 1_2b (branch pls-mtp, pls_mtp_ladder_submit.sh); post-hoc probes on each final
# checkpoint (launcher PROBE_FROM). Panels: the main model's held-out next-token loss by size; the change MTP - NTP of
# every layer's next-token probe and of its probe for the token after next, by relative depth, one line per size.
MTPL_STATUS = QUEUED
LADDER = (("130m", {"en": "130m", "zh": "130m"}), ("300m", {"en": "300m", "zh": "300m"}), ("520m", {"en": "520m", "zh": "520m"}), ("1_2b", {"en": "1.2b", "zh": "1.2b"}))
LADC = ("#8AB4F8", "#4285F4", "#1967D2", "#174EA6")   # GM2 blue 300, 500, 700, 900: smaller to larger
DEPTH_X = {"en": "Relative depth (layer / number of layers)", "zh": "相对深度（层号 / 层数）"}
DELTA = {"en": "Probe loss, NTP + MTP minus NTP (nats)", "zh": "探针损失，NTP + MTP 减 NTP（nats）"}
dview = dict(xlim=[0.0, 1.04], ylim=[-0.3, 0.15], xticks=lin_ticks(0, 1, 0.25, "{:g}"), yticks=[[t, "0" if t == 0 else f"{t:+.1f}".replace("-", "\u2212")] for t, _ in lin_ticks(-0.3, 0.1, 0.1)], marks=[], empty=MTPL_STATUS)
def mtpl_marks():
    """fig-mtpl from mtpl.json (fetch_mtpl.py): the main model's held-out next-token loss by parameter count, a line and
    dots per objective over the sizes that have both arms finished; the status word while none has."""
    M = json.load(open(HERE / "mtpl.json")) if (HERE / "mtpl.json").exists() else {}
    done = sorted((d for d in M.values() if "ntp" in d and "mtp" in d), key=lambda d: d["params"])
    if not done:
        return dict(marks=[], empty=MTPL_STATUS)
    marks = []
    for arm, c in (("ntp", RED), ("mtp", MTPC)):
        p = pts([d["params"] for d in done], [d[arm] for d in done])
        marks += [line(p, c), dots(p, c)]
    print("fig-mtpl", {f"{d['params']:.3g}": (d["ntp"], d["mtp"]) for d in done})
    return dict(marks=marks)
figs["fig-mtpl"] = dict(title={"en": "Next-token loss by model size, NTP and MTP", "zh": "按模型尺度的下一个词损失，NTP 与 MTP"},
                        legend=[[{"en": "NTP", "zh": "NTP"}, RED, "line"], [{"en": "NTP + MTP", "zh": "NTP + MTP"}, MTPC, "line"]],
                        quantity=CE_EVAL, xlabel={"en": "Parameters", "zh": "参数量"}, height=220,
                        panels=[dict(xlog=True, ylog=True, yoff=YOFF, xlim=[8e7, 2e9], ylim=[3.3, 4.1], xticks=[[1e8, "100M"], [3e8, "300M"], [1e9, "1B"]],
                                     yticks=[[a, f"{a:g}"] for a in (3.4, 3.6, 3.8, 4.0)], **mtpl_marks())])
def depth_panel(target):
    """fig-mtpl-depth from mtpl.json (fetch_mtpl.py): MTP minus NTP of each layer's post-hoc probe loss (target "next" or
    "after") at relative depth k / L, a line and dots per size whose two probe runs have finished; the status word while
    none has."""
    M = json.load(open(HERE / "mtpl.json")) if (HERE / "mtpl.json").exists() else {}
    marks = []
    for (key, _), c in zip(LADDER, LADC):
        pr = M.get(key, {}).get("probes")
        if not pr:
            continue
        L = len(pr["ntp"][target])
        p = [[round((k + 1) / L, 4), round(a - b, 4)] for k, (a, b) in enumerate(zip(pr["mtp"][target], pr["ntp"][target]))]
        marks += [line(p, c), dots(p, c)]
        print("fig-mtpl-depth", target, key, [d for _, d in p])
    d = dict(dview, marks=marks)
    if marks:
        d.pop("empty")
    return d
figs["fig-mtpl-depth"] = dict(title={"en": "Change in probe loss by relative depth, MTP minus NTP", "zh": "按相对深度的探针损失变化，MTP 减 NTP"},
                              legend=[[lab, c, "line"] for (_, lab), c in zip(LADDER, LADC)],
                              quantity=DELTA, xlabel=DEPTH_X, height=200, cols=1,
                              panels=[dict(**depth_panel("next"), label={"en": "Probes for the next token", "zh": "预测下一个词的探针"}),
                                      dict(**depth_panel("after"), label={"en": "Probes for the token after next", "zh": "预测下下个词的探针"})])

figs["fig-headft"] = dict(title={"en": "Per-layer loss before and after retraining the heads, 300m", "zh": "重训读出头前后的逐层损失，300m"},
                          legend=[[{"en": "Separate heads", "zh": "独立头"}, BLUE3, "line"], [{"en": "Separate heads, heads retrained", "zh": "独立头，重训读出头"}, BLUE7, "line"],
                                  [{"en": "Probes only", "zh": "只加探针"}, RED3, "line"], [{"en": "Probes only, heads retrained", "zh": "只加探针，重训读出头"}, RED, "line"]],
                          quantity=CE_EVAL, xlabel=XL, height=320, panels=[dict(**v10, marks=marks10)])

# Last figure (placeholder; owner popups 2026-10-17): the logit lens of Kimi K3 (93 layers, Block AttnRes in blocks of 12),
# k3_lens.py on 64 wikitext-103 test sequences of 1,024 tokens: per layer, next-token loss through the final norm and lm_head
# of three readings (the next layer's input mix, the output mix had the model stopped there, the plain sum of the blocks);
# lower panel the output mix's weights on the embedding and each block. k3lens.json fills it.
K3_STATUS = QUEUED
K3T = [[k, str(k)] for k in (1, 12, 24, 36, 48, 60, 72, 84, 93)]
figs["fig-k3lens"] = dict(title={"en": "Logit lens of Kimi K3 by layer", "zh": "Kimi K3 按层的 logit lens"},
                          legend=[[{"en": "Next layer's input", "zh": "下一层的输入"}, BLUE7, "line"], [{"en": "Output mix, stopped here", "zh": "在此停下的输出混合"}, RED, "line"],
                                  [{"en": "Plain sum of the blocks", "zh": "各块直接相加"}, GREY, "line"]],
                          quantity=CE, xlabel={"en": "Layer", "zh": "层"}, height=220, cols=1,
                          panels=[dict(xlog=False, ylog=True, yoff=0.0, xlim=[0.5, 93.5], ylim=[1.0, 14.0], xticks=K3T, yticks=[[a, f"{a:g}"] for a in (1, 2, 4, 8)],
                                       marks=[], empty=K3_STATUS, label={"en": "Next-token loss", "zh": "下一个词的损失"}),
                                  dict(xlog=False, ylog=False, xlim=[0.5, 93.5], ylim=[0, 1.05], xticks=K3T, yticks=[[a, f"{a:g}"] for a in (0, 0.25, 0.5, 0.75, 1)],
                                       marks=[], empty=K3_STATUS, label={"en": "Output-mix weight on each source", "zh": "输出混合在各来源上的权重"})])

# Layer index on the x axis of every by-layer figure (owner, 2026-10-17: "所有横轴应该改成层数"). Within one model the backbone
# compute of a readout is proportional to its layer (x = 6 N_layer k D), so dividing by one layer's compute gives k and the
# fitted curves keep their shape; each panel divides by its own model's layer (Figure 7: the design's own per-layer size).
# The training-curve figures (Figures 1, 2, 4 and 8) keep their x axes: those run over training, not over layers.
LAYER_X = {"en": "Layer", "zh": "层"}
U130, U300 = 6.0 * NL * DTOT, 6.0 * N300 * D300
def to_layers(fid, units, n):
    f = figs[fid]
    ticks = [[k, str(k)] for k in (range(1, n + 1) if n <= 12 else (1, 12, 24, 36, 48))]
    for p, u in zip(f["panels"], units):
        old = p["xlim"]
        for m in p.get("marks", []):
            if "pts" not in m:
                continue
            q = m["pts"]
            if m["type"] == "line" and len(q) == 2 and abs(q[0][0] - old[0]) < 1e-9 * old[1] and abs(q[1][0] - old[1]) < 1e-9 * old[1]:
                m["pts"] = [[0.5, q[0][1]], [n + 0.5, q[1][1]]]   # a reference line across the panel stays across it
                continue
            m["pts"] = [[round(x / u, 4), y] for x, y in q]
            if m["type"] == "dots":
                off = max(abs(x - round(x)) for x, _ in m["pts"])
                if off > 1e-3:
                    raise RuntimeError(f"{fid}: a readout point is {off:.4f} layers off an integer layer (wrong unit)")
        p.update(xlog=False, xlim=[0.5, n + 0.5], xticks=ticks)
    f["xlabel"] = LAYER_X
for fid in ("fig-heads", "fig-fix", "fig-mix", "fig-late", "fig-mtp"):
    to_layers(fid, [U130] * len(figs[fid]["panels"]), 6)
to_layers("fig-heads-300m", [U300], 12)
to_layers("fig-headft", [U300], 12)
to_layers("fig-d48", [x48("")[0]], 48)
to_layers("fig-d48-arch", [x48(k)[0] for k, _ in ARCH], 48)
to_layers("fig-lens", [U130, x48("")[0]], 6)
figs["fig-lens"]["panels"][1].update(xlim=[0.5, 48.5], xticks=[[k, str(k)] for k in (1, 12, 24, 36, 48)])

OUT.write_text("/* Generated by build_figures_data.py (pls branch, docs/della/pls_archive/blog); regenerate rather than hand-edit. */\nwindow.PLC_FIGS = " + json.dumps(figs, ensure_ascii=False, separators=(",", ":")) + ";\n")
print("N per layer", NL, "| x of layer 2..6 at the end", [f"{v:.3g}" for v in x6], "| wrote", OUT, OUT.stat().st_size, "bytes")
