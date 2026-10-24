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
figs["fig-modes"] = dict(title={"en": "Last-layer training loss of the four setups, 130m", "zh": "四种设置的最后一层训练损失，130m"},
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
figs["fig-heads"] = dict(title={"en": "Final training loss of each setup, 130m", "zh": "各设置的最终训练损失，130m"},
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
figs["fig-heads-300m"] = dict(title={"en": "Final training loss of each setup, 300m", "zh": "各设置的最终训练损失，300m"},
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
figs["fig-d48"] = dict(title={"en": "Final training loss, 48 layers of width 512", "zh": "最终训练损失，48 层、宽 512"},
                       legend=LEG48 + ([[{"en": "Ordinary models by depth", "zh": "各深度的普通模型"}, GREY, "dot"]] if REF48 else []),
                       quantity=CE, xlabel=XL, height=320, panels=[p48])
figs["fig-d48-arch"] = dict(title={"en": "Final training loss under nine more designs, 48 layers", "zh": "另外九种结构的最终训练损失，48 层"},
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
figs["fig-d48-gap"] = dict(title={"en": "Separate heads minus probes for each design, 48 layers", "zh": "各结构的独立头减只加探针，48 层"},
                           legend=[[{"en": n, "zh": n}, DCOL[a], "line"] for a, n in drawn],
                           quantity={"en": "Training loss difference (nats)", "zh": "训练损失之差（nats）"}, xlabel={"en": "Layer", "zh": "层"}, height=340,
                           panels=[dict(xlim=[-3, 49], ylim=[-2.5, 0.5], xticks=LX, yticks=lin_ticks(-2.5, 0.5, 0.5),
                                        marks=[line([[0, 0], [49, 0]], GREY, 1.0)] + [line(pts(k48, gaps[a]), DCOL[a], 1.8) for a, _ in drawn])])
figs["fig-d48-abs"] = dict(title={"en": "Final training loss of each design, 48 layers", "zh": "各结构的最终训练损失，48 层"},
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
figs["fig-hc-write"] = dict(title={"en": "Last-layer cost of separate heads for each design, 48 layers", "zh": "各结构中独立头的最后一层代价，48 层"},
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
figs["fig-attnres-seed"] = dict(title={"en": "AttnRes (Full) layer-48 training loss, two seeds, 48 layers", "zh": "AttnRes (Full) 第 48 层训练损失，两个种子，48 层"},
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
figs["fig-fix"] = dict(title={"en": "Layer-loss weight and gradient reach, 130m", "zh": "层损失权重与梯度范围，130m"},
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
figs["fig-mix"] = dict(title={"en": "Weight shape and gradient surgery, 130m", "zh": "权重形状与梯度手术，130m"},
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
figs["fig-late"] = dict(title={"en": "Separate heads switched on late, 130m", "zh": "晚启用的独立头，130m"},
                        legend=legend11, quantity=CE, xlabel=XL, height=320, panels=[p11])

# Figure after fig-late (placeholder; owner popup 2026-10-16): ten more attempts at the last-layer cost, 130m, branch
# pls-auto10 (pls_auto10_pair.sbatch). Per attempt, the final training loss (mean of the last 50 logged steps, as in
# Figure 3) of layer 1 and of layer 6 minus the ordinary model of the same depth (grey dots of Figure 3), beside separate
# heads at weight 1. auto10.json (fetch_auto10.py, once runs finish): {key: [final loss of layers 1..6]}; a key without
# data shows its status word under its bar.
AUTO10 = [("sep", {"en": "Separate heads", "zh": "独立头"}, None), ("os", {"en": "1 Own clip", "zh": "1 单独裁剪"}, None),
          ("mix", {"en": "2 Mix 0.2", "zh": "2 混合 0.2"}, None), ("kd", {"en": "3 Distill", "zh": "3 自蒸馏"}, None),
          ("conf", {"en": "4 Conf. weight", "zh": "4 置信度"}, None), ("ad", {"en": "5 Adapter", "zh": "5 适配器"}, None),
          ("mlpo", {"en": "6 MLP only", "zh": "6 只更新 MLP"}, None), ("first", {"en": "7 First 256", "zh": "7 前 256"}, None),
          ("randl", {"en": "8 One layer", "zh": "8 随机单层"}, None), ("hc", {"en": "9 HC", "zh": "9 HC"}, None),
          ("attnres", {"en": "10 AttnRes", "zh": "10 AttnRes"}, None),
          # 11, 12 (owner popups 2026-10-18, no extra parameters): only the readouts of layers 1-3 / 1-2 reach the backbone
          ("upto3", {"en": "11 Layers 1-3", "zh": "11 只监督第 1-3 层"}, None), ("upto2", {"en": "12 Layers 1-2", "zh": "12 只监督第 1-2 层"}, None),
          # 13, 14 (owner popup 2026-10-18): only layer 1's readout; layers 1-2 again with init seed 1 (data order unchanged)
          ("upto1", {"en": "13 Layer 1", "zh": "13 只监督第 1 层"}, None), ("upto2s1", {"en": "14 Seed 1", "zh": "14 种子 1"}, None)]
A10 = json.load(open(HERE / "auto10.json")) if (HERE / "auto10.json").exists() else {}
A10["sep"] = [float(v) for v in Z["-pls1-sep/final"]]
gap10 = {k: [float(A10[k][j] - cy[j]) for j in range(6)] for k in A10}
cats10 = [lab for k, lab, st in AUTO10]
# the status of bars without data goes in the panel name ("Layer 1 · 3, 13, 14 queued"): a status word under each of two
# neighbouring bars is wider than its slot and runs into the next one
wait10 = {}
for k, lab, st in AUTO10:
    if k not in gap10:
        wait10.setdefault((st["en"].lower(), st["zh"]), []).append(lab["en"].split()[0])
wait10_en = "; ".join(", ".join(ns) + " " + en for (en, zh), ns in wait10.items())
wait10_zh = "；".join("、".join(ns) + " " + zh for (en, zh), ns in wait10.items())


def auto10_panel(j, label, lo, hi, step):
    vals = [round(gap10[k][j], 4) if k in gap10 else None for k, _, _ in AUTO10]
    assert all(v is None or lo <= v for v in vals), f"fig-auto10 layer {j + 1}: a bar below {lo}: {vals}"
    # a bar above the axis is drawn to its top and its label carries the value (3 Distill at layer 6: 0.70 nats)
    cats = [{"en": c["en"] + f" ({v:.2f}, off scale)", "zh": c["zh"] + f"（{v:.2f}，超出坐标）"} if v is not None and v > hi else c for c, v in zip(cats10, vals)]
    vals = [min(v, 0.98 * hi) if v is not None else None for v in vals]
    if wait10:
        label = {"en": label["en"] + " · " + wait10_en, "zh": label["zh"] + " · " + wait10_zh}
    return dict(xlim=[-1.3, len(AUTO10) - 0.4], ylim=[lo, hi], yticks=lin_ticks(lo, hi, step, "{:.2f}"), cats=cats, label=label,
                marks=[dict(type="bars", color=BLUE7 if k == "sep" else "#8AB4F8", vals=[v if i == n else None for n, v in enumerate(vals)]) for i, (k, _, _) in enumerate(AUTO10)])


figs["fig-auto10"] = dict(title={"en": "More attempts at the last-layer cost, 130m", "zh": "降低最后一层代价的更多尝试，130m"},
                          legend=[[{"en": "Separate heads, weight 1", "zh": "独立头，权重 1"}, BLUE7, "box"], [{"en": "One attempt", "zh": "一种尝试"}, "#8AB4F8", "box"]],
                          quantity={"en": "Final training loss minus the ordinary model of the same depth (nats)", "zh": "最终训练损失减去同深度普通模型（nats）"},
                          height=200, cols=1, panels=[auto10_panel(0, {"en": "Layer 1", "zh": "第 1 层"}, 0, 1.2, 0.2), auto10_panel(5, {"en": "Layer 6, the output", "zh": "第 6 层，即输出"}, 0, 0.3, 0.05)])
print("fig-auto10", {k: [round(g, 3) for g in (v[0], v[5])] for k, v in gap10.items()})

# Figure after fig-auto10 (placeholder; owner popups 2026-10-17): the logit lens of every finished model. Per checkpoint,
# held-out (Paloma) loss by layer through the layer's own readout (its separate head or probe: eval/L{k}/loss of the source
# run) and through the model's final norm and lm_head (the logit lens: launcher LENS_FROM, an eval of the frozen
# checkpoint). Panels: 130m (6 layers) and 48 layers of width 512 (Sandwich-LN); own readouts in the dark tone, the
# lens in the light tone of the same colour. lens.json (fetch_lens.py) fills it.
LENS_STATUS = QUEUED
LENS = json.load(open(HERE / "lens.json")) if (HERE / "lens.json").exists() else {}


def lens_panel(key, xs, label):
    """One panel of fig-lens: per setup with both runs finished, its own readouts (dark) and the logit lens (light) by
    layer; the status word while none has."""
    P_ = LENS.get(key, {})
    if not P_:
        return dict(**empty_view(list(xs), 3.6, 9, LENS_STATUS, ys=(4, 5, 6, 7, 8)), label=label)
    marks, ys = [], []
    for setup, dark, light in (("probes", RED, "#F28B82"), ("sep", BLUE7, "#8AB4F8")):
        if setup not in P_:
            continue
        for kind, c in (("lens", light), ("own", dark)):
            y = np.array(P_[setup][kind], float); assert len(y) == len(xs), (key, setup, kind, len(y), len(xs))
            marks.append(line(pts(xs, y), c, 1.4))
            if len(xs) <= 12:
                marks.append(dots(pts(xs, y), c))
            ys.append(y)
    yy = np.concatenate(ys)
    v = view(xs, yy, endpoints_only=True)
    v["ylim"] = [0.97 * float(yy.min()), 1.45 * float(yy.max())]   # log scale: the top band holds the panel name clear of layer 1
    v["yticks"] = [[a, f"{a:g}"] for a in (4, 5, 6, 8, 12) if v["ylim"][0] <= a <= v["ylim"][1]]
    print("fig-lens", key, {k: {kk: [vv[0], vv[-1]] for kk, vv in d.items()} for k, d in P_.items()})
    return dict(**v, marks=marks, label=label)
figs["fig-lens"] = dict(title={"en": "Logit lens and per-layer readouts", "zh": "logit lens 与每层读出"},
                        legend=[[{"en": "Ordinary model, probes", "zh": "普通模型，探针"}, RED, "line"], [{"en": "Ordinary model, logit lens", "zh": "普通模型，logit lens"}, "#F28B82", "line"],
                                [{"en": "Separate heads, own heads", "zh": "独立头，自己的头"}, BLUE7, "line"], [{"en": "Separate heads, logit lens", "zh": "独立头，logit lens"}, "#8AB4F8", "line"]],
                        quantity={"en": "Held-out cross-entropy (nats, log scale)", "zh": "留出数据上的交叉熵（nats，对数坐标）"}, xlabel=XL, height=220, cols=1,
                        panels=[lens_panel("130m", x6, {"en": "6 layers, 130m", "zh": "6 层，130m"}),
                                lens_panel("300m", np.asarray(x300), {"en": "12 layers, 300m", "zh": "12 层，300m"}),
                                lens_panel("d48", np.asarray(x48("")), {"en": "48 layers, Sandwich-LN", "zh": "48 层，Sandwich-LN"})])
# Third panel: the logit lens of ordinary models of 12, 24, 36 layers (GM2 grey 400, 600, 800: context) beside the 48-layer
# ordinary model (the probes' backbone, light red as in the other panels); lens.json "depth" (fetch_lens.py)
DEPTH_COL = (("d12", "#BDC1C6", {"en": "Ordinary model of 12 layers, logit lens", "zh": "12 层普通模型，logit lens"}),
             ("d24", "#80868B", {"en": "Ordinary model of 24 layers, logit lens", "zh": "24 层普通模型，logit lens"}),
             ("d36", "#3C4043", {"en": "Ordinary model of 36 layers, logit lens", "zh": "36 层普通模型，logit lens"}))
def depth_panel():
    lab = {"en": "Ordinary models by depth, Sandwich-LN", "zh": "按深度的普通模型，Sandwich-LN"}
    D_, xs = LENS.get("depth", {}), np.asarray(x48(""))
    series = [(D_[k]["lens"], c) for k, c, _ in DEPTH_COL if k in D_]
    if "probes" in LENS.get("d48", {}):
        series.append((LENS["d48"]["probes"]["lens"], "#F28B82"))
    if not series:
        return dict(**empty_view(list(xs), 3.6, 9, LENS_STATUS, ys=(4, 5, 6, 8)), label=lab)
    marks = [line(pts(xs[:len(y)], np.array(y, float)), c, 1.4) for y, c in series]
    yy = np.concatenate([np.array(y, float) for y, _ in series])
    v = view(xs, yy, endpoints_only=True)
    v["ylim"] = [0.97 * float(yy.min()), 1.45 * float(yy.max())]
    v["yticks"] = [[a, f"{a:g}"] for a in (4, 5, 6, 8, 12) if v["ylim"][0] <= a <= v["ylim"][1]]
    return dict(**v, marks=marks, label=lab)
figs["fig-lens"]["panels"].append(depth_panel())
figs["fig-lens"]["legend"] += [[n, c, "line"] for _, c, n in DEPTH_COL]
# Figure 17 (owner popup 2026-10-18): fig-lens for the other nine 48-layer designs, one column as fig-d48-arch; lens.json
# "d48-<arch>"; a design whose lens runs have not finished shows the status word (AttnRes, AttnRes block, MoDA: lens-48c)
figs["fig-lens-arch"] = dict(title={"en": "Logit lens and per-layer readouts, nine more designs", "zh": "logit lens 与每层读出，另外九种结构"},
                             legend=figs["fig-lens"]["legend"][:4], quantity=figs["fig-lens"]["quantity"], xlabel=XL, height=200, cols=1,
                             panels=[lens_panel(f"d48-{k}", np.asarray(x48(k)), {"en": n, "zh": n}) for k, n in ARCH])

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
figs["fig-mtp"] = dict(title={"en": "Final probe loss, NTP and MTP, 130m", "zh": "最终探针损失，NTP 与 MTP，130m"},
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
dview = dict(xlim=[0.0, 1.04], ylim=[-0.4, 0.2], xticks=lin_ticks(0, 1, 0.25, "{:g}"), yticks=[[t, "0" if t == 0 else f"{t:+.1f}".replace("-", "\u2212")] for t, _ in lin_ticks(-0.4, 0.2, 0.1)], marks=[], empty=MTPL_STATUS)
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
    assert all(3.0 <= d[a_] <= 4.1 for d in done for a_ in ("ntp", "mtp")), "fig-mtpl: a point outside the y range"
    print("fig-mtpl", {f"{d['params']:.3g}": (d["ntp"], d["mtp"]) for d in done})
    return dict(marks=marks)
figs["fig-mtpl"] = dict(title={"en": "Next-token loss, NTP and MTP, four sizes", "zh": "下一个词损失，NTP 与 MTP，四个尺度"},
                        legend=[[{"en": "NTP", "zh": "NTP"}, RED, "line"], [{"en": "NTP + MTP", "zh": "NTP + MTP"}, MTPC, "line"]],
                        quantity=CE_EVAL, xlabel={"en": "Parameters", "zh": "参数量"}, height=220,
                        panels=[dict(xlog=True, ylog=True, yoff=YOFF, xlim=[8e7, 2e9], ylim=[3.0, 4.1], xticks=[[1e8, "100M"], [3e8, "300M"], [1e9, "1B"]],
                                     yticks=[[a, f"{a:g}"] for a in (3.2, 3.4, 3.6, 3.8, 4.0)], **mtpl_marks())])
MTPL_PROBES = json.load(open(HERE / "mtpl.json")) if (HERE / "mtpl.json").exists() else {}
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
figs["fig-mtpl-depth"] = dict(title={"en": "Change in probe loss, MTP minus NTP", "zh": "探针损失的变化，MTP 减 NTP"},
                              # sizes with both probe runs in mtpl.json; the 520m probe runs fail fetch_mtpl.py's frozen-head check (2026-10-21)
                              legend=[[lab, c, "line"] for (key, lab), c in zip(LADDER, LADC) if MTPL_PROBES.get(key, {}).get("probes")],
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
K3 = json.load(open(HERE / "k3lens.json"))["layers"] if (HERE / "k3lens.json").exists() else None   # k3-lens 15736820, merged.json
K3T = [[k, str(k)] for k in (1, 12, 24, 36, 48, 60, 72, 84, 93)]
K3P = [dict(xlog=False, ylog=True, yoff=0.0, xlim=[0.5, 93.5], ylim=[0.3, 45.0], xticks=K3T, yticks=[[a, f"{a:g}"] for a in (0.5, 1, 2, 4, 8, 16)],
            marks=[], label={"en": "Next-token loss", "zh": "下一个词的损失"}),
       dict(xlog=False, ylog=False, xlim=[0.5, 93.5], ylim=[0, 1.25], xticks=K3T, yticks=[[a, f"{a:g}"] for a in (0, 0.25, 0.5, 0.75, 1)],
            marks=[], label={"en": "Output-mix weight on each source", "zh": "输出混合在各来源上的权重"})]
if K3 is None:
    for q in K3P:
        q["empty"] = K3_STATUS
else:
    assert [r["layer"] for r in K3] == list(range(93)), "k3lens.json: layers 0..92 expected"
    xs = [r["layer"] + 1 for r in K3]
    for key, col in (("sum_ce", GREY), ("exit_ce", RED), ("next_ce", BLUE7)):
        K3P[0]["marks"].append(dict(type="line", color=col, pts=[[x, round(r[key], 4)] for x, r in zip(xs, K3)]))
    # exit_probs: [embedding, finished blocks..., the current block's partial sum]; the finished blocks are summed
    emb = [r["exit_probs"][0] for r in K3]; cur = [r["exit_probs"][-1] for r in K3]
    fin = [1.0 - a - b for a, b in zip(emb, cur)]
    assert all(-1e-3 <= f <= 1 + 1e-3 for f in fin), "k3lens.json: exit_probs do not sum to 1"
    for vals, col in ((emb, GREEN), (fin, PURPLE), (cur, YELLOW9)):
        K3P[1]["marks"].append(dict(type="line", color=col, pts=[[x, round(max(v, 0.0), 4)] for x, v in zip(xs, vals)]))
    print("fig-k3lens next/exit/sum at 93", [round(K3[-1][k], 3) for k in ("next_ce", "exit_ce", "sum_ce")], "top-1", round(K3[-1]["next_acc"], 3))
figs["fig-k3lens"] = dict(title={"en": "Logit lens of Kimi K3", "zh": "Kimi K3 的 logit lens"},
                          legend=[[{"en": "Next layer's input", "zh": "下一层的输入"}, BLUE7, "line"], [{"en": "Output mix, stopped here", "zh": "在此停下的输出混合"}, RED, "line"],
                                  [{"en": "Plain sum of the blocks", "zh": "各块直接相加"}, GREY, "line"],
                                  [{"en": "Weight: embedding", "zh": "权重：embedding"}, GREEN, "line"], [{"en": "Weight: finished blocks", "zh": "权重：已完成的块"}, PURPLE, "line"],
                                  [{"en": "Weight: current block", "zh": "权重：当前块"}, YELLOW9, "line"]],
                          quantity=CE, xlabel={"en": "Layer", "zh": "层"}, height=220, cols=1, panels=K3P)

# AttnRes check-up (owner popup 2026-10-19 "按方案做", "越全面越好"; separate-heads models added the same day): the same
# measurements on Kimi K3 (k3_checkup.py, Paloma c4_en re-tokenized, 64 x 1,024 tokens) and on our 48-layer AttnRes (Block)
# and AttnRes (Full), ordinary and separate heads (attnres_checkup.py, CPU, Paloma c4_en, 16 x 1,024 tokens). Data:
# checkup/<key>.json, one per model (schema in the two scripts); a model without its file shows its status.
CK_STATUS = QUEUED
CK = [("k3", {"en": "Kimi K3 (93 layers, blocks of 12)", "zh": "Kimi K3（93 层，每块 12 层）"}),
      ("block", {"en": "Ours, AttnRes (Block), ordinary model", "zh": "我们的 AttnRes (Block)，普通模型"}),
      ("block-sep", {"en": "Ours, AttnRes (Block), separate heads", "zh": "我们的 AttnRes (Block)，独立头"}),
      ("full", {"en": "Ours, AttnRes (Full), ordinary model", "zh": "我们的 AttnRes (Full)，普通模型"}),
      ("full-sep", {"en": "Ours, AttnRes (Full), separate heads", "zh": "我们的 AttnRes (Full)，独立头"})]
CKD = {k: json.load(open(HERE / "checkup" / f"{k}.json")) for k, _ in CK if (HERE / "checkup" / f"{k}.json").exists()}
GRN, PUR, ORA, PINK = GREEN, PURPLE, YELLOW9, PINK7
GRN3, PUR3, ORA3, PINK3 = "#81C995", "#C58AF9", "#FDD663", "#F48FB1"


def ck_axis(k):
    n = 93 if k == "k3" else 48
    t = [1, 12, 24, 36, 48, 60, 72, 84, 93] if n == 93 else [1, 6, 12, 18, 24, 30, 36, 42, 48]
    return n, [[a, str(a)] for a in t]


def ck_groups(d, r, key):
    """[embedding, older blocks, the last finished block, the current block] from a probs list (the grouped list for Full).
    Finished blocks before layer k's attention and MLP: k // B. After layer k (the head's mix): our models close a block
    after its last MLP, (k + 1) // B; K3 closes it when the next block's first attention starts, k // B. The current
    block is the last entry when the list is longer than the embedding plus the finished blocks."""
    p = r.get(key + "_grouped", r[key])
    bl, k = d.get("block_layers") or 6, r["layer"]
    nfin = (k + 1) // bl if key == "exit_probs" and d.get("model") != "Kimi-K3" else k // bl
    cur = len(p) > 1 + nfin
    emb, rest = p[0], p[1:]
    current = rest[-1] if cur else 0.0
    fin = rest[:-1] if cur else rest
    last = fin[-1] if fin else 0.0
    older = sum(fin[:-1]) if len(fin) > 1 else 0.0
    return emb, older, last, current


def ck_panels(lo, hi, ylog, yt, fill):
    out = []
    for k, lab in CK:
        n, xt = ck_axis(k)
        q = dict(xlog=False, ylog=ylog, xlim=[0.5, n + 0.5], ylim=[lo, hi], xticks=xt, yticks=[[a, f"{a:g}"] for a in yt], marks=[], label=lab)
        if ylog:
            q["yoff"] = 0.0
        if k in CKD:
            q["marks"] = fill(CKD[k])
        else:
            q["empty"] = CK_STATUS
        out.append(q)
    return out


def line(d, f, col):
    return dict(type="line", color=col, pts=[[r["layer"] + 1, round(float(f(r)), 4)] for r in d["layers"]])


def ck_lens(d):
    m = [line(d, lambda r: r["sum_ce"], GREY), line(d, lambda r: r["exit_ce"], RED), line(d, lambda r: r["next_ce"], BLUE7)]
    if "own_ce" in d["layers"][0]:
        m.append(line(d, lambda r: r["own_ce"], GREEN))
    return m


def ck_weights(key):
    def fill(d):
        return [line(d, lambda r, j=j: ck_groups(d, r, key)[j], c) for j, c in enumerate((GRN, PUR, PINK, ORA))]
    return fill


def ck_read_of(key):
    def fill(d):
        return [line(d, lambda r, j=j: ck_groups(d, r, key)[j], c) for j, c in enumerate((GRN, PUR, PINK, ORA))]
    return fill


def ck_query(d):
    """Sublayer queries in order (attention of layer 1, its MLP, ..., then the head): lengths relative to the 90th
    percentile of the model's sublayer query lengths (cut at 1.2), and cosines with the head's query."""
    q = d["queries"]; n = len(q["norm"]) - 1
    p90 = float(np.percentile(q["norm"][:n], 90)) or 1.0
    out = []
    for off, cn, cc in ((0, BLUE7, RED), (1, BLUE3, RED3)):
        idx = range(off, n, 2)
        out.append(dict(type="line", color=cn, pts=[[i // 2 + 1, round(min(q["norm"][i] / p90, 1.2), 4)] for i in idx]))
        out.append(dict(type="line", color=cc, pts=[[i // 2 + 1, round(q["cos_out"][i], 4)] for i in idx]))
    return out


def ck_write(d):
    """Write sizes relative to the 90th percentile of the model's write sizes, cut at 1.2 (AttnRes (Full)'s last attention
    write is ~45 x the others); cosines with the output query as they are."""
    p90 = float(np.percentile([v for r in d["layers"] for v in (r["attn_write_rms"], r["mlp_write_rms"])], 90)) or 1.0
    return [line(d, lambda r: min(r["attn_write_rms"] / p90, 1.2), BLUE7), line(d, lambda r: min(r["mlp_write_rms"] / p90, 1.2), BLUE3),
            line(d, lambda r: r["attn_write_cos_out"], RED), line(d, lambda r: r["mlp_write_cos_out"], RED3)]


W4 = [[{"en": "Embedding", "zh": "embedding"}, GRN, "line"], [{"en": "Older blocks", "zh": "更早的块"}, PUR, "line"],
      [{"en": "Last finished block", "zh": "刚完成的块"}, PINK, "line"], [{"en": "Current block", "zh": "当前块"}, ORA, "line"]]
figs["fig-ck-lens"] = dict(title={"en": "AttnRes check-up: logit lens", "zh": "AttnRes 体检：logit lens"},
                           legend=[[{"en": "Next layer's input", "zh": "下一层的输入"}, BLUE7, "line"], [{"en": "Output mix, stopped here", "zh": "在此停下的输出混合"}, RED, "line"],
                                   [{"en": "Plain sum of the sources", "zh": "各来源直接相加"}, GREY, "line"], [{"en": "The layer's own readout (probe or separate head)", "zh": "该层自己的读出头（探针或独立头）"}, GREEN, "line"]],
                           quantity=CE, xlabel={"en": "Layer", "zh": "层"}, height=200, cols=1, panels=ck_panels(1.5, 40.0, True, (2, 4, 8, 16), ck_lens))
figs["fig-ck-mix"] = dict(title={"en": "AttnRes check-up: output-mix weights", "zh": "AttnRes 体检：输出混合权重"}, legend=W4,
                          quantity={"en": "Weight (mean over tokens)", "zh": "权重（token 平均）"}, xlabel={"en": "Layer", "zh": "层"}, height=170, cols=1,
                          panels=ck_panels(0, 1.25, False, (0, 0.25, 0.5, 0.75, 1), ck_weights("exit_probs")))
figs["fig-ck-read"] = dict(title={"en": "AttnRes check-up: where each layer's attention reads from", "zh": "AttnRes 体检：每层的注意力从哪里读"}, legend=W4,
                           quantity={"en": "Weight of the attention input (mean over tokens)", "zh": "注意力输入的权重（token 平均）"},
                           xlabel={"en": "Layer", "zh": "层"}, height=170, cols=1, panels=ck_panels(0, 1.25, False, (0, 0.25, 0.5, 0.75, 1), ck_read_of("attn_in_probs")))
figs["fig-ck-read-mlp"] = dict(title={"en": "AttnRes check-up: where each layer's MLP reads from", "zh": "AttnRes 体检：每层的 MLP 从哪里读"}, legend=W4,
                               quantity={"en": "Weight of the MLP input (mean over tokens)", "zh": "MLP 输入的权重（token 平均）"},
                               xlabel={"en": "Layer", "zh": "层"}, height=170, cols=1, panels=ck_panels(0, 1.25, False, (0, 0.25, 0.5, 0.75, 1), ck_read_of("mlp_in_probs")))
figs["fig-ck-query"] = dict(title={"en": "AttnRes check-up: the learned queries", "zh": "AttnRes 体检：学到的查询向量"},
                            legend=[[{"en": "Attention query length / 90th percentile", "zh": "注意力查询长度 / 第 90 百分位"}, BLUE7, "line"], [{"en": "MLP query length / 90th percentile", "zh": "MLP 查询长度 / 第 90 百分位"}, BLUE3, "line"], [{"en": "Attention query · output query (cosine)", "zh": "注意力查询与输出查询的余弦"}, RED, "line"], [{"en": "MLP query · output query (cosine)", "zh": "MLP 查询与输出查询的余弦"}, RED3, "line"]],
                            quantity={"en": "Value", "zh": "数值"}, xlabel={"en": "Layer", "zh": "层"}, height=170, cols=1, panels=ck_panels(-0.5, 1.25, False, (-0.5, 0, 0.5, 1), ck_query))
figs["fig-ck-write"] = dict(title={"en": "AttnRes check-up: what each sublayer writes", "zh": "AttnRes 体检：每个子层写入了什么"},
                            legend=[[{"en": "Attention write size / 90th percentile", "zh": "注意力写入大小 / 第 90 百分位"}, BLUE7, "line"], [{"en": "MLP write size / 90th percentile", "zh": "MLP 写入大小 / 第 90 百分位"}, BLUE3, "line"],
                                    [{"en": "Attention write · output query (cosine)", "zh": "注意力写入与输出查询的余弦"}, RED, "line"], [{"en": "MLP write · output query (cosine)", "zh": "MLP 写入与输出查询的余弦"}, RED3, "line"]],
                            quantity={"en": "Value", "zh": "数值"}, xlabel={"en": "Layer", "zh": "层"}, height=170, cols=1, panels=ck_panels(-0.5, 1.25, False, (-0.5, 0, 0.5, 1), ck_write))
ABL = [("uniform_all", {"en": "Uniform", "zh": "均匀"}), ("sum_all", {"en": "Plain sum", "zh": "直接相加"}), ("no_embedding", {"en": "No embedding", "zh": "去掉 embedding"}),
       ("recent_only", {"en": "Recent only", "zh": "只读最近"})] + [(f"uniform_block_{b}", {"en": f"Block {b} uniform", "zh": f"第 {b} 块均匀"}) for b in range(1, 9)]
abl_panels = []
for k, lab in CK:
    q = dict(xlim=[-0.7, len(ABL) - 0.3], ylim=[0, 8], yticks=lin_ticks(0, 8, 2, "{:.0f}"), cats=[c for _, c in ABL], label=lab, marks=[])
    if k in CKD:
        a = CKD[k]["ablations"]; b = a["baseline"]["ce"]
        vals = [round(a[n]["ce"] - b, 4) if n in a else None for n, _ in ABL]
        q["marks"] = [dict(type="bars", color=BLUE3, vals=[min(max(v, 0), 7.9) if v is not None else None for v in vals])]
        q["cats"] = [{"en": c["en"] + (f" ({v:.1f})" if v is not None and v > 8 else ""), "zh": c["zh"] + (f"（{v:.1f}）" if v is not None and v > 8 else "")} for (_, c), v in zip(ABL, vals)]
    else:
        q["empty"] = CK_STATUS
    abl_panels.append(q)
figs["fig-ck-ablate"] = dict(title={"en": "AttnRes check-up: loss when the mixing weights are replaced", "zh": "AttnRes 体检：替换混合权重后的损失"},
                             legend=[[{"en": "Loss increase over the model's own weights", "zh": "相对模型自己权重的损失增加"}, BLUE3, "box"]],
                             quantity={"en": "Increase in next-token loss (nats)", "zh": "下一个词损失的增加（nats）"}, height=170, cols=1, panels=abl_panels)
print("fig-ck", sorted(CKD))

# DepthBench internal medicine (owner popup 2026-10-19 "按方案做"): the same read-only measures on all ten 48-layer designs,
# ordinary model (red) and separate heads (blue), Paloma C4 (depthmed.py; data depthmed/<arch>-<ord|sep>.json, schema in
# that script). A design without its files shows its status.
DM_STATUS = QUEUED
DM_ARCH = [("sandwich", "Sandwich-LN (baseline)")] + ARCH
DMD = {}
for a, _ in DM_ARCH:
    for st in ("ord", "sep"):
        f = HERE / "depthmed" / f"{a}-{st}.json"
        if f.exists():
            DMD[(a, st)] = json.load(open(f))
DMX = [[k, str(k)] for k in (1, 12, 24, 36, 48)]


def dm_fig(fid, title, quantity, lo, hi, yt, ylog, series, legend, height=150):
    """series: [(setup, key or function of a layer row, colour)]; one panel per design."""
    panels = []
    for a, name in DM_ARCH:
        q = dict(xlog=False, ylog=ylog, xlim=[0.5, 48.5], ylim=[lo, hi], xticks=DMX, yticks=[[t, f"{t:g}"] for t in yt], marks=[], label={"en": name, "zh": name})
        if ylog:
            q["yoff"] = 0.0
        have = False
        for st, key, col in series:
            d = DMD.get((a, st))
            if d is None:
                continue
            have = True
            f = key if callable(key) else (lambda r, key=key: r[key])
            pts = [[r["layer"], round(float(f(r)), 5)] for r in d["layers"] if f(r) is not None]
            q["marks"].append(dict(type="line", color=col, pts=pts))
        if not have:
            q["empty"] = DM_STATUS
        panels.append(q)
    figs[fid] = dict(title=title, legend=legend, quantity=quantity, xlabel={"en": "Layer", "zh": "层"}, height=height, cols=2, panels=panels)


DM_LEG = [[{"en": "Ordinary model", "zh": "普通模型"}, RED, "line"], [{"en": "Separate heads", "zh": "独立头"}, BLUE7, "line"]]
DM_LEG4 = [[{"en": "Ordinary, attention", "zh": "普通模型，注意力"}, RED, "line"], [{"en": "Ordinary, MLP", "zh": "普通模型，MLP"}, RED3, "line"],
           [{"en": "Separate heads, attention", "zh": "独立头，注意力"}, BLUE7, "line"], [{"en": "Separate heads, MLP", "zh": "独立头，MLP"}, BLUE3, "line"]]
dm_fig("fig-dm-skip", {"en": "Loss when one layer is deleted, 48 layers", "zh": "删掉一层后的损失，48 层"},
       {"en": "Increase in next-token loss (nats, log scale)", "zh": "下一个词损失的增加（nats，对数坐标）"}, 0.001, 300, (0.001, 0.01, 0.1, 1, 10), True,
       [(st, (lambda r, st=st, a=None: None), c) for st, c in ()], DM_LEG)
dm_fig("fig-dm-write", {"en": "Write size relative to the stream, 48 layers", "zh": "写入相对残差流的大小，48 层"},
       {"en": "RMS of the branch output / RMS of the state it read (log scale)", "zh": "子层输出 RMS / 它读到的状态 RMS（对数坐标）"}, 0.001, 8000, (0.001, 0.1, 10, 1000), True,
       [(st, (lambda r, k=k: max(r[k], 1e-3)), c) for st, k, c in (("ord", "attn_write_ratio", RED), ("ord", "mlp_write_ratio", RED3), ("sep", "attn_write_ratio", BLUE7), ("sep", "mlp_write_ratio", BLUE3))], DM_LEG4)
dm_fig("fig-dm-dla", {"en": "Direct contribution of each write to the correct token, 48 layers", "zh": "每次写入对正确词的直接贡献，48 层"},
       {"en": "Direct logit attribution (logit units)", "zh": "直接 logit 贡献（logit 单位）"}, -3, 5.5, (-2, 0, 2, 4), False,
       [("ord", "attn_dla_w", RED), ("ord", "mlp_dla_w", RED3), ("sep", "attn_dla_w", BLUE7), ("sep", "mlp_dla_w", BLUE3)], DM_LEG4)
dm_fig("fig-dm-lens", {"en": "Logit lens on the same text, 48 layers", "zh": "同一份文本上的 logit lens，48 层"}, CE, 2.5, 30, (3, 4, 6, 8, 12, 16), True,
       [("ord", "lens_ce", RED), ("sep", "lens_ce", BLUE7), ("ord", "own_ce", RED3), ("sep", "own_ce", BLUE3)],
       [[{"en": "Ordinary, logit lens", "zh": "普通模型，logit lens"}, RED, "line"], [{"en": "Separate heads, logit lens", "zh": "独立头，logit lens"}, BLUE7, "line"],
        [{"en": "Ordinary, own probe", "zh": "普通模型，自己的探针"}, RED3, "line"], [{"en": "Separate heads, own head", "zh": "独立头，自己的头"}, BLUE3, "line"]])
dm_fig("fig-dm-shape", {"en": "Shape of the per-layer state, 48 layers", "zh": "每层状态的形状，48 层"},
       {"en": "Effective rank over 2,048 tokens (log scale)", "zh": "2,048 个 token 上的有效秩（对数坐标）"}, 1, 5000, (1, 10, 100, 1000), True,
       [("ord", "eff_rank", RED), ("sep", "eff_rank", BLUE7)], DM_LEG)
# fig-dm-skip: the increase over the model's own loss, floored at 1e-3 for the log axis
for i, (a, _) in enumerate(DM_ARCH):
    q = figs["fig-dm-skip"]["panels"][i]
    for st, col in (("ord", RED), ("sep", BLUE7)):
        d = DMD.get((a, st))
        if d is not None:
            q.pop("empty", None)
            q["marks"].append(dict(type="line", color=col, pts=[[r["layer"], round(max(r.get("skip_ce_delete", r["skip_ce"]) - d["base_ce"], 1e-3), 5)] for r in d["layers"]]))   # the true deletion (depthmed.py --delete)
print("fig-dm", sorted(DMD))

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
to_layers("fig-lens", [U130, U300, x48("")[0], x48("")[0]], 6)
figs["fig-lens"]["panels"][1].update(xlim=[0.5, 12.5], xticks=[[k, str(k)] for k in range(1, 13)])
for _p in figs["fig-lens"]["panels"][2:]:
    _p.update(xlim=[0.5, 48.5], xticks=[[k, str(k)] for k in (1, 12, 24, 36, 48)])
to_layers("fig-lens-arch", [x48(k)[0] for k, _ in ARCH], 48)

# The Lean session's follow-ups (owner popups 2026-10-21; placeholders until the runs finish). L1 and L2: 48-layer ordinary
# models with probes, one new arm each (pls_arch2_fetchable.sbatch), beside the -vs-d48-g4 controls. L5: the 130m NTP
# probes again with their weights kept (pls-mtp, mtpl-probe-130m-ntp-keep).
ARCH2_STATUS = QUEUED
L48T = [[k, str(k)] for k in (1, 12, 24, 36, 48)]
# Figure 36 (L1, wq w15 done 2026-10-22): upper, each run's mean trained pre-norm scale over Pre-LN's (gains.json,
# fetch_gains.py; LNS converted by 1/sqrt(l)); lower (owner popup 2026-10-22: per-layer differences, not three layer-48
# values), each run's final probe training loss minus Pre-LN's (d48g4.npz; layer 48 in the caption).
# w61 and w62 (owner popup 2026-10-22, "做到和理论一样"): Pre-LN with LNS's start, step size and clip (theory: LNS's
# curve up to rounding; dashed so it shows on top of LNS), and Pre-LN with LNS's step size and clip only. A run not
# fetched yet keeps its legend entry, marked queued.
GAIN_RUNS = [("preln", "apreln-vs-d48-g4", RED, None, {"en": "Pre-LN", "zh": "Pre-LN"}),
             ("lns", "alns-vs-d48-g4", BLUE7, None, {"en": "LayerNorm Scaling, converted", "zh": "LayerNorm Scaling（换算后）"}),
             ("preln-gilns", "apreln-gilns-vs-d48", GREEN, None, {"en": "Pre-LN, scales start at 1/\u221a\u2113", "zh": "Pre-LN，增益初始化为 1/\u221a\u2113"}),
             ("preln-lnsequiv", "apreln-gilnsbf-slns-vs-d48", PURPLE, "5 4", {"en": "Pre-LN with LNS's start, step size and clip", "zh": "Pre-LN，取 LNS 的初始值、步长和裁剪"}),
             ("preln-slns", "apreln-slns-vs-d48", YELLOW9, None, {"en": "Pre-LN with LNS's step size and clip", "zh": "Pre-LN，取 LNS 的步长和裁剪"})]
GAIN_MARKS = ([], [])
_g = json.loads((HERE / "gains.json").read_text()) if (HERE / "gains.json").exists() else {}
_gt = {n: f"-sep-bbfrozen-{t}/final" for n, t, *_ in GAIN_RUNS}
_have = {n for n, *_ in GAIN_RUNS if n in _g and _gt[n] in G4}
if {"preln", "lns"} <= _have:
    _p = np.array(_g["preln"]["mean"])
    _ln = lambda c, d: dict(type="line", color=c, width=1.6, alpha=1.0, **({"dash": d} if d else {}))
    GAIN_MARKS = ([dict(_ln(c, d), pts=[[k + 1, round(float(v), 4)] for k, v in enumerate(np.array(_g[n]["mean"]) / _p)]) for n, t, c, d, _ in GAIN_RUNS if n in _have],
                  [dict(_ln(c, d), pts=[[k + 1, round(float(v), 4)] for k, v in enumerate(G4[_gt[n]] - G4[_gt["preln"]])]) for n, t, c, d, _ in GAIN_RUNS if n in _have])
    print("Fig 36 layer 48: final probe loss", {n: round(float(G4[_gt[n]][47]), 3) for n in _have}, "scale ratio range",
          {n: (round(float(min(np.array(_g[n]["mean"]) / _p)), 2), round(float(max(np.array(_g[n]["mean"]) / _p)), 2)) for n in _have - {"preln"}})
GAIN_LEGEND = [[lab if n in _have else {k: f"{v} ({QUEUED[k].lower()})" if k == "en" else f"{v}（{QUEUED[k]}）" for k, v in lab.items()}, c, "dash" if d else "line"] for n, t, c, d, lab in GAIN_RUNS]
LOSS48 = {"en": "Layer-48 training loss (nats)", "zh": "第 48 层训练损失（nats）"}
figs["fig-arch-gain"] = dict(title={"en": "Pre-norm scales, 48 layers", "zh": "pre-norm 增益，48 层"},
                             legend=GAIN_LEGEND,
                             quantity={"en": "Pre-norm scale relative to Pre-LN's", "zh": "pre-norm 增益相对 Pre-LN 的比值"}, xlabel=LAYER_X, height=200, cols=1,
                             panels=[dict(xlim=[0.5, 48.5], ylim=[0.0, 1.2], xticks=L48T, yticks=[[t, f"{t:g}"] for t in (0, 0.4, 0.8, 1.2)], marks=GAIN_MARKS[0], empty=None if GAIN_MARKS[0] else ARCH2_STATUS,
                                          label={"en": "Mean pre-norm scale by layer", "zh": "每层 pre-norm 增益的均值"}),
                                     dict(xlim=[0.5, 48.5], ylim=[-0.6, 0.1], xticks=L48T, yticks=[[t, f"{t:g}"] for t in (-0.6, -0.4, -0.2, 0)], marks=GAIN_MARKS[1], empty=None if GAIN_MARKS[1] else ARCH2_STATUS,
                                          label={"en": "Final probe training loss minus Pre-LN's", "zh": "探针最终训练损失减 Pre-LN"})])
# Figure 37 (L2, wq w27 HC done 2026-10-24, w28 AttnRes): each probes run's layer-48 training loss by step (d48g4.npz
# "ce", 50-step running mean, 400 points); static routing keeps only the static part of the read / mixing / write.
ROUTE37 = [("hc", "ahc-vs-d48-g4", "ahc-rstatic-vs-d48", {"en": "Hyper-Connections", "zh": "Hyper-Connections"}),
           ("attnres", "aattnres-vs-d48-g4", "aattnres-rstatic-vs-d48", {"en": "AttnRes (Full)", "zh": "AttnRes (Full)"})]
def _route_line(tag, col):
    st, ce = G4.get(f"-sep-bbfrozen-{tag}/step"), G4.get(f"-sep-bbfrozen-{tag}/ce")
    if st is None or ce is None:
        return None
    xs, y = smooth(np.asarray(st, float), np.asarray(ce, float)[:, 47])   # values above the panel are clipped by the chart
    idx = np.unique(np.linspace(0, len(xs) - 1, 400).astype(int)); return dict(type="line", color=col, width=1.6, alpha=1.0, pts=pts(xs[idx], y[idx]))
_rp = []
for _a, _dyn, _sta, _lab in ROUTE37:
    _m = [m for m in (_route_line("apreln-vs-d48-g4", GREY), _route_line(_dyn, BLUE7), _route_line(_sta, GREEN)) if m is not None]
    _q = G4.get(f"-sep-bbfrozen-{_sta}/final") is None
    _rp.append(dict(xlim=[0, 5000], ylim=[2.8, 4.0], xticks=[[t, f"{t:,}"] for t in (0, 2500, 5000)], yticks=[[t, f"{t:g}"] for t in (2.8, 3.2, 3.6, 4.0)],
                    marks=_m, empty=None if _m else ARCH2_STATUS,
                    label={k: v + ((" · static routing queued" if k == "en" else " · 静态路由排队中") if _q else "") for k, v in _lab.items()}))
    print("Fig 37", _a, "layer-48 final probe loss dynamic / static",
          *(None if G4.get(f"-sep-bbfrozen-{t}/final") is None else round(float(G4[f"-sep-bbfrozen-{t}/final"][47]), 3) for t in (_dyn, _sta)))
figs["fig-arch-route"] = dict(title={"en": "Training loss with dynamic and static routing, 48 layers", "zh": "动态路由与静态路由的训练损失，48 层"},
                              legend=[[{"en": "Dynamic routing", "zh": "动态路由"}, BLUE7, "line"], [{"en": "Static routing", "zh": "静态路由"}, GREEN, "line"], [{"en": "Pre-LN", "zh": "Pre-LN"}, GREY, "line"]],
                              quantity=LOSS48, xlabel={"en": "Training step", "zh": "训练步数"}, height=200, panels=_rp)
# Figures 38 and 39 (owner popups 2026-10-22: the follow-ups and their two placeholders; pls_arch3_fetchable.sbatch, wq
# w76-w83). The separate-heads cost of a design = its separate-heads run's final last-layer training loss minus its probes
# run's (d48g4.npz "final": the -vs-d48(-g4) runs; at 6 layers auto10.json, mean of the last 50 steps). A variant not
# fetched yet is named queued in the panel label.
def sep_cost(tag, layers=48):
    a, b = G4.get(f"-sep-{tag}/final"), G4.get(f"-sep-bbfrozen-{tag}/final")
    return None if a is None or b is None else round(float(a[layers - 1] - b[layers - 1]), 4)


VAR38 = [("amhc-vs-d48-g4", {"en": "mHC", "zh": "mHC"}, GREY),
         ("amhc-mhcid-vs-d48", {"en": "mHC, residual mixing = I", "zh": "mHC，残差混合 = I"}, BLUE7),
         ("aattnres-vs-d48-g4", {"en": "AttnRes (Full), seed 0", "zh": "AttnRes (Full)，种子 0"}, GREY),
         ("aattnres-vs-d48-s1-g4", {"en": "AttnRes (Full), seed 1", "zh": "AttnRes (Full)，种子 1"}, GREY),
         ("aattnres-qclip10-vs-d48", {"en": "AttnRes (Full), query scale \u2264 10", "zh": "AttnRes (Full)，query 尺度 \u2264 10"}, BLUE7)]
v38 = [sep_cost(t) for t, _, _ in VAR38]
w38 = [lab for (t, lab, _), v in zip(VAR38, v38) if v is None]
lab38 = {"en": "Layer 48" + (" · " + "; ".join(l["en"] for l in w38) + " queued" if w38 else ""), "zh": "第 48 层" + (" · " + "；".join(l["zh"] for l in w38) + " 排队中" if w38 else "")}
figs["fig-arch-var"] = dict(title={"en": "Separate-heads cost of design variants, 48 layers", "zh": "结构变体的独立头代价，48 层"},
                            legend=[[{"en": "Control", "zh": "对照"}, GREY, "box"], [{"en": "Variant", "zh": "变体"}, BLUE7, "box"]],
                            quantity={"en": "Last-layer training loss, separate heads minus probes (nats)", "zh": "最后一层训练损失，独立头减探针（nats）"}, height=200, cols=1,
                            panels=[dict(xlim=[-0.7, len(VAR38) - 0.3], ylim=[-0.05, 0.15], yticks=lin_ticks(-0.05, 0.15, 0.05, "{:.2f}"), cats=[l for _, l, _ in VAR38], label=lab38,
                                         marks=[dict(type="bars", color=c, vals=[v if i == n else None for n, v in enumerate(v38)]) for i, (_, _, c) in enumerate(VAR38)])])
print("Fig 38 separate-heads cost", dict(zip([t for t, _, _ in VAR38], v38)))
HCD = {6: round(A10["hc"][5] - A10["hc-probes"][5], 4) if {"hc", "hc-probes"} <= set(A10) else None,
       12: sep_cost("ahc-vs-d12", 12), 24: sep_cost("ahc-vs-d24", 24), 48: sep_cost("ahc-vs-d48-g4")}
OTH48 = [sep_cost("vs-d48-g4")] + [sep_cost(f"a{k}-vs-d48-g4") for k, _ in ARCH if k != "hc"]  # Sandwich-LN, then the other eight
w39 = [str(d) for d, v in HCD.items() if v is None]
figs["fig-hc-depth"] = dict(title={"en": "Separate-heads cost of Hyper-Connections by depth", "zh": "Hyper-Connections 的独立头代价与深度"},
                            legend=[[{"en": "Hyper-Connections", "zh": "Hyper-Connections"}, BLUE7, "line"], [{"en": "Other designs, 48 layers", "zh": "其他结构，48 层"}, GREY, "dot"]],
                            quantity={"en": "Last-layer training loss, separate heads minus probes (nats)", "zh": "最后一层训练损失，独立头减探针（nats）"},
                            xlabel={"en": "Layers", "zh": "层数"}, height=200, cols=1,
                            panels=[dict(xlim=[0, 52], ylim=[-0.05, 0.15], xticks=[[d, str(d)] for d in (6, 12, 24, 48)], yticks=lin_ticks(-0.05, 0.15, 0.05, "{:.2f}"),
                                         label={"en": "Last layer" + (" · " + ", ".join(w39) + " layers queued" if w39 else ""), "zh": "最后一层" + (" · " + "、".join(w39) + " 层排队中" if w39 else "")},
                                         marks=[dict(type="line", color=BLUE7, width=1.6, alpha=1.0, pts=[[d, v] for d, v in HCD.items() if v is not None]),
                                                dict(type="dots", color=BLUE7, pts=[[d, v] for d, v in HCD.items() if v is not None]),
                                                dict(type="dots", color=GREY, pts=[[48, v] for v in OTH48 if v is not None])])])
print("Fig 39 HC cost by depth", HCD, "others at 48", OTH48)
# Figure 40 (owner popups 2026-10-23): is the separate-heads cost the readouts seeing the whole state the later layers
# read? Pre-LN separate heads whose intermediate readouts see only the first f of the hidden coordinates (PLS_VIS, wq
# w87/w88), against the Pre-LN probes run; seed 1 of both (w89/w90) shows the seed noise of the cost at f = 1. HC (its
# readout, the sum of 4 streams, sees a quarter of the stream space) as a grey reference line.
VIS = {1.0: sep_cost("apreln-vs-d48-g4"), 0.5: None, 0.25: None}
if G4.get("-sep-apreln-vis0.5-vs-d48/final") is not None:
    VIS[0.5] = round(float(G4["-sep-apreln-vis0.5-vs-d48/final"][47] - G4["-sep-bbfrozen-apreln-vs-d48-g4/final"][47]), 4)
if G4.get("-sep-apreln-vis0.25-vs-d48/final") is not None:
    VIS[0.25] = round(float(G4["-sep-apreln-vis0.25-vs-d48/final"][47] - G4["-sep-bbfrozen-apreln-vs-d48-g4/final"][47]), 4)
VIS_S1 = sep_cost("apreln-vs-d48-s1")
VX = {1.0: 0, 0.5: 1, 0.25: 2}
w40 = [lab for f, lab in ((0.5, "1/2"), (0.25, "1/4")) if VIS[f] is None] + (["seed 1"] if VIS_S1 is None else [])
figs["fig-vis"] = dict(title={"en": "Separate-heads cost when the readouts see part of the state, Pre-LN, 48 layers", "zh": "读出只看部分状态时的独立头代价，Pre-LN，48 层"},
                       legend=[[{"en": "Pre-LN, seed 0", "zh": "Pre-LN，种子 0"}, BLUE7, "dot"], [{"en": "Pre-LN, seed 1", "zh": "Pre-LN，种子 1"}, "#8AB4F8", "dot"], [{"en": "Hyper-Connections", "zh": "Hyper-Connections"}, GREY, "line"]],
                       quantity={"en": "Last-layer training loss, separate heads minus probes (nats)", "zh": "最后一层训练损失，独立头减探针（nats）"},
                       xlabel={"en": "Share of coordinates the intermediate readouts see", "zh": "中间层读出看到的坐标比例"}, height=200, cols=1,
                       panels=[dict(xlim=[-0.5, 2.5], ylim=[-0.05, 0.15], xticks=[[0, "1"], [1, "1/2"], [2, "1/4"]], yticks=lin_ticks(-0.05, 0.15, 0.05, "{:.2f}"),
                                    label={"en": "Layer 48" + (" · " + ", ".join(w40) + " queued" if w40 else ""), "zh": "第 48 层" + (" · " + "、".join(w.replace("seed 1", "种子 1") for w in w40) + " 排队中" if w40 else "")},
                                    marks=([dict(type="line", color=GREY, width=1.2, alpha=1.0, pts=[[-0.5, HCD[48]], [2.5, HCD[48]]])] if HCD[48] is not None else [])
                                    + [dict(type="dots", color=BLUE7, pts=[[VX[f], v] for f, v in VIS.items() if v is not None])]
                                    + ([dict(type="dots", color="#8AB4F8", pts=[[0.12, VIS_S1]])] if VIS_S1 is not None else []))])
print("Fig 40 visible fraction", VIS, "seed 1", VIS_S1)
figs["fig-mtp-dirs"] = dict(title={"en": "Where each layer writes, read by the two probes, 130m", "zh": "每层写入的方向与两种探针，130m"},
                            legend=[[{"en": "Next-token probe's directions", "zh": "下一个词探针的方向"}, RED, "line"], [{"en": "Probe for the token after next", "zh": "下下个词探针的方向"}, MTPC, "line"]],
                            quantity={"en": "Share of the layer's write", "zh": "占这一层写入的比例"}, xlabel=LAYER_X, height=200,
                            panels=[dict(xlim=[0.5, 6.5], ylim=[0.0, 1.0], xticks=[[k, str(k)] for k in range(1, 7)], yticks=[[t, f"{t:g}"] for t in (0, 0.5, 1.0)], marks=[], empty=ARCH2_STATUS)])

OUT.write_text("/* Generated by build_figures_data.py (pls branch, docs/della/pls_archive/blog); regenerate rather than hand-edit. */\nwindow.PLC_FIGS = " + json.dumps(figs, ensure_ascii=False, separators=(",", ":")) + ";\n")
print("N per layer", NL, "| x of layer 2..6 at the end", [f"{v:.3g}" for v in x6], "| wrote", OUT, OUT.stat().st_size, "bytes")
