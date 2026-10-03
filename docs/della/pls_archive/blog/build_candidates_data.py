"""Data for the candidate figures of wenhaochai.com/blogs/per-layer-contribution.html, redrawn on the page in the
writing:plot (Epoch AI) style -> assets/data/plc-candidates.js (generated; regenerate, never hand-edit).
Each candidate repeats the logic of the matplotlib script that drew it first:
  1 layer_probe.py         2 layer_backbone.py (130m, separate heads)   3 heads_compare.py
  4 layer_backbone.py (300m, shared head)   5 layer_probe_300m_progress.py (live W&B)   6 final_gap.py (W&B)
  7, 8 layer_backbone_paloma.py (130m, 300m)   9 layer_train.py (baseline drawn solid, as the owner asked later)
Every figure is a list of panels with marks (lines, dots, bars), view ranges and ticks already chosen here, so the page
script only maps and draws. Sources: flops.npz, bbfrozen.npz, depth.json, baselines.json, eval_macro.json, layers.npz,
W&B reself/marin-della."""
import json, sys
from pathlib import Path
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import curve_fit
HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1])
TOK, V = 128 * 4096, 128256
D = dict(np.load(HERE / "flops.npz")); Z = np.load(HERE / "bbfrozen.npz")
B = json.load(open(HERE / "baselines.json")); DEPTH = json.load(open(HERE / "depth.json")); M = json.load(open(HERE / "eval_macro.json"))
_raw = np.load(HERE / "layers.npz", allow_pickle=True)
LY = {k: np.array([[np.nan if v is None else v for v in r] for r in _raw[k]], float) if _raw[k].ndim == 2 else np.array([np.nan if v is None else v for v in _raw[k]], float) for k in _raw.files}
GM = {("blue", 200): "#AECBFA", ("blue", 300): "#8AB4F8", ("blue", 400): "#669DF6", ("blue", 500): "#4285F4", ("blue", 600): "#1A73E8",
      ("blue", 700): "#1967D2", ("blue", 800): "#185ABC", ("blue", 900): "#174EA6", ("red", 600): "#D93025", ("green", 600): "#1E8E3E",
      ("purple", 600): "#9334E6", ("grey", 600): "#80868B"}
BLUE7, RED, GREEN, PURPLE, GREY = GM[("blue", 700)], GM[("red", 600)], GM[("green", 600)], GM[("purple", 600)], GM[("grey", 600)]
def ramp(t):  # blue 200 -> 900, the one place a tone is interpolated (writing:plot rule 10)
    a = np.array([int(GM[("blue", 200)][i:i + 2], 16) for i in (1, 3, 5)]); b = np.array([int(GM[("blue", 900)][i:i + 2], 16) for i in (1, 3, 5)])
    c = (a + (b - a) * t).round().astype(int); return "#%02X%02X%02X" % tuple(c)
SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
law = lambda c, E, A, a: E + A * (c / 1e18) ** (-a)
def fit(x, y):
    p, _ = curve_fit(law, x, y, p0=(y.min() - 0.3, 0.3, 0.5), bounds=([0, 0, 0], [y.min(), 10, 5]), maxfev=20000)
    g = np.geomspace(x.min(), x.max(), 60); return pts(g, law(g, *p))
def pchip(x, y):
    f = PchipInterpolator(np.log(x), np.log(y)); g = np.geomspace(x.min(), x.max(), 60); return pts(g, np.exp(f(np.log(g))))
def pts(x, y): return [[float(f"{a:.5g}"), round(float(b), 4)] for a, b in zip(x, y)]
def smooth(x, y, w=50): return x[w - 1:], np.convolve(y, np.ones(w) / w, mode="valid")
def thin(x, y, n=160, log=True):
    idx = np.unique((np.geomspace(1, len(x), n) if log else np.linspace(1, len(x), n)).astype(int) - 1); return pts(np.asarray(x)[idx], np.asarray(y)[idx])
def logview(ex, ey, top=1.15, yticks=(2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.5, 5)):
    xmin, xmax = min(ex) / 4, max(ex) * 2; ylo, yhi = 0.97 * min(ey), max(ey) * top
    xt = [[v, "10" + str(int(np.log10(v))).translate(SUP)] for v in (1e16, 1e17, 1e18, 1e19, 1e20) if xmin <= v <= xmax]
    return dict(xlog=True, ylog=True, xlim=[xmin, xmax], ylim=[ylo, yhi], xticks=xt, yticks=[[v, f"{v:g}"] for v in yticks if ylo <= v <= yhi / 1.05])
def line(p, color, width=1.6, alpha=1.0): return dict(type="line", color=color, width=width, alpha=alpha, pts=p)
def dots(p, color): return dict(type="dots", color=color, pts=p)
def nicestep(span, bands=3):
    raw = span / bands; mag = 10 ** np.floor(np.log10(raw)); return float(min(m * mag for m in (1, 2, 2.5, 3, 5, 10) if m * mag >= raw))
def lin_ticks(lo, hi, digits):
    st = nicestep(hi - lo, 4); a = np.floor(lo / st) * st; b = np.ceil(hi / st) * st
    return [a, b], [[round(float(v), 6), f"{v:.{digits}f}"] for v in np.arange(a, b + st / 2, st)]
XL = {"en": "Backbone compute up to the layer (FLOPs)", "zh": "到该层为止的骨干算力（FLOPs）"}
CE = {"en": "Cross-entropy (nats, log scale)", "zh": "交叉熵（nats，对数坐标）"}
figs = {}

# 1, 3: final training loss by layer against backbone compute, 130m
Fb = B["130m"]["backbone_flops"]; x6 = np.array([(k + 1) / 6 * Fb for k in range(1, 6)])
sep, probe = Z["-pls1-sep/final"][1:], Z["-pls1-sep-bbfrozen/final"][1:]
shared, ro = Z["-pls1/final"][1:], Z["-pls1-dh/final"][1:]
plain = np.array([DEPTH[str(d)]["loss"] for d in range(2, 6)] + [B["130m"]["loss"]])
figs["cand-01"] = dict(
    title={"en": "Per-layer training loss against backbone compute, 130m", "zh": "逐层训练损失与骨干算力，130m"},
    legend=[[{"en": "Separate heads, fit", "zh": "独立头，拟合"}, BLUE7, "line"], [{"en": "Probe of the baseline", "zh": "基线的探针"}, RED, "line"],
            [{"en": "Baseline by depth, fit", "zh": "各深度的基线，拟合"}, GREY, "line"]],
    quantity=CE, xlabel=XL, height=300,
    panels=[dict(**logview(np.r_[x6, x6, x6], np.r_[sep, probe, plain]), marks=[
        line(fit(x6, sep), BLUE7), dots(pts(x6, sep), BLUE7), line(pchip(x6, probe), RED), dots(pts(x6, probe), RED),
        line(fit(x6, plain), GREY), dots(pts(x6, plain), GREY)])])
figs["cand-03"] = dict(
    title={"en": "Final training loss by layer and head type, 130m", "zh": "按层和头的类型的最终训练损失，130m"},
    legend=[[{"en": "Separate", "zh": "独立头"}, BLUE7, "line"], [{"en": "Shared", "zh": "共享头"}, GREEN, "line"],
            [{"en": "Read-only head", "zh": "只读共享头"}, PURPLE, "line"], [{"en": "Baseline", "zh": "基线"}, GREY, "line"]],
    quantity=CE, xlabel=XL, height=300,
    panels=[dict(**logview(np.r_[x6, x6, x6, x6], np.r_[sep, shared, ro, plain], yticks=(3.25, 3.5, 3.75, 4)), marks=sum(
        ([line(fit(x6, y), c), dots(pts(x6, y), c)] for y, c in ((sep, BLUE7), (shared, GREEN), (ro, PURPLE), (plain, GREY))), []))])

# 2, 4: training curves against backbone compute, endpoints with a fit (layer_backbone.py)
def backbone(size, arm, L, hidden, curves, label, controls):
    F = D[f"{size}-base/gflops"][-1] * 1e9 / (D[f"{size}-base/step"][-1] * TOK) / 3 - 2 * hidden * V
    s, ce = D[f"{size}-{arm}/step"], D[f"{size}-{arm}/ce"]; m = s >= 100
    marks, ends = [], []
    for k in range(1, L):
        xs, y = smooth((s * TOK * 3 * (k + 1) / L * F)[m], ce[m, k])
        if curves is None or k in curves: marks.append(line(thin(xs, y), ramp((k - 1) / max(L - 2, 1)), 1.0, 0.15))
        ends.append((xs[-1], y[-1]))
    ex, ey = map(np.array, zip(*ends))
    marks += [line(fit(ex, ey), BLUE7), dots(pts(ex, ey), BLUE7)]
    own = B[f"{size}m"]; vx, vy = list(ex), list(ey) + [own["loss"]]
    if controls:
        cx = np.array([d / 6 * own["backbone_flops"] for d in range(2, 6)] + [own["backbone_flops"]]); cy = np.array([DEPTH[str(d)]["loss"] for d in range(2, 6)] + [own["loss"]])
        marks += [line(fit(cx, cy), GREY), dots(pts(cx, cy), GREY)]; vx += list(cx); vy += list(cy)
    else:
        marks.append(dots([[own["backbone_flops"], own["loss"]]], GREY)); vx.append(own["backbone_flops"])
    v = logview(vx, vy, yticks=(2.75, 3, 3.25, 3.5, 3.75, 4, 4.5, 5))
    v["xlim"] = [min(ex) / 4, max(ex) * 2] if not controls else v["xlim"]
    v["ylim"] = [0.97 * min(min(ey), own["loss"]), max(ey) * 1.15]
    v["yticks"] = [[a, f"{a:g}"] for a in (2.75, 3, 3.25, 3.5, 3.75, 4, 4.5, 5) if v["ylim"][0] <= a <= v["ylim"][1] / 1.05]
    return dict(**v, label=label, marks=marks)
figs["cand-02"] = dict(
    title={"en": "Per-layer training loss against backbone compute, 130m", "zh": "逐层训练损失与骨干算力，130m"},
    legend=[[{"en": "Final loss by layer, fit", "zh": "各层最终损失，拟合"}, BLUE7, "line"], [{"en": "Baseline by depth, fit", "zh": "各深度的基线，拟合"}, GREY, "line"]],
    quantity=CE, xlabel=XL, height=300,
    panels=[backbone("130", "own", 6, 512, None, {"en": "Separate heads", "zh": "独立头"}, True)])
figs["cand-04"] = dict(
    title={"en": "Per-layer training loss against backbone compute, 300m", "zh": "逐层训练损失与骨干算力，300m"},
    legend=[[{"en": "Final loss by layer, fit", "zh": "各层最终损失，拟合"}, BLUE7, "line"], [{"en": "Baseline", "zh": "基线"}, GREY, "dot"]],
    quantity=CE, xlabel=XL, height=300,
    panels=[backbone("300", "shared", 12, 768, {1, 3, 5, 7, 9, 11}, {"en": "Shared head", "zh": "共享头"}, False)])

# 5: 300m at the latest step of the separate-heads-on-a-detached-backbone run (live from W&B)
import wandb
api = wandb.Api(timeout=300)
keys = [f"train/pls/L{k}" for k in range(12)]
run = api.run("reself/marin-della/muonh-qwen3-300m-della4xh100-pls1-sep-bbfrozen")
r = [x for x in run.scan_history(keys=["_step", *keys], page_size=5000)]
ps = np.array([x["_step"] for x in r]); pce = np.array([[x[k] for k in keys] for x in r]); now = int(ps.max()); lo = now - 99
win = lambda s: (s >= lo) & (s <= now)
pr = pce[win(ps)].mean(0)[1:]; sh = D["300-shared/ce"][win(D["300-shared/step"])].mean(0)[1:]; bs = D["300-base/ce"][win(D["300-base/step"])].mean()
F3 = D["300-base/gflops"][-1] * 1e9 / (D["300-base/step"][-1] * TOK) / 3 - 2 * 768 * V
x12 = np.array([3 * (k + 1) / 12 * F3 * (now + 1) * TOK for k in range(1, 12)])
done = run.state == "finished"
figs["cand-05"] = dict(
    title={"en": f"Per-layer training loss at step {now:,} of 11,444, 300m" + ("" if done else " (draft)"), "zh": f"第 {now:,} 步（共 11,444 步）的逐层训练损失，300m" + ("" if done else "（草图）")},
    legend=[[{"en": "Shared head, fit", "zh": "共享头，拟合"}, BLUE7, "line"], [{"en": "Probe of the baseline", "zh": "基线的探针"}, RED, "line"], [{"en": "Baseline", "zh": "基线"}, GREY, "dot"]],
    quantity=CE, xlabel=XL, height=300,
    panels=[dict(**logview(np.r_[x12, x12], np.r_[sh, pr, bs], yticks=(3, 3.25, 3.5, 3.75, 4, 4.5, 5, 5.5)), marks=[
        line(fit(x12, sh), BLUE7), dots(pts(x12, sh), BLUE7), line(pchip(x12, pr), RED), dots(pts(x12, pr), RED), dots([[x12[-1], bs]], GREY)])])
print("300m probe run at step", now, "state", run.state)

# 6: final-layer c4_en bpb minus the baseline (final_gap.py)
KEY = "eval/paloma/c4_en-marin-tokenizer/bpb"; P = "reself/marin-della/muonh-qwen3-"
bpb = lambda rid: float(api.run(P + rid).summary[KEY])
pool = [bpb("130m-della4xh100" + s) for s in ("", "-s1", "-s2", "-s3", "-ema0.999", "-ema0.999-s1", "-ema0.999-s2", "-ema0.999-s3")]
b130, b300 = float(np.mean(pool)), bpb("300m-della4xh100")
always = [({"en": "130m shared", "zh": "130m 共享头"}, bpb("130m-della4xh100-pls1") - b130), ({"en": "130m read-only", "zh": "130m 只读"}, bpb("130m-della4xh100-pls1-dh") - b130),
          ({"en": "130m own heads", "zh": "130m 独立头"}, bpb("130m-della4xh100-pls1-sep") - b130), ({"en": "300m shared", "zh": "300m 共享头"}, bpb("300m-della4xh100-pls1") - b300)]
early = [({"en": "off at step 80", "zh": "第 80 步关掉"}, bpb("130m-della4xh100-pls1-off80") - b130), ({"en": "off at step 140", "zh": "第 140 步关掉"}, bpb("130m-della4xh100-pls1-off140") - b130),
         ({"en": "linear 0 to 150", "zh": "0 到 150 步线性"}, bpb("130m-della4xh100-pls1-off0-150") - b130)]
def barpanel(items, label, digits):
    vals = [v for _, v in items]; ylim, yt = lin_ticks(0, max(vals), digits)
    return dict(xlog=False, ylog=False, cats=[c for c, _ in items], xlim=[-0.6, len(items) - 0.4], ylim=ylim, yticks=yt, xticks=[], label=label,
                marks=[dict(type="bars", color=GM[("blue", 400)], vals=[round(v, 6) for v in vals])])
figs["cand-06"] = dict(
    title={"en": "Final-layer c4_en bpb minus the baseline", "zh": "最终层 c4_en bpb 减基线"}, legend=[],
    quantity={"en": "Gap (bpb)", "zh": "差值（bpb）"}, xlabel=None, height=200,
    panels=[barpanel(always, {"en": "Always on", "zh": "全程开启"}, 2), barpanel(early, {"en": "Switched off early", "zh": "提前关掉"}, 4)])
print("final gap", [round(v, 4) for _, v in always + early])

# 7, 8: Paloma macro loss minus 1.5 by layer (layer_backbone_paloma.py)
def paloma(size, L, hidden, curves):
    F = D[f"{size}-base/gflops"][-1] * 1e9 / (D[f"{size}-base/step"][-1] * TOK) / 3 - 2 * hidden * V
    st = np.array(M[f"{size}m-shared"]["step"], float); mac = np.array(M[f"{size}m-shared"]["macro"]); marks, ends = [], []
    for k in range(1, L):
        x = st * TOK * 3 * (k + 1) / L * F; y = mac[:, k] - 1.5
        if curves is None or k in curves: marks.append(line(pts(x, y), ramp((k - 1) / max(L - 2, 1)), 1.0, 0.15))
        ends.append((x[-1], y[-1]))
    ex, ey = map(np.array, zip(*ends)); marks.append(dots(pts(ex, ey), BLUE7))
    by = M[f"{size}m-base"]["macro"][-1] - 1.5; bx = M[f"{size}m-base"]["step"][-1] * TOK * 3 * F; marks.append(dots([[bx, by]], GREY))
    v = logview(ex, list(ey) + [by], top=1.12, yticks=(2.25, 2.5, 2.75, 3, 3.25, 3.5))
    return dict(**v, label={"en": "Shared head", "zh": "共享头"}, marks=marks)
for n, (size, L, hid, cv) in {"cand-07": ("130", 6, 512, None), "cand-08": ("300", 12, 768, {1, 3, 5, 7, 9, 11})}.items():
    figs[n] = dict(title={"en": f"Per-layer Paloma loss against backbone compute, {size}m", "zh": f"逐层 Paloma 损失与骨干算力，{size}m"},
                   legend=[[{"en": "Final loss by layer", "zh": "各层最终损失"}, BLUE7, "dot"], [{"en": "Baseline", "zh": "基线"}, GREY, "dot"]],
                   quantity={"en": "Paloma macro loss minus 1.5 (nats, log scale)", "zh": "Paloma macro 损失减 1.5（nats，对数坐标）"}, xlabel=XL, height=300,
                   panels=[paloma(size, L, hid, cv)])

# 9: per-layer training loss over steps, four setups (layer_train.py; the baseline solid grey)
COLS = [GM[("blue", g)] for g in (300, 400, 500, 600, 700, 900)]
bsx, bly = smooth(LY["base/step"], LY["base/ce"]); mb = bsx >= 200
panels = []
for n, name in [("-pls1", {"en": "Shared head", "zh": "共享头"}), ("-pls1-dh", {"en": "Read-only shared head", "zh": "只读共享头"}),
                ("-pls1-sep", {"en": "Own heads", "zh": "独立头"}), ("-pls1-off80", {"en": "Switched off at step 80", "zh": "第 80 步关掉"})]:
    s, ce = LY[n + "/step"], LY[n + "/ce"]; marks, hi = [], 0
    for k in range(6):
        x, y = smooth(s, ce[:, k]); m = (x >= 200) & np.isfinite(y)
        marks.append(line(thin(x[m], y[m], 220, log=False), COLS[k], 1.2)); hi = max(hi, float(np.nanmax(y[m])))
    marks.append(line(thin(bsx[mb], bly[mb], 220, log=False), GREY, 1.2))
    if hi > 6: ylim, yt = [2, 10], [[v, f"{v:.0f}"] for v in (2, 4, 6, 8, 10)]
    else: ylim, yt = lin_ticks(float(bly[mb].min()), hi, 1)
    panels.append(dict(xlog=False, ylog=False, xlim=[200, 5000], ylim=ylim, yticks=yt, xticks=[[v, f"{v // 1000}k"] for v in (1000, 2000, 3000, 4000, 5000)], label=name, marks=marks))
figs["cand-09"] = dict(title={"en": "Per-layer training loss, 130m", "zh": "逐层训练损失，130m"},
                       legend=[[f"L{k}", c, "line"] for k, c in enumerate(COLS)] + [[{"en": "Baseline", "zh": "基线"}, GREY, "line"]],
                       quantity={"en": "Cross-entropy (nats)", "zh": "交叉熵（nats）"}, xlabel=None, height=170, cols=2, panels=panels)

OUT.write_text("/* Generated by build_candidates_data.py (pls branch, docs/della/pls_archive/blog); regenerate rather than hand-edit. */\nwindow.PLC_CANDIDATES = " + json.dumps(figs, ensure_ascii=False, separators=(",", ":")) + ";\n")
print("wrote", OUT, OUT.stat().st_size, "bytes")
