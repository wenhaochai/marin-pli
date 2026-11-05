"""Figure specs of the post "Deeper Models Gain from Residual Paths, Not from Learned Routing", drawn by C.epoch (assets/article-charts.js). Data: tmp/pls_blog/d48g4.npz via facts.json, tmp/pls_blog/depth_final.json, tmp/pls_hcid/scale_depth.json. Usage: build_figs.py OUT.js"""
import json, sys
import numpy as np
from pathlib import Path
HERE = Path(__file__).parent
F = json.load(open(HERE / "facts.json"))["runs"]
z = np.load("/scratch/gpfs/GROUP/USER/tmp/pls_blog/d48g4.npz", allow_pickle=True)
BLUE7, BLUE3, GREY, INK2 = "#1967D2", "#8AB4F8", "#80868B", "var(--ink-2)"
ho = lambda g: [r["heldout"] for r in F[g]]
WITH, WITHOUT = {"en": "With learned routing", "zh": "带学到的路由"}, {"en": "Routing removed", "zh": "去掉路由"}
ONE, MANY = {"en": "One path", "zh": "一条通路"}, {"en": "Several paths, no mixing", "zh": "多条通路，没有混合"}
LOSS = {"en": "Held-out loss of the last layer (nats)", "zh": "最后一层的留出损失（nats）"}
figs = {}
def dotrows(rows, xlim, ticks, labx, label=None, rowh=30):
    """rows: [(name en, name zh, [(values, colour), ...])]"""
    marks, names, R = [], [], []
    by = {}
    for i, (en, zh, sets) in enumerate(rows):
        names.append(([labx, -i], {"en": en, "zh": zh}))
        pts = []
        for vals, col in sets:
            by.setdefault(col, []).extend([[v, -i] for v in vals])
            pts += [[v, col] for v in vals]
        R.append(dict(y=-i, name={"en": en, "zh": zh}, pts=pts))
    marks.append(dict(type="text", color=INK2, anchor="end", pts=[q for q, _ in names], labels=[l for _, l in names]))
    for col, pts in by.items():
        marks.append(dict(type="dots", color=col, pts=pts, r=4.5))
    n = len(rows)
    return dict(height=round(rowh * (n + 0.4)), panels=[dict(xlim=xlim, ylim=[-(n - 1) - 0.7, 0.7], xticks=[[v, f"{v:.2f}"] for v in ticks], yticks=[], rows=R, marks=marks)])
# Figure: removing the routing, three designs
figs["rr-removal"] = dict(title={"en": "Removing the routing", "zh": "去掉路由"}, legend=[[WITH, BLUE7, "dot"], [WITHOUT, BLUE3, "dot"]], quantity=LOSS, cols=1,
                          **dotrows([("Hyper-Connections", "Hyper-Connections", [(ho("hc_mixing"), BLUE7), (ho("hc_no_mixing"), BLUE3)]),
                                     ("mHC", "mHC", [(ho("mhc"), BLUE7), (ho("mhc_no_mixing"), BLUE3)]),
                                     ("Attention Residuals", "Attention Residuals", [(ho("attnres"), BLUE7), (ho("attnres_static"), BLUE3)])], [3.40, 3.76], [3.55, 3.60, 3.65, 3.70, 3.75], 3.525, rowh=40))
# Figure: the depth test
DF = json.load(open("/scratch/gpfs/GROUP/USER/tmp/pls_blog/depth_final.json"))
RED, PUR3, PUR6, PUR9 = "#D93025", "#C58AF9", "#9334E6", "#681DA8"
_L = ("12", "24", "48")
_end = lambda runs: [r[-1][1] for r in runs]
_m = []
for vals, col in (({l: DF["preln"][l] for l in _L}, GREY), ({l: _end(DF["no_mixing"][l]) for l in _L}, BLUE3), ({l: _end(DF["learned"][l]) for l in _L}, BLUE7)):
    _m.append(dict(type="line", color=col, width=1.8, alpha=1, pts=[[i, round(float(np.mean(vals[l])), 4)] for i, l in enumerate(_L)]))
    _m.append(dict(type="dots", color=col, r=3.2, pts=[[i, v] for i, l in enumerate(_L) for v in vals[l]]))
_d = [dict(type="line", color=GREY, width=1.0, alpha=1, pts=[[0, 0], [5000, 0]])]
_t = []
for l, col in zip(_L, (PUR3, PUR6, PUR9)):
    st = [q[0] for q in DF["no_mixing"][l][0]]
    p = [[st[j], round(float(np.mean([r[j][1] for r in DF["no_mixing"][l]]) - np.mean([r[j][1] for r in DF["learned"][l]])), 4)] for j in range(len(st))]
    _d += [dict(type="line", color=col, width=1.8, alpha=1, pts=p), dict(type="dots", color=col, pts=p)]
    _t.append(dict(type="text", color=col, anchor="start", dx=9, pts=[p[-1]], labels=[{"en": f"{l} layers", "zh": f"{l} 层"}]))
figs["rr-depth"] = dict(title={"en": "Paths and learned mixing at three depths", "zh": "三个深度上的通路和学到的混合"},
                        legend=[[{"en": "One path", "zh": "一条通路"}, GREY, "line"], [{"en": "Four streams, no mixing", "zh": "4 条流，没有混合"}, BLUE3, "line"], [{"en": "Four streams, learned mixing", "zh": "4 条流，学到的混合"}, BLUE7, "line"]],
                        quantity=[LOSS, {"en": "Gain from learned mixing, by step (nats)", "zh": "学到的混合带来的收益，按步数（nats）"}], height=240, cols=2,
                        panels=[dict(xlim=[-0.4, 2.4], ylim=[3.5, 3.9], yticks=[[v, f"{v:.1f}"] for v in (3.5, 3.6, 3.7, 3.8, 3.9)], cats=[{"en": f"{l} layers", "zh": f"{l} 层"} for l in _L], marks=_m),
                                dict(xlim=[0, 6400], ylim=[-0.08, 0.08], xticks=[[v, f"{v:,}"] for v in (0, 2500, 5000)], yticks=[[v, f"{v:+.2f}"] for v in (-0.08, -0.04, 0, 0.04, 0.08)], marks=_d + _t)])
# Figure: the scale change of the mixing compounds with depth (cumulative gain of the stream sum through the mixing)
SC = json.load(open("/scratch/gpfs/GROUP/USER/tmp/pls_hcid/scale_depth.json"))
_ms = [dict(type="line", color=GREY, width=1.0, alpha=1, pts=[[0, 1], [48, 1]])]
_tl = []
for l, col in zip(_L, (PUR3, PUR6, PUR9)):
    q = [[(i + 1) / 2, float(f"{10 ** v:.4g}")] for i, v in enumerate(SC[f"HC/{l}"]["cum"])]
    _ms.append(dict(type="line", color=col, width=1.8, alpha=1, pts=q))
    _tl.append(dict(type="text", color=col, anchor="start", dx=7, pts=[q[-1]], labels=[{"en": f"{l} layers", "zh": f"{l} 层"}]))
figs["rr-scale"] = dict(title={"en": "What the learned mixing does to the stream sum", "zh": "学到的混合对流之和的作用"}, legend=[],
                        quantity={"en": "Product of the mixing's gains up to this layer (log scale)", "zh": "到这一层为止混合增益的乘积（对数坐标）"}, xlabel={"en": "Layer", "zh": "层"}, height=250, cols=1,
                        panels=[dict(ylog=True, xlim=[0, 58], ylim=[1e-5, 10], xticks=[[v, str(v)] for v in (1, 12, 24, 36, 48)], yticks=[[v, s] for v, s in ((1e-4, "0.0001"), (1e-2, "0.01"), (1, "1"))], marks=_ms + _tl)])
# Figure: the cost follows the scale change, at every depth and with every kind of mixing
_gain = lambda k, ref: round(ref - SC[k]["heldout"], 3)
_NM = {"12": 3.7837, "24": 3.6599, "48": 3.5801}
_pts_depth = [[round(SC[f"HC/{l}"]["orders"], 2), {"12": 0.016, "24": -0.020, "48": -0.056}[l]] for l in _L]
_keep = [("HC moves/48", {"en": "Moves content", "zh": "只移动内容"}, 10, 0.005), ("HC keeps sum/48", {"en": "Moves content, signed", "zh": "只移动内容，权重有正负"}, 10, -0.002)]
_chg = [("HC scalar/48", {"en": "One learned number", "zh": "一个学到的数"}, -10, 0.010, "end"), ("HC static/48", {"en": "Constant mixing", "zh": "常量混合"}, 10, -0.005, "start"), ("HC half dose/48", {"en": "Half-dose shrink", "zh": "半剂量缩小"}, 10, -0.001, "start"), ("HC only shrinks/48", {"en": "Only shrinks", "zh": "只缩小"}, -10, -0.001, "end")]
_mk = [dict(type="line", color=GREY, width=1.0, alpha=1, pts=[[0, 0], [7, 0]])]
_mk.append(dict(type="dots", color=BLUE3, r=4.5, pts=[[round(SC[k]["orders"], 2), _gain(k, _NM["48"])] for k, *_ in _keep]))
_mk.append(dict(type="dots", color=BLUE7, r=4.5, pts=[[round(SC[k]["orders"], 2), _gain(k, _NM["48"])] for k, *_ in _chg]))
_mk.append(dict(type="dots", color=PUR6, r=4.5, pts=_pts_depth))
for k, name, dx, dy in _keep:
    _mk.append(dict(type="text", color=INK2, anchor="start", dx=dx, pts=[[round(SC[k]["orders"], 2), _gain(k, _NM["48"]) + dy]], labels=[name]))
for k, name, dx, dy, anc in _chg:
    _mk.append(dict(type="text", color=INK2, anchor=anc, dx=dx, pts=[[round(SC[k]["orders"], 2), _gain(k, _NM["48"]) + dy]], labels=[name]))
for (x, y), l in zip(_pts_depth, _L):
    _mk.append(dict(type="text", color=PUR6, anchor="start", dx=10, pts=[[x, y + (0.004 if l == "12" else 0.002)]], labels=[{"en": f"Learned mixing, {l} layers", "zh": f"学到的混合，{l} 层"}]))
figs["rr-cost"] = dict(title={"en": "Gain from the mixing against its scale change", "zh": "混合带来的收益和它的尺度变化"},
                       legend=[[{"en": "Learned mixing at 12, 24, 48 layers", "zh": "学到的混合，12、24、48 层"}, PUR6, "dot"], [{"en": "Keeps the stream sum, 48 layers", "zh": "保持流之和，48 层"}, BLUE3, "dot"], [{"en": "Changes the stream sum, 48 layers", "zh": "改变流之和，48 层"}, BLUE7, "dot"]],
                       quantity={"en": "Gain over no mixing at the same depth, held-out (nats)", "zh": "相对同深度没有混合的收益，留出损失（nats）"}, xlabel={"en": "Scale change of the stream sum (orders of magnitude)", "zh": "流之和的尺度变化（数量级）"}, height=280, cols=1,
                       panels=[dict(xlim=[-0.4, 8.2], ylim=[-0.13, 0.05], xticks=[[v, str(v)] for v in (0, 2, 4, 6)], yticks=[[v, f"{v:+.2f}"] for v in (-0.12, -0.08, -0.04, 0, 0.04)], marks=_mk)])
# Figure: paths
figs["rr-paths"] = dict(title={"en": "One path against several", "zh": "一条通路和多条通路"}, legend=[[ONE, GREY, "dot"], [MANY, BLUE3, "dot"]], quantity=LOSS, cols=1,
                        **dotrows([("Pre-LN", "Pre-LN", [(ho("pre_ln"), GREY)]), ("Sandwich-LN", "Sandwich-LN", [(ho("one_path_other")[:1], GREY)]), ("LayerNorm Scaling", "LayerNorm Scaling", [(ho("one_path_other")[1:2], GREY)]), ("Hyper-Connections, 1 stream", "Hyper-Connections，1 条流", [(ho("hc_streams_with_mixing")[:1], GREY)]),
                                   ("Hyper-Connections, 4 streams", "Hyper-Connections，4 条流", [(ho("hc_no_mixing"), BLUE3)]), ("mHC, 4 streams", "mHC，4 条流", [(ho("mhc_no_mixing"), BLUE3)]), ("Attention Residuals, every earlier layer", "Attention Residuals，读所有更早的层", [(ho("attnres_static"), BLUE3)])], [3.36, 3.72], [3.55, 3.60, 3.65, 3.70], 3.525, rowh=26))
Path(sys.argv[1]).write_text("/* Generated by tmp/pls_routing/build_figs.py; regenerate rather than hand-edit. */\nwindow.RR_FIGS = " + json.dumps(figs, ensure_ascii=False, separators=(",", ":")) + ";\n")
print("wrote", sys.argv[1], Path(sys.argv[1]).stat().st_size)
