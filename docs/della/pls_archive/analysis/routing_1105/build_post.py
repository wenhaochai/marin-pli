"""Build blogs/residual-paths-not-routing.html (a preview: noindex, no tile on blogs.html) from draft_v1.json (Argon) + draft2.json (Argon, after my audit) + the hand fixes below, the figure specs (rr-figs.js) and the film. Usage: build_post.py SITE_WORKTREE"""
import html, json, re, sys
from pathlib import Path
HERE = Path(__file__).parent
SITE = Path(sys.argv[1])
SLUG = "residual-paths-not-routing"
v1 = json.load(open(HERE / "draft_v1.json")); d2 = json.load(open(HERE / "draft2.json"))
A0 = json.load(open("/scratch/gpfs/GROUP/USER/tmp/pls_hcid/a0_l12.json"))["mlp"]
sec = {s["id"]: s for s in v1["sections"]}

# ---------- text: draft2 passages replace the audited ones ----------
def P(en, zh): return {"en": en, "zh": zh}
sec["motivation"]["paras"] = [d2["motivation"]]
sec["setup"]["paras"] = [d2["setup"]]
sec["depth"]["paras"] = [d2["depth"]]
sec["controls"]["paras"][0] = d2["controls_1"]
sec["paths"]["paras"][0] = d2["paths_1"]
sec["limitations"]["paras"] = [d2["limitations"]]
sec["takeaways"]["paras"][0] = d2["takeaways_1"]
scale = {"id": "scale", "heading": d2["scale_heading"], "paras": [d2["scale_1"], d2["scale_2"], d2["scale_3"]], "figure": "rr-scale"}
cost = {"id": "cost", "heading": d2["cost_heading"], "paras": [d2["cost_1"], d2["cost_2"], d2["cost_3"]], "figure": "rr-cost"}
order = ["motivation", "background", "setup", "depth", "controls", "scale", "cost", "paths", "limitations", "takeaways", "open-questions", "related-work"]
sec["scale"], sec["cost"] = scale, cost
# hand fixes after the audit (2026-11-05)
FIX = [
    ("zh", "所以论文的结果和我们在 48 层的结果并不矛盾。", "所以论文的结果和我们在 48 层的结果并不一定矛盾。"),
    ("en", "That would explain the late writes above, and the early writes in the model with fixed gains.", "That would explain why the late writes above count so little, and why the early writes count little in the model with fixed gains, whose product of gains falls by 6.5 orders of magnitude."),
    ("zh", "这能同时解释前文后期写入占比极小的现象，以及固定增益模型中早期写入的变化。", "这能解释前面最后的写入为什么几乎不起作用，也能解释固定增益模型里早期的写入为什么同样几乎不起作用：它的增益乘积下降了 6.5 个数量级。"),
    ("zh", "流形约束超连接（mHC）", "Manifold-Constrained Hyper-Connections（mHC）"), ("zh", "超连接（HC）", "Hyper-Connections（HC）"), ("zh", "注意力残差（AttnRes）", "Attention Residuals（AttnRes）"),
    ("zh", "搬移", "移动"), ("zh", "评测点", "评估点"), ("zh", "单位阵", "单位矩阵"),
    ("en", 'href="https://wenhaochai.com/pls-routing/"', 'href="../exps/per-layer-contribution.html#fig-arch-var"'),
    ("zh", 'href="https://wenhaochai.com/pls-routing/"', 'href="../exps/per-layer-contribution.html#fig-arch-var"'),
]
MATH = [("$F$", "<i>F</i>"), ("$h \\leftarrow h + F(\\text{RN}(h))$", "<i>h</i> ← <i>h</i> + <i>F</i>(RN(<i>h</i>))"), ("$h$", "<i>h</i>"), ("$m$", "<i>m</i>"),
        ("$m = 4$", "<i>m</i> = 4"), ("$m \\times m$", "<i>m</i> × <i>m</i>"), ("$A$", "<i>A</i>"), ("$1 \\times 1$", "1 × 1"), ("$1.8 \\times 10^{19}$", "1.8 × 10<sup>19</sup>")]
# the custom schematic is not numbered, so the chart numbers move down by one (descending, so no number is moved twice)
RENUM = [("Figure 2 shows", "The diagram below shows"), ("图 2 展示了", "下图展示了"), ("Figure 3", "Figure 2"), ("图 3", "图 2"), ("Figure 4", "Figure 3"), ("图 4", "图 3"),
         ("Figure 5", "Figure 4"), ("图 5", "图 4"), ("Figure 6", "Figure 5"), ("图 6", "图 5")]
def fix(lang, s):
    for l, a, b in FIX:
        if l == lang: s = s.replace(a, b)
    for a, b in sorted(MATH, key=lambda x: -len(x[0])): s = s.replace(a, b)
    for a, b in RENUM: s = s.replace(a, b)
    s = s.replace("1.8e19", "1.8 × 10<sup>19</sup>")
    if lang == "en": s = re.sub(r"(?<![\w.])-(\d)", "−\\1", s)
    else: s = re.sub(r"(?<![\w.])-(\d)", "−\\1", s)
    assert "$" not in s, s[:120]
    return s
for k in order:
    for p in sec[k]["paras"]:
        p["en"], p["zh"] = fix("en", p["en"]), fix("zh", p["zh"])
# related work: three sentences replaced by draft2's
REL = [("en", r'Xie et al\.<sup class="fn">2</sup>.*?Sinkhorn mixing\.', d2["related_mhc"]["en"]), ("zh", r'Xie 等人<sup class="fn">2</sup>.*?Sinkhorn 混合。', d2["related_mhc"]["zh"]),
       ("en", r'Wang et al\.<sup class="fn">5</sup>.*?static routing\.', d2["related_depthbench"]["en"]), ("zh", r'Wang 等人<sup class="fn">5</sup>.*?静态路由。', d2["related_depthbench"]["zh"]),
       ("en", r'Jianlin Su<sup class="fn">10</sup>.*?across depth\.', d2["related_su"]["en"]), ("zh", r'苏剑林<sup class="fn">10</sup>.*?这两点。', d2["related_su"]["zh"])]
for lang, pat, new in REL:
    n = 0
    for p in sec["related-work"]["paras"]:
        p[lang], k = re.subn(pat, lambda m: fix(lang, new), p[lang], flags=re.S); n += k
    assert n == 1, (lang, pat, n)
# figure text: subtitles and captions from draft2 (schematic: mine)
FIGTXT = {f: {"dek": d2[f"dek_{k}"], "cap": d2[f"cap_{k}"]} for f, k in (("rr-removal", "removal"), ("rr-depth", "depth"), ("rr-scale", "scale"), ("rr-cost", "cost"), ("rr-paths", "paths"))}
FIGTXT["rr-cost"]["cap"]["zh"] = "每个点代表一个深度下的一种混合：横轴是增益乘积跨越的数量级，纵轴是相对没有混合的留出收益。"
FIGTXT["rr-scale"]["cap"]["zh"] = "截至每一层，学到的混合作用在流之和上的增益乘积；取最终检查点在训练 token 上的中位数。"
SCHEMA = {"title": P("One sublayer of Pre-LN and of Hyper-Connections", "Pre-LN 与 Hyper-Connections 的一个子层"),
          "dek": P("Pre-LN keeps one running sum. Hyper-Connections keeps four streams and mixes them with a matrix A.", "Pre-LN 只有一个累加和。Hyper-Connections 维护四条流，并用矩阵 A 混合它们。"),
          "cap": P("Right: HC's learned A at the MLP of layer 12 in the 48-layer model, beside A fixed to the identity. Row i: how much of stream i goes to each stream; the number beside it is the row's sum.",
                   "右：48 层模型第 12 层 MLP 处学到的 A，旁边是固定为单位矩阵的 A。第 i 行：第 i 条流有多少进入每条流；旁边的数是这一行的和。")}

# ---------- the schematic (Epoch style, inline SVG, both languages) ----------
INK2, GREY, AXIS, BLUE, RED = "#3C4043", "#5F6368", "#DADCE0", "#1967D2", "#D93025"
def mixc(a, b, p):
    h = lambda c: [int(c[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * p):02x}" for x, y in zip(h(a), h(b)))
def T(x, y, en, zh, size=12.5, anchor="start", weight=400, fill=INK2):
    a = f'x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}" style="fill:{fill}"'
    return f'<text lang="en" {a}>{html.escape(en)}</text><text lang="zh" {a}>{html.escape(zh)}</text>' if en != zh else f'<text {a}>{html.escape(en)}</text>'
def sublayer(x, y, en, zh): return f'<rect x="{x}" y="{y}" width="78" height="30" rx="6" fill="#fff" stroke="{GREY}" stroke-width="1.5"/>' + T(x + 39, y + 19.5, en, zh, 12, "middle", 600)
def arrow(x1, y, x2, col, w=1.8):
    d = 6 if x2 < x1 else -6
    return f'<line x1="{x1}" y1="{y}" x2="{x2 + d * 0.5}" y2="{y}" stroke="{col}" stroke-width="{w}"/><path d="M{x2},{y} l{d},-4 l0,8 z" fill="{col}"/>'
svg = []
# Pre-LN
svg += [T(10, 18, "Pre-LN: one path", "Pre-LN：一条通路", 14, "start", 600), f'<line x1="56" y1="262" x2="56" y2="50" stroke="{INK2}" stroke-width="4"/>', T(56, 40, "running sum h", "累加和 h", 12, "middle", 400, GREY)]
for y, en, zh in ((176, "MLP", "MLP"), (96, "Attention", "注意力")):
    svg += [f'<line x1="56" y1="{y + 22}" x2="96" y2="{y + 22}" stroke="{AXIS}" stroke-width="1.6"/>', sublayer(96, y, en, zh), arrow(96, y + 8, 58, BLUE)]
svg += [T(10, 292, "h ← h + F(RN(h))", "h ← h + F(RN(h))", 13, "start", 500)]
# Hyper-Connections
XS = [238, 256, 274, 292]
svg += [T(214, 18, "Hyper-Connections: four streams", "Hyper-Connections：四条流", 14, "start", 600)]
svg += [f'<line x1="{x}" y1="262" x2="{x}" y2="54" stroke="{INK2}" stroke-width="2.6"/>' for x in XS]
svg += [f'<line x1="{XS[0]}" y1="54" x2="{XS[-1]}" y2="54" stroke="{INK2}" stroke-width="2.6"/>', f'<line x1="265" y1="54" x2="265" y2="46" stroke="{INK2}" stroke-width="2.6"/>', T(265, 38, "stream sum", "流之和", 12, "middle", 400, GREY)]
for y, en, zh in ((176, "MLP", "MLP"), (96, "Attention", "注意力")):
    svg += [f'<line x1="{XS[0]}" y1="{y + 22}" x2="330" y2="{y + 22}" stroke="{AXIS}" stroke-width="1.4"/>', sublayer(330, y, en, zh)]
    svg += [arrow(330, y + 8, x + 2, BLUE, 1.4) for x in XS[-1:]] + [f'<line x1="{XS[0]}" y1="{y + 8}" x2="{XS[-1]}" y2="{y + 8}" stroke="{BLUE}" stroke-width="1.4"/>'] + [f'<circle cx="{x}" cy="{y + 8}" r="2.6" fill="{BLUE}"/>' for x in XS]
    svg += [f'<rect x="{XS[0] - 9}" y="{y - 22}" width="{XS[-1] - XS[0] + 18}" height="15" rx="4" fill="#E8F0FE" stroke="{BLUE}" stroke-width="1.2"/>', T(265, y - 11, "A", "A", 11, "middle", 600, BLUE)]
svg += [T(214, 292, "read, write into each stream, mix with A", "读取，写入每条流，用 A 混合", 12.5, "start", 500)]
# the two mixing matrices
def matrix(x0, y0, M, title_en, title_zh, sums):
    out = [T(x0, y0 - 12, title_en, title_zh, 12.5, "start", 600)]
    c = 24
    for i in range(4):
        for j in range(4):
            v = M[i][j]; m = min(1.0, abs(v))
            fill = mixc("#FFFFFF", BLUE if v >= 0 else RED, 0.75 * m)
            out.append(f'<rect x="{x0 + j * (c + 2)}" y="{y0 + i * (c + 2)}" width="{c}" height="{c}" rx="3" fill="{fill}" stroke="#E8EAED"/>')
            txt = f"{v:.2f}".replace("-", "−").replace("0.", ".") if abs(v) < 1 else f"{v:.2f}".replace("-", "−")
            out.append(T(x0 + j * (c + 2) + c / 2, y0 + i * (c + 2) + 16, txt, txt, 8.6, "middle", 500, "#fff" if m > 0.62 else INK2))
        s = sums[i]; out.append(T(x0 + 4 * (c + 2) + 22, y0 + i * (c + 2) + 16.5, s, s, 12, "middle", 600, RED if s.startswith("−") else INK2))
    out.append(T(x0 + 4 * (c + 2) + 22, y0 - 1, "row sum", "行和", 10.5, "middle", 400, GREY))
    return out
svg += matrix(446, 44, A0, "Learned mixing, layer 12", "学到的混合，第 12 层", [f"{sum(r):.2f}".replace("-", "−") for r in A0])
svg += matrix(446, 186, [[1 if i == j else 0 for j in range(4)] for i in range(4)], "No mixing: A = I", "没有混合：A = I", ["1.00"] * 4)
SVG = f'<svg viewBox="0 0 640 300" role="img" aria-label="Pre-LN keeps one running sum; Hyper-Connections keeps four streams, writes into each and mixes them with a matrix A; HC\'s learned A at layer 12 has row sums 0.81, 0.10, −0.47 and 0.86, and A = I keeps every row at 1">' + "".join(svg) + "</svg>"
# the row sums printed on the schematic must be the post's
assert [f"{sum(r):.2f}" for r in A0] == ["0.81", "0.10", "-0.47", "0.86"], [f"{sum(r):.2f}" for r in A0]

# ---------- notes: short author lists ----------
NOTES = {n["n"]: n["en"] for n in v1["notes"]}
SHORT = {1: "Zhu et al.", 2: "Xie et al.", 3: "Yang and Gao.", 4: "Kimi Team.", 5: "Wang et al.", 6: "He, Zhang, Ren, and Sun.", 7: "Srivastava, Greff, and Schmidhuber.", 8: "Huang, Liu, van der Maaten, and Weinberger.", 9: "Pagliardini, Mohtashami, Fleuret, and Jaggi.", 10: "Jianlin Su.", 11: "Xiao, Meng, Li, and Yuan."}
for n, s in NOTES.items():
    i = s.index(". <a ")
    NOTES[n] = SHORT[n] + s[i + 1:]
seen = set()
def para_block(p):
    out = f'    <p lang="en">{p["en"]}</p>\n    <p lang="zh">{p["zh"]}</p>\n'
    ns = [int(x) for x in re.findall(r'<sup class="fn">(\d+)</sup>', p["en"] + p["zh"])]
    new = [n for n in dict.fromkeys(ns) if n not in seen]; seen.update(new)
    if ns:
        out = '  <div class="fnp">\n' + out + "".join(f'    <aside class="fnote" data-n="{n}">{NOTES[n]}</aside>\n' for n in new) + "  </div>\n"
    return out
def figure(fid):
    if fid == "schematic":
        return (f'  <figure class="rr-schema">\n    <h4><span lang="en">{SCHEMA["title"]["en"]}</span><span lang="zh">{SCHEMA["title"]["zh"]}</span></h4>\n'
                f'    <p class="dek"><span lang="en">{SCHEMA["dek"]["en"]}</span><span lang="zh">{SCHEMA["dek"]["zh"]}</span></p>\n    {SVG}\n'
                f'    <figcaption><span lang="en">{SCHEMA["cap"]["en"]}</span><span lang="zh">{SCHEMA["cap"]["zh"]}</span></figcaption>\n  </figure>\n')
    F = json.loads(re.search(r"window\.RR_FIGS = (.*);\s*$", (HERE / "rr-figs.js").read_text(), re.S).group(1))[fid]
    t = FIGTXT[fid]; e = lambda s: html.escape(s, quote=True)
    return (f'  <figure class="viz" id="{fid}" data-src="" data-title-en="{e(F["title"]["en"])}" data-title-zh="{e(F["title"]["zh"])}" data-dek-en="{e(t["dek"]["en"])}" data-dek-zh="{e(t["dek"]["zh"])}" '
            f'data-cap-en="{e(t["cap"]["en"])}" data-cap-zh="{e(t["cap"]["zh"])}"></figure>\n')
body, toc = [], []
for k in order:
    s = sec[k]; h = s["heading"]
    body.append(f'<section id="{k}">\n  <h2><span lang="en">{h["en"]}</span><span lang="zh">{h["zh"]}</span></h2>\n')
    toc.append(f'    <li><a href="#{k}"><span lang="en">{h["en"]}</span><span lang="zh">{h["zh"]}</span></a></li>')
    paras = s["paras"]
    fig = s.get("figure")
    # the figure goes after the paragraph that first cites it (the schematic after the paragraph that introduces it)
    at = len(paras) - 1
    if fig:
        key = {"rr-removal": "Figure 1", "rr-depth": "Figure 2", "rr-scale": "Figure 3", "rr-cost": "Figure 4", "rr-paths": "Figure 5", "schematic": "The diagram below"}[fig]
        at = next((i for i, p in enumerate(paras) if key in p["en"]), at)
    for i, p in enumerate(paras):
        body.append(para_block(p))
        if fig and i == at: body.append(figure(fig))
    if k == "motivation":
        pass
    body.append("</section>\n")
    if k == "takeaways":   # the read-more card after the takeaways, as in the last post
        body.append('''<a class="vr-more" href="../exps/per-layer-contribution.html">
  <span class="tile-thumb"><img src="../assets/img/exp-cards/per-layer-contribution.webp" alt="" loading="lazy" decoding="async"></span>
  <span class="vr-more-body">
    <span class="tile-meta"><span lang="en">Read more · Experiment page</span><span lang="zh">延伸阅读 · 实验主页</span></span>
    <span class="tile-title">Quantifying the Per-layer Contribution of LLMs</span>
    <span class="tile-desc"><span lang="en">The full record of this study: code, training curves, and every question we tested, finished and still running.</span><span lang="zh">这项研究的完整记录：代码、训练曲线，以及我们测过的每一个问题，包括已有结论的和还在跑的。</span></span>
  </span>
</a>
''')
for n in NOTES: assert n in seen, f"note {n} is never cited"
title, stand = v1["title"], v1["standfirst"]
VOC = (SITE / "blogs/vocab-multi-epoch.html").read_text(encoding="utf-8")
style = VOC[VOC.index("<style>"):VOC.index("</style>") + 8]
style = style.replace("</style>", ".rr-schema{margin:36px 0}.rr-schema h4{font-family:var(--font-sans);font-size:19px;font-weight:600;line-height:1.3;margin:0 0 6px;color:var(--ink);text-wrap:balance}"
                      ".rr-schema .dek{font-family:var(--font-sans);font-size:15px;line-height:1.45;color:var(--ink-3);margin:0 0 14px}.rr-schema svg{display:block;width:100%;height:auto;overflow:visible}"
                      ".rr-schema svg text{font-family:var(--font-sans)}.rr-schema figcaption{margin-top:12px;font-size:14px;line-height:1.5;color:var(--ink-3)}\n</style>")
desc = stand["en"]
FILM_ALT = ("A short animation of the setup and the result: Pre-LN keeps one running sum while Hyper-Connections keeps four streams and mixes them with a matrix A; at layer 12 of the 48-layer model the learned A has row sums 0.81, 0.10, −0.47 and 0.86, "
            "so this sublayer shrinks the stream sum to a third, while no mixing keeps every row at 1; across 12, 24 and 48 layers the gain from paths is +0.061, +0.085 and +0.090 nats, and the gain from learned mixing is +0.016, −0.020 and −0.056")
page = f'''<!DOCTYPE html>
<html lang="en" data-lang="en">
<head>
<meta charset="utf-8">
<meta name="robots" content="noindex">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="icon" type="image/png" sizes="32x32" href="../assets/img/favicons/favicon-32x32.png">
<link rel="icon" type="image/png" sizes="16x16" href="../assets/img/favicons/favicon-16x16.png">
<link rel="shortcut icon" href="../assets/img/favicons/favicon.ico">
<link rel="apple-touch-icon" sizes="180x180" href="../assets/img/favicons/apple-touch-icon.png">
<link rel="manifest" href="../assets/img/favicons/site.webmanifest">
<title>{title["en"]} | Wenhao Chai</title>
<meta name="description" content="{html.escape(desc, quote=True)}">
<meta name="theme-color" content="#FFFFFF">
<meta property="og:type" content="article">
<meta property="og:site_name" content="Wenhao Chai">
<meta property="og:title" content="{title["en"]}">
<meta property="og:description" content="{html.escape(desc, quote=True)}">
<meta property="og:url" content="https://wenhaochai.com/blogs/{SLUG}.html">
<meta property="og:image" content="https://wenhaochai.com/assets/img/og/card.png?v=3">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta property="og:image:type" content="image/png">
<meta property="article:published_time" content="2026-11-05">
<link rel="canonical" href="https://wenhaochai.com/blogs/{SLUG}.html">
<link rel="alternate" type="application/rss+xml" title="Wenhao Chai — Blog" href="https://wenhaochai.com/feed.xml">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:site" content="@wenhaocha1">
<meta name="twitter:title" content="{title["en"]}">
<meta name="twitter:description" content="{html.escape(desc, quote=True)}">
<link rel="preload" href="../assets/fonts/HankenGrotesk-normal-latin.woff2" as="font" type="font/woff2" crossorigin>
<link rel="stylesheet" href="../assets/site.css?v=106">
{style}
</head>
<body data-active="blog" data-base="../" data-sidenav>

<a class="skip-link" href="#main">Skip to content</a>

<div id="site-header"></div>

<main>
<div class="ar-page">

<div class="ar-head" id="main" tabindex="-1">
  <div class="ar-crumb"><b><a href="../blogs.html">Blog</a></b><span>›</span><span>{title["en"]}</span></div>
  <div class="ar-eyebrow"><span>Nov 5, 2026</span>
    <div class="ar-lang" role="group" aria-label="Language"><button type="button" data-lang="en" aria-pressed="true">EN</button><button type="button" data-lang="zh" aria-pressed="false">中文</button></div>
  </div>
  <h1><span lang="en">{title["en"]}</span><span lang="zh">{title["zh"]}</span></h1>
  <p class="ar-standfirst" lang="en">{stand["en"]}</p>
  <p class="ar-standfirst" lang="zh">{stand["zh"]}</p>
  <div class="ar-authors"><span class="av"><img src="../assets/img/images/avatar-800.jpg" alt=""></span><span lang="en">By Wenhao Chai</span><span lang="zh">作者 Wenhao Chai</span></div>
</div>

<aside class="ar-toc" aria-label="Contents">
  <h3 lang="en">Contents</h3><h3 lang="zh">目录</h3>
  <ol>
{chr(10).join(toc)}
    <li><a href="#ack"><span lang="en">Acknowledgements</span><span lang="zh">致谢</span></a></li>
    <li><a href="#notes"><span lang="en">References</span><span lang="zh">参考文献</span></a></li>
    <li><a href="#cite">Citation</a></li>
  </ol>
</aside>

<div class="ar-main">
<!-- the opening film: the setup, the compared variables on screen, ending on the result (tmp/pls_routing/film) -->
<figure class="ar-video" id="overview">
  <div class="rpf-film" role="img" aria-label="{html.escape(FILM_ALT, quote=True)}"></div>
</figure>

{"".join(body)}
<section id="ack">
  <h2><span lang="en">Acknowledgements</span><span lang="zh">致谢</span></h2>
  <p lang="en">Compute came from Princeton Language and Intelligence on the Della cluster. We thank the Marin project for its codebase, tokenizer, and training recipe.</p>
  <p lang="zh">实验算力来自普林斯顿语言与智能中心（PLI）的 Della 集群。感谢 Marin 项目的代码库、分词器和训练配方。</p>
</section>

<section id="notes">
  <h2><span lang="en">References</span><span lang="zh">参考文献</span></h2>
</section>

<section id="cite">
  <h2>Citation</h2>
  <pre>@misc{{chai2026residualpaths,
  title        = {{{title["en"]}}},
  author       = {{Chai, Wenhao}},
  year         = {{2026}},
  howpublished = {{Blog post}},
  url          = {{https://wenhaochai.com/blogs/{SLUG}.html}}
}}</pre>
  <div class="article-foot">
    <a href="../blogs.html"><span lang="en">All posts</span><span lang="zh">全部文章</span></a>
    <a href="#main"><span lang="en">Back to top</span><span lang="zh">回到顶部</span></a>
  </div>
</section>

</div><!-- /.ar-main -->
</div><!-- /.ar-page -->
</main>

<div id="site-footer"></div>
<script src="../assets/site.js?v=96"></script>
<script src="../assets/article-charts.js?v=36"></script>
<script src="../assets/data/{SLUG}-figs.js?v=1"></script>
<script src="../assets/{SLUG}-film.js?v=1"></script>
<script>RPFilm.mount(document.querySelector("#overview .rpf-film"));</script>
<script>
/* Figure specs from assets/data/{SLUG}-figs.js (tmp/pls_routing/build_figs.py), drawn by C.epoch. */
ArticleCharts.define(function (C, t) {{
  var S = window.RR_FIGS || {{}};
  Object.keys(S).forEach(function (id) {{ if (document.getElementById(id)) C.epoch(id, S[id]); }});
}});
</script>
</body>
</html>
'''
(SITE / "blogs" / f"{SLUG}.html").write_text(page, encoding="utf-8")
(SITE / "assets/data" / f"{SLUG}-figs.js").write_text((HERE / "rr-figs.js").read_text().replace("tmp/pls_routing/build_figs.py", f"tmp/pls_routing/build_figs.py for blogs/{SLUG}.html"), encoding="utf-8")
(SITE / "assets" / f"{SLUG}-film.js").write_text((HERE / "film/residual-paths-film.js").read_text(), encoding="utf-8")
# word counts and the numbers that are in the text, for the audit
def strip(s): return re.sub(r"<[^>]+>", "", s)
en = " ".join(strip(p["en"]) for k in order for p in sec[k]["paras"]); zh = "".join(strip(p["zh"]) for k in order for p in sec[k]["paras"])
print("words EN", len(en.split()), "| CJK chars ZH", len(re.findall(r"[一-鿿]", zh)))
json.dump({"en": [strip(p["en"]) for k in order for p in sec[k]["paras"]], "zh": [strip(p["zh"]) for k in order for p in sec[k]["paras"]]}, open(HERE / "post/text.json", "w"), ensure_ascii=False)
print("wrote", SITE / "blogs" / f"{SLUG}.html")
