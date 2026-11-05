"""Merge the captions of Figs 38 and 50 into the page after w960, w962, w972, w1006 (Argon draft, corrected by hand, 2026-11-05); bump the data versions. Each old string must occur exactly once in its attribute; nothing is written otherwise."""
import html, re
P = "/scratch/gpfs/GROUP/USER/tmp/plc_site_wt/exps/per-layer-contribution.html"
s = open(P, encoding="utf-8").read()
e = lambda t: html.escape(t, quote=True)
def edit(fid, name, old, new):
    global s
    i = s.index(f'id="{fid}"'); a = s.rfind('<figure class="viz"', 0, i); b = s.index('</figure>', i)
    blk = s[a:b]
    m = re.search(name + r'="([^"]*)"', blk); assert m, name
    v = html.unescape(m.group(1))
    assert v.count(old) == 1, (fid, name, old, v.count(old))
    v = v.replace(old, new)
    s = s[:a] + blk[:m.start(1)] + e(v) + blk[m.end(1):] + s[b:]

F = "fig-arch-var"
edit(F, "data-more-en", "It scales the whole state and moves nothing between streams. In the second,",
     "It scales the whole state and moves nothing between streams. It pays 0.157 nats. In the second,")
edit(F, "data-more-zh", "它只缩放整个状态，不在流之间移动任何东西。第二个变体",
     "它只缩放整个状态，不在流之间移动任何东西。它的代价是 0.157 nats。第二个变体")
edit(F, "data-more-en", "The bars of the first, second and fifth variants are still empty, because their runs are still training. So far, neither half-dose shrinking nor moving-only mixing reaches the 0.047 nats of full learned mixing.",
     "The bars of the second and fifth variants are still empty, because their separate-heads models are still training. So far, scalar mixing, half-dose shrinking and moving-only mixing all pay more than the 0.047 nats of full learned mixing.")
edit(F, "data-more-zh", "第一、第二和第五个变体的柱子还是空的，因为它们还在训练。到目前为止，半剂量缩小和只移动混合都没有降到完整学到混合的 0.047 nats。",
     "第二和第五个变体的柱子还是空的，因为它们的独立头模型还在训练。到目前为止，标量混合、半剂量缩小和只移动混合的代价都高于完整学到混合的 0.047 nats。")

F = "fig-sep-probes"
edit(F, "data-cap-en", "Identity or moving-only mixing brings its ordinary model near mHC's and raises its separate-heads loss.",
     "Identity, scalar or moving-only mixing brings its ordinary model near mHC's and raises its separate-heads loss.")
edit(F, "data-cap-zh", "单位混合或只移动混合让它的普通模型接近 mHC，同时抬高独立头模型的损失。",
     "单位混合、标量混合或只移动混合让它的普通模型接近 mHC，同时抬高独立头模型的损失。")
edit(F, "data-more-en", "apart from the Hyper-Connections rows with identity, half-dose or moving-only mixing described below.",
     "apart from the Hyper-Connections rows with identity, scalar, half-dose or moving-only mixing described below.")
edit(F, "data-more-zh", "（下文单位混合、半剂量和只移动的 Hyper-Connections 除外）",
     "（下文单位混合、标量混合、半剂量和只移动的 Hyper-Connections 除外）")
edit(F, "data-more-en", "It is 359 with half-dose shrinking, at layer 2, and 612 with Pre-LN's fixed gains, at layer 12.",
     "It is 359 with half-dose shrinking, at layer 2, 612 with Pre-LN's fixed gains, at layer 12, and 97 with scalar mixing, at layer 10.")
edit(F, "data-more-zh", "Pre-LN 固定增益是 612，在第 12 层。", "Pre-LN 固定增益是 612，在第 12 层。标量混合是 97，在第 10 层。")
edit(F, "data-more-en", "inside the range of the mHC runs. Its separate-heads model is still training. With moving-only mixing, the ordinary model reaches 2.925 nats, also inside that range,",
     "inside the range of the mHC runs. Its evaluation loss is 3.565 nats, against 3.638 for Hyper-Connections. The separate-heads model reaches 3.075 nats, and the cost is 0.157 nats, just below the 0.169 and 0.240 of identity mixing. With moving-only mixing, the ordinary model reaches 2.925 nats, also inside the range of the mHC runs,")
edit(F, "data-more-zh", "落在 mHC 运行的范围内。它的独立头模型还在训练。只移动混合时，普通模型是 2.925 nats，同样落在这个范围内，",
     "落在 mHC 运行的范围内。它的评估损失是 3.565 nats，Hyper-Connections 是 3.638。独立头模型是 3.075 nats，代价是 0.157 nats，略低于单位混合的 0.169 和 0.240。只移动混合时，普通模型是 2.925 nats，同样落在 mHC 运行的范围内，")
edit(F, "data-more-en", "With Pre-LN's fixed gains, the separate-heads model reaches 3.191 nats",
     "With mixing that only shrinks the stream sum, the ordinary model reaches 3.030 nats, above half-dose shrinking. With mixing that keeps the stream sum and allows negative weights, the ordinary model reaches 2.928 nats, close to identity mixing. The separate-heads models of these two variants are still training. With Pre-LN's fixed gains, the separate-heads model reaches 3.191 nats")
edit(F, "data-more-zh", "Pre-LN 固定增益时，独立头模型是 3.191 nats",
     "只缩小各流之和时，普通模型是 3.030 nats，高于半剂量缩小。保持各流之和、同时允许负权重时，普通模型是 2.928 nats，接近单位混合。这两个变体的独立头模型还在训练。Pre-LN 固定增益时，独立头模型是 3.191 nats")
for a, b in (("plc-figures.js?v=139", "plc-figures.js?v=140"), ("plc-record.js?v=111", "plc-record.js?v=112")):
    assert s.count(a) == 1, a
    s = s.replace(a, b)
open(P, "w", encoding="utf-8").write(s)
print("ok")
