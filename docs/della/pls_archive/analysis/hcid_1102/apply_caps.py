"""Merge the captions of Figs 38 and 50 (Argon draft, audited and corrected by hand, 2026-11-03) into the page; bump the data versions."""
import html, re
P = "/scratch/gpfs/GROUP/USER/tmp/plc_site_wt/exps/per-layer-contribution.html"
s = open(P, encoding="utf-8").read()
e = lambda t: html.escape(t, quote=True)
def block(fid):
    i = s.index(f'id="{fid}"'); a = s.rfind('<figure class="viz"', 0, i); b = s.index('</figure>', i)
    return a, b
def attr(blk, name, new):
    m = re.search(name + r'="([^"]*)"', blk)
    assert m, name
    return blk[:m.start(1)] + e(new) + blk[m.end(1):]
def insert_after(blk, name, anchor, text):
    m = re.search(name + r'="([^"]*)"', blk); assert m, name
    v = m.group(1); a = e(anchor); assert v.count(a) == 1, anchor
    v2 = v.replace(a, a + " " + e(text)) if name.endswith("en") else v.replace(a, a + e(text))
    return blk[:m.start(1)] + v2 + blk[m.end(1):]
def sub(blk, name, old, new):
    m = re.search(name + r'="([^"]*)"', blk); assert m, name
    v = m.group(1); assert v.count(e(old)) == 1, old
    return blk[:m.start(1)] + v.replace(e(old), e(new)) + blk[m.end(1):]

F38_SHORT_EN = "Identity mixing and bounded read-write maps do not explain mHC's full cost. In Hyper-Connections, identity mixing raises the cost most and one shared write leaves it about the same. Clipped AttnRes queries still spike."
F38_SHORT_ZH = "单位混合和有界读写解释不了 mHC 的全部代价。在 Hyper-Connections 中，单位混合让代价升得最多，共用一个写入时代价差不多。裁剪 query 的 AttnRes 仍有尖峰。"
F38_EN = ("Hyper-Connections with its mixing fixed to the identity pays 0.169 nats, the largest cost of any 48-layer design we trained. Figure 50 splits this cost into its two models. "
          "In the ordinary Hyper-Connections model, the static part of the learned mixing shrinks the stream sum in the early layers, to about half per layer over layers 2–11, whereas identity and Sinkhorn mixing leave the sum unchanged. "
          "Three more Hyper-Connections variants are still training, and their bars are empty. In the first, the dynamic parts of read and write come from one joint projection of all streams, as in mHC. "
          "In the second, the mixing keeps the stream sum, because each stream's outgoing weights add up to one. In the third, the mixing is one learned number per sublayer, which scales the whole state and moves nothing between streams. "
          "The last two test whether the effect of the mixing on the stream sum is what matters.")
F38_ZH = ("固定为单位混合的 Hyper-Connections 代价是 0.169 nats，是我们训练过的所有 48 层结构中最大的。图 50 把这个代价拆到两个模型上。"
          "在普通 Hyper-Connections 模型中，学到的混合的静态部分会在前面几层缩小各流之和，第 2–11 层每层缩小到约一半。单位混合和 Sinkhorn 混合都不改变各流之和。"
          "另外三个 Hyper-Connections 变体还在训练，它们的柱子是空的。第一个变体的读取和写入，动态部分像 mHC 一样来自所有流的同一个联合投影。"
          "第二个变体的混合保持各流之和不变，因为每条流流出的权重之和等于 1。第三个变体的混合是每个子层一个学到的数，它只缩放整个状态，不在流之间移动任何东西。"
          "后两个变体用来检验混合对各流之和的影响是不是关键。")
F50_SHORT_EN = "Every mHC ordinary model beats every Hyper-Connections one. With identity mixing, the ordinary model of Hyper-Connections is close to mHC's, and its separate-heads model has the highest loss here."
F50_SHORT_ZH = "所有 mHC 普通模型都优于所有 Hyper-Connections 普通模型。改用单位混合后，Hyper-Connections 的普通模型接近 mHC，独立头模型的损失却是图中最高。"
F50_EN = ("Fixing the mixing of Hyper-Connections to the identity tests whether the learned mixing separates its ordinary model from mHC's. "
          "On the newer code, the ordinary model falls from 2.956 to 2.931 nats, below every Hyper-Connections row with learned mixing and above every mHC row. "
          "Its evaluation loss reaches mHC's level, at 3.579 nats against 3.576 for mHC with identity mixing and linear read and write, and 3.638 for Hyper-Connections. "
          "The probes change far more than the last layer. At layers 10–32 they end 1.03–1.23 nats above Hyper-Connections, and at layers 12–48 they stay within about 0.23 nats of that mHC design. "
          "The separate-heads model moves the other way and rises from 3.003 to 3.100 nats, the highest in this figure. The cost rises from 0.047 to 0.169 nats. "
          "The separate-heads gap to Hyper-Connections stays at 0.02–0.03 nats from step 1,000 to about step 2,750 of 4,959. It then grows to about 0.11 nats by step 4,000, while the gradients of the first three layers grow. "
          "At the final checkpoint, about 3% of the tokens have a state norm more than 100 times the median. For about half of them this starts with one MLP write at layer 5, about 100,000 times the median write, and identity mixing keeps it through every later layer. "
          "These tokens account for about a fifth of the rise on 4,096 tokens of held-out text. We have not yet found the source of the rest.")
F50_ZH = ("把 Hyper-Connections 的混合固定为单位矩阵，可以检验学到的混合是不是它的普通模型与 mHC 拉开差距的原因。"
          "在新代码上，普通模型从 2.956 降到 2.931 nats，低于所有学到混合的 Hyper-Connections 行，高于所有 mHC 行。"
          "它的评估损失是 3.579 nats，到了 mHC 的水平：单位混合加线性读写的 mHC 是 3.576，Hyper-Connections 是 3.638。"
          "探针的变化比最后一层大得多。第 10–32 层的探针最终比 Hyper-Connections 高 1.03–1.23 nats，第 12–48 层与上面那个 mHC 结构最多相差约 0.23 nats。"
          "独立头模型朝相反方向变化，从 3.003 升到 3.100 nats，是本图最高。代价从 0.047 升到 0.169 nats。"
          "从第 1,000 步到约第 2,750 步（共 4,959 步），独立头模型比 Hyper-Connections 高 0.02–0.03 nats。随后差距扩大，到第 4,000 步约为 0.11 nats，同时前三层的梯度也在变大。"
          "在最后的检查点上，约 3% 的 token 的状态范数超过中位数的 100 倍。其中约一半始于第 5 层的一次 MLP 写入，大小约是写入中位数的 10 万倍，单位混合让它一直留到后面每一层。"
          "在 4,096 个留出文本的 token 上，这些 token 贡献了上升量的约五分之一。其余部分的来源我们还没有找到。")

a, b = block("fig-arch-var"); blk = s[a:b]
blk = attr(blk, "data-cap-en", F38_SHORT_EN); blk = attr(blk, "data-cap-zh", F38_SHORT_ZH)
blk = insert_after(blk, "data-more-en", "Hyper-Connections with one write shared across streams pays 0.042 nats, close to Hyper-Connections.", F38_EN)
blk = insert_after(blk, "data-more-zh", "各流共用一个写入的 Hyper-Connections 代价是 0.042 nats，与原版差不多。", F38_ZH)
s = s[:a] + blk + s[b:]
a, b = block("fig-sep-probes"); blk = s[a:b]
blk = attr(blk, "data-cap-en", F50_SHORT_EN); blk = attr(blk, "data-cap-zh", F50_SHORT_ZH)
blk = sub(blk, "data-more-en", "Between Hyper-Connections and mHC, the separate-heads models are about equally good, at 2.99–3.03 nats with four or more streams.",
          "Between Hyper-Connections and mHC, the separate-heads models are about equally good, at 2.99–3.03 nats with four or more streams, apart from the identity-mixing Hyper-Connections row described below.")
blk = sub(blk, "data-more-en", "the eight Hyper-Connections runs reach 2.936–2.979", "the eight Hyper-Connections runs with learned mixing reach 2.936–2.979")
blk = sub(blk, "data-more-zh", "都在 2.99–3.03 nats，差距在普通模型：4 个 mHC 运行在 2.913–2.926 nats，8 个 Hyper-Connections 运行在 2.936–2.979。",
          "都在 2.99–3.03 nats（下文单位混合的 Hyper-Connections 除外）。差距在普通模型：4 个 mHC 运行在 2.913–2.926 nats，8 个学到混合的 Hyper-Connections 运行在 2.936–2.979。")
blk = insert_after(blk, "data-more-en", "and a far better ordinary model, at 2.918 against 2.979.", F50_EN)
blk = insert_after(blk, "data-more-zh", "普通模型好得多，为 2.918 对 2.979。", F50_ZH)
s = s[:a] + blk + s[b:]
assert s.count("plc-record.js?v=105") == 1 and s.count("plc-figures.js?v=133") == 1
s = s.replace("plc-record.js?v=105", "plc-record.js?v=106").replace("plc-figures.js?v=133", "plc-figures.js?v=134")
open(P, "w", encoding="utf-8").write(s)
print("ok")
