"""Merge the captions of Figs 38 and 50 (Argon draft, audited and corrected by hand, 2026-11-04) into the page; bump the data versions."""
import html, re
P = "/scratch/gpfs/GROUP/USER/tmp/plc_site_wt/exps/per-layer-contribution.html"
s = open(P, encoding="utf-8").read()
e = lambda t: html.escape(t, quote=True)
def block(fid):
    i = s.index(f'id="{fid}"'); a = s.rfind('<figure class="viz"', 0, i); b = s.index('</figure>', i)
    return a, b
def attr(blk, name, new):
    m = re.search(name + r'="([^"]*)"', blk); assert m, name
    return blk[:m.start(1)] + e(new) + blk[m.end(1):]
def span(blk, name, start, end, new):
    """Replace the text from start through end (both plain text, on the unescaped attribute) with new."""
    m = re.search(name + r'="([^"]*)"', blk); assert m, name
    v = html.unescape(m.group(1))
    i = v.index(start); j = v.index(end, i) + len(end)
    assert v.count(start) == 1, start
    return blk[:m.start(1)] + e(v[:i] + new + v[j:]) + blk[m.end(1):]
F38_EN = ("With the dynamic parts of read and write taken from one joint projection of all streams, as in mHC, and the learned mixing kept, Hyper-Connections pays 0.009 nats, against 0.047 for Hyper-Connections itself. Figure 50 shows that most of this drop comes from a worse ordinary model. Five more Hyper-Connections variants are still training, and their bars are empty. In the first, the mixing is one learned number per sublayer. It scales the whole state and moves nothing between streams. In the second, the mixing only shrinks the stream sum. Its gains are fixed, one per sublayer, and come from the ordinary Hyper-Connections model. Every difference between streams stays as it is. The third is the same with half the dose. We replace each gain with its square root. In the fourth, the mixing only moves content between streams. The outgoing weights of each stream are non-negative and add up to one. The stream sum stays unchanged, and repeated mixing cannot make the state grow without bound. In the fifth, the mixing keeps the stream sum and allows negative weights. With the streams stored in bfloat16, its ordinary model diverged at step 27, and we stopped that run. We run it again with the streams stored in float32. These five variants test whether the effect of the mixing on the stream sum is what matters. The Pre-LN variant applies the same fixed gains to its running sum and has no streams. Its ordinary model never trained. At step 2,019 its last-layer loss was 7.74 nats, the entropy of the token frequencies is 7.75, and Pre-LN is at 3.49. We stopped that run, so this bar stays empty. We think the gains cut the residual path, so that the last layer's loss alone cannot reach the earlier layers. The separate-heads model of this variant has a loss at every layer and does train. It is still running.")
F38_ZH = ("如果像 mHC 一样，用所有流的同一个联合投影算出读写的动态部分，同时保留学到的混合，Hyper-Connections 的代价只有 0.009 nats，原版 Hyper-Connections 是 0.047。图 50 表明，这一降幅主要来自普通模型变差。另外五个 Hyper-Connections 变体还在训练，它们的柱子是空的。第一个变体的混合是每个子层一个学到的数，它只缩放整个状态，不在流之间移动任何东西。第二个变体的混合只缩小各流之和。它的增益是固定的，每个子层一个，来自 Hyper-Connections 普通模型。各流之间的所有差异都保持原样。第三个变体和它相同，只是剂量减半。我们把每个增益换成它的平方根。第四个变体的混合只在流之间移动内容。每条流向外分配的权重都非负，而且加起来等于一。各流之和保持不变，反复混合也不会让状态无限增长。第五个变体的混合保持各流之和不变，同时允许负权重。把流存成 bfloat16 时，它的普通模型在第 27 步发散，我们停掉了它。我们把流存成 float32 重新跑。这五个变体都用来检验混合对各流之和的影响是不是关键。Pre-LN 变体把相同的固定增益用在它的累加和上，而且没有流。它的普通模型根本训练不起来。到第 2,019 步，它最后一层的损失是 7.74 nats，token 频率的熵是 7.75，Pre-LN 是 3.49。我们停掉了这次运行，所以这根柱子一直是空的。我们认为这些增益切断了残差路径，只有最后一层的损失时，梯度到不了前面的层。这个变体的独立头模型每层都有损失，能够正常训练，它还在运行。")
F50_EN = ("At the final checkpoint, about 3% of the tokens have a state norm more than 100 times the median. For about half of them, this starts with one MLP write at layer 5, about 100,000 times the median write, and identity mixing keeps it through every later layer. These are easy tokens, often with an almost certain next token, such as \" able\" before \" to\". The MLP at these layers has one isolated high-gain direction. The product of the top singular values of the gate, up and down weights is 4,517 at layer 5, against 5–35 at healthy layers, and every layer where many tokens first blow up has this direction. The pre-norm scale at those layers grows while the gap to Hyper-Connections widens. In the other separate-heads designs, the early layers show no such outlier. These tokens account for about a fifth of the rise on 4,096 tokens of held-out text. We have not yet found the source of the rest. With the dynamic parts of read and write taken from mHC's joint projection, the separate-heads model stays at 2.996 nats, against 3.003 for Hyper-Connections, and the ordinary model rises to 2.987, against 2.956. The cost therefore falls to 0.009 nats because the ordinary model gets worse. With the mixing reduced to one learned number per sublayer, the ordinary model reaches 2.918 nats, inside the range of the mHC runs. Its separate-heads model is still training.")
F50_ZH = ("在最后的检查点上，约 3% 的 token 的状态范数超过中位数的 100 倍。其中约一半始于第 5 层的一次 MLP 写入，大小约是写入中位数的 10 万倍，单位混合让它一直留到后面每一层。这些都是容易的 token，下一个 token 往往几乎确定，例如 \" able\" 后面的 \" to\"。这些层的 MLP 有一个孤立的高增益方向。gate、up、down 权重最大奇异值的乘积在第 5 层是 4,517，正常的层只有 5–35，大量 token 首次超过这条线的层都有这个方向。随着与 Hyper-Connections 的差距拉大，这些层的前置归一化尺度也在增长。其他独立头结构的前面几层没有这样的离群值。在 4,096 个留出文本的 token 上，这些 token 贡献了上升量的约五分之一。其余部分的来源我们还没有找到。读写的动态部分取自 mHC 的联合投影时，独立头模型保持在 2.996 nats，Hyper-Connections 是 3.003；普通模型升到 2.987，Hyper-Connections 是 2.956。所以代价降到 0.009 nats，是因为普通模型变差了。混合只剩每个子层一个学到的数时，普通模型到了 2.918 nats，落在 mHC 运行的范围内。它的独立头模型还在训练。")
a, b = block("fig-arch-var")
blk = s[a:b]
blk = span(blk, "data-more-en", "Six more Hyper-Connections variants", "All but the first test whether the effect of the mixing on the stream sum is what matters.", F38_EN)
blk = span(blk, "data-more-zh", "另外六个 Hyper-Connections 变体", "除第一个之外的所有变体都用来检验混合对各流之和的影响是不是关键。", F38_ZH)
blk = attr(blk, "data-cap-en", "Identity mixing and bounded read-write maps do not explain mHC's full cost. In Hyper-Connections, identity mixing raises the cost most, and a joint read-write computation lowers it most.")
blk = attr(blk, "data-cap-zh", "单位混合和有界读写映射解释不了 mHC 的全部代价。在 Hyper-Connections 中，单位混合让代价升得最多，联合计算读写让代价降得最多。")
s = s[:a] + blk + s[b:]
a, b = block("fig-sep-probes")
blk = s[a:b]
blk = span(blk, "data-more-en", "At the final checkpoint, about 3% of the tokens", "We have not yet found the source of the rest.", F50_EN)
blk = span(blk, "data-more-zh", "在最后的检查点上，约 3% 的 token", "其余部分的来源我们还没有找到。", F50_ZH)
blk = span(blk, "data-more-en", "the eight Hyper-Connections runs with learned mixing reach 2.936–2.979", "2.936–2.979", "the nine Hyper-Connections runs with learned mixing reach 2.936–2.987")
blk = span(blk, "data-more-zh", "8 个学到混合的 Hyper-Connections 运行在 2.936–2.979", "2.936–2.979", "9 个学到混合的 Hyper-Connections 运行在 2.936–2.987")
blk = attr(blk, "data-cap-en", "mHC ordinary models beat Hyper-Connections with full learned mixing. Identity or single-number mixing brings its ordinary model near mHC's. Identity mixing gives separate heads the highest loss.")
blk = attr(blk, "data-cap-zh", "mHC 普通模型优于完整学到混合的 Hyper-Connections。改用单位混合或单个学到的数后，它的普通模型接近 mHC。单位混合下它的独立头模型损失最高。")
s = s[:a] + blk + s[b:]
for old, new in (("plc-record.js?v=109", "plc-record.js?v=110"), ("plc-figures.js?v=137", "plc-figures.js?v=138")):
    assert s.count(old) == 1, old
    s = s.replace(old, new)
open(P, "w", encoding="utf-8").write(s)
print("caption lengths", len("mHC ordinary models beat Hyper-Connections with full learned mixing. Identity or single-number mixing brings its ordinary model near mHC's. Identity mixing gives separate heads the highest loss."), len("mHC 普通模型优于完整学到混合的 Hyper-Connections。改用单位混合或单个学到的数后，它的普通模型接近 mHC。单位混合下它的独立头模型损失最高。"), len("单位混合和有界读写映射解释不了 mHC 的全部代价。在 Hyper-Connections 中，单位混合让代价升得最多，联合计算读写让代价降得最多。"))
