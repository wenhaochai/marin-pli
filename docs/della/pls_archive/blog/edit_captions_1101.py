"""2026-11-01: rewrite the Fig 38 (fig-arch-var) and Fig 49 (fig-workspace) captions of exps/per-layer-contribution.html
after the causal-test pairs wq w711-w714 finished. Usage: edit_captions_1101.py <site worktree> <U1> <RS>
U1 = the shared-write HC's stream-difference use on its separate-heads checkpoint (mean collapse, l <= 16, nats);
RS = Spearman of that use against the separate-heads L48 loss over the multi-stream designs (use_vs_terms.py).
Each old string must occur exactly once; nothing is written unless every replacement applies."""
import sys
from pathlib import Path

site, U1, RS = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
p = site / "exps/per-layer-contribution.html"
s = p.read_text()
A = "&#x27;"
R = [
    # Fig 38, short captions
    ("The signed, unbounded maps lower mHC" + A + "s cost only a little. Two bars will test Figure 49.",
     "The signed, unbounded maps lower mHC" + A + "s cost only a little, and identity mixing on top of them adds nothing. "
     "Hyper-Connections with one shared write pays about what Hyper-Connections pays."),
    ("可取负、无界的读写映射只让 mHC 的代价降低一点。两根新柱将检验图 49。",
     "可取负、无界的读写映射只让 mHC 的代价降低一点，再加单位混合也没有更低。各流共用一个写入的 Hyper-Connections 付的代价与 Hyper-Connections 差不多。"),
    # Fig 38, long captions
    ("The two bars after them will test the reading of Figure 49: mHC with identity mixing and linear read and write, and "
     "Hyper-Connections with one write shared by every stream.",
     "The two bars after them test the reading of Figure 49 on the newer attention and cross-entropy code, where "
     "Hyper-Connections pays 0.047 nats (0.048 on the old code). mHC with identity mixing and linear read and write pays "
     "0.085 nats, about what mHC with linear read and write pays (0.080), so identity mixing adds nothing on top of the "
     "signed maps. Hyper-Connections with one write shared by every stream pays 0.042 nats, about what Hyper-Connections pays."),
    ("后面两根柱将检验图 49 的解读：用单位混合和线性读写的 mHC，以及各流共用一个写入的 Hyper-Connections。",
     "后面两根柱检验图 49 的解读，用的是较新的注意力和交叉熵代码；Hyper-Connections 在新代码上付 0.047 nats（旧代码上 0.048）。"
     "用单位混合和线性读写的 mHC 付 0.085 nats，与 mHC 线性读写（0.080）差不多，所以在可取负的读写映射之上，单位混合不再降低代价。"
     "各流共用一个写入的 Hyper-Connections 付 0.042 nats，与 Hyper-Connections 差不多。"),
    # Fig 49, short captions
    ("pay less for separate heads. Two runs will test whether this is cause.",
     "pay less for separate heads. Opening that part to mHC raised its use fourfold. Its cost fell only to 0.085 nats, "
     "where the trend of the other designs predicts 0.06 and Hyper-Connections pays 0.034–0.049."),
    ("结构付的独立头代价越少。两次运行将检验这是不是原因。",
     "结构付的独立头代价越少。把这一部分打开给 mHC 后，它的使用升到 4 倍，代价只降到 0.085 nats；"
     "按其他结构的趋势应为约 0.06，Hyper-Connections 付 0.034–0.049。"),
    # Fig 49, long captions
    ("This is a correlation. Two runs will test it: mHC with identity mixing and signed writes, which opens the hidden part "
     "to mHC, and Hyper-Connections with one write shared by every stream, which closes it.",
     "This is a correlation, and two runs tested it. The first is mHC with identity mixing and signed, linear read and "
     "write maps, which opens the hidden part to mHC. Its separate-heads model uses the stream differences four times as "
     "much as mHC" + A + "s (1.00 against 0.25 nats), and its cost fell only to 0.085 nats. The trend of the earlier "
     "designs predicts 0.06 at this use, and Hyper-Connections pays 0.034–0.049. The second run is Hyper-Connections with "
     "one write shared by every stream, meant to close the hidden part. Its mixing still makes each stream a different "
     "weighted sum of the embedding and the earlier writes, so the streams still differ and the model still uses them "
     f"({U1} nats); it pays 0.042 nats. This run therefore does not test the reading. Splitting the cost into its two "
     "terms shows why the first run missed. Across the multi-stream designs, the use tracks the separate-heads model" + A +
     f"s own loss (rank correlation {RS}), and the new mHC" + A + "s separate-heads model is as good as that of "
     "Hyper-Connections (3.003 against 2.992–3.003 nats). Every mHC ordinary model is better than every Hyper-Connections "
     "one (2.913–2.926 against 2.936–2.979 nats), and the cost is the difference of the two models."),
    ("这是相关。两次运行将检验它：mHC 用单位混合和可取负的写入，向 mHC 打开这个看不到的部分；Hyper-Connections 让各流共用一个写入，把它关上。",
     "这是相关，我们用两次运行检验了它。第一次是用单位混合和可取负的线性读写的 mHC，它向 mHC 打开这个看不到的部分。"
     "它的独立头模型对各流之差的使用是 mHC 的 4 倍（1.00 对 0.25 nats），代价只降到 0.085 nats。"
     "按之前各结构的趋势，这个使用对应约 0.06 nats；Hyper-Connections 付 0.034–0.049。"
     "第二次是各流共用一个写入的 Hyper-Connections，本意是关上这个部分。它的混合仍让每条流成为嵌入和之前各次写入的不同加权和，"
     f"所以各流仍然不同，模型仍在使用各流之差（{U1} nats），代价是 0.042 nats。所以这一次检验不了这个解读。"
     f"把代价拆成两项可以看出第一次为什么没有达到预测。在多流结构中，使用跟着独立头模型自己的损失走（秩相关 {RS}）；"
     "新 mHC 的独立头模型和 Hyper-Connections 的一样好（3.003 对 2.992–3.003 nats）。"
     "所有 mHC 普通模型都比所有 Hyper-Connections 普通模型好（2.913–2.926 对 2.936–2.979 nats），而代价是这两个模型之差。"),
]
for old, new in R:
    n = s.count(old)
    if n != 1:
        sys.exit(f"{n} matches for: {old[:80]}")
for old, new in R:
    s = s.replace(old, new)
p.write_text(s)
print(f"{len(R)} replacements written to {p}")
