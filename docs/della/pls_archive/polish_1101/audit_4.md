# 第 4 部分审计：润色是否改变了意思

范围：review_4.md 的 54 个单元（27 对，u102 到 u128）。每个修正都是对 NEW 的最小改动，全文写在 overrides_4.json。所有修正保留 NEW 的全部数字和标签。长图注和段落的修正长度不超过 OLD。

## 问题

1. u102zh，成对。OLD 和英文都说“几个普通模型”里最后一个 MLP 压低正确的词。NEW 写成“普通模型最后一个 MLP 常压低正确的词”，泛化到所有普通模型。修正：改为“几个普通模型的最后一个 MLP 会压低正确的词”。

2. u104en，意思。OLD：“the model's own next-token loss”。NEW 删了 “the model's own”，中文 NEW 仍保留“模型自身的”。修正：把 “the model's own” 加回去。

3. u106en，意思。(a) OLD：“A few outlier tokens”，NEW：“Outlier tokens”。“少数”这个限定词丢了，中文仍有“少数”。(b) OLD 说 0.68 变成 99.8% 是“因为”源的大小（“its size … turns that 0.68 into”）。NEW 写成 “At 53,000 times the median source, that 0.68 becomes …”，句子没有主语，因果关系看不清。修正：恢复 “A few”，改为 “Because the source is 53,000 times the median source”。

4. u106zh，意思。(a) OLD 和英文都是“最多低 64 倍”（up to）。NEW 写“比均值低 64 倍”，丢了“最多”，把上限写成了定值。(b) “单独把输出传给第 48 层 MLP”读起来像“第 44 层独自传送”。原意是“只传给第 48 层 MLP”（to … alone）。修正：加“最多”；“单独”改为“只”。

5. u107zh，意思。与第 4 条 (b) 相同：“单独把输出传给”改为“只把输出传给”。

6. u108zh，意思（hedge 丢失）。OLD：“我们把它理解为对过度自信的残差流的纠正”，这是作者的解读。NEW：“相当于纠正过度自信的残差流”，读成了事实。英文 NEW 保留了 “which we read as”。修正：恢复“我们把它理解为……的纠正”。

7. u109en，意思（hedge 丢失，加入因果）。OLD：“which we read as a correction of an overconfident stream”。NEW：“pushes against the correct token to correct an overconfident stream”，把解读写成了目的或事实。修正：“, which we read as correcting an overconfident stream”。

8. u109zh，意思。NEW：“用来纠正过度自信的残差流”，问题同第 7 条。修正：“我们理解为在纠正过度自信的残差流”。

9. u111en，成对与意思。中文 OLD 和 NEW 都写“在多数结构里”。英文 NEW 写成 “reads middle layers nearly as well as their own heads”，没有范围限定。长图注说的是“十种结构中的八种”。修正：加 “in most designs”。为了让短图注不超过约 200 个字符，“Dark lines use …, and light lines use …”压缩为 “Dark lines use the output head, light lines each layer's readout”，“unlike in ordinary models”改为 “unlike ordinary models”。

10. u112en，意思。(a) OLD：“an effective rank near 380 to 400”，NEW 删了 “near”（中文仍有“左右”）。(b) OLD：“at some late layers”，NEW：“at several late layers”，数量被改了（短图注 u113 仍写 “some”）。修正：恢复 “near” 和 “some”。

11. u112zh，意思（新加因果）。OLD：“那里一个超大的来源主导了混合”，是位置描述。NEW：“因为一个极大的源主导了混合”，把相关写成了原因。修正：“那里一个极大的源主导了混合”。

12. u115en，意思。OLD：“Dropping only its branches costs 0.62 nats”。NEW 删了 “only”，读者可能以为去掉子层就等于删层。中文仍有“只去子层”。修正：加 “only”。

13. u115zh，中文与意思。“以及最依赖第 44 层的普通 AttnRes (Full)，达 8.99 nats”读起来像 AttnRes (Full) 删第 1 层的代价是 8.99 nats。实际上 8.99 nats 是删第 44 层的代价。修正：“以及普通 AttnRes (Full)：它最依赖的是第 44 层，删掉这一层的代价达 8.99 nats”。（这一单元标记了“numbers lost {'1': 1}”：OLD 的“第 1 层的大部分价值”里有一个“1”，NEW 改成了“第 1 层的作用”，数字仍在，只是少了一处重复，没有内容丢失。）

14. u117en，意思（hedge 丢失）。OLD：“as the theory predicts, its scales and its probe losses follow those of LayerNorm Scaling”。NEW 删了 “as the theory predicts”，中文 NEW 仍有“如理论预测”。修正：加回 “as the theory predicts”。

15. u118en，名字。长图注和图例都用 “start”（“Pre-LN, scales start at 1/√ℓ”，“LNS's start, step size and clip”）。NEW 短图注改成了 “the initial value”，同一个东西用了两个名字。修正：改回 “the start”。

16. u119zh，意思。(a) OLD：“尖峰需要依赖 token 的路由”，是必要条件。NEW：“可见尖峰来自依赖 token 的路由”，变成了充分原因。英文 NEW “require” 是对的。(b) NEW：“都低于 Pre-LN”，把静态和动态两条线都说成低于 Pre-LN。OLD 只说静态路由低于 Pre-LN。(c) NEW：“只是因为普通模型有尖峰”，加了 OLD 没有的“只”。修正：“可见尖峰需要依赖 token 的路由才会出现”；删“都”；删“只”。

17. u120en，意思。OLD：“Static routing ends near dynamic routing”。NEW：“matches dynamic routing”，说法变强了。修正：恢复 “ends near”。

18. u120zh，意思。“这些尖峰来自依赖 token 的路由”，问题同第 16 条 (a)。修正：“这些尖峰需要依赖 token 的路由才会出现”。

19. u121en，意思，共 5 处。
    (a) “leaves its cost unchanged” 改回 OLD 的“about the same”（修正为 “leaves its cost about the same”）。
    (b) OLD 的一句 “most of it comes from another part of mHC” 丢了，中文 NEW 仍有。修正：补回。
    (c) “test the mechanism of Figure 49”：OLD 是 “the reading of”（中文 NEW 是“解读”）。“mechanism”说法更强。修正：改回 “reading”。
    (d) “pays 0.042 nats, matching Hyper-Connections”：OLD 是 “about what Hyper-Connections pays”。修正：“close to Hyper-Connections”。
    (e) “Sharp reads do not cause the spikes, as Figure 41 confirms”：OLD 只把图 41 作为出处，NEW 加了 “confirms”。修正：改回 “(Figure 41)”。
    另外 NEW 开头的 “while two designs depart from the rest” 改为 “and two designs depart from it”，这样把补回的句子放进去后总长度不超过 OLD。

20. u121zh，意思。(a) OLD：“在第 659 步发散，所以已停止”。NEW：“发散而停止”，读起来像运行自己停了，其实是我们停的。(b) NEW：“图 41 证实尖锐读取不是尖峰的原因”，问题同第 19 条 (e)。修正：“并在第 659 步发散，我们停止了这次运行。尖锐的读取不是尖峰的原因（图 41）。”

21. u122en，意思。(a) OLD：“explains mHC's full cost”，NEW：“explain mHC's cost”。mHC 的代价有一部分由读写映射解释（同一句的后半句就这样说），所以必须是“full”。(b) “One shared write matches Hyper-Connections”：主语不清楚，“matches”也比 OLD 的 “about what … pays” 更强。修正：“full cost”；“HC with one shared write pays about the same”。修正后 203 个字符，在“约 200 个字符”之内。

22. u122zh，意思。(a) “解释不了 mHC 的代价”改为“全部代价”，理由同第 21 条 (a)。(b) “共用一个写入时 Hyper-Connections 代价不变”，OLD 是“差不多”。修正：“代价差不多”。

23. u127en，意思。OLD 用冒号连接两句，NEW 改成 “because the smaller cost came from spikes”，倒过来说“代价变一致是因为尖峰”，因果方向错了，还丢了 “earlier”。修正：“The separate-heads cost of AttnRes then matches the other designs. The earlier, smaller cost came from spikes in the ordinary models.”

## 已检查、无需修改

- u123en/zh 与 u124en/zh（标记 figure numbers lost {'48', '12'}）：短图注里的 12 和 48 都在 NEW 长图注里（“At 12 layers … still pays the full cost”，“a second seed of Hyper-Connections at 48 layers”，“The low cost at 48 layers does not depend on the implementation”；中文相同）。“第二个种子和另一种实现下代价仍低”这句已在长图注里。除此之外没有内容丢失。
- 名字：图例 “Ordinary, logit lens / own probe”、“Separate heads, own head”、“HC, mixing = I”、“HC, one shared write”、“Pre-LN, scales start at 1/√ℓ” 都和正文对得上。“final head”改为“output head”符合术语表。
- u123 里的 “fa4thd” 改成 “FA4 attention kernel”，属于把代码名改成通用说法，意思不变。

## 统计

- 检查单元：54 个（27 对）。
- 有问题的单元：23 个（意思 20 个，成对 2 个，名字 1 个；u111en 既是成对问题也是意思问题，按成对计）。
- 全部 23 个都用最小改动修好了，没有一个需要退回 OLD。
