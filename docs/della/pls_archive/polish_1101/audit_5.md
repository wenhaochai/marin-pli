# 第 5 部分审核：含义改变检查

范围：review_5.md 中的全部 45 个单元（9 张图的短注和长注 36 个，相关工作 9 个）。每处修改都在 overrides_5.json 里，改动只针对 NEW 文本，并且保留每个数字、标签和链接。

## 问题

1. **u135en｜含义（断言变强）**
   - OLD："New runs **can use** faster kernels"
   - NEW："New runs **use** faster kernels"
   - 原因：OLD 说新运行可以选用这两种 kernel，是一个可选项。NEW 说所有新运行都在用。这与下文冲突：下文把种子 0 在两种注意力 kernel 上各重跑一次，其中一次用的是 JAX flash。中文 u135zh 仍写"可用"，所以这一对的中英文也不一致。
   - 修改：恢复 "can use"。

2. **u141en｜含义（比较对象错了）**
   - OLD："its cost is **the common one** once its ordinary model has no spikes"
   - NEW："its cost **matches the others** once …"
   - 原因：OLD 说 AttnRes 的代价是多数结构共有的那个代价。NEW 说它与"其他结构"相同。但同一段里刚说过，Hyper-Connections 的代价最小，mHC 承担全部代价，所以"与其他结构都相同"是错的。
   - 修改："its cost matches that of most designs once …"。

3. **u141zh｜配对 / 含义（同上）**
   - OLD 和 NEW 都写"它的代价与其他结构相同"。这个错误在 OLD 里就有，NEW 保留了它。
   - 修改："它的代价与多数结构相同"，与修改后的英文一致。

4. **u143zh｜含义（中文加了断言、改了 hedge）**
   - OLD："这是相关，我们用两次运行**检验了它**。" NEW："这只是相关，我们用两次运行**检验因果**。"
     - 原因：英文只说用两次运行检验这个相关。"检验因果"给结论加了一个英文没有的框架。
     - 修改："我们用两次运行检验它。"
   - OLD："所以这一次**检验不了**这个解读。" NEW："这次运行**无法直接检验**上述解读"
     - 原因："直接"暗示这次运行能间接检验这个解读。英文 "does not test the reading" 是完全否定。
     - 修改："这次运行检验不了上述解读"。

5. **u144en｜含义（比较丢了）**
   - OLD："Its cost fell only to 0.085 nats, **where the trend of the other designs predicts 0.06** and Hyper-Connections pays 0.034–0.049."
   - NEW："yet lowers its cost **only a little**."
   - 原因：这几个数字按规则移到了长注，这一点没有问题。0.085、0.06、0.034–0.049 都还在 u143en 里。但 OLD 的论点是一个比较：代价的下降比其他结构的趋势预测的小。"only a little" 换成了一个绝对的判断，丢掉了与趋势的比较，而这个比较正是这张图检验相关的依据。
   - 修改："yet its cost falls less than the trend predicts."（共 199 个字符，在 200 个字符的上限内）

6. **u144zh｜含义（同上）**
   - OLD："代价只降到 0.085 nats；按其他结构的趋势应为约 0.06" NEW："代价却只降了一点。"
   - 修改："代价却降得比趋势预测的少。"（65 个字符）

7. **u146zh｜中文（主语不明，可能读成错误的论断）**
   - OLD："所以 **mHC** 付得更多。" NEW："因此代价更高。"
   - 原因：前一个分句的主语是"所有 mHC 普通模型"，宾语是"Hyper-Connections 普通模型"。所以"代价更高"可以读成"普通模型的代价更高"，读不出是 mHC 这个结构的独立头代价更高。
   - 修改："因此 mHC 的代价更高。"

8. **u147zh｜含义（中文把论断缩窄了）**
   - OLD："发现**有帮助**。" NEW："发现**有助于训练**。"
   - 原因：英文 "find that it helps" 没有说帮助的是哪一方面。Deeply-Supervised Nets 报告的是分类精度和收敛速度两方面的提升。"有助于训练"只说了其中一方面。
   - 修改：恢复"发现有帮助。"

## 已核对、未发现问题的要点

- u133/u134：从短注删掉的数字 "0"（Full 的种子 0）仍在长注 u133en 中（"Seed 0 of Full"），也在 u133zh 中（"Full 种子 0"）。尖峰的说明完整地移进了长注。
- u135/u136：丢掉的数字 "4" 来自代码名 "fa4thd"。按规则可以删掉代码名。"FA4" 仍在 u135en 和 u135zh 中。
- u143/u144：0.085、0.06、0.034–0.049 都在 u143en 和 u143zh 中。
- 名称：本部分用到的结构名和系列名（Pre-LN、Hyper-Connections/HC、mHC、AttnRes (Block)/(Full)、"mixing fixed to the identity"）都与 plc-figures.js 中的标题和图例一致。中文术语符合词表：独立头、骨干、读出、代价、种子、各流之差、下下个词。

## 统计

- 检查的单元：45 个。
- 发现的问题：8 个单元，9 处。
  - 含义：6 个单元（u135en、u141en、u143zh 两处、u144en、u144zh、u147zh）。
  - 配对：1 个单元（u141zh）。
  - 中文：1 个单元（u146zh）。
- 全部用小改动修复。没有单元退回 OLD 文本。
