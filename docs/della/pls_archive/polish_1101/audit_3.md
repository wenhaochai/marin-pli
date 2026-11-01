# 第 3 部分审查：润色后的含义变化

所有修正都是对 NEW 的最小改动。每个单元的数字、标签和链接都与 NEW 一致。overrides_3.json 中没有单元退回 OLD。

## 问题

1. u082en（含义）
   - OLD："far more than a model predicts on text it has not seen"
   - NEW："far above its accuracy on unseen text"
   - 原因：OLD 拿一般模型在未见文本上的水平作比较。NEW 改成拿 K3 自己在未见文本上的准确率作比较。页面没有测 K3 在未见文本上的准确率。修正为 "far more than a model predicts on unseen text"。

2. u082zh（含义，与 1 同一处）
   - OLD："远高于模型在没见过的文本上的水平"
   - NEW："远高于它在未见文本上的水平"
   - 原因：同第 1 条，比较对象从一般模型变成了 K3。修正为 "远高于模型在未见文本上的水平"。

3. u084en（含义）
   - OLD："which K3 has very likely seen"
   - NEW："which K3 has likely seen"
   - 原因：NEW 弱化了限定语。u083en 和 u084zh 都写 "very likely" 或 "很可能"。修正时恢复 "very likely"。

4. u090zh、u091zh（含义、成对）
   - OLD："MLP 跟随它们的注意力"
   - NEW："MLP 紧跟同层注意力"（u090zh），"读取规律紧跟同层注意力"（u091zh）
   - 原因："紧跟" 把结论说强了。英文是 "follow"。而且独立头下 MLP 读较早块的权重是 0.51 和 0.53，注意力是 0.70 和 0.79，两者并不接近。修正为 "跟随"。

5. u093zh（含义）
   - OLD："后面各层的注意力查询"
   - NEW："后半段各层的注意力查询"
   - 原因：长描述写的是 Block 从第 15 层起、Full 从第 18 层起，48 层中这已经在前半段。"后半段" 给出了错误的范围。修正时恢复 "后面各层"。

6. u094en（含义）
   - OLD："The block-end writes look like input for the next block, which the head then leaves out."
   - NEW："The block-end writes act as input for the next block, which the head leaves out."
   - 原因：OLD 是 "look like"（推测），NEW 是 "act as"（断言），限定语丢了。修正时恢复 "look like" 和 "then"。

7. u094zh（含义）
   - OLD："块末的写入看起来是给下一块的输入，输出头随后把它们排除在外。"
   - NEW："块末写入像是给下一块的输入，因此被输出头排除在外。"
   - 原因：OLD 写的是两件先后发生的事。NEW 的 "因此" 加了一个因果关系，英文没有这个因果。修正为 "块末写入像是给下一块的输入，输出头随后把它们排除在外。"

8. u098en（含义、英文无法读懂）
   - OLD："On Paloma C4, which K3 has likely not seen (final loss 2.05 nats), ... Its two mixes stay close, 0.31 nats apart on average"
   - NEW："On Paloma C4, likely unseen by K3 at final loss 2.05 nats, ... Its two mixes stay 0.31 nats apart on average"
   - 原因 1："likely unseen by K3 at final loss 2.05 nats" 读起来像 "未见过" 取决于损失值。这样 2.05 nats 就挂错了对象。
   - 原因 2：NEW 删掉了 "close"。"close" 是这句话的结论，中文 "平均只差" 也保留了这个意思。
   - 修正为 "On Paloma C4, which K3 has likely not seen and where its final loss is 2.05 nats, ..."，并恢复 "stay close, 0.31 nats apart on average"。

## 已检查、无问题的单元

- 名称：Block、Full、AttnRes (Block)、AttnRes (Full)、Kimi K3、普通模型、独立头、输出头 都与图标题和术语表一致。
- u080zh 中的 "所以/因此" 在 OLD 中已经有，不是新加的因果。
- u085zh 加了 "损失"，与图的内容一致。
- u081en 没有改动。u081zh 与它一致。

## 统计

- 检查单元：41 个（u080en 到 u101zh，包括 u081zh、u091zh、u097zh 三个只改了中文的单元）。
- 发现问题：8 条，涉及 9 个单元：u082en、u082zh、u084en、u090zh、u091zh、u093zh、u094en、u094zh、u098en。
