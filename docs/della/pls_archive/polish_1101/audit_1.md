# 第 1 部分审核：润色是否改变了含义

每条问题的修正文本写在 overrides_1.json 里。每条修正只改 NEW 中出问题的几个词，润色后的其余措辞保留。所有数字、标签和链接与 NEW 完全一致。没有单元需要退回 OLD。

## 问题

1. **u000zh**｜含义
   - NEW：「不含 embedding 和输出头」
   - 修正：「不含 embedding 和头」
   - 原因：英文写的是 "without the embedding and the heads"，指所有的头，独立头也包括在内。NEW 只排除输出头，N 的定义因此变了。

2. **u012zh**｜含义
   - NEW：「是否改变每一层学到的内容」
   - 修正：「是否改变每一层学到多少」
   - 原因：英文是 "how much each layer knows"，问的是多少，不是内容是什么。图上画的是损失，表示的是多少。
   - NEW：「连最后一层都高于 4 层普通模型」
   - 修正：「它们的最后一层甚至高于 4 层普通模型」
   - 原因：「连……都」把强调放在了"最后一层"上。原文强调的是比较对象"连更浅的 4 层模型都不如"。NEW 的说法读起来像在说"最后一层本来最不该高"。

3. **u022en**｜含义（限定词丢失）
   - OLD："was first proposed"
   - NEW："was proposed"
   - 原因：丢了 "first"。这是一个历史限定："最初"是为深网络提出的。中文 u022zh 仍保留「最初」，中英文因此也对不上。

4. **u022zh**｜含义、配对
   - OLD：「每多一个读出头增加 65.7M」
   - NEW：「每多一个独立头增加 65.7M」
   - 原因：英文是 "each extra head"。这张图里独立头和探针都给每层加了头，所以这句话对两种设置都成立。NEW 把它缩小到只说独立头。

5. **u023en**｜含义（程度减弱）
   - OLD："make the shallow and middle layers much lower"
   - NEW："lower the shallow and middle layers"
   - 原因：丢了 "much"。中文 u023zh 仍是「大幅降低」，中英文对不上。修正为 "make the shallow and middle layers much lower but raise the last layers"。

6. **u034en**｜含义（转折丢失）
   - OLD："follows DepthBench's definition and initialization but trains with the baseline's MuonH recipe"
   - NEW："... and trains with ..."
   - 原因：原文用 "but" 标出一个偏离：本页的训练配方和 DepthBench 不同。后面一句（MuonH 让初始缩小保持整个训练）正是依赖这个偏离。改成 "and" 后，偏离看不出来了。中文仍是「但」。

7. **u034zh**｜含义（2 处）
   - NEW：「初始化时的缩放会贯穿整个训练」
   - 修正：「初始化时的缩小会贯穿整个训练」
   - 原因：英文是 "down-scaling"。「缩放」不分放大和缩小，方向丢了。
   - NEW：「探针点在层间跳动，因此该面板用单独的 y 轴范围」
   - 修正：去掉「因此」
   - 原因：OLD 的英文和中文都只是并列两件事。NEW 加了一个因果连接，英文没有。

8. **u036zh**｜中文（读成错误的说法）
   - NEW：「换第二个初始化重复两次运行」
   - 修正：「换第二个初始化重复这两个运行」
   - 原因：「重复两次运行」读起来像"把运行重复做两遍"。原意是把普通模型和独立头这两个运行各换种子重做一次。

9. **u037zh**｜含义、配对
   - NEW：「种子 0 下看似消失的代价」
   - 修正：「种子 0 下看似消失的最后一层代价」
   - 原因：英文是 "the missing last-layer cost"。NEW 丢了"最后一层"这个范围。在种子 0 下，其他各层的代价并没有消失。

10. **u039zh**｜含义
    - NEW：「Hyper-Connections 在最后一层代价约减半」
    - 修正：「在最后一层的代价约为其他结构的一半」
    - 原因：英文是 "ends about half as high"，即和其他结构比。「减半」读起来像它自己的代价从某个值降了一半，比较对象丢了。

11. **u042en**｜含义（比较被改写）
    - OLD："The test leaves the cause open; mHC also differs from HC in its mixing, which it keeps non-negative."
    - NEW："... since mHC also keeps its mixing non-negative."
    - 原因：原文的要点是"mHC 和 HC 还有第二处不同"（混合是否非负），所以原因还不能确定。NEW 去掉了 "differs from HC"，这个对比关系没有写出来。中文仍写「mHC 与 HC 在混合上也不同」。修正时把这个对比放回句中，并保留 "since"。

12. **u042zh**｜含义
    - NEW：「最后一根柱子像 mHC 那样让所有写入保持为正、其余不变」
    - 修正：「最后一根柱子是像 mHC 那样让所有写入保持为正、其余不变的 HC」
    - 原因：NEW 没有说最后一根柱子是哪个结构。读者会以为它是某种 mHC 变体。图例写的是 "HC with positive writes"。

## 已核对、没有问题的单元

- 名称：图中名称和 plc-figures.js 的图例一致，包括 "Probes only"、"Hyper-Connections"、"mHC"、"AttnRes (Block)"、"HC with positive writes"，以及 fig-d48 的红线（Probes only）。术语表也已遵守：骨干、独立头、探针、普通模型、运行、种子、读出。
- 本部分没有标为 "accepted despite: figure numbers lost" 的单元。
- 共享头的说法：u003en 把 "the model's output head" 改成 "the shared output head"。这个面板就是共享头设置，所以说法正确。

## 统计

- 核对单元数：84 个（42 对，有些对只有一种语言改动）。
- 发现问题：12 个单元，共 14 处。
- 退回 OLD 的单元：0 个。
