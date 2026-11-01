# 第 2 部分审核：含义变化

review_2.md 里没有标“accepted despite: figure numbers lost”的单元。所有修正都写在 overrides_2.json 里，都是对 NEW 的最小改动。数字、标签和链接都和 NEW 相同。

## 问题

1. **u048en｜含义**
   - OLD："Two explanations compete: the layer losses may dilute the final loss's updates"
   - NEW："The layer losses may dilute the final update"
   - 原因：NEW 删掉了“有两种相互竞争的解释”这层框架。"the final update" 还可以读成“最后一步更新”。原文说的是最终损失的更新。
   - 修正：恢复 "Two explanations compete." 和 "the final loss's updates"。

2. **u048zh｜含义**
   - OLD："把下面各层拉向浅层预测"
   - NEW："把浅层拉向浅层预测"
   - 原因：原文是“下面各层”（英文 lower layers）。NEW 改成“浅层”，范围变了。读起来也像同义反复。
   - 修正：改回“下面各层”。

3. **u052zh｜中文**
   - NEW："直到第 300 步或第 1,500 步普通模型已形成上下文电路"
   - 原因：这句话读成“直到普通模型已形成电路”。读者看不出两个步数是时间点，也看不出“普通模型已形成电路”是对这两个时间点的说明。
   - 修正：改成“直到第 300 步或第 1,500 步，这时普通模型已形成上下文电路”。

4. **u053zh｜含义**
   - OLD："更晚的那个用前几层换来最后两层"
   - NEW："更晚启用的方案只是牺牲前几层来换取最后两层"
   - 原因：NEW 加了“只是”。“只是”带贬低意味，英文没有这层意思。
   - 修正：删掉“只是”。

5. **u054en｜含义**
   - OLD："Fourteen further changes attack the cost from other sides … which readouts train the backbone at all"
   - NEW："We test fourteen further changes"
   - 原因：NEW 删掉了这些改法的目的（从其他方面处理代价）。
   - 修正：恢复 "that attack the cost from other sides"。

6. **u054zh｜含义**
   - OLD："另外十四种改法从其他方面处理这个代价"
   - NEW："我们又测试了十四种改法"
   - 原因：和 u054en 相同，目的被删掉了。
   - 修正：改成“我们又从其他方面测试了十四种处理这个代价的改法”。

7. **u055zh｜含义／成对**
   - OLD："这里另外十四种改法从其他方面处理这个代价"
   - NEW："这里再测试十四种改法"
   - 原因：目的被删掉了。英文 NEW 保留了 "attack the cost from other sides"，所以中英两句不再一致。
   - 修正：改成“这里再从其他方面测试十四种处理这个代价的改法”。

8. **u056zh｜含义／中文**
   - OLD："让第 6 层达到普通模型、同时让第 1 层留在独立头附近"
   - NEW："同时保住第 1 层并让第 6 层追上普通模型"
   - 原因：“保住第 1 层”说得含糊，丢掉了“留在独立头附近”这个具体标准。
   - 修正：改成“能让第 6 层追上普通模型，同时让第 1 层留在独立头附近”。

9. **u061en｜含义**
   - OLD："change 2 orthogonalizes the final loss's update apart from theirs"
   - NEW："change 2 makes the final loss's update orthogonal to theirs"
   - 原因：原文的意思是把最终损失的更新单独做正交化，不和读出的更新混在一起。这里的优化器是 MuonH，相关项目名是 optsplit。NEW 说的是把一个更新投影到与另一个更新正交的方向，这是另一种操作。
   - 修正：这一处改回 OLD 的措辞。

10. **u061zh｜成对／含义**
    - NEW："先把最终损失的更新与读出的更新正交化"
    - 原因：这句话可以读成“使两者正交”，和 u061en 修正后的意思不一致。OLD 的中文本来就有这个歧义。
    - 修正：改成“先单独对最终损失的更新做正交化，不混入读出的更新”。
    - 说明：这一条依据我对 "apart from theirs" 的理解。理解的依据是 MuonH 优化器和 optsplit 项目名。我没有看到改法 2 的代码。建议负责人确认。

11. **u067zh｜含义／成对**
    - OLD："探针读前几层比独立头差"
    - NEW："探针读前几层远不如独立头"
    - 原因：英文只说 "worse"。NEW 加了“远”，说法变强了。
    - 修正：改成“不如”。

12. **u071zh｜含义**
    - OLD："把下一个词的信息移到别的层"
    - NEW："推向更早的层"
    - 原因：英文是 "to other layers"。NEW 把“别的层”缩成“更早的层”，限定了方向。
    - 修正：改回“移到别的层”。

13. **u073zh｜含义**
    - OLD："我们把 520m 的差距看作运行之间的波动"
    - NEW："因此 520m 的差距属于运行间波动"
    - 原因：原文是作者的解读（we read as）。NEW 把它写成了已确定的事实。英文 NEW 保留了 "we read"。
    - 修正：改成“因此我们把 520m 的差距看作运行间波动”。

14. **u074en｜含义**
    - OLD：三个尺度上两线重合，520m 有小差距（看作波动）
    - NEW："give the same held-out next-token loss across model sizes"
    - 原因：520m 上差 0.005 nats，所以“the same”说得过满。数字移到长图注是规则允许的，长图注 u073en 里也有这些数字。但短图注的说法不能比数据更强。中文 NEW 写的是“基本重合”，英文和中文不一致。
    - 修正：改成 "nearly the same"。

## 已检查但未修改

- u057zh 的“提前进入”和 u058zh、u060zh 的“直接读出预测”：这两处只是把原意说得更明白，没有改变结论。
- u059zh 删掉了“在这项评测中”，英文 NEW 也改成了 "here"。范围变化很小，不影响结论。
- u077en 用 "Yet" 代替了 "so"，前提和结论都还在。
- 名称：短图注里的 "output head" 替换了 "final head"，符合术语表。"Probes only"、"Separate heads from step 300" 等图例名没有被改。

## 统计

- 检查的单元：71 个（u044 到 u079，其中 u072 只有中文）。
- 发现的问题：14 个，都用小改动修正，没有单元退回 OLD。
