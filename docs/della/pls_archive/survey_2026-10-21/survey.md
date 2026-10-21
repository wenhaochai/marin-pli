# 独立头把预测前移：十种残差设计的内部调研

调研日期：2026-10-21。输入：`/home/USER/research_notes/独立头与残差结构调研/` 下的 6 份笔记（项目发现、残差设计与路由、深度与深监督、kexue.fm、优化器与归一化、MTP）。

## 0. 摘要

最重要的结论有 5 条。

1. 独立头（每层一个 RMSNorm 加 lm_head，权重 1）的最后一层代价是已知现象。LayerSkip 明确写出：所有层所有步都加早退损失会降低最后一层精度。MSDNet 把它叫作中间分类器对特征生成的"干扰"。没有文献在从零开始的 LM 预训练中消除这个代价。项目 14 个修复全部失败，这与文献一致，而且是更严格的对照版本（[LayerSkip](https://arxiv.org/html/2404.16710v4)；[MSDNet](https://arxiv.org/abs/1703.09844)）。
2. 项目的新颖处在"内部"一侧。文献中没有人在同一架构上对比"有/无逐层损失"的层可删除性，没有人在 10 种残差设计上做逐层监督，也没有人公开分析过 Kimi K3 的 AttnRes 路由。这三点是项目最可能独有的贡献。
3. 多条文献把项目的几个异常连成一条机制线：AttnRes 用 RMSNorm 键做 softmax，键的大小不进入打分，所以"谁被读"只由 query 长度决定（苏剑林 9889：稀疏度 S(a) = exp(−σ²‖q‖²/2)）。项目的 query 用 Adam 训练且不加权重衰减，长度 4–8，读法近似二值（logit 差 32–67）。K3 的 query 长度只有 0.13–0.33，读法是软的、局部的。DepthBench 的 Block AttnRes"死段"是同一症状。值的范数不进入打分，所以 53,000× 的巨大源可以安全存在。
4. 对排队中的运行，文献给出可检验的预测。静态 HC 应保留动态 HC 增益的约 70–100%。静态 Full AttnRes 应保留约 30–60%，并且不应再出现 10⁴× 量级的源。Pre-LN 配 1/√ℓ 初始增益的运行，最终增益应更接近 Pre-LN（步长主导），损失与 Pre-LN、LNS 的差在约 0.01 nats 以内。
5. 最便宜、信息量最大的下一步是在现有 20 个 48 层检查点上做分析，不训练新模型。优先项：AttnRes query 长度与读熵、第 44 层巨大源是否为"常数偏置"、最后一层 MLP 反向修正是否为校准（温度）行为、最后一层隐状态"跳变"ζ_L、HC 有效路由矩阵。

---

## 1. 项目发现与文献的对照表

符号说明。"ord" = 普通模型（探针分离，不回传梯度）。"sep" = 独立头。"代价" = sep 损失减去 ord 或同深度普通模型的损失。状态列：**已知** = 文献报告过同一现象；**已解释** = 文献给出可用的机制；**矛盾** = 文献结果方向相反或数值不符；**新** = 未找到先例。

| # | 项目发现（来源） | 文献 / kexue.fm 的说法 | 状态 | 具体下一步实验 |
|---|---|---|---|---|
| A1 | 6 层 130m：sep 每层比同深度普通模型高 0.07–0.17 nats；L6 3.370 对 3.260（[YAML](/scratch/gpfs/GROUP/USER/project/marin-pls/logs/pls_depth_controls.yaml)）。48 层 Sandwich：12/24/36 层高 0.10/0.07/0.09 nats | LayerSkip：全层全步早退损失"降低最后一层精度"；Llama3-8B 续训 MMLU 66.5→60.5（[LayerSkip](https://arxiv.org/html/2404.16710v4)）。MSDNet："classifiers throughout interfere with the feature generation"（[MSDNet](https://arxiv.org/abs/1703.09844)）。苏剑林 8907：把主次任务平行对待"可能会明显降低主任务的效果"（[8907](https://kexue.fm/archives/8907)） | 已知 | 不再找"修复"。改测机制：在 sep 与 ord 上测逐层本征维数（TwoNN），检验代价是否与中段"高本征维抽象阶段"缩短同步（[Cheng et al.](https://arxiv.org/abs/2405.15471)） |
| A2 | 代价在主干特征，不在头：冻结主干重训头 2,000 步，没有更好的头（Fig 18）；stop-grad 共享头不减代价 | MSDNet 也把原因放在特征生成上。没有文献把"特征与头"分开测过 | 已解释（方向一致），分解方法为新 | 无需新实验。写论文时把它作为与 LayerSkip 的区别点 |
| A3 | 代价随训练增长：共享头 Δbpb 在 1,000 步为 +0.023，最终 +0.041（[PLSMD](/scratch/gpfs/GROUP/USER/project/marin-pls/docs/della/pls.md)） | Wei et al. 2026：训练越久，层冗余越低，决策集中到最后几层（[2603.23701](https://arxiv.org/abs/2603.23701)）。Shibata：最后一层跳变 ζ_L 随预训练单调增长，Pythia-1B 1.03→11.2（[2601.18302](https://arxiv.org/html/2601.18302v1)） | 已知，已解释 | 在已有中间检查点上画"代价 vs token 数"，并同时画 ζ_L。预测：两者同步上升 |
| A4 | 14 个修复全部失败；权重降低时最后一层变好、浅层变差；只让少数层回传梯度时最后一层代价变小 | LayerSkip 用二次层权重、轮转/渐进课程、层 dropout，仍需调参"避免最后一层精度下降"。DeeBERT 两阶段、BERxiT 交替训练、Gradient Equilibrium 都是同类修复（[DeeBERT](https://arxiv.org/abs/2004.12993)；[BERxiT](https://aclanthology.org/2021.eacl-main.8)；[Li et al.](https://arxiv.org/abs/1908.06294)） | 已知（项目结果更强） | 不再提"修复"。若要再试一项无参数方法，选 JREG（见 §4 B3），因为它不读出中间层 |
| A5 | 批级梯度余弦已为正，在训练中保持 0.8–0.9，测不到交叉点；PCGrad 式按 token 投影几乎无效 | 苏剑林 8907 的主次投影：λ = relu(−⟨g₀,g₁⟩)/‖g₁‖²。余弦为正时 λ = 0，投影不起作用 | 已解释：8907 投影在本项目中预计为空操作 | 不做 8907 投影训练。kexue 笔记推荐了它，但项目数据已预言其结果 |
| A6 | 早期关闭逐层监督（80/140 步，或 0–150 步线性降到 0）最终回到基线池，差 +0.0003–0.0006 bpb | Al-Rfou et al. 用中间损失辅助训练 64 层模型（逐步淡出的细节来自记忆，未核实）（[1808.04444](https://arxiv.org/abs/1808.04444)） | 部分已知 | 晚期淡出（最后 20% 训练把权重降到 0）尚未测试。它区分"持久特征扭曲"与"实时容量竞争"。成本：1 个 130m 运行 |
| A7 | CNN 深监督可提升最终层，LM 不行 | DSN、BYOT（+2.65%）在 CNN 上提升最终精度（[DSN](https://arxiv.org/abs/1409.5185)；[BYOT](https://arxiv.org/abs/1905.08094)）。单轮 LM 预训练是容量受限区，dropout 也有害（[2505.24788](https://arxiv.org/pdf/2505.24788)） | 已解释（区间依赖） | 多轮小语料运行（8 遍以上）：预测 sep 代价变小或转正。成本：1–2 个 130m 运行 |
| B1 | 48 层十种设计中 8 种的 L48 代价为 0.09–0.11 nats；HC 只付 0.048；Full AttnRes seed 1 付 0.061 | MSDNet：密集连接（后层能读到未被中间分类器改写的早期特征）减轻干扰。按此预测，HC、AttnRes、MoDA 应付更少 | 部分支持，部分矛盾：HC 与 Full 符合，Block AttnRes（0.113）、MoDA（0.103）、mHC（0.102）不符 | 在现有检查点上计算"最后一层对早期状态的读权重"，与代价做相关。若无相关，MSDNet 的解释不适用于 LM |
| B2 | HC 只付一半代价；隐藏写入（零和写入）理论在状态层面失败；hc_pos 正写入仍为 0.035；6 层 HC 付全价 0.125 | 没有文献研究 HC 上的深监督。苏剑林 11664：HC 展开后 a_{t+1,s} = H^pre·H^res 乘积·H^post，tanh 生成的 H^res 乘积"有爆炸或坍缩的风险"，系数可为负（[11664](https://kexue.fm/archives/11664)）。DepthBench：mHC 的双随机乘积趋向 (1/m)11ᵀ，传输有效秩 1.44–1.65，HC 为 2.53–2.81（[DepthBench](https://arxiv.org/html/2609.32534)） | 新 | (a) 分析：算出 HC/mHC 的有效路由矩阵 A 及其每 token 方差，与 AttnRes 的 A 并排画。(b) 训练：HC 用非负混合加有符号写入（hc_pos 的反面）。(c) 训练：HC 6/12/24 层深度扫描，定位半价出现的深度 |
| B3 | 普通 HC 在 3/4 深度就进入输出头坐标（Fig 17）；HC-ord 在 46–48 层 bf16 吸收损失最多 75.8% 写入能量 | DepthBench 的 logit lens：HC 早期表示"离最终预测更远"。两者测法不同（项目测与头坐标对齐；DepthBench 测四流之和解码后的 KL） | 矛盾（测法不同） | 用同一探针在 HC-ord 上同时报告两种量。若 HC 写入有符号抵消，求和 lens 会隐藏早期对齐 |
| B4 | HC 第 1 层混合把 4 份嵌入拷贝缩放为约 0.39、2.76、2.84、−0.81；删分支只损 0.62 nats，删整层损 5.69 | HC 原文：Λ 形路由，底层（如第 0、2 层）被反复读取；冻结宽度连接几乎回到基线（2.804），说明静态混合和读映射承载收益（[HC](https://arxiv.org/html/2409.19606v3)）。DenseFormer：嵌入权重早期为正、后期为负（[DenseFormer](https://arxiv.org/abs/2402.02622)） | 已知（模式一致），负系数的作用已解释 | 无需新训练。在 L2 静态 HC 中检查第 1 层是否学到同样的有符号缩放 |
| B5 | mHC 的 ord 模型最好（L48 2.922），但 sep 付全价；mHC 删除 layer 1 只损 0.37 nats | DepthBench：mHC 在 L=24 最优，之后传输秩坍缩。知乎实验称 H_res = I 已有竞争力（经 AttnRes 与 11664 转引，原帖未打开）（[Zhihu](https://zhuanlan.zhihu.com/p/2010852389670908320)） | 部分已知 | 训练 mHC 配 H_res = I（只学 pre/post）。检验"秩坍缩"是否解释 mHC 付全价 |
| C1 | Block AttnRes ord 第 1 块读法近似二值：4 个子层读嵌入（权重 0.62–1.0），7 个只读部分和；logit 差 32–67；query 长度 4–8 | DepthBench："第一块输出保存前，深度 softmax 对运行和给出几乎 0 权重……网络反复重读嵌入"；死段在 L=16/24/32 为 2/5/8 层（[DepthBench](https://arxiv.org/html/2609.32534)）。苏剑林 9889：注意力集中度只由 ‖q‖ 与键的离散度决定，RMSNorm 键固定了离散度（[9889](https://kexue.fm/archives/9889)） | 已知且已解释 | 分析：每个子层画 ‖w‖ 对读熵与最大 logit 差。训练（无参数）：对 query 范数做 Post Clip（[11729](https://kexue.fm/archives/11729)）。预测：读法变软，损失变化在噪声内 |
| C2 | K3 读当前块（注意力平均 0.73），query 长度 0.13–0.33；我们的 ord 模型不读本地；sep 的 query 与头 query 对齐（cos > 0.9） | AttnRes 原文 16 层图：每层最强地读紧邻前一层，嵌入始终有可观权重；块化后分布"更锐利"（[AttnRes](https://arxiv.org/pdf/2603.15031)）。K3 用 Moonlight 版 Muon（有权重衰减）（[11848](https://kexue.fm/archives/11848)；[Moonlight](https://arxiv.org/html/2502.16982)） | K3 的局部性已知；"对齐使两种混合一致"为新 | 分析：对比 K3 与我们模型的 ‖w‖ 分布。预测：query 长度的差别（权重衰减有无）解释"本地软读"与"二值远读"的差别 |
| C3 | Full AttnRes ord 第 44 层 MLP 写入为读入的 1,339×，为源中位数的 53,000×；只被第 48 层 MLP 以 0.68 权重读取；移除它损 8.95 nats | Massive activations：少数激活达中位数约 10⁴–10⁵×，近似与输入无关，置零灾难性，替换为均值无损，作用是偏置（[Sun et al.](https://arxiv.org/abs/2402.17762)）。OASIS：AttnRes 中 softmax 没有"空"通道，sink 与离群值更重（[OASIS](https://arxiv.org/html/2605.17887)）。AttnRes 只归一化键不归一化值，打分对大小不敏感 | 机制已部分解释；AttnRes 中按源测的范数比为新 | 分析：(i) 该源方向对 token 的余弦是否 > 0.9；(ii) 用 token 均值替换它的损失；(iii) 在 Full seed 1 检查点上检查是否复现。训练：深度 softmax 加固定 0 logit 的空槽（Softmax₁，无参数） |
| C4 | kexue 笔记推测"巨大源是嵌入" | 项目数据显示巨大源是第 44 层 MLP 的输出，不是嵌入。嵌入的大权重出现在 Block-ord 头混合的早期（第 6 层以前 0.88–0.92） | 矛盾（推测与数据不符） | 不做。报告中更正此推测 |
| C5 | 均匀权重消融代价巨大：ord Block 7.51、Full 13.52 nats；sep 1.53/3.12 | 苏剑林 11664：输入有 RMSNorm 时，均匀平均 ≡ 等权求和，所以零 query 的 AttnRes 在初始化时就是 Pre-LN 函数 | 已解释：该代价测的是协同适应，不是混合本身的价值 | 不做新实验。写作时把该数字表述为"协同适应代价" |
| C6 | 冻结路由到 token 均值（不重训）避免 78–95% 的初始点代价 | 静态混合在匹配设置下保留动态增益约 0–100%，多数为 40–70%：HC n=4 为 55–67%，AttnRes 全静态 59%，MUDDFormer 41–48%，DCA 逐维静态 96%，DenseFormer 在 AttnRes 框架中 0%（[HC](https://arxiv.org/html/2409.19606v3)；[AttnRes](https://arxiv.org/pdf/2603.15031)；[MUDDFormer](https://arxiv.org/abs/2502.12170)；[DCA](https://arxiv.org/abs/2502.06785)） | 已知（冻结测试偏高，因为权重已协同适应） | 已排队为 L2。预测见 §2 |
| D1 | 8/10 设计的 sep 模型有 13–28 层删除代价 < 0.005 nats；ord 只有 0–7 层 | LayerDrop、LayerSkip 让任意深度子网可用（[LayerDrop](https://arxiv.org/abs/1909.11556)）。Veit：残差网络像浅路径集成（[1605.06431](https://arxiv.org/abs/1605.06431)）。Jastrzębski：残差块近似逐块梯度下降（[1710.04773](https://arxiv.org/abs/1710.04773)）。苏剑林 9009：Pre-Norm"深度有水分"（[9009](https://kexue.fm/archives/9009)） | 现象类比已知；同架构有/无逐层损失的对照为新 | 分析：对每个可删层同时报告写入/流范数比与下游影响。若可删层的写入仍大，可删性来自"迭代估计的冗余"，不是"近恒等块"。并报告"跳过 vs 截断"差（[Glavas et al.](https://arxiv.org/abs/2410.20022)） |
| D2 | ord 的 Pre-LN、LNS、KEEL、mHC、MoDA 最后一层 MLP 对正确 token 推 −1.57 到 −2.79 logits；sep 在 ±0.04 内 | 残差锐化与抑制神经元（[Lad et al.](https://arxiv.org/html/2406.19384v3)）；熵神经元通过最终 LayerNorm 作温度（[Stolfo et al.](https://arxiv.org/html/2406.16254v2)）；置信度修正阶段（[Joshi et al.](https://arxiv.org/abs/2511.00280v1)）；LayerSkip 训练的模型 ζ_L = 4.62，普通 Llama-3.2-1B 为 20.6（[Shibata](https://arxiv.org/html/2601.18302v1)）。苏剑林 8747：最后一个 LN 的 gamma 会偏大，作为温度旋钮（[8747](https://kexue.fm/archives/8747)） | 机制已知；"平均推向正确 token 的反方向"在小型从零模型中未见报告 | 分析：(1) 按 L−1 层 lens 置信度分桶；(2) 用单一温度替换最后 MLP，看恢复多少 NLL；(3) 把写入分到反嵌入行空间与零空间；(4) 比较最终增益与其他层增益 |
| D3 | 20 个运行中 18 个的最终范数增益在最大能量坐标上训练到约 0 | Massive activations 在最后几层消退（[Sun et al.](https://arxiv.org/html/2402.17762v2)）。暗信号在反嵌入谱尾（[Cancedda](https://arxiv.org/abs/2402.09221)）。最终 LN 塌缩只在玩具回归模型中报告过（[Hoogland et al.](https://arxiv.org/html/2402.02364)） | LLM 中为新 | 分析：回归每 token 熵对该坐标大小；重算不含该坐标的最终 RMS，看 Δloss。若 Δloss 大，该坐标是温度旋钮 |
| D4 | sep HC 的流和有效秩从 400 降到 64；ord Full AttnRes 后期秩为 1 | 无文献测多流和的秩。Muon 权重谱更平（SVD 熵更高，[Moonlight](https://arxiv.org/html/2502.16982)） | 新 | 先检查 64 是否等于结构数字：模型头维是 64（8×64 头）。再测各流的秩与跨流余弦 |
| E1 | ord 模型在最后 1/4 深度才进入输出头坐标；sep 在中段已进入 | Csordás：后半层贡献少，中段有相变（[2505.13898](https://arxiv.org/html/2505.13898v1)）；Lad：预测集成在约 85% 深度达峰；Curse of Depth：近一半层"不如预期有效"（[2502.05795](https://arxiv.org/abs/2502.05795)） | 已知（48×512 窄深模型可能更晚） | 分析：画 ord 48×512 与 130m–1.2b 模型的 lens KL 对相对深度。Csordás 的"宽度瓶颈"假说预测窄深模型形成更晚 |
| F1 | LNS 与 Pre-LN 是同一函数类；Pre-LN 训练后增益是 LNS 换算增益的 2.1–3.5×；L48 ord：Pre-LN 2.983，LNS 2.996 | LNS 论文：130M–1B 困惑度下降（如 1B 17.02→15.71），并建议去掉缩放初始化（[2502.05795](https://arxiv.org/pdf/2502.05795)）。DepthBench：LNS 需 5× 学习率，深度变大不持续受益。Hyperball 论文不分析增益（[2606.16899](https://arxiv.org/html/2606.16899v1)） | 与 LNS 论文矛盾（MuonH 下 LNS 不优于 Pre-LN）；增益动力学为新 | 已排队为 L1。补第四格：初始 1、步长 η/√ℓ（nGPT 式 s_init/s_scale），组成 2×2 析因 |
| F2 | DeepNorm 两条曲线都上移约 0.27 nats；DeepNorm 有可测 bf16 代价，f32 统计恢复 75–92% | 苏剑林 8978：DeepNorm 的 α、λ 取决于优化器；LAMB 型（更新正比 ‖θ‖）行为 α = 1（[8978](https://kexue.fm/archives/8978)）。MuonH 固定 ‖W‖_F，所以 β 缩小的初始化永久保留。LNS 论文中 DeepNorm 在 350M–1B 发散。没有文献测 DeepNorm 的 bf16 代价 | 方向已解释；bf16 数值为新 | 分析（廉价）：在全部十种设计上开 f32 统计做对照。预测：只有 DeepNorm 恢复明显 |
| G1 | MTP（DeepSeek 式，权重 0.3）在 130m–1.2b 不降低 NTP 损失 | DeepSeek-V3：BPB 0.729→0.729，0.658→0.657（[2412.19437](https://arxiv.org/html/2412.19437v1)）。Gloeckle 并行头在 0.3B/0.6B 变差，1.3–3B 才交叉（[2404.19737](https://arxiv.org/html/2404.19737v1)） | 已知 | 无需新训练。可选：300m 加 Gloeckle 并行头对照，隔离"看见 x_{t+1}"的作用 |
| G2 | 后后一 token（after-next）增益只在约 60% 深度以后出现；1.2b 的 after-next 探针在顶层变平 | Future Lens：NTP 模型的 t+2 信息在中层达峰（GPT-J 第 14/28 层）（[2311.04897](https://arxiv.org/abs/2311.04897)）。EAGLE-3：顶层特征"对应下一 token 的 logits"（[2503.01840](https://arxiv.org/html/2503.01840)）。V-信息：探针曲线测的是"线性可读"信息，不是 Shannon 信息（[Xu et al.](https://arxiv.org/abs/2002.10689)） | 已知（顶层专化）；从零训练的 NTP vs MTP 逐层对照为新 | 分析：边缘化上限（[Mehra et al.](https://arxiv.org/html/2502.09419)）与条件探针（读 Emb(x_{t+1})，[Hewitt et al. 2021](https://arxiv.org/abs/2109.09234)） |
| G3 | 1.2b 中层 next-token 探针在 MTP 下差至 0.15 nats，最后一层恢复 | NextLat：MTP/JTP 降低最后一层 offset-1 探针（[2511.05963](https://arxiv.org/html/2511.05963)）。AdaMTP 称梯度干扰（[2608.00434](https://arxiv.org/abs/2608.00434)）。无文献测中层 | 新 | 需第二个种子。可选：λ=0.1、最后 10% 降到 0、在 0.6L 以下截断 MTP 梯度 |

---

## 2. 排队运行的预测

### 2.1 L1：Pre-LN 配 1/√ℓ 初始增益（作业链 15954024 → 15954025）

这个运行在初始化时与 LNS 计算同一个函数。区别在训练中。LNS 的有效增益 g = γ/√ℓ，Adam 每步改 γ 约 η，所以 g 每步改约 η/√ℓ。L1 中 g 每步改约 η。因此 L1 等价于"LNS 的增益学习率乘以 √ℓ"，最后一层乘约 √48 ≈ 6.9。这是调研笔记作者自己的推导，没有论文验证过；[nGPT](https://arxiv.org/html/2410.01131) 用 s_init/s_scale 把这两个量分开。

**预测 1：最终增益更接近 Pre-LN 的剖面，不是 LNS 的剖面。** 理由如下。Adam 的步长与参数大小无关，0.14 量级的增益每步收到相对很大的更新（苏剑林 8620 认为这种缩放不应共享主学习率，[8620](https://kexue.fm/archives/8620)）。MuonH 锁住矩阵范数，所以增益是唯一的尺度通道（[Hyperball](https://arxiv.org/html/2606.16899v1)）。Moonlight 报告 RMSNorm γ 不加权重衰减时每层输出 RMS 会过大（[Moonlight](https://arxiv.org/html/2502.16982)）。项目配置不对增益加权重衰减。置信度：中等。

**预测 2：L48 损失与 Pre-LN（2.983）和 LNS（2.996）的差都在约 0.01 nats 内。** MuonH 下 LNS 本来就不优于 Pre-LN，所以这次运行在损失上没有多少空间。

**判别量。** 记录深层增益 RMS 随训练的变化。若深层增益停在 1/√ℓ 的 2 倍以内，说明初始化主导；若增长到 O(1)，说明步长主导。同时记录每组相对步长 ‖Δγ‖/‖γ‖ 和最大注意力 logit（增益以平方进入 q·k）。

**风险。** 若增益参数以 bf16 存储或在 bf16 中相乘，0.14 附近的小更新会被舍入吞掉（bf16 在 1 附近间距 2⁻⁷ ≈ 7.8e-3；[Zamirai et al.](https://arxiv.org/abs/2010.06192)），会人为保持初始值。运行前应确认增益为 fp32。另一混杂因素：LNS 论文脚注说缩放初始化会削弱 LNS（[LNS](https://arxiv.org/pdf/2502.05795)）。笔记未确认项目的 LNS 是否用了缩放初始化。

### 2.2 L2：HC 与 Full AttnRes 的静态路由，从零训练（15954026 → 27；15954028 → 29 → 30）

参照数字（48 层 ord，L48 训练损失）：Pre-LN 2.983，HC 2.944（增益 0.039），Full AttnRes seed 1 2.955（增益 0.028），seed 0 3.045（有 11 次尖峰）（[PLSMD](/scratch/gpfs/GROUP/USER/project/marin-pls/docs/della/pls.md)）。

**静态 HC：预测保留 HC 增益的 70–100%，L48 约 2.944–2.956。** 依据：HC 原文 n=2 时静态 ≥ 动态，n=4 时静态保留 55–67%（[HC Table 2](https://arxiv.org/html/2409.19606v3)）。项目冻结测试避免 78–95%。本运行只有 2.6B tokens，动态项从 0.01 起步，增长时间短。静态 HC 能保留第 1 层的有符号混合，而项目发现这一混合承载大部分价值。风险：动态 tanh 项可能起稳定作用。DepthBench 在 500M 最深形状上看到 HC 梯度爆炸（[DepthBench](https://arxiv.org/html/2609.32534)）。

**静态 Full AttnRes：预测保留增益的 30–60%，L48 约 2.966–2.975。** 依据：Kimi 消融中全静态保留 59%（16 层）（[AttnRes Table 4](https://arxiv.org/pdf/2603.15031)）。48 层有 97 个源，选择更重要，所以预测更低。另一面：苏剑林说更复杂的设计"大多不如"静态 query 的简单版本（[11664](https://kexue.fm/archives/11664)），这是"动态部分作用小"的先验。两者冲突，所以区间宽。

**静态 Full AttnRes 的内部预测。** 静态权重不能只对部分 token 排除一个巨大源。要让 α 低于 1e-5，需要约 11.5 nats 的 logit 差。所以预测：静态模型中没有源超过中位数的约 10² 倍。这一预测是笔记作者的推理，不来自论文。若静态模型保留 ≥ 80% 增益，则说明键的动态依赖不重要，AttnRes 主要是"带 softmax 正性的静态 DenseNet"。

**分辨力警告。** 静态 Full 与 Pre-LN 的预测差只有 0.008–0.017 nats。48 层的种子噪声没有测过；130m 的 9 次运行在 0.004 内，但 Full AttnRes 两个种子相差 0.09（尖峰导致）。所以静态 Full 的结论只有在差值 > 0.02 nats 时可靠。静态 HC 与 Pre-LN 的预测差 ≥ 0.027 nats，较易分辨。

### 2.3 L5：130m 后验探针保留权重（作业 15953858）

问题：最后几层是否把写入方向"避开"after-next 探针的方向。

**预测：是。** 130m ord 中 after-next 探针从 L5 到 L6 只降 0.09 nats，next-token 探针在最后一层降 0.23 nats（[MTPL](/scratch/gpfs/GROUP/USER/tmp/pls_blog/mtpl.json)）。文献支持顶层专化于下一 token：残差锐化阶段压制给下一 token 预测加噪的特征（[Lad et al.](https://arxiv.org/html/2406.19384v3)）；EAGLE-3 说顶层特征"天然只限于下一 token"（[EAGLE-3](https://arxiv.org/html/2503.01840)）。预测第 6 层写入在 W_next 行空间中的份额明显高于在 W_after 行空间中的份额，且后者在最后一层下降。

**附加检验。** 对最后一层 MLP 做零/均值消融。若 after-next 探针损失在消融后变好，说明最后一层主动压制 t+2 可读方向。

**解释限制。** 这些探针是"RMSNorm + 线性"族。结论应写成"线性可读性"，不是"信息量"（[Xu et al.](https://arxiv.org/abs/2002.10689)；[Pimentel et al.](https://arxiv.org/abs/2004.03061)）。Future Lens 中线性探针（27.5–29.2%）远低于学习提示（48.4%），说明线性读出会大幅低估 t+2 信息（[Future Lens](https://arxiv.org/abs/2311.04897)）。

### 2.4 520m 探针重跑（15952439/40，PARAM_MAP=vocab）

**预测 1：两道闸门都通过。** 130m/300m/1.2b 在冻结头检查中都通过。之前 520m 的失败与 48 层 FSDP 误编译的特征相同（冻结头读 9.8 nats，评估为 3.547），PARAM_MAP=vocab 在 48 层已修复该问题（[YAML](/scratch/gpfs/GROUP/USER/project/marin-pls/logs/pls_depth_controls.yaml)）。

**预测 2：after-next 增益从约 60% 深度（24 层中约第 14 层）开始，最后一层增益在 0.26–0.38 nats 之间**（300m 与 1.2b 之间）。Wu et al. 报告预缓存（pre-caching）随规模增大（[2404.00859](https://arxiv.org/abs/2404.00859)），支持单调性。

**预测 3：中层 next-token 探针在 MTP 下的损失增加在 0–0.15 nats 之间。** 置信度低：只有 1.2b 一个种子显示该效应。

---

## 3. 建议实验的优先级

规则提醒。(1) 针对最后一层代价的修复不能增加参数（所有者 2026-10-18）。(2) 每张新图要先通过弹窗得到所有者批准。(3) 优先在现有检查点上做分析。(4) 所有者已拒绝的项不再提出：48 层无探针 Sandwich 对照、Sandwich seed 1、AttnRes Full 稳定化头 softmax、第二个宽深形状（[PLSMD](/scratch/gpfs/GROUP/USER/project/marin-pls/docs/della/pls.md)）。(5) 20 个 48 层检查点必须保留到检查完成（[MEM](/home/USER/.claude/projects/-home-USER/memory/project_pls_per_layer_supervision_closed.md)）。

成本等级：**S** = 现有检查点上的 CPU/单 GPU 分析，数小时以内；**M** = 1 个 130m 训练运行；**L** = 1 个 48 层训练运行（约 2.6B tokens）。

### 3.1 第一组：现有检查点上的分析

| 优先级 | 实验 | 回答的问题 | 成本 | 针对发现 |
|---|---|---|---|---|
| A1 | 每个 AttnRes 子层的 ‖w‖ 对读熵与最大 logit 差；ord、sep、K3 三组并排 | 二值读法与 K3 的软本地读法是否只由 query 长度（即"有无权重衰减"）决定 | S | C1、C2 |
| A2 | 第 44 层巨大源：方向对 token 的余弦；替换为 token 均值的损失；在 Full seed 1 检查点上复查 | 巨大源是否是"常数偏置通道"（massive activation 型）；是否随种子复现 | S | C3 |
| A3 | 最后一层 MLP 反向修正：按 L−1 置信度分桶；单一温度替换；行空间/零空间分解；最终增益对其他层增益 | 反向修正是否为校准（温度）行为；sep 为何去掉它 | S | D2、D3 |
| A4 | 20 个检查点上的最后一层跳变 ζ_L（Shibata 定义） | ord 是否高、sep 是否低（LayerSkip 模型 4.62 对 Llama 20.6） | S | D1、D2 |
| A5 | HC/mHC 有效路由矩阵 A 及每 token 方差；H^res 乘积范数；HC-ord 用同一探针报告"头坐标对齐"与"四流和 KL" | HC 半价是否与有符号路由或乘积爆炸相关；解决与 DepthBench 的 lens 矛盾 | S | B2、B3 |
| A6 | 可删层：写入/流比对下游影响；"跳过 vs 截断"差，按设计与 ord/sep 分列 | sep 的可删性来自"近恒等"还是"迭代冗余" | S | D1 |
| A7 | HC-sep 秩 64：各流秩、跨流余弦；先核对是否等于头维 64 | 秩 64 是学到的还是结构决定的 | S | D4 |
| A8 | 全部十种设计开 f32 归一化统计做评估 | bf16 代价是否只属于 DeepNorm | S | F2 |
| A9 | 逐层本征维数（TwoNN/GRIDE）与提示熵，ord 对 sep | sep 是否缩短中段高本征维阶段，并与最后一层代价同步 | S | A1 |
| A10 | MTP：边缘化上限与条件探针（读 Emb(x_{t+1})） | MTP 增加的是 t+2 边缘信息，还是与 x_{t+1} 互补的特征 | S（需训练探针） | G2、G3 |
| A11 | K3 或 Block-ord：从下一块的读取中去掉块末写入，对比从头读取中去掉 | 块末大写入是否为"下一块的输入" | S | C2 |
| A12 | 最后一层读早期状态的权重与 sep 代价做相关 | MSDNet 的"密集连接减轻干扰"是否适用于 LM | S | B1 |

### 3.2 第二组：新训练运行

| 优先级 | 实验 | 回答的问题 | 成本 | 针对发现 |
|---|---|---|---|---|
| B1 | Full AttnRes 深度 softmax 加固定 0 logit 空槽（Softmax₁，无参数） | 巨大源与 32–67 的 logit 差是否因"没有空通道"而生 | L | C1、C3 |
| B2 | AttnRes query 范数 Post Clip（无参数） | 二值读法是否为温度伪影；损失是否不变 | L | C1 |
| B3 | JREG（惩罚逐层位移，向最后层加权，λ=1）替代独立头 | 不读出中间层也能把工作推向中段吗；若能且最后一层无代价，代价与"输出坐标可解码"绑定 | M，后续 L | A1、D1 |
| B4 | HC 非负混合 + 有符号写入；HC 6/12/24 层深度扫描 | HC 半价来自混合符号吗；半价在什么深度出现 | M 到 L | B2 |
| B5 | mHC 配 H_res = I | mHC 付全价是否因双随机乘积秩坍缩 | L | B5 |
| B6 | Block AttnRes 无死段布局（第 1 块在 1–2 层后保存） | 去掉死段后，深度收益与二值读法是否改变 | L | C1 |
| B7 | 增益 2×2 析因的第四格：初始 1、步长 η/√ℓ | 分开初始化与步长对增益的作用 | L | F1 |
| B8 | 多轮小语料（≥ 8 遍）上的 sep 对 ord | 数据受限时 sep 代价是否变小或转正 | M | A7 |
| B9 | 晚期淡出：最后 20% 训练把逐层损失权重降到 0 | 代价是持久扭曲还是实时竞争 | M | A6 |
| B10 | 300m Gloeckle 并行头对照 | 隔离"看见 x_{t+1}"对 NTP 无损的作用 | M | G1 |

降级项。8907 主次投影：项目已测到批级余弦为正（0.8–0.9），投影系数恒为 0，预计无效。1/sg(L_l) 加权：它压低浅层权重，与已测的"随深度上升权重"同类，预计落在已知的权衡曲线上（[8870](https://kexue.fm/archives/8870)）。门控注意力（Qwen 输出门）能把最大激活降约 10×（[Qiu et al.](https://arxiv.org/html/2505.06708)），但它增加参数，只能用于理解内部，不能作为最后一层代价的修复。

---

## 4. 项目可能的新颖之处

以下各点在 6 份笔记的检索范围内没有找到已发表的先例。"没有找到"不等于"不存在"。笔记作者没有检索形式化验证（Lean）方面的文献，所以 Lean 部分的新颖性未评估。

1. **十种残差设计上的逐层独立头。** 没有文献在 HC、AttnRes、MoDA、mHC 等设计上训练逐层深监督。也没有文献解释"HC 只付一半代价"。
2. **同一架构有/无逐层损失、匹配计算量、从零训练的层删除对照。** LayerSkip 报告早退困惑度，不报告"保留顶层、删除中层"的结果（[LayerSkip](https://arxiv.org/html/2404.16710v4)）。
3. **"代价在特征，不在头"的分解**（冻结主干重训头）。文献没有这一分解。
4. **Kimi K3 的内部路由分析。** 没有公开论文测过 K3 的 AttnRes 权重、块末写入或 query 长度。
5. **AttnRes 中按源测量的范数比**（53,000× 源，私有通道）。OASIS 只报告峰度与 ‖x‖∞，且是在改装的 LLaMA 上（[OASIS](https://arxiv.org/html/2605.17887)）。
6. **最终范数增益在 massive 坐标上归零**（18/20 次运行）。只有玩具回归 Transformer 中有类似报告（[Hoogland et al.](https://arxiv.org/html/2402.02364)）。
7. **范数固定权重下 Adam 训练的增益动力学**，以及"Pre-LN 配 1/√ℓ 初始增益"这一运行。Hyperball 论文不分析增益（[Hyperball](https://arxiv.org/html/2606.16899v1)）。
8. **DeepNorm 的 bf16 代价数值**（1–2e-3 nats，f32 统计恢复 75–92%）。
9. **从零训练的 NTP vs MTP 逐层 t+2 探针对照**，以及 MTP 使 1.2b 中层 next-token 探针变差。NextLat 只探最后一层，Future Lens 只研究 NTP 模型。
10. **≥ 48 层、Muon 族优化器下的静态 vs 动态路由对比**（L2）。现有对比都在 16 层左右，且用 AdamW。

表述建议。第 1–3 点是项目论点的核心，写成"首个受控测量"。不要写成"首次发现最后一层代价"，因为该代价已知。

---

## 5. 研究者标出的差异与注意事项

### 5.1 项目内部

- **第 44 层的数字有四个版本。** 只从第 48 层 MLP 的读取中移除该源：8.95 nats。删除第 44 层：8.99 nats。DEPTHMED 中删除后的绝对 CE：12.34（约等于基线 + 8.99，隐含基线约 3.35）。Lean 引用 12.43 与 12.44 nats，看起来是绝对 CE，不是增量，尚不确定（[DEPTHMED](/scratch/gpfs/GROUP/USER/tmp/pls_blog/depthmed/attnres-ord.json)；[LEAN](/scratch/gpfs/GROUP/USER/project/resnext-lean/README.md)）。写作前需统一定义。
- **单种子。** 几乎所有臂只跑一次。例外：130m 基线池、AttnRes Full 两个种子、upto2 两个种子。Full 的两个种子在 ord L48 上差 0.09 nats。130m 的 0.004 噪声不能外推到 48 层。
- **AttnRes Full seed 0 的"负代价"是种子伪影**：第 2106 步头部深度 softmax 饱和，最后读出从 3.41 跳到 8.32。
- **裁剪组耦合。** 48 层时 47 个探针头与输出头共用梯度裁剪组，输出头梯度在多数步被缩放到 0.6–0.8。损失影响未测，所有者拒绝了对照。Lean 另证明共用裁剪组使重参数化等价在训练中不精确。
- **配方差异。** 所有设计按 DepthBench 定义与初始化，但用 MuonH 训练（DepthBench 用 AdamW，权重衰减 0.1）。MuonH 让 DeepNorm、HC、mHC 的初始缩小在整个运行中保持。只跑了一个形状（48×512）。
- **基线设计是 Sandwich-LN**（`hybrid_norm=True`），不是 Pre-LN。
- **K3 数据。** 93 层这一数字没有在 K3 论文摘要或苏剑林文章中得到确认；DepthBench 说 K3 用 8 个边界，与"每块 12 层、93 层"大致一致。WikiText 上 top-1 为 0.90，可能在训练数据中；Paloma C4 只读了 64×1024 tokens。
- **FSDP 误编译。** 48 层读出与旧的 520m 探针运行受影响；旧 520m 结果已判无效并从 Fig 21 删除。
- **未从数据文件读取的数字。** Fig 12 的权重 0.1 数字与 Figs 13–14 的逐层值只来自图注。
- **DLA 列表与删除列表不一致。** DeepNorm 在删除列表中，不在 DLA 列表中；Sandwich、HC、AttnRes 两个列表都不在。
- **HC 隐藏写入理论是事后提出的。** 状态层面的预测失败；hc_pos 测试没有确定原因。
- **头的优化器组。** 笔记写的是头用 AdamH（范数固定）。苏剑林认为 LM 头应使用列归一化 SGD，初始化与学习率按 Θ(1/d) 缩放（[11647](https://kexue.fm/archives/11647)）。冻结主干重训头没有找到更好的头，所以这一混杂因素的影响可能小。

### 5.2 文献与笔记

- **kexue 笔记的推测与项目数据矛盾**：它推测巨大源是嵌入；项目数据显示是第 44 层 MLP 的输出。
- **AttnRes 滑动窗口**：论文 Table 4 为 1.764（基线 1.766），苏剑林说"甚至还不如 Residuals"。两者略有冲突。
- **AttnRes "input-independent mixing" 行**没有说明标量是否经过 softmax、如何初始化。
- **苏剑林"25% 的收益"**在原文中没有定义。
- **OASIS 页面内部不一致**（正文 ‖x‖∞ ≈ 28，表中 84.43）。
- **Hyperball 论文中模型规模**：正文写"88B"，二手摘要写 8B，未确认。"Hyperball May Not Be a Free Lunch"在 arXiv 上没有找到。
- **DeepSeek-V4 用可学习最终混合器**这一点来自 DepthBench 的转述，未在 V4 原文中确认。
- **"mHC 可能不需要 m"知乎帖**未打开，结论为二手。Frac-Connections 只读到摘要片段。
- **来自记忆、未重新核实的内容**：MSDNet Fig. 3 数字、DeeBERT 两阶段与 BERxiT 交替的细节、Al-Rfou 的淡出计划、CALM 的层权重、Admin 与 Xiong 的细节、超级权重的位置、CogView PB-relax 细节、Gurnee 预测/抑制神经元细节。
- **二手数字**：Gromov et al. 的各模型崩溃阈值、Medusa 每头精度、Skean "高达 16%"、Wei et al. 关于 Pythia 的全文结论。
- **Future Lens Table 2 的列在 PDF 抽取时错乱**；48.4% 与摘要一致，其他单元格的 N 对应不确定。
- **KORMo 的 MTP 变体**（并行或串行）未确认。
- **DeepNet 本身的 Adam 更新界**没有打开原文，用的是苏剑林的 SGD 转述。
- **静态/动态百分比**是笔记作者根据论文表格自行计算的。

---

## 6. 结论

本次调研改变了项目的定位。最后一层代价不再是"待修复的问题"，而是已知现象的一个受控、更强的版本。项目真正的空白在另一侧：独立头怎样重排网络内部。这里有三个文献没有覆盖的测量：同架构的层可删性对照、十种设计上的代价差异（尤其 HC 半价）、K3 的路由。

调研还给出一条可以证伪的统一假说：AttnRes 系列的大部分异常（二值读法、死段、巨大源、K3 与我们模型的差别）都来自"RMSNorm 键 + 无权重衰减的 query 长度"。A1、A2 两项分析只需几小时。若它们支持该假说，B1、B2 两个无参数训练可以给出因果证据。若不支持，项目应放弃这条线，转向 HC 的有符号路由（A5、B4）。

---

## 7. 参考文献（按主题）

### 7.1 项目内部来源
- PAGE：`/scratch/gpfs/GROUP/USER/tmp/plc_site_wt/exps/per-layer-contribution.html`（[文件](/scratch/gpfs/GROUP/USER/tmp/plc_site_wt/exps/per-layer-contribution.html)）
- LEANPAGE：`/scratch/gpfs/GROUP/USER/tmp/plc_site_wt/exps/per-layer-contribution-lean.html` 与 `assets/data/plc-lean.js`（[文件](/scratch/gpfs/GROUP/USER/tmp/plc_site_wt/assets/data/plc-lean.js)）
- YAML：`/scratch/gpfs/GROUP/USER/project/marin-pls/logs/pls_depth_controls.yaml`（[文件](/scratch/gpfs/GROUP/USER/project/marin-pls/logs/pls_depth_controls.yaml)）
- PLSMD：`/scratch/gpfs/GROUP/USER/project/marin-pls/docs/della/pls.md`（[文件](/scratch/gpfs/GROUP/USER/project/marin-pls/docs/della/pls.md)）
- AUDIT：`/scratch/gpfs/GROUP/USER/project/marin-pls/docs/della/pls_audit_2026-10-07.md`
- MEM：`/home/USER/.claude/projects/-home-USER/memory/project_pls_per_layer_supervision_closed.md`
- LEAN：`/scratch/gpfs/GROUP/USER/project/resnext-lean/README.md`；HANDOFF：`/scratch/gpfs/GROUP/USER/project/resnext-lean/docs/HANDOFF.md`
- MTPL：`/scratch/gpfs/GROUP/USER/tmp/pls_blog/mtpl.json`；DEPTHMED：`/scratch/gpfs/GROUP/USER/tmp/pls_blog/depthmed/<design>-<ord|sep>.json`

### 7.2 残差设计与深度路由
- Zhu, Huang, Huang, Zeng, Mao, Wu, Min, Zhou (2024/2025). Hyper-Connections. arXiv:2409.19606, ICLR 2025. [link](https://arxiv.org/abs/2409.19606)
- Xie, Wei, Cao et al., DeepSeek-AI (2025/2026). mHC: Manifold-Constrained Hyper-Connections. arXiv:2512.24880. [link](https://arxiv.org/abs/2512.24880)
- DeepSeek-AI (2026). DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence. arXiv:2606.19348. [link](https://arxiv.org/abs/2606.19348)
- Zhu, Huang, Zhou et al. (2025). Frac-Connections. arXiv:2503.14125. [link](https://arxiv.org/pdf/2503.14125)
- Yang, Gao (2026). mHC-lite. arXiv:2601.05732. [link](https://arxiv.org/abs/2601.05732)
- Tian Xie (2026). Your DeepSeek mHC Might Not Need the "m"（知乎）. [link](https://zhuanlan.zhihu.com/p/2010852389670908320)
- mHC 用于参数高效微调 (2026). arXiv:2607.18130. [link](https://arxiv.org/html/2607.18130)
- Kimi Team: Chen, Zhang, Su et al. (2026). Attention Residuals. arXiv:2603.15031. [link](https://arxiv.org/abs/2603.15031)
- Kimi Team (2026). Kimi K3: Open Frontier Intelligence. arXiv:2607.24653. [link](https://arxiv.org/abs/2607.24653)
- Pagliardini, Mohtashami, Fleuret, Jaggi (2024). DenseFormer. arXiv:2402.02622. [link](https://arxiv.org/abs/2402.02622)
- Xiao, Meng, Li, Yuan (2025). MUDDFormer. arXiv:2502.12170, ICML 2025. [link](https://arxiv.org/abs/2502.12170)
- Heddes, Javanmard, Axiotis, Fu, Bateni, Mirrokni (2025). DeepCrossAttention. arXiv:2502.06785, ICML 2025. [link](https://arxiv.org/abs/2502.06785)
- Zhu, Fang, Liao, …, Wang (2026). MoDA: Mixture-of-Depths Attention. arXiv:2603.15619. [link](https://arxiv.org/abs/2603.15619)
- Zhou, Wu, Jiang, Obeid, Lan (2024/2025). Value Residual Learning. arXiv:2410.17897, ACL 2025. [link](https://arxiv.org/abs/2410.17897)
- Menghani, Kumar, Kumar (2024/2025). LAuReL. arXiv:2411.07501, ICML 2025. [link](https://arxiv.org/abs/2411.07501)
- AttnRes 参考文献中列出但未打开：MRLA arXiv:2302.03985；ANCRe arXiv:2602.09009；Dreamer arXiv:2601.21582；DDL arXiv:2601.00417；SiameseNorm arXiv:2602.08064；Residual Matrix Transformers arXiv:2506.22696；HybridNorm arXiv:2503.04598；LIMe arXiv:2502.09245。[AttnRes refs](https://arxiv.org/pdf/2603.15031)
- Wang, Huang, Kang, González-Martínez, Liu, Liu (2026). DepthBench. arXiv:2609.32534. [link](https://arxiv.org/html/2609.32534)
- Huang et al. (2016). DenseNet. arXiv:1608.06993. [link](https://arxiv.org/abs/1608.06993)
- Srivastava, Greff, Schmidhuber (2015). Highway Networks. arXiv:1505.00387. [link](https://arxiv.org/abs/1505.00387)

### 7.3 归一化、缩放与参数化
- Sun, Song, Li, Yin, Zheng, Liu (2025). The Curse of Depth in LLMs (LNS). arXiv:2502.05795, NeurIPS 2025. [link](https://arxiv.org/abs/2502.05795)
- Wang, Ma, Dong, Huang, Zhang, Wei (2022). DeepNet. arXiv:2203.00555. [link](https://arxiv.org/abs/2203.00555)
- Chen, Wei (2026). Post-LayerNorm Is Back (KEEL). arXiv:2601.19895. [link](https://arxiv.org/abs/2601.19895)
- Ding et al. (2021). CogView (Sandwich-LN). arXiv:2105.13290. [link](https://arxiv.org/abs/2105.13290)
- Kim et al. (2025). Peri-LN. arXiv:2502.02732, ICML 2025. [link](https://arxiv.org/html/2502.02732v3)
- Li, Yin, Liu (2024). Mix-LN. arXiv:2412.13795, ICLR 2025. [link](https://arxiv.org/abs/2412.13795)
- Liu, Liu, Gao, Chen, Han (2020). Admin. arXiv:2004.08249. [link](https://arxiv.org/abs/2004.08249)
- Xiong et al. (2020). On Layer Normalization in the Transformer Architecture. arXiv:2002.04745. [link](https://arxiv.org/abs/2002.04745)
- Bachlechner et al. (2020). ReZero. arXiv:2003.04887. [link](https://arxiv.org/abs/2003.04887)
- Xie et al. (2023). ResiDual. arXiv:2304.14802. [link](https://arxiv.org/abs/2304.14802)
- Wang et al. (2022). Foundation Transformers (Sub-LN/Magneto). arXiv:2210.06423. [link](https://arxiv.org/abs/2210.06423)
- Dey et al. (2025). CompleteP. arXiv:2505.01618. [link](https://arxiv.org/html/2505.01618)
- Yang, Yu, Zhu, Hayou (2023). Tensor Programs VI (Depth-μP). arXiv:2310.02244. [link](https://arxiv.org/abs/2310.02244)
- Bordelon, Noci, Li, Hanin, Pehlevan (2023). arXiv:2309.16620. [link](https://arxiv.org/abs/2309.16620)
- Everett et al. (2024). arXiv:2407.05872. [link](https://arxiv.org/abs/2407.05872)
- Blake, Orr, Luschi (2023). Unit Scaling. arXiv:2303.11257. [link](https://arxiv.org/abs/2303.11257)；Blake et al. (2024). u-μP. arXiv:2407.17465. [link](https://arxiv.org/abs/2407.17465)
- Salimans, Kingma (2016). Weight Normalization. arXiv:1602.07868. [link](https://arxiv.org/abs/1602.07868)

### 7.4 优化器与范数约束
- Wen, Dang, Lyu, Ma, Liang (2026). Hyperball (MuonH/AdamH). arXiv:2606.16899. [link](https://arxiv.org/html/2606.16899v1)；[Modelwire 摘要](https://themodelwire.com/article/fantastic-pretraining-optimizers-and-where-to-find-them-ii-hyperball-optimizatio-01KV77X7SP050113R8XZG2JTE1)；"Hyperball May Not Be a Free Lunch"（未确认）[BAAI Hub](https://hub.baai.ac.cn/view/56966)；[Hyperball blog](https://whenwen.github.io/wd_blog/public/hyperball-part-1.html)
- Wen, Hall, Ma, Liang (2025). Fantastic Pretraining Optimizers and Where to Find Them. arXiv:2509.02046. [link](https://arxiv.org/abs/2509.02046)
- Liu et al., Moonshot (2025). Muon is Scalable for LLM Training. arXiv:2502.16982. [link](https://arxiv.org/html/2502.16982)
- Kimi Team (2025). Kimi K2. arXiv:2507.20534. [link](https://arxiv.org/html/2507.20534)
- Fu et al. (2025). Nemotron-Flash. arXiv:2511.18890. [link](https://arxiv.org/abs/2511.18890)
- Xie et al. (2026). Spectral Sphere Optimizer. arXiv:2601.08393. [link](https://arxiv.org/abs/2601.08393)
- Loshchilov, Hsieh, Sun, Ginsburg (2024). nGPT. arXiv:2410.01131. [link](https://arxiv.org/html/2410.01131)
- Pethick et al. (2025). Scion. arXiv:2502.07529. [link](https://arxiv.org/abs/2502.07529)
- Large et al. (2024). Modular norm. arXiv:2405.14813. [link](https://arxiv.org/html/2405.14813)；Bernstein, Newhouse (2024). Old Optimizer, New Norm. arXiv:2409.20325. [link](https://arxiv.org/abs/2409.20325)
- Newhouse et al. (2025). Lipschitz transformers. arXiv:2507.13338. [link](https://arxiv.org/abs/2507.13338)
- Filatov et al. (2025). Optimal Scaling Needs Optimal Norm. arXiv:2510.03871. [link](https://arxiv.org/abs/2510.03871)
- Kosson, Messmer, Jaggi (2024). Rotational equilibrium. arXiv:2305.17212. [link](https://arxiv.org/abs/2305.17212)；Kosson et al. (2024). Warmup. arXiv:2410.23922. [link](https://arxiv.org/abs/2410.23922)
- Heo et al. AdamP. arXiv:2006.08217. [link](https://arxiv.org/pdf/2006.08217)
- Karras et al. (2023). EDM2. arXiv:2312.02696. [link](https://arxiv.org/abs/2312.02696)
- Park et al. (2025). Outlier-Safe Pre-training. arXiv:2506.19697. [link](https://arxiv.org/abs/2506.19697)
- He, Noci, Paliotta, Schlag, Hofmann (2024). Outlier features / OP block + SOAP. arXiv:2405.19279. [link](https://arxiv.org/abs/2405.19279)
- Huang (2026). Muon as a Residual Connection. arXiv:2607.01124. [link](https://arxiv.org/abs/2607.01124)

### 7.5 深度利用、logit lens 与层删除
- Csordás, Manning, Potts (2025). Do Language Models Use Their Depth Efficiently? arXiv:2505.13898. [link](https://arxiv.org/html/2505.13898v1)；[NeurIPS poster](https://neurips.cc/virtual/2025/poster/118586)
- Hu, Zhou, Zhang (2025). What Affects the Effective Depth of LLMs? arXiv:2512.14064. [link](https://arxiv.org/abs/2512.14064)
- Gupta, Yeung, Anumanchipalli, Ivanova (2025/2026). How Do LLMs Use Their Depth? arXiv:2510.18871. [link](https://arxiv.org/abs/2510.18871v1)
- Lad, Lee, Gurnee, Tegmark (2024). The Remarkable Robustness of LLMs: Stages of Inference? arXiv:2406.19384. [link](https://arxiv.org/html/2406.19384v3)
- Veit, Wilber, Belongie (2016). arXiv:1605.06431. [link](https://arxiv.org/abs/1605.06431)
- Jastrzębski et al. (2017). Residual Connections Encourage Iterative Inference. arXiv:1710.04773. [link](https://arxiv.org/abs/1710.04773)
- Greff, Srivastava, Schmidhuber (2016). arXiv:1612.07771. [link](https://arxiv.org/abs/1612.07771)
- nostalgebraist (2020). interpreting GPT: the logit lens. [LessWrong](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)
- Belrose et al. (2023). Tuned Lens. arXiv:2303.08112. [link](https://arxiv.org/abs/2303.08112v4)
- Wendler, Veselovsky, Monea, West (2024). Do Llamas Work in English? arXiv:2402.10588. [link](https://arxiv.org/abs/2402.10588)
- Cheng, Doimo, Kervadec, Macocco et al. (2024). High-Dimensional Abstraction Phase. arXiv:2405.15471. [link](https://arxiv.org/abs/2405.15471)
- Skean, Arefin, Zhao, Patel et al. (2025). Layer by Layer. arXiv:2502.02013. [link](https://arxiv.org/abs/2502.02013)
- Razzhigaev et al. (2024). Your Transformer is Secretly Linear. arXiv:2405.12250. [link](https://arxiv.org/abs/2405.12250)；Razzhigaev et al. (2023). arXiv:2311.05928. [link](https://arxiv.org/abs/2311.05928)
- Gromov, Tirumala, Shapourian, Glorioso, Roberts (2024). The Unreasonable Ineffectiveness of the Deeper Layers. arXiv:2403.17887. [link](https://arxiv.org/abs/2403.17887)
- Men, Xu, Zhang, Wang et al. (2024). ShortGPT. arXiv:2403.03853. [link](https://arxiv.org/abs/2403.03853)
- Sun, Pickett, Nain, Jones (2024). Transformer Layers as Painters. arXiv:2407.09298. [link](https://arxiv.org/abs/2407.09298)
- He, Sun, Shen, Li (2024). What Matters in Transformers? arXiv:2406.15786. [link](https://arxiv.org/abs/2406.15786)
- Elhage et al. (2021). A Mathematical Framework for Transformer Circuits. [link](https://transformer-circuits.pub/2021/framework/)
- Dong, Cordonnier, Loukas (2021). Rank collapse. arXiv:2103.03404. [link](https://arxiv.org/abs/2103.03404)；Noci et al. (2022). arXiv:2206.03126. [link](https://arxiv.org/abs/2206.03126)
- Godey, de la Clergerie, Sagot (2024). Anisotropy. arXiv:2401.12143. [link](https://arxiv.org/abs/2401.12143)

### 7.6 深监督与早退
- Lee, Xie, Gallagher, Zhang, Tu (2015). Deeply-Supervised Nets. arXiv:1409.5185. [link](https://arxiv.org/abs/1409.5185)
- Huang, Chen, Li, Wu, van der Maaten, Weinberger (2017). MSDNet. arXiv:1703.09844. [link](https://arxiv.org/abs/1703.09844)；后续工作引用的干扰表述 [link](https://arxiv.org/pdf/1911.09418)
- Li, Zhang, Qi, Yang, Huang (2019). Improved Techniques for Training Adaptive Deep Networks. arXiv:1908.06294. [link](https://arxiv.org/abs/1908.06294)
- Kaya, Hong, Dumitras (2019). Shallow-Deep Networks. arXiv:1810.07052. [link](https://arxiv.org/abs/1810.07052)；[PMLR](https://proceedings.mlr.press/v97/kaya19a.html)
- Zhang, Song, Gao, Chen, Bao, Ma (2019). Be Your Own Teacher. arXiv:1905.08094. [link](https://arxiv.org/abs/1905.08094)
- Al-Rfou, Choe, Constant, Guo, Jones (2018). Character-Level LM with Deeper Self-Attention. arXiv:1808.04444. [link](https://arxiv.org/abs/1808.04444)
- Xin, Tang, Lee, Yu, Lin (2020). DeeBERT. arXiv:2004.12993. [link](https://arxiv.org/abs/2004.12993)
- Zhou, Xu, Ge, McAuley, Xu, Wei (2020). BERT Loses Patience (PABEE). arXiv:2006.04152. [link](https://arxiv.org/abs/2006.04152)
- Xin, Tang, Yu, Lin (2021). BERxiT. EACL 2021. [link](https://aclanthology.org/2021.eacl-main.8)
- Elbayad, Gu, Grave, Auli (2019). Depth-Adaptive Transformer. arXiv:1910.10073. [link](https://arxiv.org/abs/1910.10073)
- Schuster, Fisch, Gupta, Dehghani et al. (2022). CALM. arXiv:2207.07061. [link](https://arxiv.org/abs/2207.07061)
- Elhoushi et al. (2024). LayerSkip. arXiv:2404.16710, ACL 2024. [link](https://arxiv.org/html/2404.16710v4)
- Chen, Pan, Li, Ding, Zhou (2023). EE-LLM. arXiv:2312.04916. [link](https://arxiv.org/abs/2312.04916)
- Kavehzadeh, Valipour, Tahaei, Ghodsi et al. (2023). Sorted LLaMA. arXiv:2309.08968. [link](https://arxiv.org/abs/2309.08968)
- Kapadia et al. (2026). LEAP. arXiv:2605.01058. [link](https://arxiv.org/pdf/2605.01058)
- Glavas, Chataoui, Regol et al. (2024). Dynamic layer selection in decoder-only transformers. arXiv:2410.20022. [link](https://arxiv.org/abs/2410.20022)
- Wei, Du, Yu, Tiwari, Li, Xu, Wang (2026). The Diminishing Returns of Early-Exit Decoding. arXiv:2603.23701. [link](https://arxiv.org/abs/2603.23701)
- Lawson, Aitchison (2025). Learning to Skip the Middle Layers. arXiv:2506.21103. [link](https://huggingface.co/papers/2506.21103)
- Fan, Grave, Joulin (2019). LayerDrop. arXiv:1909.11556. [link](https://arxiv.org/abs/1909.11556)
- Drop Dropout on Single-Epoch LM Pretraining (2025). arXiv:2505.24788. [link](https://arxiv.org/pdf/2505.24788)

### 7.7 最后一层行为、massive activations 与 sink
- Stolfo, Wu, Gurnee, Belinkov, Song, Sachan, Nanda (2024). Confidence Regulation Neurons. arXiv:2406.16254. [link](https://arxiv.org/html/2406.16254v2)
- Gurnee et al. (2024). Universal Neurons in GPT2. arXiv:2401.12181. [link](https://arxiv.org/abs/2401.12181)
- McDougall, Conmy, Rushing, McGrath, Nanda (2023). Copy Suppression. arXiv:2310.04625. [link](https://arxiv.org/abs/2310.04625)
- Joshi, Ahmad, Modi (2025). Calibration Across Layers. arXiv:2511.00280. [link](https://arxiv.org/abs/2511.00280v1)
- Halawi, Denain, Steinhardt (2023). Overthinking the Truth. arXiv:2307.09476. [link](https://arxiv.org/abs/2307.09476)
- Shibata, Yano, Takahashi, Lee, Ikeda, Suzuki (2026). Suppressing Final Layer Hidden State Jumps (JREG). arXiv:2601.18302. [link](https://arxiv.org/html/2601.18302v1)
- Geva, Caciularu, Wang, Goldberg (2022). FFN Layers Promote Concepts. arXiv:2203.14680. [link](https://arxiv.org/abs/2203.14680)
- Sun, Chen, Kolter, Liu (2024). Massive Activations. arXiv:2402.17762. [link](https://arxiv.org/abs/2402.17762)
- Luo et al. (2026). OASIS: Attention Sinks and Outliers in Attention Residuals. arXiv:2605.17887. [link](https://arxiv.org/html/2605.17887)
- A Unified View of Attention and Residual Sinks (2026). arXiv:2601.22966. [link](https://arxiv.org/html/2601.22966v1)
- Do New Attention Mechanisms Actually Fix Attention Sinks at Million-Token Context? (2026). arXiv:2609.08574. [link](https://arxiv.org/pdf/2609.08574)
- Shi et al. (2026). Massive Emergence Layer. arXiv:2605.08504. [link](https://arxiv.org/abs/2605.08504)
- Owen et al. (2025). arXiv:2503.22329. [link](https://arxiv.org/abs/2503.22329)
- Vemula (2026). Commitment stream. arXiv:2606.20743. [link](https://arxiv.org/abs/2606.20743)
- Jin et al. (2025). Massive values in Q/K. arXiv:2502.01563. [link](https://arxiv.org/abs/2502.01563)
- Darcet et al. (2023). Vision Transformers Need Registers. arXiv:2309.16588. [link](https://arxiv.org/abs/2309.16588)
- Xiao et al. (2023). StreamingLLM. arXiv:2309.17453. [link](https://arxiv.org/abs/2309.17453)
- Gu et al. (2025). When attention sinks emerge. arXiv:2410.10781. [link](https://arxiv.org/html/2410.10781v2)
- Barbero et al. (2025). arXiv:2504.02732. [link](https://arxiv.org/abs/2504.02732)
- Qiu et al., Qwen (2025). Gated attention. arXiv:2505.06708. [link](https://arxiv.org/html/2505.06708)
- Bondarenko, Nagel, Blankevoort (2023). Quantizable Transformers (no-op heads). arXiv:2306.12929. [link](https://arxiv.org/abs/2306.12929)
- An et al. (2025). Systematic outliers. arXiv:2502.06415. [link](https://arxiv.org/abs/2502.06415)
- Dettmers et al. (2022). LLM.int8(). arXiv:2208.07339. [link](https://arxiv.org/abs/2208.07339)
- Kovaleva et al. (2021). BERT outlier dimensions. arXiv:2105.06990. [link](https://arxiv.org/abs/2105.06990)
- Yu et al. (2024). Super weights. arXiv:2411.07191. [link](https://arxiv.org/abs/2411.07191)
- Fishman et al. (2024). Smooth-SwiGLU. arXiv:2409.12517. [link](https://arxiv.org/abs/2409.12517)
- Cancedda (2024). Dark signals. arXiv:2402.09221. [link](https://arxiv.org/abs/2402.09221)
- Hoogland et al. (2024/2025). Final-LayerNorm collapse. arXiv:2402.02364. [link](https://arxiv.org/html/2402.02364)
- Heimersheim (2024). Removing LayerNorm from GPT-2. arXiv:2409.13710. [link](https://arxiv.org/abs/2409.13710)

### 7.8 低精度
- PyTorch 2.14 AMP 文档（rms_norm 不在 fp32 自动转换列表中）. [link](https://docs.pytorch.org/docs/2.14/amp.html)
- HF transformers LlamaRMSNorm. [link](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/llama/modeling_llama.py)
- Unsloth, Fixing All Gemma Bugs (2024). [link](https://unsloth.ai/blog/gemma-bugs)；Gemma 3 docs [link](https://docs.unsloth.ai/basics/gemma-3-how-to-run-and-fine-tune)
- Megatron-LM `--fp32-residual-connection`（ORNL 镜像）. [commit](https://code.ornl.gov/candle/Megatron-LM/-/commit/deffcb6a41c1017ff09a6d4c4d3b8817dae386fd)；[commit](https://code.ornl.gov/candle/Megatron-LM/-/commit/62632d39364e499bbfe9e143d2856dbe3f3a6802)
- tt-metal issue #57861. [link](https://github.com/tenstorrent/tt-metal/issues/57861)
- Liger-Kernel PR #1498. [link](https://github.com/linkedin/Liger-Kernel/pull/1498)
- Zamirai et al. (2020). arXiv:2010.06192. [link](https://arxiv.org/abs/2010.06192)；Kalamkar et al. (2019). arXiv:1905.12322. [link](https://arxiv.org/abs/1905.12322)
- Wortsman et al. (2023). arXiv:2304.13013. [link](https://arxiv.org/abs/2304.13013)；Wortsman et al. (2023). Small-scale proxies. arXiv:2309.14322. [link](https://arxiv.org/abs/2309.14322)
- KV 缓存项目的 fp32 残差报告（个案）. [link](https://github.com/lecuong1502/Semantic-Aware_Adaptive_KV_Cache_Precision/issues/36)

### 7.9 多 token 预测与未来 token 探针
- Qi et al. (2020). ProphetNet. arXiv:2001.04063. [link](https://arxiv.org/abs/2001.04063)
- Stern, Shazeer, Uszkoreit (2018). Blockwise Parallel Decoding. arXiv:1811.03115. [link](https://arxiv.org/abs/1811.03115)
- Goodman et al. (2020). TeaForN. arXiv:2010.03494（经 kexue 7818 链接）
- Gloeckle et al. (2024). Better & Faster LLMs via Multi-token Prediction. arXiv:2404.19737. [link](https://arxiv.org/abs/2404.19737)
- DeepSeek-AI (2024). DeepSeek-V3. arXiv:2412.19437. [link](https://arxiv.org/html/2412.19437v1)
- Kim et al. (2025). KORMo. arXiv:2510.09426. [link](https://arxiv.org/html/2510.09426)
- Zuhri, Fuadi, Aji (2025/2026). Token Order Prediction. arXiv:2508.19228. [link](https://arxiv.org/html/2508.19228v2)
- Aynetdinov, Akbik (2025). MTP curriculum. arXiv:2505.22757. [link](https://arxiv.org/html/2505.22757)
- Cui et al. (2026). AdaMTP. arXiv:2608.00434. [link](https://arxiv.org/abs/2608.00434)
- Cai, Li, Geng, Peng, Lee, Chen, Dao (2024). Medusa. arXiv:2401.10774. [link](https://arxiv.org/html/2401.10774)
- Li, Wei, Zhang, Zhang (2024). EAGLE. arXiv:2401.15077. [link](https://arxiv.org/abs/2401.15077)；EAGLE-3 (2025). arXiv:2503.01840. [link](https://arxiv.org/html/2503.01840)
- Mehra, Alonso Garcia, Mauch (2025). On MTP for efficient inference. arXiv:2502.09419. [link](https://arxiv.org/html/2502.09419)
- Samragh et al. (2025). Your LLM Knows the Future. arXiv:2507.11851. [link](https://arxiv.org/abs/2507.11851)
- Gerontopoulos, Gidaris, Komodakis (2025). MuToR. arXiv:2505.10518. [link](https://arxiv.org/html/2505.10518)
- Yin, Ding, Song, Zhang (2024). Semformer. arXiv:2409.11143. [link](https://arxiv.org/abs/2409.11143v1)
- Thankaraj, Jiang, Kolter, Bisk (2025). TRELAWNEY. arXiv:2504.11336. [link](https://export.arxiv.org/abs/2504.11336)
- Ahn, Lamb, Langford (2025). JTP. arXiv:2503.21801. [link](https://arxiv.org/abs/2503.21801v1)
- Mahajan et al. (2025). Future Summary Prediction. arXiv:2510.14751. [link](https://arxiv.org/abs/2510.14751)
- Bachmann, Nagarajan (2024). Pitfalls of next-token prediction. arXiv:2403.06963. [link](https://arxiv.org/abs/2403.06963)
- Zhong et al. (2025). Planning via MTP. arXiv:2509.23186. [link](https://arxiv.org/html/2509.23186v1)
- Pal, Sun, Yuan, Wallace, Bau (2023). Future Lens. arXiv:2311.04897. [link](https://arxiv.org/abs/2311.04897)
- Wu, Morris, Levine (2024). Do Language Models Plan for Future Tokens? arXiv:2404.00859. [link](https://arxiv.org/abs/2404.00859)
- Xu, Zhao, Song, Stewart, Ermon (2020). Usable Information (V-information). arXiv:2002.10689. [link](https://arxiv.org/abs/2002.10689)
- Pimentel et al. (2020). Information-Theoretic Probing. arXiv:2004.03061. [link](https://arxiv.org/abs/2004.03061)
- Hewitt, Liang (2019). Control tasks. arXiv:1909.03368. [link](https://arxiv.org/abs/1909.03368)
- Hewitt, Ethayarajh, Liang, Manning (2021). Conditional probing. arXiv:2109.09234. [link](https://arxiv.org/abs/2109.09234)
- Teoh et al. (2025/2026). Next-Latent Prediction. arXiv:2511.05963. [link](https://arxiv.org/html/2511.05963)
- Men et al. (2024). Unlocking the Future. arXiv:2406.16033. [link](https://arxiv.org/abs/2406.16033)
- Dong et al. (2025). Emergent Response Planning. arXiv:2502.06258. [link](https://arxiv.org/abs/2502.06258)
- Anthropic (2025). Tracing the thoughts of a large language model. [link](https://www.anthropic.com/research/tracing-thoughts-language-model)

### 7.10 kexue.fm（苏剑林）
- 9009 为什么Pre Norm的效果不如Post Norm？(2022-03-29) [link](https://kexue.fm/archives/9009)
- 8620 浅谈Transformer的初始化、参数化与标准化 (2021-08-17) [link](https://kexue.fm/archives/8620)
- 8747 模型优化漫谈：BERT的初始标准差为什么是0.02？(2021-11-08) [link](https://kexue.fm/archives/8747)
- 8978 训练1000层的Transformer究竟有什么困难？(2022-03-09) [link](https://kexue.fm/archives/8978)
- 8994 为什么需要残差？一个来自DeepNet的视角 (2022-03-19) [link](https://kexue.fm/archives/8994)
- 8998 RoFormerV2 (2022-03-21) [link](https://kexue.fm/archives/8998)
- 8990 门控注意力单元（GAU）还需要Warmup吗？(2022-03-11) [link](https://kexue.fm/archives/8990)
- 8027 RealFormer (2020-12-24) [link](https://kexue.fm/archives/8027)
- 11664 Attention Residuals 回忆录 (2026-03-19) [link](https://kexue.fm/archives/11664)
- 11848 简单谈谈K3的MoE和Attention (2026-08-04) [link](https://kexue.fm/archives/11848)
- 9889 注意力机制真的可以"集中注意力"吗？(2023-12-12) [link](https://kexue.fm/archives/9889)
- 8907 多任务学习漫谈（三）：分主次之序 (2022-02-14) [link](https://kexue.fm/archives/8907)
- 8896 多任务学习漫谈（二）：行梯度之事 (2022-02-08) [link](https://kexue.fm/archives/8896)
- 8870 多任务学习漫谈（一）：以损失之名 (2022-01-18) [link](https://kexue.fm/archives/8870)
- 7818 TeaForN (2020-10-27) [link](https://kexue.fm/archives/7818)
- 11340 MuP之上：1. 好模型的三个特征 (2025-10-21) [link](https://kexue.fm/archives/11340)
- 11605 MuP之上：2. 线性层与最速下降 (2026-02-15) [link](https://kexue.fm/archives/11605)
- 11647 MuP之上：3. 特殊情况特殊处理 (2026-03-02) [link](https://kexue.fm/archives/11647)
- 11729 MuP之上：4. 坚守参数的稳定性 (2026-04-24) [link](https://kexue.fm/archives/11729)
- 11416 Muon优化器指南 (2025-11-19) [link](https://kexue.fm/archives/11416)
- 10739 Muon续集 (2025-02-27) [link](https://kexue.fm/archives/10739)
- 11241 流形上的最速下降：4. Muon + 谱球面 (2025-08-21) [link](https://kexue.fm/archives/11241)
- 11196 流形上的最速下降：1. SGD + 超球面 (2025-08-01) [link](https://kexue.fm/archives/11196)
- 11549 为什么我们偏爱各向同性？(2026-01-20) [link](https://kexue.fm/archives/11549)
- 11126 QK-Clip (2025-07-12) [link](https://kexue.fm/archives/11126)
- 11371 低精度Attention可能存在有偏的舍入误差 (2025-10-27) [link](https://kexue.fm/archives/11371)
- 9698 语言模型输出端共享Embedding的重新探索 (2023-07-20) [link](https://kexue.fm/archives/9698)
- 11833 解构Scaling Law (2026-07-29) [link](https://kexue.fm/archives/11833)
- 11854 除了交叉熵，LM Loss还有什么选择？(2026-08-09，未细读) [link](https://kexue.fm/archives/11854)
- 10831 DyT（找到但未映射）[link](https://kexue.fm/archives/10831)
- 10770（说明 MuP 分析未含归一化与残差）[link](https://kexue.fm/archives/10770)
