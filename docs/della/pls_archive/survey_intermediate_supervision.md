# 中间层监督（deep supervision / 逐层 LM loss）文献综述

> 面向实验：在 dense Qwen3-style decoder（130M–1.2B，Muon 系优化器，marin/levanter）上，训练早期对**每一层**临时施加经共享 final norm + LM head 的 next-token loss，再退火移除（即 LLAL 的"逐层版"）。
> 日期 2026-09-29。核实方式见 §0.3，相关度 1–5 的定义见 §0.2。原始笔记在 `survey_notes/`（其中 `pdftxt/` 是用来核对关键数字的全文文本）。

> **主 session 审计附注（2026-09-29）**：以下 12 项承重结论已逐字对照原文（全文文本或 arXiv 原文）核实，数字全部存在且与上下文相符：
> Al-Rfou（bpc 1.062/1.158、l/2n 移除）、LayerSkip（"all layers at all iterations … reduces the accuracy of the last layer"原话；MMLU 46.0/43.1）、RRT（12.85→13.24、51.7→50.2、−1.5；i/Σi −1.2；0.1+KD 后训练 +0.8）、Kubaty（71.61/68.39）、HASTE（Table 3：termination 一列 8.1→5.3）、SOLO（70.62/75.34/67.08；logit-lens 一致率 16–51%）、EE-LLM（1/4、1/2 深度，权重 1/4、1/2）、Multi-Layer Softmax（34.87/34.87）、DAT（36.2/35.9）、InterCTC（WSJ eval92 16.5→13.6、13.9→12.4；w=0.3 全程）、HLD-LLM（arXiv 2605.11513 存在）、BitSkip（2510.23766 存在）。
> **更正三处**：(1) RRT 的 −1.5 那一行，中间输出系数是 0.1（在 uptraining 中联合训练），不是"等权或偏浅层"——小权重也伤末层，这对 decoder LM 是更强的负证据；(2) "τ=400K → 7.4" 出自 Figure 1（只有 REPA），与 Table 3 的 HASTE 8.1→5.3 不是同一设置；(3) InterCTC "共享 final LayerNorm"来自 ESPnet 实现，论文正文未写。
> **一句话结论需要收窄**："全层等权、全程常开会可重复地伤害末层"对 decoder LM 成立（LayerSkip、RRT），但在 NMT decoder 上不成立：DAT 所有 6 层等权常开，末层 36.2 对 35.9（+0.3 BLEU）；Multi-Layer Softmax 34.87 对 34.87（持平）；InterCTC 单个中间层 w=0.3 常开，末层明显变好。

**一句话结论**：在 145 篇已核实的、含中间层监督的工作里，给中间层加 same-label / LM loss，只有在"权重小或偏向深层，且只在训练早期或分阶段施加、之后移除"时，对末层是中性到正面的。例如 Al-Rfou 等人给 64 层字符 LM 的每层加 loss、训练过半前逐层全部移除，text8 dev bpc 从 1.158 降到 1.062；HASTE 在合适时刻停掉中间层对齐，FID 从 8.1 降到 5.3。反过来，"全层等权或偏向浅层、全程常开"会可重复地伤害末层：LayerSkip 作者明确报告了这一点，RRT 联合训练使平均准确率 −1.5，Kubaty 等人在 ViT-T/ImageNet 上测得 −3.2。至今没有工作在从零训练的 dense decoder LM 上测过"逐层 + 早期退火移除"这个组合，本实验填的正是这个空白，但先验上效应会很小。

---

## 0. 速览

### 0.1 对本实验最重要的三条结论

1. **"临时"是保护末层的关键，窗口长度应当作为主变量。**
   - 全程常开、权重不小的逐层监督反复被报告伤害末层：
     - LayerSkip 原文："adding early exit loss of all layers at all iterations during training slows down training and reduces the accuracy of the last layer"，他们因此引入 curriculum。
     - RRT：在 uptraining 期间联合训练中间循环的 LM loss，末层平均 few-shot 从 51.7 降到 50.2，SlimPajama PPL 从 12.85 升到 13.24。
     - Kubaty 等人：从零 joint 训练，ViT-T/IN-1k 末出口从 71.61 降到 68.39，BERT/20NG 从约 85.75 降到约 84.3。
     - DEED：只用 L_avg 时末层 decoder 退化。
   - 早期或分阶段、之后移除的做法是中性到正面：
     - Al-Rfou：第 l 层在训练进度 l/(2n) 处停止，训练过半时全部移除，bpc −0.096。
     - HASTE：硬停对齐，FID 8.1→5.3。停在 400K 只到 7.4，停在 100K 反而变差。停止后还有一段暂时变差（dip）。
     - HLD-LLM：只在前 1–5% 预算做 hint，之后移除，得到小而稳定的增益；常开版本没有增益。
     - FitNets、TinyBERT、MobileBERT 都是"先中间、后移除"：MobileBERT 的 progressive 83.9 优于 joint 83.5，也优于常开 auxiliary 的 83.0。
   - 建议把窗口扫成 {≈1%, 3%, 10%} 的训练步数，并在移除后足够久再评估。
2. **总权重要归一，层间分布不要偏向浅层；同时要做好梯度记账。**
   - 成功的配方大多把中间层总权重压在 ≤0.3–0.5，或者按深度递增：
     - EE-LLM：0.25/0.5，或 0.1/0.2。
     - ESPnet InterCTC：(1−w)·L_final + w·mean_l L_l，w=0.3。
     - LayerSkip：e(l) ∝ Σ_{i≤l} i，按深度二次增长，e_scale 取 0.1–0.2。
     - CALM：ω_i=i/Σj。PABEE：∝ j。RAFT：γ^(N−i)。
   - 偏向浅层的配方会伤末层：
     - BitSkip 的 w_l∝1/(l+1)：val PPL 228.77→252.37（与层跳混杂）。
     - Ouro 中偏向早步的几何先验：训练 loss 平台更高。
     - RRT 的 i/Σi（只有 2 个出口时就是 1/3, 2/3）：末层 −1.2。改成 0.1 时则 +0.8（后训练多了 15B token，这部分增益有一些来自多训）。
   - 逐层监督下，第 i 层会收到 L−i+1 份辅助梯度。文献里的补救有 gradient equilibrium（IMTA/ElasticBERT）、BoostNet 的 1/(N−n+1)、DAT 的 gradient scaling。
   - 在 Muon 下，每个矩阵的更新幅度被正交化了，所以起作用的很可能是辅助梯度与主梯度的**方向占比**（这是推断，没有文献直接研究过）。建议逐层记录 cos(g_aux, g_main)：HASTE 就是用这个夹角决定何时停。
3. **共享 final norm + LM head 是标准且有效的设计，但它在跟模型的"自然解"对抗。**
   - 共享的先例：
     - LayerSkip 的 head 就是"final layer norm + linear"。
     - ESPnet 的 InterCTC/SC-CTC 让中间层输出先过 encoder 的 final LayerNorm，再进同一个 CTC 头。
     - DETR 共享 LayerNorm 和预测 FFN。CALM 和 OISD 也都共享。
   - SOLO 的读出消融表明，**共享 readout 基**本身就是逐模块 next-token loss 有效的原因：40M 模型上，共享 70.62 ppl，私有 75.34，BP 67.08。
   - 但正常训练的 LM 里，中间层会随训练越来越**不可**被 logit-lens 解码：
     - LayerSkip Fig.11：baseline 中层 PPL 随训练升到数百。
     - Pythia：中层与末层的相似度出现低谷。
     - SOLO：BP 模型 logit-lens 与末层一致率只有 16–51%。
   - 所以全程常开的逐层 LM loss 是一个很强的约束，这是"只在早期施加"的又一个理由。
   - 文献里两个便宜的旋钮：
     - 辅助 loss 对共享 head/final norm 做 stop-gradient：SOLO 用上一步的只读拷贝；Liu 等人的 RNN-T 辅助 loss 前向共享 decoder 但不更新它。
     - 给浅层用更软的目标：DTS 的置信度随深度递增，HC-CTC 用由粗到细的目标，RRT/OISD 用对 detached 末层分布的 KL/JSD。

**先验预期**（跑之前写下来）：
- 在 130M–1.2B 的现代配方下（pre-norm、合理初始化、Muon），历史上逐层 loss 修复的那类优化失败基本不存在，例如 Al-Rfou 的 ">10 层难训"、Wang 等人 "24 层语音 transformer 不收敛"。
- LLAL 解决的 "early experts silent death" 在 dense 中没有直接对应物。而且 dense 模型第一个 MLP 层本来就承载 bigram 子网络（Chang & Bergen 2025）。
- 因此合理先验是效应很小：可能有早期加速，之后逐渐消失；λ 过大或窗口过长时，末层略差。
- 项目文档给出 130M 的 c4_en bpb 种子标准差约 0.0004，要看到约 1e-3 的差异才算信号。
- LLAL 博客中有一条署名 "Sean" 的页内评论，从 "we tailored and validated our solution" 的措辞看应当来自作者方，写道："We suspect the weak signal issue is inherent to both MoE and dense architectures ... Its validity on dense architectures warrants further investigation." 所以 dense 版本是他们明确留下的开放问题。

### 0.2 相关度（1–5）定义

针对"在从零训练的 dense decoder LM 上，对每层临时、退火地施加经共享 final norm + LM head 的 next-token loss"：

| 分 | 含义 |
|---|---|
| 5 | decoder LM 预训练中的逐层 LM loss，并给出末层证据或 schedule（或 LLAL 本身） |
| 4 | Transformer 上的同标签逐层 loss，给出末层证据或权重/schedule 洞见（含 seq2seq decoder、语音、looped LM、强相关的对齐早停证据） |
| 3 | 其它架构的中间层监督，有可迁移的 schedule/权重/助害证据 |
| 2 | 以推理加速为主的 early exit，或对齐/蒸馏变体，训练动力学洞见少 |
| 1 | 背景或对照 |

### 0.3 核实方法与计数

- **元数据**：arXiv API（export.arxiv.org 批量查询）逐条核对 id、标题、作者、日期、comment/journal-ref。非 arXiv 条目用 ACL Anthology、PMLR、CVF、ECVA、AAAI OJS、NeurIPS/ICML 官网、ISCA、Crossref 核对。
- **方法细节与数字**：读全文（arxiv.org/html、ar5iv）。对关键数字，在 della-vis1 上下载 PDF，用 PyMuPDF 转成文本后逐字核对，文本在 `survey_notes/pdftxt/`。逐字核对过的包括：
  - Al-Rfou Table 4 与 schedule 原文
  - LayerSkip §4–7、Table 1–2
  - Depth-Adaptive Transformer Table 1–2
  - EE-LLM §5.1、CALM §3.4/§5、FREE §4、DEED §3–4
  - RRT Table J.1、Ouro Eq.4 与 App. A
  - HASTE Table 3–4、Kubaty Table 1
  - Multi-Layer Softmax Table 1、SOLO 方法段
- **分工**：F1/F1b/F8、F3/F5/F7、F4 由三个并行子代理按统一格式整理（笔记见 `survey_notes/family_*.md`），F2/F6/F9/F10 由我整理。我又对其中最关键的数字做了抽查（Kubaty、HASTE、Multi-Layer Softmax、SOLO），全部与原文一致。
- **规则**：无法核实的不进正文，列入 §6。

**计数（互不重复的已核实条目）：162 条，其中 145 条为含中间层监督的核心文献，17 条为背景或对照（F10）。**

| 家族 | 内容 | 条数 |
|---|---|---|
| F1 | 经典 deep supervision（CNN / ViT / MIM） | 21 |
| F1b | 迭代精修架构的逐阶段/逐层 loss（DETR、RAFT、CPM、Hourglass …，新增家族） | 6 |
| F2 | Transformer LM / seq2seq decoder 的逐层 LM loss（核心） | 19 |
| F3 | Early-exit / adaptive-depth encoder 与多出口训练研究 | 21 |
| F4 | 语音（InterCTC、iterated loss、early-exit ASR …） | 27 |
| F5 | MT / seq2seq | 4 |
| F6 | Looped / recurrent-depth / universal transformer 的逐步 loss | 16 |
| F7 | 对齐 / 蒸馏式中间层监督（REPA、HASTE、TinyBERT …） | 16 |
| F8 | Local / greedy 逐层学习 | 10 |
| F9 | MoE 相关（含 LLAL 两篇博客） | 5 |
| F10 | 背景与对照（深度利用、早期训练动力学、logit lens、仅推理的 early exit） | 17 |
| **合计** | | **162** |

另有 4 篇语音论文已核实但属边缘内容，未计分（见 F4 末尾）。arXiv 2005.08081 核实后发现没有辅助 loss，已排除。

相关度分布：5 分 4 条（Al-Rfou、LayerSkip、EE-LLM、LLAL），4 分 25 条，3 分 42 条，2 分 69 条，1 分 22 条。

### 0.4 LLAL 做法回顾（本综述的锚点）

来源：本地保存的博客静态副本 `llal1.txt` / `llal.txt`，以及线上页面 <https://mooler0410.github.io/puguJin/blog/llal/>（2026-07-27，2026-08-22 更新）和后续页面 <https://mooler0410.github.io/puguJin/blog/llal-change/>。

- **形式**：L = L_LM(h_L) + λ_aux · CE(softmax(W_lm h_ℓ), y)，其中 W_lm 与主路径**共享**。博客的公式和图都只写了 W_lm，是否经过 final RMSNorm **没有说明**。
- **位置**：
  - 60B 探索实验：接在 L5 或 L3（11k 步短跑）。接在 L5 时，下方 L3、L4 的专家范数也被保住；接在 L3 时，下方 L2 与上方 L4、L5 都不再塌缩。
  - 但把接点从 L5 挪到 L3，并没有明显改善最底层的 MoE 层 L1，所以正式实验直接接在 L1（第一个 MoE 层，L0 是 dense）。
- **Schedule**：λ 从 0.1 线性退火到 0，然后移除。
  - 60B：退火窗口 40k（Exp6-1）或 10k（Exp6-2），总 90k 步。
  - 180B：4k 步（Exp7；WSD，总 135k，约 850B token）。
  - 2k 步窗口在 40k 步时与 10k 窗口无显著差异。
- **结果**：
  - 60B（10k 窗口）：val loss Δ≈−0.024，MMLU 0.5162→0.5555，MMLU-Pro 0.1899→0.2306。
  - 180B：MMLU 0.6205→0.6489，MMLU-Pro 0.2827→0.3302。
  - 同时负载均衡改善，早期专家不再塌缩。
  - 额外算力 <0.3%。
- **机制（后续文章）**：屏蔽 routed 专家的实验显示，LLAL 让 L1/L2 专家获得了"稀有知识"功能，例如 AdamW-2T 上屏蔽 L2 时，稀有名字 ΔNLL 为 2.158，常见名字为 0.108。Engram 插在 L1 前与插在 L14 前的对照，用来佐证"早期建立有用功能"的解释。

---

## 1. 汇总表（全部 162 条已核实条目）

按家族排列，族内按相关度降序。缩写说明：
- "同标签"：与末层相同的监督目标。
- "独立头"：每个出口各有自己的分类器。
- "共享头"：与末层共用输出层。
- "norm✓"：同时共享 final norm。
- "?"：论文未说明。
- 末层效果写成"无→有"。

| # | 论文（arXiv / 出处） | 族 | 监督哪些层 | 目标 | head / final norm | 权重 · schedule | 目的 | 末层效果（数字） | 相关 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | DeepMIM — Ren 2025 WACV (2303.08817) | F1 | ViT-B 第 6/8/10 块 + 末端 | 同重建目标；浅层用更易的 hybrid 目标 | 独立解码器 | 等权，常开 | 预训练 | IN 微调 82.6→83.4 (300ep)；83.6→84.2 (1600ep)，增益随训练变长而缩小 | 4 |
| 2 | DSN — Lee 2015 AISTATS (1409.5185) | F1 | 每个 conv 隐层 | 同标签 (sq. hinge) | 独立头 | α_m；γ-hinge 使已达标层自动关闭；可选 (1−t/N) 衰减到 0 | 训练动力学 | C100 err 35.68→34.57；C10 10.41→9.78 | 3 |
| 3 | CNDS — Wang 2015 arXiv (1505.02496) | F1 | 梯度消失处加分支 | 同标签 | 独立头 | α 从 0.3 按 (1−t/N) 递推衰减 | 训练动力学 | IN top-1 err 34.7→33.8；训练 5 天 vs 6 天 | 3 |
| 4 | Inception-v3 — Szegedy 2016 CVPR (1512.00567) | F1 | 上部 1 个 side head | 同标签 | 独立头 | 0.3，常开 | 正则 | 对早期收敛"无改善"，末期平台略高；BN side head +0.4% | 3 |
| 5 | MSDNet — Huang 2018 ICLR (1703.09844) | F1 | 多深度分类器（动机实验：ResNet/DenseNet 挂中间分类器） | 同标签 | 独立头 | 均匀 w=1 | 推理 | ResNet 末层分类器最多 −7%（C100），DenseNet 受影响小得多 | 3 |
| 6 | SDN — Kaya 2019 ICML (1810.07052) | F1 | 15–90% 推理成本处 | 同标签 | 独立头 | 权重 ∝ 深度占比，从 0.01 线性 warm-up | 推理+诊断 | 未单列末层；overthinking 统计 | 3 |
| 7 | AdaLoss — Hu 2019 AAAI (1708.06832) | F1 | anytime 多出口 | 同标签 | 独立头 | ∝1/滑动平均 loss（对比等权 CONST） | 推理 | CONST 等权比单出口网络多 15–18% 相对误差，越深差距越大 | 3 |
| 8 | BYOT — Zhang 2019 ICCV (1905.08094) | F1 | 各 stage 后浅分类器 | 同标签 + KL 到最深 + hint | 独立头 | 常数 | 训练 | C100 R50 77.68→80.56；IN R50 73.56→75.24 | 3 |
| 9 | IMTA — Li 2019 ICCV (1908.06294) | F1 | 所有出口 | 同标签 + KD | 独立头 | gradient equilibrium | 两者 | 末出口 C100 +1.64，IN +1.09 | 3 |
| 10 | Contrastive DS — Zhang 2022 ECCV (2207.05306) | F1 | 中间 stage | 对比学习（替代任务 CE） | 独立投影头 | 常数 | 训练 | 同标签 DS 在 IN 上只有 +0.07~0.33；CDS 在 IN R50 上 75.30→78.25 | 3 |
| 11 | SDViT — Sultana 2022 ACCV (2207.12392) | F1 | 每步随机 1 个中间块 | KL 到末分类器 | **共享末层 head** | λ 0.1–0.5 | 训练 (OOD) | PACS DeiT-S 84.9→86.3 | 3 |
| 12 | DTS — Wang 2026 ICML (OpenReview N9kBjRH0Cx) | F1 | 中间层 | 置信度随深度递增的软目标 | ? | 目标锐度按深度呈指数 | 训练 | 摘要称 IN-1K 精度与收敛都改善（无数字） | 3 |
| 13 | GoogLeNet — Szegedy 2015 CVPR (1409.4842) | F1 | Inception 4a/4d | 同标签 | 独立头 | 0.3，推理时丢弃 | 训练动力学 | 论文无单独数字 | 3 |
| 14 | HED — Xie 2015 ICCV (1504.06375) | F1 | 5 个 stage 的 side output | 同边缘图 | 独立头 + 融合 | α=1 | 训练 + 多尺度 | 融合输出 ODS .771→.782（不是纯末层） | 2 |
| 15 | PSPNet — Zhao 2017 CVPR (1612.01105) | F1 | res4b22 后 1 个 | 同标签 | 独立头 | α=0.4（0.3–0.9 都有益） | 训练 | mIoU 35.82→37.23 | 2 |
| 16 | BranchyNet — Teerapittayanon 2016 ICPR (1709.01686) | F1 | 1–2 个侧枝 | 同标签 | 独立头 | 首枝 1.0、末枝 0.3，比等权 +1% | 推理 | 未单列末层 | 2 |
| 17 | DKS — Sun 2019 CVPR (1906.00675) | F1 | conv3_x/4_x 后 | 同标签 + 互蒸馏 | 独立头 | 1 | 训练 | 纯 DS 只 +0.08~0.92；DKS 在 IN R50 上 err 25.47→23.53 | 2 |
| 18 | DISCO — Li 2019 TPAMI (1801.03399) | F1 | 按概念层级 | 由粗到细的不同目标 | 独立头 | 常数 | 训练 | 比全放末层 +4.4% | 2 |
| 19 | LION-DG — Kim 2026 arXiv (2601.02105) | F1 | DS-CNN 辅助头 | 同标签 | 独立，零初始化 | 零初始化使辅助梯度隐式 warm-up | 训练速度 | 小幅（收敛快 8–11%） | 2 |
| 20 | UNet++ — Zhou 2018 DLMIA (1807.10165) | F1 | 4 个嵌套解码节点 | 同 mask | 独立头 | 等权 | 两者 | 4 个数据集正负参半 | 1 |
| 21 | Deep supervision 综述 — Li 2022 arXiv (2207.02376) | F1 | 综述 | — | — | — | 综述 | — | 1 |
| 22 | DETR 辅助解码 loss — Carion 2020 ECCV (2005.12872) | F1b | 每个 decoder 层（6） | 同 Hungarian loss | **共享 FFN 头 + 共享 LayerNorm（norm✓）** | 与末层等权，全程常开 | 训练 | 无开关消融；逐层 AP 从首层到末层 +8.2 | 4 |
| 23 | RAFT — Teed 2020 ECCV (2003.12039) | F1b | 每次迭代（12） | 同 flow | 更新算子共享 | γ^(N−i)，γ=0.8 | 训练 | 共享 vs 不共享：EPE 1.63 vs 1.96 | 3 |
| 24 | Deformable DETR — Zhu 2021 ICLR (2010.04159) | F1b | 每个 decoder 层 | 同 | 迭代精修时各层 head 不共享、梯度截断 | 常数 | 训练 | box refine 43.8→45.4 AP | 3 |
| 25 | CPM — Wei 2016 CVPR (1602.00134) | F1b | 每个 stage | 同 belief map | 结构相同、权重不共享 | 等权 | 梯度流 | 数字只在图中；联合 + 中间监督最好 | 3 |
| 26 | Stacked Hourglass — Newell 2016 ECCV (1603.06937) | F1b | 每个 hourglass | 同 heatmap | 不共享 | 等权 | 训练 | 2/4/8 stacks：87.4/87.8/88.1 PCKh | 2 |
| 27 | Mask2Former — Cheng 2022 CVPR (2112.01527) | F1b | 每个 decoder 层 + query | 同 | ? | 常数 | 训练 | 无单独消融 | 2 |
| 28 | **Al-Rfou T64** — Al-Rfou 2019 AAAI (1808.04444) | F2 | 64 层字符 LM 的**每个**中间层、所有位置 | 同 next-char | 独立头（仅训练期存在） | 第 l 层在训练进度 l/(2n) 处停止，训练过半时全部移除（浅层先停） | 训练动力学（>10 层难训） | text8 dev bpc 1.158→1.062（去掉中间层 loss 则 +0.096） | 5 |
| 29 | **LayerSkip** — Elhoushi 2024 ACL (2404.16710) | F2 | 所有层（rotational / gradual curriculum 选子集） | 同 next-token | **共享（final LN + linear），norm✓** | ẽ(t,l) ∝ C(t,l)·e_scale·Σ_{i≤l} i（随深度二次增长）；预训练 e_scale 0.2，R=23/31 | 推理（early exit + 自投机） | 作者：全层全步 EE loss 会降末层；续训 Llama2-7B MMLU 46.0→43.1（baseline 为原 ckpt） | 5 |
| 30 | **EE-LLM** — Chen 2024 ICML (2312.04916) | F2 | 深度 1/4、1/2 处两个出口 | 同 next-token | 1.3B：出口无 LN、embedding tied；7B：untied | 常数：0.25/0.5（1.3B），0.1/0.2（7B） | 推理 | 从零 300B/150B token：末层 loss 与标准模型持平或略低 | 5 |
| 31 | CALM — Schuster 2022 NeurIPS (2207.07061) | F2 | T5 decoder 每层 | 同 | 共享 softmax（norm ?） | ω_i=i/Σj | 推理 | "mostly preserve"（无数字）；微调设定 | 4 |
| 32 | Depth-Adaptive T. — Elbayad 2020 ICLR (1910.10073) | F2 | decoder 每层（6） | 同 | IWSLT 各层独立头，WMT tied | 均匀权重平均最好；ω=n 对末层最好 | 推理 | IWSLT n=6 BLEU：35.9→36.2（ω=1），36.3（ω=n），35.8（ω=1/n） | 4 |
| 33 | DEED — Tang 2024 NAACL-F (2311.08623) | F2 | enc-dec VL 模型 decoder 每层 | 同 | 共享 generation head + 逐层 adapter | L_avg + L_N（给末层额外加一份） | 推理 | 作者：只用 L_avg 时末层会降；预训练加 DS 使首层 +3% | 4 |
| 34 | FREE — Bae 2023 EMNLP (2310.05424) | F2 | 浅、深两个出口 + 逐层 KD | 同 + KD | 共享 | α∝层数 | 推理 | 作者：co-train 大量出口导致退化；全模型 Multi-News 37.62（CALM 式）vs 39.20 | 3 |
| 35 | LITE — Varshney 2024 NAACL-F (2310.18581) | F2 | LLaMA-2 第 8,12,…,28 层 + 末层 | 同 | 共享 RMSNorm + LM head（norm✓） | 等权 | 推理 | 指令微调；末层"质量相当"（无单独数字） | 3 |
| 36 | Sorted LLaMA — Kavehzadeh 2024 EACL-F (2309.08968) | F2 | 13B 的第 12,16,…,40 层 | 同 | 共享 RMSNorm + head（norm✓） | 平均 | 推理 | SFT；末层与 SFT 相当；第 36 层子模型 ≈ 全模型 | 3 |
| 37 | EE natural capability — Shan 2024 arXiv (2412.01455) | F2 | 分析 | — | 不加额外头 | — | 分析 | 称 joint optimization 伤害全模型（定性） | 3 |
| 38 | BitSkip — Bhuvaneswaran 2025 arXiv (2510.23766) | F2 | 12 层 85M 模型的每层 | 同 | 共享 LM head | λ=0.3，w_l∝1/(l+1)（偏浅层）+ 层跳 p_max=0.7 | 推理 | WikiText-2 val PPL 228.77→252.37（与层跳混杂、数据极小） | 3 |
| 39 | MoDE — Luo 2024 arXiv (2410.13077) | F2 | 最后 k=3 层 | 同 + KL 蒸馏 | 共享 head + 逐层可训 norm + router | λ | 微调质量 | DoRA+MoDE 55.1 vs 54.7 | 2 |
| 40 | ELMER — Li 2022 EMNLP (2210.13304) | F2 | NAR decoder 每层（每 token 随机出口层） | 同 | off-ramp 可独立可共享 | LPLM 随机排列出口层 | 质量 + 速度 | XSUM R-L 29.92（BART 30.61） | 2 |
| 41 | DOC — Takase 2018 EMNLP (1808.10143) | F2 | RNN LM 中间层（含 embedding） | 输出为多层分布的混合 | ? | 混合权重 + 防止偏向浅层的正则 | 表达力 | PTB/WT2 当时 SOTA | 2 |
| 42 | SortedNet — Valipour 2023 arXiv (2309.00255) | F2 | 嵌套子网络（深度/宽度） | 同 | 共享 | 随机采样子模型 + 梯度累积 | 多合一 | 160 个子模型 ≥ 原模型 96% | 2 |
| 43 | EE-Tuning — Pan 2024 arXiv (2402.00518) | F2 | 在预训练 LLM 上加出口层 | 同 LM | 独立出口层 | 主干冻结 | 推理 | 主干不变 | 2 |
| 44 | Balcony — Jamialahmadi 2025 arXiv (2503.05005) | F2 | 选定出口处插入额外层 | 自蒸馏 | 独立 | 主干冻结 | 推理 | 主干不变；只用 0.2% 数据 | 2 |
| 45 | EESD — Liu 2024 ACL-F (2406.03853) | F2 | 前 N 层后接 1 个出口层 | 自蒸馏 | 独立（从末层 / LM head 初始化） | 主干冻结 | 投机解码 | 主干不变 | 1 |
| 46 | Self-sup. EE heads — Valade 2024 arXiv (2407.21082) | F2 | 中间层出口头 | 模仿主模型预测 | 独立 | （仅摘要） | 推理 | — | 1 |
| 47 | Kubaty — Kubaty 2025 ICML (2407.14320) | F3 | ResNet / ViT / BERT 的 ICs | 同 | 独立头 | disjoint / joint / mixed | 训练研究 | 从零 joint 伤末层：ViT-T IN-1k 71.61→68.39；mixed 71.20；BERT 20NG 约 85.75→84.3 | 4 |
| 48 | PABEE — Zhou 2020 NeurIPS (2006.04152) | F3 | ALBERT/BERT 每层 | 同 | 独立头 | Σ j·L_j / Σ j | 两者 | ALBERT-base GLUE 84.4→85.1（patience exit） | 4 |
| 49 | BERxiT — Xin 2021 EACL (aclanthology 2021.eacl-main.8) | F3 | 每层 | 同 | 独立头 | joint / two-stage / alternating（奇数步只训末层） | 推理 | joint 使末层分类器变差（只有图） | 4 |
| 50 | ElasticBERT — Liu 2022 NAACL (2110.07038) | F3 | **预训练**时每层 MLM+SOP | 同自监督目标 | ? | 等权；分组轮转 + gradient equilibrium | 两者 | GLUE 85.6（BERT 82.9，RoBERTa 86.1；无对照） | 4 |
| 51 | DeeBERT — Xin 2020 ACL (2004.12993) | F3 | 每层 off-ramp | 同 | 独立头 | 两阶段：先训末层，冻结主干后训 off-ramp | 推理 | 作者：联合训练"generally worsening"末层 | 3 |
| 52 | Right Tool — Schwartz 2020 ACL (2004.07453) | F3 | BERT-large 第 0/4/12/23 层 | 同 | 独立头 | 等权求和 | 推理 | 末层"smaller drop"（无数字） | 3 |
| 53 | LeeBERT — Zhu 2021 ACL (aclanthology 2021.acl-long.231) | F3 | 每层 | 同 + 互蒸馏 | 独立头 | 可学习权重（双层优化） | 推理 | 1.96× 加速下仍超 full ALBERT（MNLI 85.4 vs 84.6） | 3 |
| 54 | BoostNet — Yu 2023 AAAI (2211.16726) | F3 | 所有出口 | 同（boosting 形式） | 独立头 | 回传梯度按 1/(N−n+1) 重标定 | 两者 | IN 末出口 75.08 vs 74.69 | 3 |
| 55 | LEAP — Kapadia 2026 ACL-Ind (2605.01058) | F3 | MiniLM 每个中间层 | 对齐到末层表示 | n.a. | 0.3/0.4 常数 | 推理 | STS-B 0.777→0.760（−2.2%） | 3 |
| 56 | FastBERT — Liu 2020 ACL (2004.02178) | F3 | 每层 student | 自蒸馏 KL | 独立头 | 主干冻结 | 推理 | 末层不变 | 2 |
| 57 | CATs — Schuster 2021 EMNLP (2104.08803) | F3 | 每层 | 同 + conformal | 独立头 | 主干冻结（为了不伤全模型） | 推理 | — | 2 |
| 58 | RomeBERT — Geng 2021 arXiv (2101.09755) | F3 | 所有出口 | CE + 自蒸馏 | 独立头 | 冲突梯度投影 | 推理 | RTE 69.5 | 2 |
| 59 | Ensemble-IC — Sun 2021 arXiv (2105.13792) | F3 | 每层 | 同 + 多样性项 | 独立头 | 等权 | 推理 | SST-2 93.5 | 2 |
| 60 | ZTW — Wołczyk 2021 NeurIPS (2106.05409) | F3 | 每个 block | 同 + 级联 | 独立头 | 主干冻结 | 推理 | — | 2 |
| 61 | MEViT — Bakhtiarnia 2021 BMVC (2106.15183) | F3 | ViT 各层分支 | 同 | 独立头 | 端到端时末出口权重 ×2 | 推理 | — | 2 |
| 62 | LGViT — Xu 2023 ACM MM (2308.00255) | F3 | 按等 MAC 设出口 | 自蒸馏 | 独立头 | 两阶段，第二阶段冻结主干 | 推理 | 早退 88.5 vs 90.8 | 2 |
| 63 | Meta-GF — Sun 2022 ECCV (ECVA) | F3 | 所有出口 | 同 | 独立头 | 元学习的梯度融合 | 训练 | （仅摘要） | 2 |
| 64 | HDKD — Wang 2021 AAAI (OJS 17225) | F3 | 所有出口 | 同 + 稠密蒸馏 | 独立头 | 双层优化权重 | 两者 | （仅摘要） | 2 |
| 65 | DFS — Gong 2024 ECCV (2407.13986) | F3 | 所有出口 | 同 | 部分共享 | 特征分区 | 两者 | MSDNet/C100 末出口 79.56 vs 77.47 | 2 |
| 66 | Why early exits? — Scardapane 2020 Cogn. Comput. (2004.12814) | F3 | 综述 | — | — | "常见做法是给浅层分类器降权" | 综述 | — | 2 |
| 67 | Dynamic NN survey — Han 2022 TPAMI (2102.04906) | F3 | 综述 | — | — | — | 综述 | "多个分类器会互相干扰" | 1 |
| 68 | InterCTC — Lee 2021 ICASSP (2102.03216) | F4 | 中间 1 层 ⌊L/2⌋；变体为多层 | 同 CTC | **共享 CTC 头 + 共享 final LayerNorm（ESPnet），norm✓** | (1−w)L + w·L_inter，w=0.3 常开 | 训练动力学 | WSJ eval92：12 层 16.5→13.6；24 层 13.9→12.4；48 层 13.8→12.6 | 4 |
| 69 | SC-CTC — Nozaki 2021 Interspeech (2104.02724) | F4 | 18 层中的 5 个中间层 | 同 CTC + 预测回灌 | 共享 LN + Linear（norm✓） | λ=0.5，对中间层取平均 | 训练 | TEDLIUM2：12.2→10.1（InterCTC）→9.4 | 4 |
| 70 | Deja-vu（iterated loss）— Tjandra 2020 ICASSP (1910.10324) | F4 | 每 6–12 层 | 同 | 独立 MLP | λ=0.3 常数 | 训练动力学 | 36 层 4.0/9.4→3.4/8.1；使加深重新有收益 | 4 |
| 71 | Transformer 混合声学模型 — Wang 2020 ICASSP (1910.09799) | F4 | 24 层的第 6/12/18 层 | 同 CE | 独立头 | 0.3 常数 | 训练动力学 | 24 层没有此 loss 时"not converged"，加上后 2.66/5.64 | 4 |
| 72 | RNN-T 辅助任务 — Liu 2021 SLT (2011.03109) | F4 | 第 6/12/18 层 | 同 RNN-T + KL | 前向共享 decoder，**辅助梯度不更新它** | 常数 | 训练动力学 | 24 层 2.77/6.60→2.31/5.26；无辅助任务时 24/36 层不收敛 | 4 |
| 73 | 从零 early-exit ASR — Wright 2024 ICASSP-W (2309.09546) | F4 | 隔层出口 | 同 | 独立头 | 等权求和 | 推理 | 从零：Conformer-CTC 末出口 6.5/17.7→5.1/15.1；在预训练模型上微调加出口：wav2vec2 3.4/8.6→4.3/12.2 | 4 |
| 74 | DeCRED — Polok 2025 ASRU (2508.08938) | F4 | 自回归 decoder 中间层（默认 D−2） | 同 next-token | 独立头 | β=0.4，末层 0.6 | 训练 | OOD WER 18.2→16.2；中后层最好，浅层增益很小 | 4 |
| 75 | ILO shared decoder — Zhang 2022 arXiv (2207.04177) | F4 | 1 个中间编码层（最好是 9/12） | 同 | 共享 decoder | γ=0.2 | 训练 | 口音英语 8.1→7.7 | 3 |
| 76 | HuBERT-ILS — Wang 2021 arXiv (2112.08778) | F4 | 预训练时的 {4,12} / {9,24} 层 | 同 k-means 目标 | 独立头（共享时略差，<3%） | 等权常数 | 训练 | BASE 6.3/13.2→4.7/10.1；说话人 ID 81.42→79.29 | 3 |
| 77 | OWSM-CTC — Peng 2024 ACL (2402.12654) | F4 | 27 层中的 6/12/15/21 | 下半 ASR、上半任务相关 | 共享 CTC 投影 | 平均 | 训练 | 所有层都用任务相关硬目标时发散 | 3 |
| 78 | HC-CTC — Higuchi 2022 ICASSP (2110.04109) | F4 | 第 6/12/18 层 | 由粗到细的子词 | 独立头 | 等权 | 训练 | LS-100：9.1→8.4 | 3 |
| 79 | GIC — Yang 2023 ICASSP (2205.12462) | F4 | 第 3/6/9/12/15 层 | 同 + 预测回灌 | ? | λ=0.5 | 训练 | TEDLIUM2：8.3→7.3 | 3 |
| 80 | Layer pruning on demand — Lee 2021 Interspeech (2106.09216) | F4 | InterCTC + stochastic depth | 同 | ? | ? | 推理 | 仅摘要 | 2 |
| 81 | HMTL-CTC — Sanabria 2018 SLT (1807.07104) | F4 | 各 BiLSTM 层 | 由细到粗 | 独立头 | ? | 精度 | 12.5/23.7 vs 15.6/26.7 | 2 |
| 82 | Hierarchical MTL CTC — Krishna 2018 arXiv (1807.06234) | F4 | 第 i 层 phone CTC | 不同目标 | 独立头 | λ=0.7 | 精度 | SWB 21.5→18.6 | 2 |
| 83 | 低层级辅助任务 — Toshniwal 2017 Interspeech (1704.01631) | F4 | 第 2/3 层 | 低层级单元 | 独立头 | 等权 | 精度 | 25.0/42.4→23.1/40.8 | 2 |
| 84 | InterDecoder — Komatsu 2023 SLT (DOI 10.1109/SLT54892.2023.10022760) | F4 | 中间编码层 → attention decoder | CE | ? | ? | 训练 | 最多 6% 相对（仅摘要） | 2 |
| 85 | 多语辅助 CTC — Chen 2023 ICASSP (2302.12829) | F4 | 第 3 层（LID）+ 6/9/12/15 | LID → 转写 | ? | w=0.5 | 精度 | FLEURS CER 10.1 vs 14.1 | 2 |
| 86 | LAIL — Altinok 2025 TSD (2506.22846) | F4 | 第 6/12/18/24 层 | 冻结 LLaMA-3 算的因果 LM loss | 独立 connector | α=0.3 | 训练 | LibriSpeech 1.96/3.98→1.74/2.96 | 2 |
| 87 | PMS-SSL — Wan 2022 arXiv (2212.03480) | F4 | 第 6、12 层 | 粗→细 k-means | 独立头 | 等权 | 训练 | test-other 9.4→8.27 | 2 |
| 88 | 层间注意力 CTC — Hojo 2024 Interspeech (DOI 10.21437/Interspeech.2024-1776) | F4 | 每个编码层 | 同 CTC | ? | ? | 精度 | TEDLIUM2 9.9/11.8（摘要） | 2 |
| 89 | CTC 对齐用于翻译 — Yan 2023 EACL (2210.05200) | F4 | 中间层（源 CTC）+ 末层（目标 CTC） | 不同目标 | 独立头 | 常数 | 精度 | En-De 27.7→28.3 | 2 |
| 90 | HuBERT-EE — Yoon 2024 Interspeech (2204.06328) | F4 | 微调时第 5/8/11 层分支 | 同 | 独立头 | λ=1 或两阶段 | 推理 | 末层 3.88→3.90（中性） | 2 |
| 91 | Dynamic Encoder Transducer — Shi 2021 Interspeech (2104.02176) | F4 | 多深度编码器协同学习 | 同 RNN-T | ? | ? | 推理 | 全尺寸相对提升 >8%（来源不明） | 2 |
| 92 | Splitformer — Lasbordes 2025 arXiv (2506.18035) | F4 | 每 2 层 | 同 | 独立头 | 等权 | 推理 | 第 12 层：4.8/14.7 vs 5.1/14.8 | 2 |
| 93 | Inter-KD — Yoon 2022 SLT (2211.15075) | F4 | 第 18/24/30 层 | 同 + 教师蒸馏 | 独立头 | λ=0.25 | 压缩 | 8.85→6.30 | 2 |
| 94 | InterMPL — Higuchi 2023 ICASSP (2211.00795) | F4 | 中间 CTC（伪标签） | 伪标签 | ? | ? | 半监督 | 比 MPL 最多 +12.1% 绝对 | 1 |
| 95 | Multi-Layer Softmaxing — Dabre 2019 arXiv (1908.10118) | F5 | 6-6 AR NMT 的所有（编码深度 n, 解码层 m）组合 | 同 next-token | **单一共享 softmax** | 36 项平均，从零训练 | 推理灵活 | WMT18 En→De：6-6 BLEU 34.87 = 34.87；1-1 为 27.07 vs 24.24 | 4 |
| 96 | DSLP — Huang 2022 AAAI (2110.07515) | F5 | NAT decoder 每层 | 同 + 逐层预测回灌 | 共享输出投影 | 等权，从零训练 | 两者 | 仅 DS：+0.66 BLEU（21.18→21.84）；DSLP +1.54 | 4 |
| 97 | MV-Transformer — Wang 2020 COLING (2011.01482) | F5 | 1 个中间编码层作辅助视图 | 同 + KL 一致性 | decoder 部分共享 | α 0.3–0.5 | 训练 | IWSLT De→En 34.77→35.49 | 3 |
| 98 | Hint-NAT — Li 2019 EMNLP (1909.06708) | F5 | NAT 各层 | AR 教师的 hint | n.a. | λ=5，μ=1，常数 | 训练 | 23.08→25.55 | 2 |
| 99 | **Ouro / LoopLM** — Zhu 2025 arXiv (2510.25741) | F6 | 每个循环步（T_max=4） | 同 next-token | 各步共享 LM head | Σ_t p(t\|x)·L^(t) − βH(p)，β 0.1→0.05，均匀先验 | 自适应计算 | 附录 A：偏早步的几何先验使训练 loss 平台更高 | 4 |
| 100 | **Relaxed Recursive T.** — Bae 2025 ICLR (2410.20672) | F6 | 中间循环的输出（2–3 个出口） | 同 + 对 detached 末层输出做 KD | ? | α=i/Σi 过度强调中间层；aggressive 0.1 + KD | 推理（depth-wise batching） | uptrain 时联合训练：末层 avg 51.7→50.2（−1.5）；后训练用 i/Σi −1.2，用 0.1 +0.8 | 4 |
| 101 | PonderNet — Banino 2021 ICML-W (2107.05407) | F6 | 每一步 | 同 | 共享 | Σ p_n·L_n + β·KL(p‖Geom(λ_p)) | 自适应计算 | — | 3 |
| 102 | Looped T. 学算法 — Yang 2024 ICLR (2311.12424) | F6 | 迭代窗口 [b−T, b] | 同 | 共享 | 窗口平均（如 b=20，T=15） | 算法学习 | — | 3 |
| 103 | Progressive loss — Bansal 2022 NeurIPS (2202.05826) | F6 | 随机 n+k 次迭代处 | 同 | 共享 | (1−α)L_max + α·L_prog；前 n 步不回传 | 外推 | 防 overthinking | 2 |
| 104 | HRM — Wang 2025 arXiv (2506.21734) | F6 | 每个 segment | 同 | 共享输出头 | 每段一次 loss，段间 detach | 推理任务 | 原文无单独消融 | 2 |
| 105 | ARC Prize 的 HRM 分析 — 2025 blog | F6 | HRM 外循环 | 同 | — | — | 分析 | 外循环 refinement 是主要来源；训练时用 16 步，单步推理也 >15pp | 2 |
| 106 | TRM — Jolicoeur-Martineau 2025 arXiv (2510.04871) | F6 | N_sup=16 个监督步 | 同 | 共享 | 步间 detach，最后一次递归完整回传 | 推理任务 | Sudoku-Extreme 87.4%（1-step 梯度 56.5%） | 2 |
| 107 | LoopRPT — Tang 2026 arXiv (2603.19714) | F6 | Ouro 每个 latent 步 | RL 信号 | 共享 | — | 预训练 | 每步表示质量提升 | 2 |
| 108 | Huginn（recurrent depth）— Geiping 2025 arXiv (2502.05171) | F6 | **只有**末次循环 | 同 | coda 只有一次 | r ~ log-normal Poisson（均值 32），截断回传 8 步 | 测试时计算 | 对照：无逐步 loss | 2 |
| 109 | Universal Transformers — Dehghani 2019 ICLR (1807.03819) | F6 | ACT 加权后的输出 | 同 | 共享 | ponder cost | — | 对照 | 1 |
| 110 | ACT — Graves 2016 arXiv (1603.08983) | F6 | 输出为各步加权平均 | 同 | 共享 | ponder cost τ | — | 对照 | 1 |
| 111 | Looped 长度泛化 — Fan 2025 ICLR (2409.15647) | F6 | 预设步数 T_i 的输出 | 同 | 共享 | — | 长度泛化 | 对照 | 1 |
| 112 | Latent thoughts — Saunshi 2025 ICLR (2502.17416) | F6 | looping 启发的正则 | 正则 | — | — | 推理 | 对照 | 1 |
| 113 | Mixture-of-Recursions — Bae 2025 arXiv (2507.10524) | F6 | 未见逐递归 LM loss（只有 router 辅助 loss） | — | — | — | 自适应深度 | 对照 | 1 |
| 114 | 把 TRM 看作策略改进 — Asadulaev 2025 arXiv (2511.16886) | F6 | TRM 递归步 | RL/扩散式训练 | — | — | 推理任务 | 前向次数减少 18× | 1 |
| 115 | **HASTE** — Wang 2025 NeurIPS (2505.16792) | F7 | SiT 第 8 层特征 + 第 4–7 层注意力 | 对齐 DINOv2 | 独立投影 | λ=0.5，在 τ 处**硬停** | 训练 | SiT-XL/2 @500K：FID 8.1→5.3（τ=250K）；τ=400K 7.4；停得太早反而变差 | 4 |
| 116 | HLD-LLM — Guigon 2026 arXiv (2605.11513) | F7 | 学生中间层 | 对齐教师中间层 | 独立回归器 | HLDF：前 1–5% 预算做 hint 后移除；HLDC：常开 | 预训练 | 相对纯 KD：C4 3.005→3.000（123M），2.609→2.607（735M）；常开无增益 | 4 |
| 117 | DeepFlow — Shin 2025 ICCV (2503.14494) | F7 | k 个分支 | 同 velocity 目标 | ? | 中间 β=0.2，末层 1.0 | 训练 | SiT-B/2 FID 34.4→33.0（仅 DS） | 4 |
| 118 | REPA — Yu 2025 ICLR (2410.06940) | F7 | 1 个早中层（第 8/28 层） | 对齐 DINOv2 | 独立 MLP | λ=0.5 常数 | 训练 | 训练快 >17.5×；第 8 层最好（FID 10.0；第 12 层 11.2） | 3 |
| 119 | NITP — Zhang 2026 ICML (2605.24956) | F7 | 末层隐状态（经 MLP） | 预测下一 token 在浅层（约 20% 深度）的表示 | 独立投影 | λ=1.0 常数 | 预训练 | 9B MoE MMLU-Pro 15.29→21.00；dense 3B 平均 37.18→38.53 | 3 |
| 120 | OISD — Liu 2026 EMNLP-F (2605.29089) | F7 | 1 个中间层 | 对 detached 末层分布做 JSD | **共享 LN + unembedding，norm✓** | 常数 | RL 后训练 | Qwen3-4B Avg@K 55.08→64.75 | 3 |
| 121 | DistillLens — Dhakal 2026 arXiv (2602.13567) | F7 | 部分 decoder 层 | 对教师 logit-lens 分布做 JSD | 各自的 unembedding | λ=1 | 蒸馏 SFT | ROUGE-L 17.20→21.12 | 3 |
| 122 | FitNets — Romero 2015 ICLR (1412.6550) | F7 | 1 个中间层 | 对齐教师 hint | 独立回归器 | 先做 hint 预训练，**之后移除** | 训练 + 压缩 | 深窄网络变得可训；C10 91.61% | 3 |
| 123 | MobileBERT — Sun 2020 ACL (2004.02984) | F7 | 每层 | 蒸馏 | n.a. | auxiliary 83.0 < joint 83.5 < progressive 83.9 | 压缩 | MNLI-m | 3 |
| 124 | SRA — Jiang 2026 ICLR (2505.02831) | F7 | 早层对齐晚层（EMA） | 自对齐 | 独立投影 | λ=0.2 | 训练 | SiT-B/2 33.02→29.10；同深度对齐反而变差（37.08） | 2 |
| 125 | REPR-ALIGN — Peng 2026 arXiv (2605.06885) | F7 | 每层 | 对齐 AR LM 的同一层 | n.a. | λ=10 | AR→扩散转换 | 全层对齐 pass@1 18.00，高于任何 1/3 子集 | 2 |
| 126 | TinyBERT — Jiao 2020 EMNLP-F (1909.10351) | F7 | 每层映射 | 蒸馏 | n.a. | 先中间层 20 epoch，再预测层 3 epoch | 压缩 | 去掉层蒸馏 −19.3 | 2 |
| 127 | MiniLM — Wang 2020 NeurIPS (2002.10957) | F7 | 只有末层 | 蒸馏 | n.a. | 常数 | 压缩 | 末层蒸馏 80.0，优于逐层 79.0 | 2 |
| 128 | MiniLMv2 — Wang 2021 ACL-F (2012.15828) | F7 | 学生末层 ← 教师中上层 | 蒸馏 | n.a. | 常数 | 压缩 | 用中上层作教师更好 | 2 |
| 129 | PKD — Sun 2019 EMNLP (1908.09355) | F7 | 学生中间层 | 蒸馏 | n.a. | 常数 | 压缩 | MNLI-m 80.2→81.5 | 2 |
| 130 | iREPA — Singh 2025 arXiv (2512.10794) | F7 | 同 REPA | 对齐（空间结构） | conv 投影 | 同 REPA | 训练 | （仅摘要） | 1 |
| 131 | SOLO — Yin 2026 arXiv (2609.35440) | F8 | 24 层分为 K=2/4 个梯度隔离模块，每个模块都加 next-token loss | 同 | **共享的只读末层 unembedding（上一步的拷贝）**；各自有 RMSNorm、温度和 2 层 | 等权常数 | 训练效率 | 340M K=2：29.05 vs BP 28.04 ppl；共享读出 70.62 vs 私有 75.34（40M） | 4 |
| 132 | InfoPro — Wang 2021 ICLR (2101.10832) | F8 | K 个隔离模块 | 同标签（greedy）vs 重建 | 独立头 | 常数 | 显存 | greedy SL C10 err：7.37→10.30（K=2）→24.59（K=16） | 3 |
| 133 | Greedy ImageNet — Belilovsky 2019 ICML (1812.11446) | F8 | 逐层 | 同 | 独立头 | 按层顺序训练 | 理解 | k=3 集成 top-1 71.6 | 2 |
| 134 | DGL — Belilovsky 2020 ICML (1901.08164) | F8 | K 个模块 | 同 | 独立头 | 纯局部 | 并行 | VGG-13 67.8 vs BP 66.6 | 2 |
| 135 | Local error signals — Nøkland 2019 ICML (1901.06656) | F8 | 每个隐层 | 同 CE + 相似性匹配 | 独立头 | β=0.99 | 无全局 BP | predsim 优于纯 pred | 2 |
| 136 | PGL — Bhatti 2022 ICML-W (2208.00821) | F8 | 模块 | 同 | 独立头 | 周期性恢复全局目标 | 显存 | （仅摘要） | 2 |
| 137 | AugLocal — Ma 2024 ICLR (2402.17318) | F8 | 每个隐层 | 同 | 辅助网络复制后续层的结构 | 纯局部 | 显存 | ResNet-110 C10 93.96 vs BP 94.61 | 2 |
| 138 | LoPT — Shi 2026 arXiv (2605.04913) | F8 | 在中点处梯度隔离 | 上半用任务目标、下半用重建 | ? | — | 后训练效率 | （仅摘要） | 2 |
| 139 | Greedy InfoMax — Löwe 2019 NeurIPS (1905.11786) | F8 | 3 个隔离模块 | InfoNCE | 独立头 | 纯局部 | — | STL-10 81.9 vs 80.5 | 1 |
| 140 | Greedy layer-wise — Bengio 2006 NIPS | F8 | 逐层 | 无监督 | — | 逐层训练后全局微调 | 优化 | （仅摘要） | 1 |
| 141 | **LLAL** — Jin et al. 2026 blog | F9 | 第一个 MoE 层（L1） | 同 next-token | 共享 W_lm（是否过 norm 未说明） | λ=0.1 线性退火到 0：60B 用 40k/10k 步（共 90k），180B 用 4k 步（共 135k），另有 2k | 训练动力学 | 60B：val Δ≈−0.024，MMLU +3.9，MMLU-Pro +4.0；180B：MMLU 62.05→64.89 | 5 |
| 142 | LLAL 后续：How does LLAL change the model — Jin et al. 2026 blog | F9 | 同上 | — | — | — | 机制分析 | 早期专家获得"稀有知识"功能（屏蔽实验 ΔNLL） | 4 |
| 143 | DeepSeekMoE — Dai 2024 arXiv (2401.06066) | F9 | — | — | — | — | 背景 | 首层保持 dense，因为"负载均衡在第一层收敛特别慢" | 2 |
| 144 | Engram — Cheng 2026 arXiv (2601.07372) | F9 | — | — | — | — | 背景 | 条件记忆模块；LLAL 后续用它的插入位置做对照 | 2 |
| 145 | MoE 层内/跨层正则 — Hu 2026 arXiv (2602.14159) | F9 | MoE 各层 | 专家特化 / 跨层路由耦合正则（不是 LM loss） | — | — | 背景 | 下游一致提升（摘要） | 1 |
| 146 | Don't Drop Dropout — Elhoushi 2026 ICML (2609.05275) | F10 | — | layer dropout | — | 随层递增、随时间**递减到 0** 最好 | 训练 | 同 FLOPs 下 val loss 更低，early-exit 更好（271M–8.2B） | 3 |
| 147 | ProRes — Chen 2026 arXiv (2603.05369) | F10 | — | 残差 warmup（越深越晚） | — | "early layer learns first" | 训练 | 收敛更快，下游更好（摘要） | 3 |
| 148 | Logit lens — nostalgebraist 2020 LessWrong | F10 | — | 用末层 unembedding 读中间层 | — | — | 分析 | GPT-2 各层预测逐步收敛到末层 | 2 |
| 149 | Tuned lens — Belrose 2023 arXiv (2303.08112) | F10 | — | 每层一个仿射 probe（冻结模型） | — | — | 分析 | 比 logit lens 更可靠 | 2 |
| 150 | Curse of Depth — Sun 2025 NeurIPS (2502.05795) | F10 | — | LayerNorm Scaling | — | — | 训练 | Pre-LN 使深层接近恒等映射 | 2 |
| 151 | Depth efficiency — Csordás 2025 NeurIPS (2505.13898) | F10 | — | 分析 | — | — | 分析 | Llama3.1/Qwen3/OLMo2 后半层贡献小得多 | 2 |
| 152 | Tending Towards Stability — Diehl Martinez 2024 EMNLP-F (2410.11451) | F10 | — | 分析 | — | — | 分析 | 小模型的层收敛慢且不稳 | 2 |
| 153 | Learning Less Is More — Zhu 2026 arXiv (2605.10504) | F10 | — | 早期减慢上层 QK | — | 临时干预 | 训练 | 有 gated FFN（LLaMA 式）时几乎不需要 | 2 |
| 154 | JREG — Shibata 2026 EACL-F (2601.18302) | F10 | 末层 | 抑制末层 hidden-state 的跳变 | — | — | 预训练 | 任务表现提升（摘要） | 2 |
| 155 | Diminishing EE — Wei 2026 arXiv (2603.23701) | F10 | — | 分析 | — | — | 分析 | 新一代模型的早退潜力下降；dense > MoE > SSM | 2 |
| 156 | Bigram subnetworks — Chang 2025 NeurIPS (2504.15471) | F10 | — | 分析 | — | — | 分析 | bigram 子网络集中在第一个 MLP 层，参数 <0.2% | 2 |
| 157 | Layer by Layer — Skean 2025 ICML (2502.02013) | F10 | — | 分析 | — | — | 分析 | 中间层表示常优于末层 | 1 |
| 158 | ShortGPT — Men 2024 arXiv (2403.03853) | F10 | — | 层剪枝 | — | — | 分析 | 层冗余 | 1 |
| 159 | Neurons: Dead, N-gram, Positional — Voita 2023 arXiv (2309.04827) | F10 | — | 分析 | — | — | 分析 | 早层大量神经元 dead；有 n-gram 检测器 | 1 |
| 160 | Draft & Verify — Zhang 2024 ACL (2309.08168) | F10 | — | 推理时跳层 | — | 不训练 | 推理 | — | 1 |
| 161 | SkipDecode — Del Corro 2023 arXiv (2307.02628) | F10 | — | 推理时跳层 | — | — | 推理 | — | 1 |
| 162 | Kangaroo — Liu 2024 arXiv (2404.18911) | F10 | 浅子网络 + adapter | — | 主干冻结 | — | 推理 | — | 1 |

---

## 2. 分家族详述

各家族按与本实验的相关程度排序：F2 → F9 → F6 → F4 → F5 → F3 → F7 → F1 → F1b → F8 → F10。每条给出引用和链接、关键细节、相关度，汇总表里已有的字段不再重复。

### F2 Transformer LM / seq2seq decoder 的逐层 LM loss（核心，19 条）

**概览**：
- 真正在 decoder LM **从零预训练**中给中间层加 LM loss 的只有三篇：
  - Al-Rfou：所有层，逐层移除。
  - EE-LLM：2 个出口，常开。
  - LayerSkip：所有层轮转，常开，另有 layer dropout。
- 其余大多是在预训练好的模型上做微调或指令微调（CALM、LITE、Sorted LLaMA、MoDE），或者主干冻结（EE-Tuning、Balcony、EESD）。
- 在"是否伤末层"上，这三篇给出的信号是：
  - Al-Rfou：有益，但当时 >10 层很难训。
  - EE-LLM：中性。
  - LayerSkip：全层常开会伤末层，要靠 curriculum 缓解。

- **Al-Rfou T64** — Rami Al-Rfou, Dokook Choe, Noah Constant, Mandy Guo, Llion Jones. *Character-Level Language Modeling with Deeper Self-Attention.* AAAI 2019. [arXiv:1808.04444](https://arxiv.org/abs/1808.04444)。相关度 5。
  - **设置**：64 层 Transformer 字符 LM，d=512，FFN 2048，context 512，text8/enwik8；momentum 优化器，lr 固定 0.003，4M 步。
  - **三种辅助 loss**：
    - multiple positions：在所有位置都预测，即现在标准的 LM loss。
    - intermediate layer losses：每个中间层都在所有位置预测。
    - multiple targets：额外预测更远的字符，权重 0.5。
  - **head 不共享**：原文说中间层预测的输出分类层是"只在训练时使用"的参数，推理参数与训练参数分开计数。
  - **Schedule（原文）**："Lower layers are weighted to contribute less and less to the loss as training progresses. If there are n layers total, then the l-th intermediate layer stops contributing any loss after finishing l/2n of the training. This schedule drops all intermediate losses after half of the training is done."
    - 实现上是从第 1 层开始每隔固定步数移除一层，也就是**浅层先停**。
    - 原文写的间隔是 62.5K（=4M×1/(2·64)），但 4M/128=31.25K，原文算式与数值不一致，可能是笔误。
  - **动机**：">10 层训练困难、收敛慢、精度差"，加了辅助 loss 后"sped up convergence significantly"，并假设它也起正则作用。
  - **消融**（Table 4，text8 dev，context 512）：
    - T64 基线：1.062
    - 去掉 intermediate layer losses：1.158（+0.096）
    - 去掉 multiple positions：2.482
    - 去掉 multiple targets：1.068
  - **注意**：这是 2018 年的优化环境（深层难训），头不共享，而且与其它辅助 loss 同时使用。它是唯一"所有层 + 移除"的 decoder LM 先例，但不能直接外推到现代 pre-norm + Muon。

- **LayerSkip** — Mostafa Elhoushi, Akshat Shrivastava, Diana Liskovich et al. *LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding.* ACL 2024. [arXiv:2404.16710](https://arxiv.org/abs/2404.16710)。相关度 5。
  - **Loss**：J = Σ_l ẽ(t,l)·CE(g(x_{l+1}), Y)，其中 ẽ(t,l)=C(t,l)e(l)/Σ_i C(t,i)e(i)。
    - 浅层权重 e(l)=e_scale·Σ_{i=0}^{l} i（l<L−1）。
    - 末层权重 e(L−1)=L−1+e_scale·Σ_{i=0}^{L−2} i。
    - 原文："we penalize later layers with quadratically higher weight, as predicting in later layers is easier"。
  - **共享 head**："the language model head (that consists of the model's final layer normalization and linear layer)"，不加额外 head。这正是本实验"共享 final norm + LM head"的设计。
  - **Curriculum**：
    - 关键原文："We find that adding early exit loss of all layers at all iterations during training slows down training and reduces the accuracy of the last layer. To overcome this, we introduce a curriculum"。
    - rotational C_rot,R：每 R 层启用一个出口，逐步轮转，每步只做 ⌈L/R⌉ 次 unembedding。
    - gradual C_grad：从 L−1 到 0 每 T/2L 步启用一层。
  - **超参**：
    - 从零预训练 Llama2 1.5B（24 层）：p_max=0.1，e_scale=0.2，R=23；7B：p_max=0.2，e_scale=0.2，R=31；均为 26B token。
    - 续训（52B token）：Llama2 7B 用 e_scale 0.2、R=8；13B 用 0.1、R=39。
    - 另有 layer dropout D(l)=e^{l·ln2/(L−1)}−1。从零训练时还加上时间方向的指数 curriculum。
  - **末层证据**：
    - 续训 Llama2-7B 末层：MMLU 46.0→43.1，TriviaQA 58.5→56.8，GSM8K 14.3→12.2，HumanEval 13.4→15.9，MBPP 21.0→22.4。
    - 续训 Llama2-13B：MMLU 55.2→53.7。
    - Llama3-8B：GSM8K 54.2→45.0，HumanEval 37.8→28.7，MBPP 49.0→40.0。
    - Llama3.2-1B：HumanEval 17.7→9.15。
    - **注意**：续训对照是原始发布的 checkpoint，不是用同样数据续训的对照，因此数字里混入了续训数据的影响。作者把 Llama3 的更大降幅归因于其浅层原本的 PPL 高出 2–3 个数量级（8T token）。
    - 从零训练 26B token：原文说末层"in some downstream tasks ... a slight drop"，另一些任务则更高。只有图（Fig.8），没有表格。
    - Fig.11（1.5B，The Stack）：baseline 中层（第 12 层）PPL 随训练 token 增加"drastically"上升到数百，除非加 EE loss。末层 PPL 随 token 增加而下降（图中纵轴为 2.6–2.9）；各配置之间的末层差异只有图，没有数字。
  - **局限**（原文）：p_max、e_scale、R "requires tuning in order to avoid a drop in last layer accuracy"。

- **EE-LLM** — Yanxi Chen, Xuchen Pan, Yaliang Li, Bolin Ding, Jingren Zhou. *EE-LLM: Large-Scale Training and Inference of Early-Exit LLMs with 3D Parallelism.* ICML 2024. [arXiv:2312.04916](https://arxiv.org/abs/2312.04916)。相关度 5。
  - **1.3B 设置**：GPT 24 层，在第 6、12 层各加一个"minimalistic early-exit layer **without layer normalization**"，权重 1/4 和 1/2，末层 1，embedding tied，从零训练 300B token。
  - **7B 设置**：出口在第 8、16 层，权重 0.1 和 0.2，untied，从零训练 150B token。
  - **结论**（原文）："the final-exit loss curve of each early-exit model is close to (or even slightly below) that of the standard model, suggesting that optimizing for early-exit losses might not hurt the full-model output in our setting"。
  - **附带**：框架支持"changing early-exit loss weights during training"，但文中没有给出具体的变权重实验。

- **CALM** — Tal Schuster, Adam Fisch, Jai Gupta et al. *Confident Adaptive Language Modeling.* NeurIPS 2022. [arXiv:2207.07061](https://arxiv.org/abs/2207.07061)。相关度 4。
  - **设置**：在 T5 1.1 的 8 层 decoder（附录另有 12 层）上逐任务微调，最多 500K 步。
  - **Loss**：L=Σω_i L_i，ω_i=i/Σj，理由是"favor higher layers"。
  - **共享**："We share all output embeddings for the softmax predictions ... across all decoder layers"。是否对中间层施加 final LN，文中没有说明。
  - **末层**："find this objective to mostly preserve the full model's performance compared to regular training"，没有数字。
  - **exit 分类器**：第二阶段冻结其余参数，只训练 exit 分类器。

- **Depth-Adaptive Transformer** — Maha Elbayad, Jiatao Gu, Edouard Grave, Michael Auli. ICLR 2020. [arXiv:1910.10073](https://arxiv.org/abs/1910.10073)。相关度 4。
  - **aligned training**：6 层 decoder 每层都有输出分类器（IWSLT 上 6 个独立分类器、untied；WMT 上 tied）。
  - **IWSLT De-En 权重消融**（App. A，valid BLEU，n=6 末层，baseline 为独立训练的 6 层模型 35.9）：
    - ω=1：36.2
    - ω=n：36.3
    - ω=√n：36.1
    - ω=1/√n：35.9
    - ω=1/n：35.8
    - 第 1 层上的排序正好相反：ω=1/n 为 34.7，ω=n 为 32.2。
    - 作者认为均匀权重整体最好。
  - **gradient scaling**：把第 n 块自身监督的梯度放大 γ(N−n) 倍，可以帮最低层，但会牺牲高层；"no scaling generally works very well"。
  - **mixed training**：随机 exit 路径，更贴近推理，但末层低于 aligned（M=6 时 35.9，与 baseline 持平）。

- **DEED** — Peng Tang, Pengkai Zhu, Tian Li et al. *DEED: Dynamic Early Exit on Decoder for Accelerating Encoder-Decoder Transformer Models.* Findings of NAACL 2024. [arXiv:2311.08623](https://arxiv.org/abs/2311.08623)。相关度 4。
  - **L_avg 的问题**（原文）："this approach does not optimize the model for the final decoder layer solely. As a result, the model suffers from degraded accuracy of the final decoder layer"。
  - **改法**：L = L_avg + L_N。
  - **共享 head**：共享 generation head，但必须配逐层 adaptation module；没有 adapter 时，共享 head "has inferior performance due to the mis-alignment"。
  - **预训练阶段**：也加 deep supervision，使首层 +3%。
  - **结果**：DocVQA 等任务精度持平或略升（如 DEED-L +0.3，ST-VQA-L +1.2），decoder 延迟降 40–73%。

- **FREE** — Sangmin Bae, Jongwoo Ko, Hwanjun Song, Se-Young Yun. EMNLP 2023. [arXiv:2310.05424](https://arxiv.org/abs/2310.05424)。相关度 3。
  - 复评 CALM 式全层 early exit 时发现：T5-large 用前 1–2 层 static exit，ROUGE-L 接近 0。
  - 提出 shallow-deep module，只设浅、深两个出口，加上逐层 KD，目的是"tackles the performance degradation associated with co-training numerous exiting layers"。
  - Table 3 中两种全模型的对比（CALM 式加权平均全层训练 vs FREE）：SAMSum 48.82 vs 49.11，Multi-News 37.62 vs 39.20，SQuAD 90.63 vs 91.90，CNN/DM 41.15 vs 41.09。

- **LITE** — Neeraj Varshney, Agneet Chatterjee, Mihir Parmar, Chitta Baral. Findings of NAACL 2024（ACL Anthology 的标题是 *Investigating Acceleration of LLaMA Inference by Enabling Intermediate Layer Decoding via Instruction Tuning with 'LITE'*）。[arXiv:2310.18581](https://arxiv.org/abs/2310.18581)。相关度 3。
  - 在 Alpaca 上指令微调 LLaMA-2。7B 模型在第 8/12/16/20/24/28 层和末层上加等权 loss，中间层先过 RMSNorm 再进同一个 LM head。
  - 末层生成质量"comparable"，由 Claude 评分，没有逐项数字。

- **Sorted LLaMA** — Parsa Kavehzadeh, Mojtaba Valipour, Marzieh Tahaei et al. Findings of EACL 2024. [arXiv:2309.08968](https://arxiv.org/abs/2309.08968)。相关度 3。
  - LLaMA2-13B 的第 12,16,…,40 层作为子模型，loss 取平均，共享 RMSNorm 和 head，数据为 Alpaca/TriviaQA。
  - 第 36 层子模型"almost as well as"完整 SFT 模型；末层与 SFT 相当；作者承认模型仍欠训练。

- **EE natural capability** — Weiqiao Shan, Long Meng, Tong Zheng et al. arXiv 2024. [arXiv:2412.01455](https://arxiv.org/abs/2412.01455)。相关度 3。
  - 主张 early exit 是 Transformer 的固有能力：vanilla LLaMA 上不超过 11.08% 的 token 需要末层。
  - joint optimization 只是提高了邻层分布的相似度，"despite its negative impact on the performance of the full model"。这是定性结论，没有给数字。

- **BitSkip** — Ramshankar Bhuvaneswaran, Handan Liu. arXiv 2025. [arXiv:2510.23766](https://arxiv.org/abs/2510.23766)。相关度 3。
  - 85M、12 层 Llama 式 ternary 模型，只在 WikiText-2 上训练。
  - 按 LayerSkip 做法共享 LM head，但权重偏浅层：w_l=1/(l+1)，λ_ee=0.3，再加二次增长的层跳 p_max=0.7。
  - val PPL：baseline 228.77，只加 EE 为 252.37，Hadamard 为 185.46，Hadamard+EE 为 216.06。
  - 作者归因于"gradient interference from the shared language model head"。
  - **证据质量低**：数据极小、与层跳混杂、PPL 在 200 量级。但它是"偏浅层权重 + 共享 head"方向的一个反例。

- **MoDE** — Haoyan Luo, Lucia Specia. arXiv 2024. [arXiv:2410.13077](https://arxiv.org/abs/2410.13077)。相关度 2。
  - 最后 k=3 层各过一个可训练 norm，再进共享 LM head，由 router 混合 logits，并加 λ·KL 让浅层向末层对齐。
  - 与 LoRA/DoRA 叠加，微调算术推理：55.1 vs 54.7。

- **ELMER** — Junyi Li, Tianyi Tang, Wayne Xin Zhao, Jian-Yun Nie, Ji-Rong Wen. EMNLP 2022. [arXiv:2210.13304](https://arxiv.org/abs/2210.13304)。相关度 2。
  - NAR 预训练 LM 中，每个 token 随机指定一个出口层（Layer Permutation LM），上层直接拷贝隐状态。
  - off-ramp"可以独立，也可以跨层共享"。
  - XSUM ROUGE-L 29.92（BART 30.61）。

- **DOC** — Sho Takase, Jun Suzuki, Masaaki Nagata. EMNLP 2018. [arXiv:1808.10143](https://arxiv.org/abs/1808.10143)。相关度 2。
  - RNN LM 的最终分布是多个中间层分布的混合，作者认为这类似 GoogLeNet 辅助分类器，能缓解梯度消失。
  - 作者观察到"DOC tends to assign large weights to shallow layers"，所以加了一个防止权重偏向浅层的正则。

- **SortedNet** — Mojtaba Valipour, Mehdi Rezagholizadeh, Hossein Rajabzadeh et al. arXiv 2023. [arXiv:2309.00255](https://arxiv.org/abs/2309.00255)。相关度 2。嵌套子网络随机采样 + 梯度累积，160 个子模型都能达到原模型的 ≥96%。
- **EE-Tuning** — Xuchen Pan, Yanxi Chen, Yaliang Li, Bolin Ding, Jingren Zhou. arXiv 2024. [arXiv:2402.00518](https://arxiv.org/abs/2402.00518)。相关度 2。主干冻结，只参数高效地训练出口层；按构造末层不变。
- **Balcony** — Benyamin Jamialahmadi, Parsa Kavehzadeh, Mehdi Rezagholizadeh et al. arXiv 2025. [arXiv:2503.05005](https://arxiv.org/abs/2503.05005)。相关度 2。冻结 LLM，在出口处插入额外层并做自蒸馏，只用 LLaMA3-8B 预训练数据的 0.2%；声称优于 LayerSkip 和 Flextron。
- **EESD** — Jiahao Liu, Qifan Wang, Jingang Wang, Xunliang Cai. Findings of ACL 2024. [arXiv:2406.03853](https://arxiv.org/abs/2406.03853)。相关度 1。前 N 层冻结，接 1 个可训练的出口层（从末层和 LM head 初始化）做自蒸馏，用于投机解码。
- **Self-supervised early exits** — Florian Valade. arXiv 2024. [arXiv:2407.21082](https://arxiv.org/abs/2407.21082)。相关度 1。中间层的出口头学习模仿主模型的预测；只核实了摘要。

### F9 MoE 相关（5 条）

**检索结论**：
- 检索过 "MoE shallow layers auxiliary LM loss"、"early MoE layers dead experts auxiliary"、"early-exit MoE training" 等多组关键词。
- **除 LLAL 外，没有找到专门用逐层 LM loss 挽救早期 MoE 层的已发表工作**。
- 相关的只有背景性观察：DeepSeekMoE 首层保持 dense；新一代 MoE 的早退潜力低于 dense（Diminishing EE，见 F10）；NITP 在 MoE 上用浅层表示作目标（见 F7）。

- **LLAL** — Hongye Jin, Linwei Li, Xiaotian Han, Xin Liu, Haoyang Wen, Sha Li, Chia-Yuan Chang, Tuo Zhao, Qingyu Yin, Binxuan Huang. *Mitigate Silent Expert Death in Ultra-Sparse MoE.* Blog / Notion，2026-07-27（2026-08-22 更新）。<https://mooler0410.github.io/puguJin/blog/llal/>。相关度 5。细节见 §0.4。
  - 与本实验最相关的四点：
    1. 用 W_lm 共享的**单层** LM loss，λ=0.1 线性退火到 0。
    2. 窗口从 40k 缩到 2k，效果几乎不变。
    3. 辅助 loss 接在 L5 或 L3 时，接点上下相邻的层都受益；但把接点从 L5 挪到 L3，并没有明显改善最底层的 L1，所以最后直接接在 L1。这支持"每层都加"而不是只加一层。
    4. 页内评论（应来自作者方）明确说 dense 版本有待验证。
- **LLAL 后续：How does LLAL change the model? Similar to Engram, but not the same** — 同一作者组，2026。<https://mooler0410.github.io/puguJin/blog/llal-change/>。相关度 4。
  - 屏蔽 routed 专家的实验显示，LLAL 使 L1/L2 专家获得"稀有知识"功能，之后的层也都有正向贡献。
  - Engram 插在 L1 前会让早期层更健康，插在 L14 前则加剧早期层塌缩。
- **DeepSeekMoE** — Damai Dai et al. arXiv 2024. [arXiv:2401.06066](https://arxiv.org/abs/2401.06066)。相关度 2。原文："We substitute all FFNs except for the first layer with MoE layers, since we observe that the load balance status converges especially slower for the first layer." 这说明早期 MoE 层的训练难度是已知现象。
- **Engram** — Xin Cheng, Rui Tian, Wangding Zeng, Damai Dai et al. *Conditional Memory via Scalable Lookup: A New Axis of Sparsity for LLMs.* arXiv 2026. [arXiv:2601.07372](https://arxiv.org/abs/2601.07372)。相关度 2。条件记忆模块。只作为 LLAL 后续文章的对照背景，本身不是中间层监督。
- **Synergistic Intra- and Cross-Layer Regularization Losses for MoE** — Rizhen Hu, Yuan Cao, Boao Kong, Mou Sun, Kun Yuan. arXiv 2026. [arXiv:2602.14159](https://arxiv.org/abs/2602.14159)。相关度 1。层内专家特化 loss（惩罚专家 SwiGLU 激活的余弦相似度）加跨层 top-k 路由耦合 loss。这是 LLAL 引用的"其它 MoE 辅助 loss"，不是 LM loss。

### F6 Looped / recurrent-depth / universal transformer 的逐步 loss（16 条）

**概览**：
- 权重共享的循环模型里，"每步都加任务 loss"很常见：PonderNet、Ouro、HRM/TRM 的 deep supervision、Yang 等人的迭代窗口。
- 但也有主流做法**只**在末步加 loss：Huginn、UT/ACT、Fan 等人。
- 与本实验关系最近的两条：
  - Ouro：LM 预训练，各步共享 head，权重是学出来的 p(t|x)，另加熵正则。
  - RRT：在中间循环上加 LM loss 会伤末层，要用小系数或后训练。
- 注意：权重共享会放大"中间目标与末层目标的冲突"，因为同一组参数要同时服务两个目标。所以这里观察到的伤害，放到 dense 模型上可能偏大，可视为上界。

- **Ouro / LoopLM** — Rui-Jie Zhu, Zixuan Wang, Kai Hua et al. *Scaling Latent Reasoning via Looped Language Models.* arXiv 2025. [arXiv:2510.25741](https://arxiv.org/abs/2510.25741)。相关度 4。
  - **Stage I 预训练 loss**：L=Σ_{t=1}^{T_max} p_φ(t|x)·L^(t) − β·H(p_φ)。β 在 Stage 1a 为 0.1，之后为 0.05；等价于均匀先验下的 ELBO。
  - **结构**：各步共享同一个 LM head。模型为 1.4B/2.6B，T_max=4，共 7.7T token。
  - **App. A 先验对比**：776M、T_max=4、20B token FineWeb-Edu 上，几何先验（偏早退）的训练 loss 平台更高，且 λ 越大差距越大，原文说是"weaker supervision for deeper iterations"；均匀先验最好。只有图，没有数字。
  - **启示**：把太多权重放在浅步会伤最终质量。
  - **Stage II**：冻结 LM，只训练 exit gate。
- **Relaxed Recursive Transformers (RRT)** — Sangmin Bae, Adam Fisch, Hrayr Harutyunyan, Ziwei Ji, Seungyeon Kim, Tal Schuster. ICLR 2025. [arXiv:2410.20672](https://arxiv.org/abs/2410.20672)。相关度 4。
  - 在 Gemma 转成的 2-loop 递归模型上做 early-exit 训练，对比见 Table J.1：
    - 基线：uptrain 15B token 后，末层平均 few-shot 51.7。
    - **co-training**：在 uptraining 期间就加中间循环 loss（aggressive 0.1）→ 末层 50.2（−1.5），SlimPajama PPL 12.85→13.24。原文："significantly degraded the final output performance, even with an aggressive loss coefficient strategy"。
    - **后训练 15B token**：权重 α_i=i/Σi 时末层 50.5（−1.2），原文说它"overemphasis on the training of intermediate representations"；系数 0.3、0.1、0.05、0.01 时末层分别为 52.2、52.5、52.6、52.4（+0.5~0.9，部分来自多训的 token）。
    - 第一循环的准确率随系数下降而下降：i/Σi 49.8，0.3 为 48.5，0.1 为 47.1，0.05 为 46.0，0.01 为 42.1。
    - 加上对 detached 末层输出的 KD 后，"aggressive 0.1"最稳。
- **PonderNet** — Andrea Banino, Jan Balaguer, Charles Blundell. ICML 2021 AutoML Workshop. [arXiv:2107.05407](https://arxiv.org/abs/2107.05407)。相关度 3。L=Σ_n p_n·L(y,ŷ_n) + β·KL(p_n‖p_G(λ_p))。几何先验鼓励探索，也把期望步数偏向 1/λ_p；各步使用同一网络。
- **Looped Transformers are Better at Learning Learning Algorithms** — Liu Yang, Kangwook Lee, Robert Nowak, Dimitris Papailiopoulos. ICLR 2024. [arXiv:2311.12424](https://arxiv.org/abs/2311.12424)。相关度 3。
  - loss 取迭代窗口 [b−T, b] 上的平均，例如线性回归用 b=20、T=15。
  - 理由：借鉴截断 BPTT；让模型学到不发散的不动点。T 太大时梯度会波动、训练不稳。
- **End-to-end Algorithm Synthesis with Recurrent Networks (progressive loss)** — Arpit Bansal, Avi Schwarzschild, Eitan Borgnia et al. NeurIPS 2022. [arXiv:2202.05826](https://arxiv.org/abs/2202.05826)。相关度 2。
  - 采样 n~U{0,m−1}、k~U{1,m−n}；先跑 n 步且不回传梯度，再跑 k 步计算 loss。
  - L=(1−α)L_max_iters + α·L_progressive，目的是防 overthinking、学到与迭代次数无关的行为。
- **HRM** — Guan Wang, Jin Li, Yuhao Sun et al. *Hierarchical Reasoning Model.* arXiv 2025. [arXiv:2506.21734](https://arxiv.org/abs/2506.21734)。相关度 2。每个 segment 算一次 loss，段间把隐状态 detach（1-step 近似），输出头共享，另有 ACT/Q-learning 停止机制；约 27M 参数。原文没有单独消融 deep supervision。
- **The Hidden Drivers of HRM's Performance on ARC-AGI** — ARC Prize Team, 2025-08-15. <https://arcprize.org/blog/hrm-analysis>。相关度 2。
  - 同尺寸普通 Transformer 与 HRM 相差约 5pp，分层架构不是关键。
  - 外循环 refinement 是关键：从 0 步到 1 步 +13pp；训练时用 16 步 refinement，即使推理只用 1 步也 >15pp。
- **TRM** — Alexia Jolicoeur-Martineau. *Less is More: Recursive Reasoning with Tiny Networks.* arXiv 2025. [arXiv:2510.04871](https://arxiv.org/abs/2510.04871)。相关度 2。
  - N_sup=16 个监督步，步间 detach latent，但对最后一次递归做完整回传：Sudoku-Extreme 87.4%，用 HRM 式 1-step 梯度只有 56.5%。
  - 文中引用 ARC Prize (2025a) 称"deep supervision doubled accuracy ... 19% to 39%"。这两个数字我没能在上面那篇博客正文中找到，见 §6。
- **LoopRPT** — Guo Tang, Shixin Jiang, Heng Chang et al. arXiv 2026. [arXiv:2603.19714](https://arxiv.org/abs/2603.19714)。相关度 2。在 Ouro 上把 RL 信号直接分配到各 latent 步（EMA 教师），提升了每步表示的质量，尤其是早期步。
- **Huginn（recurrent depth）** — Jonas Geiping, Sean McLeish, Neel Jain et al. arXiv 2025. [arXiv:2502.05171](https://arxiv.org/abs/2502.05171)。相关度 2（对照）。loss 只在随机 r 次循环后计算，r~log-normal Poisson（均值 32），只回传最后 8 步，coda 只出现一次。它说明大规模 looped LM 也可以不用逐步 loss。
- **Universal Transformers** — Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit, Łukasz Kaiser. ICLR 2019. [arXiv:1807.03819](https://arxiv.org/abs/1807.03819)。相关度 1。loss 加在 ACT 加权后的输出上，不是逐步 loss。
- **ACT** — Alex Graves. arXiv 2016. [arXiv:1603.08983](https://arxiv.org/abs/1603.08983)。相关度 1。输出是各步的加权平均，另加 ponder cost。
- **Looped Transformers for Length Generalization** — Ying Fan, Yilun Du, Kannan Ramchandran, Kangwook Lee. ICLR 2025. [arXiv:2409.15647](https://arxiv.org/abs/2409.15647)。相关度 1。只在每个样本预设的步数 T_i 处计算 loss。
- **Reasoning with Latent Thoughts** — Nikunj Saunshi, Nishanth Dikkala, Zhiyuan Li, Sanjiv Kumar, Sashank Reddi. ICLR 2025. [arXiv:2502.17416](https://arxiv.org/abs/2502.17416)。相关度 1。looping 启发的正则，不是逐步 loss。
- **Mixture-of-Recursions** — Sangmin Bae, Yujin Kim, Reza Bayat et al. arXiv 2025. [arXiv:2507.10524](https://arxiv.org/abs/2507.10524)。相关度 1。全文检索没有找到逐递归的 LM loss，只有 router 的辅助 loss（balancing loss、z-loss）。
- **Latent Reasoning in TRMs is Secretly a Policy Improvement Operator** — Arip Asadulaev, Rayan Banerjee, Fakhri Karray, Martin Takac. arXiv 2025. [arXiv:2511.16886](https://arxiv.org/abs/2511.16886)。相关度 1。把 TRM 的递归看作策略改进，前向次数减少 18×。

### F4 语音（27 条）

**概览**：
- 语音领域是"同标签中间层 loss + 从零训练 Transformer"证据最多的地方，而且几乎一致为正。
- 它也提供了最直接的"共享 final norm + 输出头"先例：ESPnet 的 InterCTC/SC-CTC 实现里，中间层输出先过 encoder 的 `after_norm`，再进同一个 CTC 头。
- 但**没有任何语音论文对中间 loss 做退火或移除**，全部是常开常数权重（0.3–0.5）。
- 在已训练好的模型上**事后**加出口会伤末层（Wright 等人）。
- 多层 same-label 监督的收益会饱和，除非把中间预测回灌进残差流（SC-CTC、GIC），或者给浅层更粗的目标（HC-CTC）。

- **InterCTC** — Jaesong Lee, Shinji Watanabe. *Intermediate Loss Regularization for CTC-based Speech Recognition.* ICASSP 2021. [arXiv:2102.03216](https://arxiv.org/abs/2102.03216)。相关度 4。
  - **形式**：L=(1−w)L_CTC + w·L_InterCTC，w=0.3 常开；默认监督 ⌊L/2⌋ 层。
  - **ESPnet 实现**：中间输出过 `after_norm` 后进共享的 `self.ctc`，多层时对中间 loss 取平均。
  - **WSJ eval92**：12 层 16.5→13.6，24 层 13.9→12.4，48 层 13.8→12.6。
  - **位置**：下层（6/24）12.9，差于中层的 12.4，但仍好于 baseline 13.9；多层（24 层中取 K=3）12.0。
  - **动机**：stochastic depth 只正则化了高层，低层"may rely on the remaining higher layers"。
- **SC-CTC** — Jumon Nozaki, Tatsuya Komatsu. Interspeech 2021. [arXiv:2104.02724](https://arxiv.org/abs/2104.02724)。相关度 4。
  - 18 层中 5 个中间层复用同一个 LayerNorm 和 Linear（Eq.6–7），λ=0.5，中间预测再回灌进残差流。
  - TEDLIUM2：12.2 → 10.1（InterCTC）→ 9.4。
  - Fig.3：InterCTC 增加中间 loss 的数量时，12 层"no improvement"，18 层只有少许提升。
- **Deja-vu（iterated loss）** — Andros Tjandra, Chunxi Liu, Frank Zhang et al. ICASSP 2020. [arXiv:1910.10324](https://arxiv.org/abs/1910.10324)。相关度 4。每 6–12 层接一个独立 MLP 头，λ=0.3。36 层 baseline 与 24 层相同（4.0/9.4），加 iterated loss 后 36 层到 3.4/8.1，也就是加深重新有了收益。
- **Transformer-based Acoustic Modeling for Hybrid ASR** — Yongqiang Wang, Abdelrahman Mohamed, Duc Le et al. ICASSP 2020. [arXiv:1910.09799](https://arxiv.org/abs/1910.09799)。相关度 4。在 24 层的第 6/12/18 层加独立头，权重 0.3。24 层模型没有它时"not converged"，有它时为 2.66/5.64。原文："Deep transformer models (deeper than 20 layers) often got stuck in training"。
- **Improving RNN Transducer Based ASR with Auxiliary Tasks** — Chunxi Liu, Frank Zhang, Duc Le et al. SLT 2021. [arXiv:2011.03109](https://arxiv.org/abs/2011.03109)。相关度 4。
  - 辅助 RNN-T 分支前向共享 predictor/joiner，但"we do not update the decoder parameters if the gradients are back propagated from the auxiliary RNN-T loss"。这就是"对共享头 stop-grad"的现成先例。
  - 24 层：2.77/6.60 → 2.31/5.26。没有辅助任务时，24 层和 36 层都无法收敛。
- **Training dynamic models using early exits for ASR** — George August Wright, Umberto Cappellazzo, Salah Zaiem et al. ICASSP 2024 SASB workshop. [arXiv:2309.09546](https://arxiv.org/abs/2309.09546)。相关度 4。
  - 从零训练，隔层出口等权求和，末出口与单出口模型对比：Conformer-CTC 5.1/15.1 vs 6.5/17.7；AED 2.3/6.0 vs 2.5/6.1。
  - 在已预训练的模型上加出口再微调，末层变差：wav2vec2 4.3/12.2 vs 3.4/8.6；WavLM 3.6/8.8 vs 3.0/6.5。
  - 这是"从零加中间 loss 有益、事后加有害"最清楚的对照。
- **DeCRED** — Alexander Polok, Santosh Kesiraju, Karel Beneš et al. ASRU 2025. [arXiv:2508.08938](https://arxiv.org/abs/2508.08938)（早期版本 arXiv:2410.17437）。相关度 4。
  - 在自回归 decoder（也就是模型内部的条件 LM）的中间层加 next-token CE，默认在 D−2 层，β=0.4，末层保留 0.6；各层用独立头。
  - OOD WER 18.2→16.2，内部 LM 困惑度降 36.6%。
  - 中后层最好，浅层增益很小，多个辅助头没有额外收益。
- **ILO shared decoder** — Jicheng Zhang, Yizhou Peng, Haihua Xu et al. arXiv 2022（投 Interspeech 2022，录用未确认）。[arXiv:2207.04177](https://arxiv.org/abs/2207.04177)。相关度 3。中间编码层的输出送入同一个 attention decoder，γ=0.2，最佳位置约为 9/12 层；口音英语 8.1→7.7。
- **HuBERT-ILS** — Chengyi Wang, Yu Wu, Sanyuan Chen et al. arXiv 2021（投 ICASSP 2022）。[arXiv:2112.08778](https://arxiv.org/abs/2112.08778)。相关度 3。预训练时在 {4,12} 层加 masked prediction，各层独立头（共享头时略差，<3%）。BASE 6.3/13.2→4.7/10.1，但说话人 ID 81.42→79.29，即中间监督会挤掉非目标信息。
- **OWSM-CTC** — Yifan Peng, Yui Sudo, Muhammad Shakeel, Shinji Watanabe. ACL 2024. [arXiv:2402.12654](https://arxiv.org/abs/2402.12654)。相关度 3。1B 参数、180k 小时数据，在 6/12/15/21 层加共享投影的自条件 CTC。所有中间层都用任务相关的硬目标时训练**发散**；前半 ASR、后半多任务最好。
- **HC-CTC** — Yosuke Higuchi, Keita Karube, Tetsuji Ogawa, Tetsunori Kobayashi. ICASSP 2022. [arXiv:2110.04109](https://arxiv.org/abs/2110.04109)。相关度 3。第 6/12/18 层分别用 256/2048/16384 子词，由粗到细，优于各层同一词表：LS-100 8.2/19.9 vs 8.9/21.0。
- **GIC** — Yuting Yang, Yuke Li, Binbin Du. ICASSP 2023. [arXiv:2205.12462](https://arxiv.org/abs/2205.12462)。相关度 3。每 3 层一个中间 CTC，预测回灌并加门控；TEDLIUM2 8.3（InterCTC）→ 7.3。
- **Layer Pruning on Demand with Intermediate CTC** — Jaesong Lee, Jingu Kang, Shinji Watanabe. Interspeech 2021. [arXiv:2106.09216](https://arxiv.org/abs/2106.09216)。相关度 2。InterCTC 加 stochastic depth，任意深度都可用；方法细节只核实了摘要。
- **Hierarchical Multi Task Learning With CTC** — Ramon Sanabria, Florian Metze. SLT 2018. [arXiv:1807.07104](https://arxiv.org/abs/1807.07104)。相关度 2。逐层从字符到子词，比全放顶层好 14–20% 相对。
- **Hierarchical Multitask Learning for CTC-based ASR** — Kalpesh Krishna, Shubham Toshniwal, Karen Livescu. arXiv 2018. [arXiv:1807.06234](https://arxiv.org/abs/1807.06234)。相关度 2。phone CTC 放在第 3、4 层好于放在顶层；SWB 21.5→18.6。
- **Multitask Learning with Low-Level Auxiliary Tasks** — Shubham Toshniwal, Hao Tang, Liang Lu, Karen Livescu. Interspeech 2017. [arXiv:1704.01631](https://arxiv.org/abs/1704.01631)。相关度 2。低层级目标放在低层好于放在顶层。
- **InterDecoder** — Tatsuya Komatsu, Yusuke Fujita. SLT 2022（2023 年出版），DOI 10.1109/SLT54892.2023.10022760。相关度 2。中间编码输出接 attention decoder 做正则；只核实了摘要，最多 6% 相对提升。
- **Improving Massively Multilingual ASR With Auxiliary CTC Objectives** — William Chen, Brian Yan, Jiatong Shi et al. ICASSP 2023. [arXiv:2302.12829](https://arxiv.org/abs/2302.12829)。相关度 2。第 3 层用语种 ID 作目标，其上各层用转写；FLEURS CER 10.1 vs 14.1。
- **LAIL** — Duygu Altinok. TSD 2025. [arXiv:2506.22846](https://arxiv.org/abs/2506.22846)。相关度 2。第 6/12/18/24 层经 connector 接冻结的 LLaMA-3，计算因果 LM loss，α=0.3；LibriSpeech 1.96/3.98→1.74/2.96。
- **PMS-SSL** — Genshun Wan, Tan Liu, Hang Chen et al. arXiv 2022. [arXiv:2212.03480](https://arxiv.org/abs/2212.03480)。相关度 2。中间层用粗 k-means 目标、顶层用细目标；test-other 9.4→8.27。
- **Inter-layer attention-based CTC** — Keigo Hojo, Yukoh Wakabayashi, Kengo Ohta et al. Interspeech 2024，DOI 10.21437/Interspeech.2024-1776。相关度 2。每个编码层都加 CTC（只读了摘要）。
- **CTC Alignments Improve Autoregressive Translation** — Brian Yan, Siddharth Dalmia, Yosuke Higuchi et al. EACL 2023. [arXiv:2210.05200](https://arxiv.org/abs/2210.05200)。相关度 2。中间层用源语言 CTC、末层用目标语言 CTC；En-De 27.7→28.3。
- **HuBERT-EE** — Ji Won Yoon, Beom Jun Woo, Nam Soo Kim. Interspeech 2024. [arXiv:2204.06328](https://arxiv.org/abs/2204.06328)。相关度 2。微调时联合训练出口，末层 3.88→3.90，基本中性。
- **Dynamic Encoder Transducer** — Yangyang Shi, Varun Nagaraja, Chunyang Wu et al. Interspeech 2021. [arXiv:2104.02176](https://arxiv.org/abs/2104.02176)。相关度 2。多深度编码器协同学习；只核实了摘要。
- **Splitformer** — Maxence Lasbordes, Daniele Falavigna, Alessio Brutti. arXiv 2025. [arXiv:2506.18035](https://arxiv.org/abs/2506.18035)。相关度 2。每 2 层一个出口，从零训练。
- **Inter-KD** — Ji Won Yoon, Beom Jun Woo, Sunghwan Ahn et al. SLT 2022. [arXiv:2211.15075](https://arxiv.org/abs/2211.15075)。相关度 2。中间 CTC 头同时蒸馏教师 softmax；8.85→6.30。
- **InterMPL** — Yosuke Higuchi, Tetsuji Ogawa, Tetsunori Kobayashi, Shinji Watanabe. ICASSP 2023. [arXiv:2211.00795](https://arxiv.org/abs/2211.00795)。相关度 1。

另有 4 篇已核实但属边缘、未计分：2405.17376（联邦 early-exit ASR）、2106.08595（多说话人 InterCTC）、2308.08449（CTC-AED 辅助正则）、2309.12234（双语 CTC 翻译）。

### F5 MT / seq2seq（4 条）

- **Multi-Layer Softmaxing during Training NMT** — Raj Dabre, Atsushi Fujita. arXiv 2019. [arXiv:1908.10118](https://arxiv.org/abs/1908.10118)。相关度 4。
  - 6-6 自回归 Transformer，从零训练，对所有（编码深度 n, 解码层 m）共 36 种组合求 next-token CE 再取平均，所有组合共享**同一个 softmax**。
  - WMT18 En→De：6-6 BLEU 34.87，vanilla 也是 34.87，完全持平；1-1 为 24.24，dedicated 1-1 模型为 27.07。
  - 原文："when the number of decoder layers are increased there is no statistically significant difference"。
  - 训练时间约为 9.5×。
  - 这是最接近"AR decoder、所有层、共享输出层、从零训练"的对照：**末层无损也无益**。
- **DSLP** — Chenyang Huang, Hao Zhou, Osmar R. Zaïane, Lili Mou, Lei Li. *Non-Autoregressive Translation with Layer-Wise Prediction and Deep Supervision.* AAAI 2022. [arXiv:2110.07515](https://arxiv.org/abs/2110.07515)。相关度 4。
  - NAT decoder 每层用共享输出投影、等权 CE。
  - 只加 deep supervision：21.18→21.84（+0.66）。
  - 再加逐层预测回灌：22.72（+1.54）；加 mixed training 后 24.17。
- **Layer-Wise Multi-View Learning for NMT** — Qiang Wang, Changliang Li, Yue Zhang, Tong Xiao, Jingbo Zhu. COLING 2020. [arXiv:2011.01482](https://arxiv.org/abs/2011.01482)。相关度 3。
  - 把一个中间编码层当作辅助视图，交给部分共享的 decoder（交叉注意力独立，因为"a fully shared decoder has no sufficient capacity"），再加 KL 一致性，α 取 0.3–0.5。
  - IWSLT De→En 34.77→35.49；WMT16 En→De 33.06→33.75。
- **Hint-Based Training for NAT** — Zhuohan Li, Zi Lin, Di He et al. EMNLP 2019. [arXiv:1909.06708](https://arxiv.org/abs/1909.06708)。相关度 2。用 AR 教师的隐状态和注意力做 hint，λ=5，μ=1，常开；23.08→25.55。

（已排除：arXiv 2005.08081 *Layer-Wise Multi-View Decoding*，纯架构改动，没有辅助 loss。）

### F3 Early-exit / adaptive-depth encoder 与多出口训练研究（21 条）

**概览**：
- 这一族以推理加速为目的，但留下了两类对本实验有用的信息。
- 其一，大量工作**为了保护末层而冻结主干或交替训练**（DeeBERT、FastBERT、CATs、ZTW、BERxiT），这本身就是"联合全层监督伤末层"的旁证。
- 其二，Kubaty 等人（ICML 2025）的系统比较：从零 joint 在大数据上伤末出口，"先 backbone 后 joint"的 mixed 更好。注意这与 LLAL 的"先加后撤"顺序**相反**。

- **How to Train Your Multi-Exit Model?** — Piotr Kubaty, Bartosz Wójcik, Bartłomiej Krzepkowski et al. ICML 2025. [arXiv:2407.14320](https://arxiv.org/abs/2407.14320)。相关度 4。
  - 末出口准确率，按 disjoint（只训 backbone）/ joint / mixed 排列：
    - ViT-T IN-1k：71.61 / **68.39** / 71.20
    - ViT-S IN-1k：78.38 / 76.44 / 78.33
    - ResNet-50 Tiny-IN：65.71 / 65.01 / 67.24
    - ViT-T C100：63.99 / 66.49 / 70.25
    - MSDNet C100：70.36 / 75.86 / 76.51
    - BERT-B 20NG：最高 85.75 / 84.24–84.41 / 84.99–85.25
  - 我已对照原文 Table 1 抽查这些数字。
  - **机制**："optimization focuses on subnetworks in the middle of the architecture, where gradients from intermediate classifiers exert the strongest influence"。
  - **结论**：小数据上 joint 可以有益（C100），大数据上有害（ImageNet、BERT）。
- **PABEE（BERT Loses Patience）** — Wangchunshu Zhou, Canwen Xu, Tao Ge et al. NeurIPS 2020. [arXiv:2006.04152](https://arxiv.org/abs/2006.04152)。相关度 4。L=Σ j·L_j / Σ j，按层号线性加权；ALBERT-base GLUE 84.4→85.1。注意这个数字比较的是 patience exit，而不是单独的末层。
- **BERxiT** — Ji Xin, Raphael Tang, Yaoliang Yu, Jimmy Lin. EACL 2021. <https://aclanthology.org/2021.eacl-main.8/>。相关度 4。
  - 原文："Joint treats all classifiers equally, and therefore its final classifier is less effective than that of Two-stage"。
  - 提出 alternating：奇数步只用 L_n 训练全部参数，偶数步用 Σ L_i，相当于给辅助 loss 50% 的占空比。
  - 各策略的末层数字只在图里。
- **ElasticBERT（Towards Efficient NLP）** — Xiangyang Liu, Tianxiang Sun, Junliang He et al. NAACL 2022. [arXiv:2110.07038](https://arxiv.org/abs/2110.07038)。相关度 4。
  - **预训练**时在每层加 MLM+SOP，等权；出口分组后按 batch 轮转，并用 gradient equilibrium。
  - 12 层 GLUE 85.6（BERT-base 82.9，RoBERTa-base 86.1）。
  - 没有"同配方但不加逐层 loss"的对照，所以对末层的影响未测。
- **DeeBERT** — Ji Xin, Raphael Tang, Jaejun Lee, Yaoliang Yu, Jimmy Lin. ACL 2020. [arXiv:2004.12993](https://arxiv.org/abs/2004.12993)。相关度 3。两阶段训练。原文："otherwise, transformer layers are no longer optimized solely for the last off-ramp, generally worsening its quality"。
- **The Right Tool for the Job** — Roy Schwartz, Gabriel Stanovsky, Swabha Swayamdipta et al. ACL 2020. [arXiv:2004.07453](https://arxiv.org/abs/2004.07453)。相关度 3。在第 0/4/12/23 层加分类器，等权求和联合微调；末层有"smaller drop"，没给数字。
- **LeeBERT** — Wei Zhu. ACL-IJCNLP 2021. <https://aclanthology.org/2021.acl-long.231/>。相关度 3。逐层权重可学习（双层优化）并加互蒸馏；1.96× 加速下 MNLI 85.4，高于 full ALBERT 的 84.6。
- **Boosted Dynamic Neural Networks** — Haichao Yu, Haoxiang Li, Gang Hua, Gao Huang, Humphrey Shi. AAAI 2023. [arXiv:2211.16726](https://arxiv.org/abs/2211.16726)。相关度 3。回传到第 n 块的梯度按 1/(N−n+1) 取平均而不是求和，防止浅层梯度过大；IN 末出口 75.08 vs 74.69。
- **LEAP** — Shashank Kapadia, Deep Naryan Mishra, Sujal Reddy Alugubelli et al. ACL 2026 Industry. [arXiv:2605.01058](https://arxiv.org/abs/2605.01058)。相关度 3。把每个编码层推向末层表示（余弦 hinge）后，末层 STS-B 0.777→0.760（−2.2%），BEIR 5 项中 3 项更好。
- **FastBERT** — Weijie Liu, Peng Zhou, Zhe Zhao et al. ACL 2020. [arXiv:2004.02178](https://arxiv.org/abs/2004.02178)。相关度 2。主干冻结，逐层 student 自蒸馏。
- **CATs** — Tal Schuster, Adam Fisch, Tommi Jaakkola, Regina Barzilay. EMNLP 2021. [arXiv:2104.08803](https://arxiv.org/abs/2104.08803)。相关度 2。原文："We fix F rather than train it jointly ... to avoid any reduction in F's performance"。
- **RomeBERT** — Shijie Geng, Peng Gao, Zuohui Fu, Yongfeng Zhang. arXiv 2021. [arXiv:2101.09755](https://arxiv.org/abs/2101.09755)。相关度 2。自蒸馏梯度与微调梯度冲突（夹角 >90°）时做投影。
- **Early Exiting with Ensemble Internal Classifiers** — Tianxiang Sun, Yunhua Zhou, Xiangyang Liu et al. arXiv 2021. [arXiv:2105.13792](https://arxiv.org/abs/2105.13792)。相关度 2。
- **Zero Time Waste** — Maciej Wołczyk, Bartosz Wójcik, Klaudia Bałazy et al. NeurIPS 2021. [arXiv:2106.05409](https://arxiv.org/abs/2106.05409)。相关度 2。主干冻结。
- **Multi-Exit Vision Transformer** — Arian Bakhtiarnia, Qi Zhang, Alexandros Iosifidis. BMVC 2021. [arXiv:2106.15183](https://arxiv.org/abs/2106.15183)。相关度 2。端到端训练时末出口权重是其它出口的 2 倍。
- **LGViT** — Guanyu Xu, Jiawei Hao, Li Shen et al. ACM MM 2023. [arXiv:2308.00255](https://arxiv.org/abs/2308.00255)。相关度 2。两阶段，第二阶段冻结主干。
- **Meta-GF** — Yi Sun, Jian Li, Xin Xu. ECCV 2022（[ECVA](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/6337_ECCV_2022_paper.php)）。相关度 2。元学习融合各出口的梯度；原文说"exits usually interfere with each other"。
- **HDKD** — Xinglu Wang, Yingming Li. AAAI 2021（[OJS](https://ojs.aaai.org/index.php/AAAI/article/view/17225)）。相关度 2。稠密蒸馏 + 双层优化权重。
- **Deep Feature Surgery** — Cheng Gong, Yao Chen, Qiuyang Luo et al. ECCV 2024. [arXiv:2407.13986](https://arxiv.org/abs/2407.13986)。相关度 2。把特征分区以消除出口间相反的梯度方向。
- **Why should we add early exits to neural networks?** — Simone Scardapane, Michele Scarpiniti, Enzo Baccarelli, Aurelio Uncini. Cognitive Computation 2020. [arXiv:2004.12814](https://arxiv.org/abs/2004.12814)。相关度 2。原文："it is customary to weight earlier classifiers less"。
- **Dynamic Neural Networks: A Survey** — Yizeng Han, Gao Huang, Shiji Song et al. TPAMI 2022. [arXiv:2102.04906](https://arxiv.org/abs/2102.04906)。相关度 1。

### F7 对齐 / 蒸馏式中间层监督（16 条）

**与本实验的区别**：
- 这里的中间层目标不是同一个标签，而是外部表示（DINOv2、教师层）或自身其它层的表示，一般还要经过独立投影头。约束更"软"，也不强迫中间层可被 LM head 解码。
- 但它们提供了关于**时间 schedule** 最有力的定量证据：
  - HASTE：中间层监督早期有益，中期与主梯度正交，后期与主梯度冲突，适时停掉优于常开。
  - HLD-LLM：在 LM 预训练里，前 1–5% 做 hint 后移除，优于常开。
  - FitNets / TinyBERT / MobileBERT：先中间后移除。

- **HASTE（REPA Works Until It Doesn't）** — Ziqiao Wang, Wangbo Zhao, Yuhao Zhou et al. NeurIPS 2025. [arXiv:2505.16792](https://arxiv.org/abs/2505.16792)。相关度 4。
  - **做法**：在第 8 层做 REPA 特征对齐，在第 4–7 层做注意力对齐，λ=0.5；到 τ 处**硬停**，之后只用去噪 loss。
  - **梯度余弦的三个阶段**：0–200K "ignition" 阶段较高；200–400K 近正交；>400K 为负。
  - **SiT-XL/2 数字**（Table 4，已对照原文）：
    - 500K 时：不停 8.1；τ=250K 5.3；τ=400K 7.4。
    - 400K 时：不停 5.5；τ=250K 7.3。也就是停止后会先暂时变差，之后反超。
  - **其它规模**：SiT-B/2 在 400K 时 21.3→19.6（τ=100K）；SiT-L/2 8.9→7.9（τ=250K）。
  - **Table 3**：100 epoch 时 REPA-only 7.5；holistic 常开 8.1；holistic + 终止 5.3。
  - 停在 100K 过早，反而有害。
  - 只测了硬停，没有测线性退火。
- **HLD-LLM（A Study on Hidden Layer Distillation for LLM Pre-Training）** — Maxime Guigon, Lucas Dixon, Michaël E. Sander. arXiv 2026. [arXiv:2605.11513](https://arxiv.org/abs/2605.11513)。相关度 4。
  - 教师 Gemma3 3.4B，学生 123M/735M，在 C4 上**从零**训练（最多 168B token），把学生中间层对齐到教师中间层（normalized MSE）。
  - HLDF：前 1–5% 预算做 hint，然后移除，只做 KD。C4 指标 3.005→3.000（123M），2.609→2.607（735M）；原文标注为 perplexity，但数值像 log-ppl。
  - HLDC（常开）：没有增益。
  - 下游"no method dominates KD"。
- **DeepFlow** — Inkyu Shin, Chenglin Yang, Liang-Chieh Chen. ICCV 2025. [arXiv:2503.14494](https://arxiv.org/abs/2503.14494)。相关度 4。各分支用**同一个** velocity 目标，中间权重 β=0.2、末层 1.0；只加 DS 时 SiT-B/2 FID 34.4→33.0。
- **REPA** — Sihyun Yu, Sangkyung Kwak, Huiwon Jang et al. ICLR 2025. [arXiv:2410.06940](https://arxiv.org/abs/2410.06940)。相关度 3。
  - SiT-XL/2 第 8/28 层接 MLP 投影，对齐 DINOv2，λ=0.5 常开，训练快 >17.5×。
  - 对齐深度消融：第 6/8/10/12 层 FID 分别为 10.3/10.0/10.5/11.2。
  - 原文："limiting regularization to the first few layers further enhances generation performance"。
- **NITP** — Xiangdong Zhang, Debing Zhang, Shaofeng Zhang et al. ICML 2026. [arXiv:2605.24956](https://arxiv.org/abs/2605.24956)。相关度 3。
  - **监督的是末层**：末层隐状态经 MLP 投影 P 后，预测下一个 token 在固定浅层（约 20% 深度）的 stop-gradient 表示，loss 为 1−cos；λ=1.0（9B MoE 与 3B dense 为 0.8），常开。
  - 结果：9B MoE MMLU-Pro 15.29→21.00；dense 3B 平均 37.18→38.53。
  - 目标层消融：浅层 L4 最好，23.58，高于 L14 的 22.16 和 L8 的 21.22。
  - 动机："bottom-up convergence pattern"。
- **OISD** — Xinyu Liu, Darryl Cherian Jacob, Yang Zhou et al. Findings of EMNLP 2026. [arXiv:2605.29089](https://arxiv.org/abs/2605.29089)。相关度 3。
  - RL 后训练时，对一个中间层（Qwen3 第 6 层）做 logit lens：p^l=softmax(LN(h^l)E_u^T)，即**共享 final LN + unembedding**；与 detached 末层分布做 JSD，按 GRPO advantage 加权。
  - Qwen3-4B Avg@K 55.08→64.75。
- **DistillLens** — Manish Dhakal, Uthman Jinadu, Anjila Budathoki et al. arXiv 2026. [arXiv:2602.13567](https://arxiv.org/abs/2602.13567)。相关度 3。
  - 学生与教师在部分层上的 logit-lens 分布做 JSD，λ=1，在 SFT 蒸馏阶段使用。
  - ROUGE-L：GPT-2-120M 17.20→21.12；GPT-2-340M 20.42→23.72。
- **FitNets** — Adriana Romero, Nicolas Ballas, Samira Ebrahimi Kahou et al. ICLR 2015. [arXiv:1412.6550](https://arxiv.org/abs/1412.6550)。相关度 3。先用 hint 预训练到中间层，再对整网做 KD，hint loss 就此移除。原本无法训练的深窄网络因此可训。
- **MobileBERT** — Zhiqing Sun, Hongkun Yu, Xiaodan Song et al. ACL 2020. [arXiv:2004.02984](https://arxiv.org/abs/2004.02984)。相关度 3。逐层蒸馏的三种安排，MNLI-m 上：auxiliary（常开）83.0 < joint 83.5 < progressive（逐层后冻结）83.9。
- **SRA** — Dengyang Jiang, Mengmeng Wang, Liuzhuozheng Li et al. ICLR 2026. [arXiv:2505.02831](https://arxiv.org/abs/2505.02831)。相关度 2。早层对齐自身 EMA 的晚层（如 3→8），SiT-B/2 33.02→29.10；同深度对齐（3→3）反而变差，为 37.08。
- **REPR-ALIGN（Don't Retrain, Align）** — Fred Zhangzhi Peng, Alexis Fox, Anru R. Zhang et al. arXiv 2026. [arXiv:2605.06885](https://arxiv.org/abs/2605.06885)。相关度 2。在 AR→扩散 LM 转换中逐层对齐同架构 AR LM；全层 pass@1 18.00，高于任何 1/3 子集（中间 1/3 为 15.73）。
- **TinyBERT** — Xiaoqi Jiao, Yichun Yin, Lifeng Shang et al. Findings of EMNLP 2020. [arXiv:1909.10351](https://arxiv.org/abs/1909.10351)。相关度 2。先做中间层蒸馏 20 epoch，再做预测层蒸馏 3 epoch。
- **MiniLM** — Wenhui Wang, Furu Wei, Li Dong et al. NeurIPS 2020. [arXiv:2002.10957](https://arxiv.org/abs/2002.10957)。相关度 2。只蒸馏末层（80.0），好于逐层蒸馏（79.0）。
- **MiniLMv2** — Wenhui Wang, Hangbo Bao, Shaohan Huang, Li Dong, Furu Wei. Findings of ACL 2021. [arXiv:2012.15828](https://arxiv.org/abs/2012.15828)。相关度 2。
- **Patient KD** — Siqi Sun, Yu Cheng, Zhe Gan, Jingjing Liu. EMNLP 2019. [arXiv:1908.09355](https://arxiv.org/abs/1908.09355)。相关度 2。
- **iREPA（What matters for Representation Alignment）** — Jaskirat Singh, Xingjian Leng, Zongze Wu et al. arXiv 2025. [arXiv:2512.10794](https://arxiv.org/abs/2512.10794)。相关度 1。

### F1 经典 deep supervision：CNN / ViT / MIM（21 条）

**要点**：
- 同标签 DS 在 CNN 上的增益普遍很小（ImageNet 上约 +0.1~0.9），有时为负。
- MSDNet 与 InfoPro 给出了机制："把浅层特征推向短期可分类，会塌缩后层需要的信息"。ResNet 式加性残差受影响最大，DenseNet 较轻。decoder LM 的残差流是加性的，属于前者。
- 退火先例：DSN 的 γ-hinge 自关闭与 (1−t/N) 衰减；CNDS 的 0.3·(1−t/N)。
- 反例：Inception-v3 发现 aux head 对早期收敛**没有**帮助，只在后期起正则作用，这与"早期有用"的故事相反。
- 浅层用更软或更易的目标普遍更好：DTS、DeepMIM hybrid、DISCO、BYOT/DKS 的自蒸馏。

- **DeepMIM** — Sucheng Ren, Fangyun Wei, Samuel Albanie, Zheng Zhang, Han Hu. WACV 2025. [arXiv:2303.08817](https://arxiv.org/abs/2303.08817)。相关度 4。
  - ViT-B 第 6/8/10 块各接独立解码器，做同样的像素重建；浅层用更易的 hybrid 目标（α=0,1/3,2/3）。
  - 300 epoch 时 +0.8（hybrid 目标 +1.0）；1600 epoch 时 +0.6。增益随训练变长而缩小。
- **DSN** — Chen-Yu Lee, Saining Xie, Patrick Gallagher, Zhengyou Zhang, Zhuowen Tu. AISTATS 2015. [arXiv:1409.5185](https://arxiv.org/abs/1409.5185)。相关度 3。
  - companion loss 带 γ 阈值：某层的辅助 loss 低于 γ 后梯度为 0，即自动关闭。
  - 可选让 α_m 按 α_m·0.1·(1−t/N) 衰减，使辅助项在若干轮后消失。
  - C100 err 35.68→34.57。
- **CNDS** — Liwei Wang, Chen-Yu Lee, Zhuowen Tu, Svetlana Lazebnik. arXiv 2015. [arXiv:1505.02496](https://arxiv.org/abs/1505.02496)。相关度 3。只在梯度消失处加分支；α 从 0.3 按 α←α(1−t/N) 衰减；IN top-1 err 34.7→33.8，训练 5 天 vs 6 天。
- **Inception-v3** — Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna. CVPR 2016. [arXiv:1512.00567](https://arxiv.org/abs/1512.00567)。相关度 3。原文："did not result in improved convergence early in the training ... Near the end of training, the network with the auxiliary branches starts to overtake"；删掉下部 aux head 对最终质量没有影响。
- **MSDNet** — Gao Huang, Danlu Chen, Tianhong Li et al. ICLR 2018. [arXiv:1703.09844](https://arxiv.org/abs/1703.09844)。相关度 3。
  - 原文："the introduction of an intermediate classifier harms the final ResNet classifier ..., reducing its accuracy by up to 7%"（C100）。
  - 机制："collapses information required to generate high quality features in later layers"。DenseNet 受影响小得多。
- **SDN** — Yigitcan Kaya, Sanghyun Hong, Tudor Dumitras. ICML 2019. [arXiv:1810.07052](https://arxiv.org/abs/1810.07052)。相关度 3。权重 ∝ 深度占比（0.15…0.9，末层 1），从 0.01 线性 warm-up，方向与 LLAL 的衰减相反。
- **AdaLoss（Anytime prediction）** — Hanzhang Hu, Debadeepta Dey, Martial Hebert, J. Andrew Bagnell. AAAI 2019. [arXiv:1708.06832](https://arxiv.org/abs/1708.06832)。相关度 3。
  - 等权常数下，anytime 网络比单出口网络多 15–18% 相对误差，且越深差距越大。
  - 改为按 1/running-loss 自适应加权，等价于最小化 Σ log L_i。浅层 LM loss 远大于深层时，这种归一值得借鉴。
- **BYOT** — Linfeng Zhang, Jiebo Song, Anni Gao et al. ICCV 2019. [arXiv:1905.08094](https://arxiv.org/abs/1905.08094)。相关度 3。浅层用"CE + KL 到最深 + hint"；IN R50 73.56→75.24，好于纯 DSN。
- **IMTA** — Hao Li, Hong Zhang, Xiaojuan Qi, Ruigang Yang, Gao Huang. ICCV 2019. [arXiv:1908.06294](https://arxiv.org/abs/1908.06294)。相关度 3。gradient equilibrium 让回传梯度的方差不随出口数增长；末出口 C100 +1.64，IN +1.09。
- **Contrastive Deep Supervision** — Linfeng Zhang, Xin Chen, Junbo Zhang, Runpei Dong, Kaisheng Ma. ECCV 2022. [arXiv:2207.05306](https://arxiv.org/abs/2207.05306)。相关度 3。
  - 原文："this conflict sometimes leads to accuracy degradation in the final classifier"。
  - 同标签 DS 在 IN 上只有 +0.07~0.33；改用对比目标后 R50 75.30→78.25。
- **SDViT** — Maryam Sultana, Muzammal Naseer, Muhammad Haris Khan et al. ACCV 2022. [arXiv:2207.12392](https://arxiv.org/abs/2207.12392)。相关度 3。
  - 每步只随机选一个中间块，经**共享的末层 head** 对末层做 KL 自蒸馏。原文说一次蒸馏所有块"poses optimization difficulties"。
  - PACS 84.9→86.3。
- **DTS（Deep Trajectory Supervision: Deep Supervision Strikes Back）** — Han Wang, Weijie Wang, Jiaqi Liu, Hilde Kuehne, Nicu Sebe. ICML 2026，<https://icml.cc/virtual/2026/poster/64479>，OpenReview N9kBjRH0Cx；未找到 arXiv 版本。相关度 3。
  - 用 Tuned Lens 观察到证据的积累随深度呈指数；据此给中间层"置信度随深度递增"的软目标。
  - 只核实了摘要，数字未取得。
- **GoogLeNet** — Christian Szegedy, Wei Liu, Yangqing Jia et al. CVPR 2015. [arXiv:1409.4842](https://arxiv.org/abs/1409.4842)。相关度 3。aux loss 权重 0.3，推理时丢弃；论文中没有单独的效果数字。
- **HED** — Saining Xie, Zhuowen Tu. ICCV 2015. [arXiv:1504.06375](https://arxiv.org/abs/1504.06375)。相关度 2。
- **PSPNet** — Hengshuang Zhao, Jianping Shi, Xiaojuan Qi, Xiaogang Wang, Jiaya Jia. CVPR 2017. [arXiv:1612.01105](https://arxiv.org/abs/1612.01105)。相关度 2。α 在 0.3–0.9 之间都有益，0.4 最好，结果对权重不敏感。
- **BranchyNet** — Surat Teerapittayanon, Bradley McDanel, H. T. Kung. ICPR 2016. [arXiv:1709.01686](https://arxiv.org/abs/1709.01686)。相关度 2。
- **DKS** — Dawei Sun, Anbang Yao, Aojun Zhou, Hao Zhao. CVPR 2019. [arXiv:1906.00675](https://arxiv.org/abs/1906.00675)。相关度 2。
- **DISCO（Deep Supervision with Intermediate Concepts）** — Chi Li, M. Zeeshan Zia, Quoc-Huy Tran et al. TPAMI 2019. [arXiv:1801.03399](https://arxiv.org/abs/1801.03399)。相关度 2。
- **LION-DG** — Hyunjun Kim. arXiv 2026. [arXiv:2601.02105](https://arxiv.org/abs/2601.02105)。相关度 2。
  - aux head 零初始化，使辅助梯度在初始时刻为 0，之后逐步"苏醒"。
  - 提示：本实验用的是共享且逐渐训练好的 head，没有这种隐式 warm-up，辅助梯度从第 0 步起就是全强度。
- **UNet++** — Zongwei Zhou, Md Mahfuzur Rahman Siddiquee, Nima Tajbakhsh, Jianming Liang. DLMIA 2018. [arXiv:1807.10165](https://arxiv.org/abs/1807.10165)。相关度 1。
- **A Comprehensive Review on Deep Supervision** — Renjie Li, Xinyi Wang, Guan Huang et al. arXiv 2022. [arXiv:2207.02376](https://arxiv.org/abs/2207.02376)。相关度 1。

### F1b 迭代精修架构的逐阶段 / 逐层 loss（新增家族，6 条）

- **DETR** — Nicolas Carion, Francisco Massa, Gabriel Synnaeve et al. ECCV 2020. [arXiv:2005.12872](https://arxiv.org/abs/2005.12872)。相关度 4。
  - 原文："All predictions FFNs share their parameters. We use an additional shared layer-norm to normalize the input to the prediction FFNs from different decoder layers"。
  - 官方代码中，每个中间输出都经过 decoder 的 final norm，辅助 loss 权重与末层相同，全程常开，整个 DETR 家族沿用至今。
  - 这是"共享 final norm + 共享 head、每层等权"在 Transformer 中最典型的先例。
- **RAFT** — Zachary Teed, Jia Deng. ECCV 2020. [arXiv:2003.12039](https://arxiv.org/abs/2003.12039)。相关度 3。L=Σ γ^(N−i)·‖f_gt−f_i‖，γ=0.8，12 次迭代中第一次的权重约 0.086。这是一个被广泛使用的"指数偏向后层"的权重曲线。
- **Deformable DETR** — Xizhou Zhu, Weijie Su, Lewei Lu et al. ICLR 2021. [arXiv:2010.04159](https://arxiv.org/abs/2010.04159)。相关度 3。做迭代框精修时，各层 head **不共享**，并截断梯度。这与 DETR 形成对照：当各层是在精修上一层的输出时，共享 head 并不合适。
- **Convolutional Pose Machines** — Shih-En Wei, Varun Ramakrishna, Takeo Kanade, Yaser Sheikh. CVPR 2016. [arXiv:1602.00134](https://arxiv.org/abs/1602.00134)。相关度 3。每个 stage 等权监督，理由是缓解梯度消失。
- **Stacked Hourglass** — Alejandro Newell, Kaiyu Yang, Jia Deng. ECCV 2016. [arXiv:1603.06937](https://arxiv.org/abs/1603.06937)。相关度 2。
- **Mask2Former** — Bowen Cheng, Ishan Misra, Alexander G. Schwing et al. CVPR 2022. [arXiv:2112.01527](https://arxiv.org/abs/2112.01527)。相关度 2。

### F8 Local / greedy 逐层学习（10 条）

- **SOLO** — Bojian Yin, Shurong Wang, Yuqi Pan et al. *SOLO: Pretraining Billion-Parameter Language Models with Shared-Output Local Learning.* arXiv 2026（2026-09-28 提交）。[arXiv:2609.35440](https://arxiv.org/abs/2609.35440)。相关度 4。
  - **做法**：340M–2B decoder LM，15B token。把 24 层分成 K=2 或 4 个梯度隔离模块，每个模块输出都加 next-token loss。
  - **读出**：每个辅助头都经过"a shared, read-only copy of the final module's readout"，即上一步末层 unembedding 的拷贝，只有末层 loss 会更新它。每个头另有 2 个 Transformer 块、自己的 RMSNorm 和温度，所以 final norm **不共享**。
  - **读出消融**（40M，WikiText ppl）：随机读出 98.88，私有读出 75.34，共享读出 70.62，BP 67.08。
  - **与 BP 的 ppl 差距**：340M K=2 为 29.05 vs 28.04，K=4 为 31.19；2B K=2 为 21.57 vs 20.93。
  - BP 模型 logit-lens 与末层预测的一致率只有 16–51%，SOLO 的出口为 71–75%。
  - **注意**：这是没有全局梯度的 local learning，权重常数，没有 schedule。
- **InfoPro** — Yulin Wang, Zanlin Ni, Shiji Song, Le Yang, Gao Huang. ICLR 2021. [arXiv:2101.10832](https://arxiv.org/abs/2101.10832)。相关度 3。
  - 同标签 greedy 局部 loss 使 ResNet-32 C10 err 从 7.37 升到 10.30（K=2），再到 24.59（K=16）。
  - 原因是它"collapse[s] both I(h,x) and I(h,y) in their first few modules"。这是 MSDNet 那类伤害的机制性量化。
- **Greedy Layerwise Learning Can Scale to ImageNet** — Eugene Belilovsky, Michael Eickenberg, Edouard Oyallon. ICML 2019. [arXiv:1812.11446](https://arxiv.org/abs/1812.11446)。相关度 2。
- **Decoupled Greedy Learning of CNNs** — Eugene Belilovsky, Michael Eickenberg, Edouard Oyallon. ICML 2020. [arXiv:1901.08164](https://arxiv.org/abs/1901.08164)。相关度 2。
- **Training Neural Networks with Local Error Signals** — Arild Nøkland, Lars Hiller Eidnes. ICML 2019. [arXiv:1901.06656](https://arxiv.org/abs/1901.06656)。相关度 2。纯同标签 CE（pred）不如加上相似性目标（predsim）。
- **PGL** — Hasnain Irshad Bhatti, Jaekyun Moon. ICML 2022 HAET workshop. [arXiv:2208.00821](https://arxiv.org/abs/2208.00821)。相关度 2。周期性恢复全局目标。
- **AugLocal** — Chenxiang Ma, Jibin Wu, Chenyang Si, Kay Chen Tan. ICLR 2024. [arXiv:2402.17318](https://arxiv.org/abs/2402.17318)。相关度 2。辅助网络越像"网络的剩余部分"，局部学习越接近 BP。从这个角度看，共享 final norm + LM head 只是"剩余网络"的最后一小段。
- **LoPT** — Hengyu Shi, Tianyang Han, Peizhe Wang et al. arXiv 2026. [arXiv:2605.04913](https://arxiv.org/abs/2605.04913)。相关度 2。
- **Greedy InfoMax** — Sindy Löwe, Peter O'Connor, Bastiaan S. Veeling. NeurIPS 2019. [arXiv:1905.11786](https://arxiv.org/abs/1905.11786)。相关度 1。
- **Greedy Layer-Wise Training of Deep Networks** — Yoshua Bengio, Pascal Lamblin, Dan Popovici, Hugo Larochelle. NIPS 2006（[proceedings](https://proceedings.neurips.cc/paper/2006/hash/5da713a690c067105aeb2fae32403405-Abstract.html)）。相关度 1。

### F10 背景与对照（17 条；计入总数，但不算核心）

- **Don't Drop Dropout** — Mostafa Elhoushi, Alex Pretko, Nolan Dey et al. ICML 2026. [arXiv:2609.05275](https://arxiv.org/abs/2609.05275)。相关度 3。
  - 271M–8.2B 模型，共 2400+ 次实验。
  - 原文 Finding 4："a schedule decaying from a maximum rate to zero consistently achieves the highest accuracy"。layer dropout 随层递增、随时间递减，同 FLOPs 下 val loss 更低，early-exit loss 也更好。
  - 这是另一种"只在早期施加的深度干预"，适合作为本实验的对照或组合对象；LayerSkip 就是把两者合用。
- **ProRes** — Tianhao Chen, Xin Xu, Lu Yin et al. arXiv 2026. [arXiv:2603.05369](https://arxiv.org/abs/2603.05369)。相关度 3。每层残差乘一个从 0 warm-up 到 1 的系数，越深的层 warm-up 越久，也就是"early layer learns first"，另一种早期干预对照。
- **Logit lens** — nostalgebraist, *interpreting GPT: the logit lens*, LessWrong 2020-08-31，<https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens>。相关度 2。
- **Tuned lens** — Nora Belrose, Igor Ostrovsky, Lev McKinney et al. arXiv 2023. [arXiv:2303.08112](https://arxiv.org/abs/2303.08112)。相关度 2。每层一个仿射 translator，比 logit lens 更可靠。这提示了一个折中方案：给每层加一个可学习的仿射或 gain，减轻共享 head 带来的约束。
- **The Curse of Depth in LLMs** — Wenfang Sun, Xinyuan Song, Pengxiang Li et al. NeurIPS 2025. [arXiv:2502.05795](https://arxiv.org/abs/2502.05795)。相关度 2。Pre-LN 使深层接近恒等映射；用 LayerNorm Scaling 修正，130M–7B 都有效。
- **Do Language Models Use Their Depth Efficiently?** — Róbert Csordás, Christopher D. Manning, Christopher Potts. NeurIPS 2025. [arXiv:2505.13898](https://arxiv.org/abs/2505.13898)。相关度 2。Llama 3.1 / Qwen 3 / OLMo 2 的后半层贡献小得多；可以作为本实验训练后"深度利用"的评估工具。
- **Tending Towards Stability** — Richard Diehl Martinez, Pietro Lesci, Paula Buttery. Findings of EMNLP 2024. [arXiv:2410.11451](https://arxiv.org/abs/2410.11451)。相关度 2。Pythia 大模型的层在前 20% 训练内稳定，小模型的层收敛慢且不稳。
- **Learning Less Is More** — Jinchang Zhu, Jindong Li, Yuwen Hao et al. arXiv 2026. [arXiv:2605.10504](https://arxiv.org/abs/2605.10504)。相关度 2。
  - GPT 式模型中，上层注意力在下层特征稳定之前就过早特化；早期临时减慢上层 Q/K 可以改善 PPL。
  - 在带 gated FFN 的 LLaMA 式架构中几乎不需要这一干预。Qwen3 用 SwiGLU，属于这一类。
- **JREG（Suppressing Final Layer Hidden State Jumps）** — Keigo Shibata, Kazuki Yano, Ryosuke Takahashi et al. Findings of EACL 2026. [arXiv:2601.18302](https://arxiv.org/abs/2601.18302)。相关度 2。预训练时惩罚末层 hidden state 的突变，让中间层的能力使用更均衡，任务表现提升。
- **The Diminishing Returns of Early-Exit Decoding in Modern LLMs** — Rui Wei, Rui Du, Hanfei Yu et al. arXiv 2026. [arXiv:2603.23701](https://arxiv.org/abs/2603.23701)。相关度 2。
  - 新一代模型的早退潜力下降；dense 高于 MoE，MoE 高于 SSM。
  - Pythia 训练后期中层与末层的相似度出现低谷，说明关键决策集中到了末几层。
- **Bigram Subnetworks** — Tyler A. Chang, Benjamin K. Bergen. NeurIPS 2025. [arXiv:2504.15471](https://arxiv.org/abs/2504.15471)。相关度 2。原文："Bigram subnetworks are concentrated in the first Transformer MLP layer"，参数占比 <0.2%。LLAL 引用了它。
- **Layer by Layer** — Oscar Skean, Md Rifat Arefin, Dan Zhao et al. ICML 2025. [arXiv:2502.02013](https://arxiv.org/abs/2502.02013)。相关度 1。
- **ShortGPT** — Xin Men, Mingyu Xu, Qingyu Zhang et al. arXiv 2024. [arXiv:2403.03853](https://arxiv.org/abs/2403.03853)。相关度 1。
- **Neurons in LLMs: Dead, N-gram, Positional** — Elena Voita, Javier Ferrando, Christoforos Nalmpantis. arXiv 2023. [arXiv:2309.04827](https://arxiv.org/abs/2309.04827)。相关度 1。OPT-66B 某些早层中 >70% 的神经元是 dead 的；存在 n-gram 检测器。LLAL 引用了它。
- **Draft & Verify** — Jun Zhang, Jue Wang, Huan Li et al. ACL 2024. [arXiv:2309.08168](https://arxiv.org/abs/2309.08168)。相关度 1。
- **SkipDecode** — Luciano Del Corro, Allie Del Giorno, Sahaj Agarwal et al. arXiv 2023. [arXiv:2307.02628](https://arxiv.org/abs/2307.02628)。相关度 1。
- **Kangaroo** — Fangcheng Liu, Yehui Tang, Zhenhua Liu et al. arXiv 2024. [arXiv:2404.18911](https://arxiv.org/abs/2404.18911)。相关度 1。

---

## 3. decoder LM 中逐层 LM loss 对末层的正反证据（含数字）

本节只列与"自回归 / decoder、next-token 或同标签、经输出层"相近的证据，按与本实验的接近程度排序。**加粗**表示从零训练的 decoder LM。

### 3.1 中性到正面

| 证据 | 设置 | 做法 | 末层结果 | 局限 |
|---|---|---|---|---|
| **Al-Rfou 2019** | **64 层字符 LM，从零，text8** | 每层独立头；第 l 层在训练进度 l/(2n) 处移除 | dev bpc 1.158→1.062（−0.096） | 2018 年的深层难训环境；头不共享 |
| **EE-LLM 2024** | **GPT 1.3B/7B，从零 300B/150B token** | 2 个出口（1/4、1/2 深度），权重 0.25/0.5 或 0.1/0.2，常开 | 末层 loss 与标准模型持平或略低 | 只有 2 个出口；只给了曲线 |
| **HLD-LLM 2026** | **LM 123M/735M，从零，C4，蒸馏设定** | 前 1–5% 预算做中间层 hint，然后移除 | 相对纯 KD：C4 3.005→3.000、2.609→2.607；常开版本无增益 | 目标是教师中间层，不是 LM loss |
| LLAL 2026 | MoE 60B/180B，从零 | 第一个 MoE 层，λ 0.1→0，2k–40k 步 | val −0.024；MMLU +2.8~3.9；MMLU-Pro +4.0~4.8 | MoE 特有问题；单层 |
| Multi-Layer Softmax 2019 | AR NMT 6-6，从零 | 所有 decoder 层，共享 softmax，等权 | 全深度 BLEU 34.87 = 34.87 | 无益，但也无损 |
| Depth-Adaptive T. 2020 | 6 层 decoder MT | 每层独立头，等权 | 36.2 vs 35.9（ω=n 时 36.3） | 小模型、小数据 |
| DSLP 2022 | NAT decoder，从零 | 每层共享投影，等权 | +0.66 BLEU | 非自回归 |
| DeCRED 2025 | ASR 的 AR decoder | 中后 1 层，next-token，β=0.4 | OOD WER 18.2→16.2 | 独立头；encoder-decoder |
| InterCTC / SC-CTC 2021 | ASR encoder，从零 | 中间层，共享 final LN + head，w=0.3/0.5 | WSJ 16.5→13.6 等 | encoder、CTC |
| Wright 2024 | Conformer，从零 | 隔层等权 | 6.5/17.7→5.1/15.1 | 事后在预训练模型上加则有害 |
| SOLO 2026 | **LM 340M–2B** | 模块级 next-token，共享只读 unembedding | 与 BP 差 +0.64~3.15 ppl，但**没有全局梯度** | local learning；只能作上界参考 |
| NITP 2026 | **LM 0.5B–9B，从零** | 末层预测浅层表示（不是中间层 loss） | dense 3B 平均 +1.35；9B MoE MMLU-Pro +5.7 | 监督位置不同 |

### 3.2 伤害末层

| 证据 | 设置 | 做法 | 末层结果 | 局限 |
|---|---|---|---|---|
| **LayerSkip 2024** | **Llama 1.5B/7B 从零 26B token；Llama2/3 续训** | 所有层、共享 final LN + head、二次深度权重、轮转 | 原文：全层全步会降末层；续训 Llama2-7B MMLU 46.0→43.1，Llama3-8B GSM8K 54.2→45.0、HumanEval 37.8→28.7；从零时部分任务略降 | 续训的 baseline 是原始 ckpt；同时有 layer dropout |
| RRT 2025 | 递归 Gemma（2 loop），uptrain | 中间循环 LM loss | co-train −1.5 avg（PPL 12.85→13.24）；i/Σi 后训练 −1.2 | 权重共享，冲突被放大 |
| DEED 2024 | enc-dec VL，预训练 + 微调 | L_avg 等权 | 原文：末层 decoder 退化，需加 L_N | 数字只在图里 |
| FREE 2023 | T5 微调 | CALM 式全层 | 与只训 2 个出口相比，全模型 Multi-News 37.62 vs 39.20，SQuAD 90.63 vs 91.90 | 两边的 KD 设置也不同 |
| BitSkip 2025 | 85M，WikiText-2 | 1/(l+1) 偏浅层，共享 head | val PPL 228.77→252.37 | 与层跳混杂；数据极小 |
| Early-exit natural 2024 | LLaMA 分析 | joint optimization | 定性："negative impact on ... full model" | 无数字 |
| Ouro App. A 2025 | LoopLM 776M，20B token | 偏早步的几何先验 | 训练 loss 平台更高 | 只有图 |
| Kubaty 2025 | ViT/ResNet/BERT | joint 从零 | ViT-T IN-1k −3.2；BERT 20NG 约 −1.4 | 非 LM；独立头 |
| CALM 2022 | T5 微调 | ω_i=i/Σj | "mostly preserve"，暗示有轻微损失 | 无数字 |
| MSDNet / InfoPro / AdaLoss / CDS | CNN | 同标签 DS / 局部 | 最多 −7%；局部 K=16 时 err 24.59；+15–18% 相对误差；有时降末层 | CNN |
| Wright 2024（事后） | wav2vec2 / WavLM | 在预训练模型上加出口再微调 | 3.4/8.6→4.3/12.2 | 语音 |
| HASTE（常开对照） | SiT-XL/2 | 常开对齐到 500K | Table 4：400K 时 FID 5.5，500K 时 8.1（冲突期） | 对齐目标 |

### 3.3 规律：什么时候有益，什么时候有害

1. **时间**：
   - 早期或分阶段施加、然后移除，结果为正或中性：Al-Rfou、LLAL、HASTE、HLDF、FitNets、MobileBERT、DSN/CNDS 的衰减。
   - 全程常开在大数据或长训练下多为负或零：LayerSkip 原文、RRT co-train、Kubaty joint、DEED、HASTE 常开、HLDC。
   - 两个例外：模型本身难以优化时，常开也有益（深层语音 Transformer 上的 InterCTC、iterated loss，都是常开 0.3）；DETR 类检测模型常开也有益。
   - 时机要合适：HASTE 显示停得太早（100K）会损失增益，而且停止后会先暂时变差。
2. **从零 vs 事后**：
   - 从零就加：中性或正面（Wright、EE-LLM、Multi-Layer Softmax）。
   - 在已收敛的模型上加：负面（Wright 的 wav2vec2；LayerSkip 在 Llama3 上续训降幅大于 Llama2，作者归因于 Llama3 浅层原本的 PPL 高出 2–3 个数量级）。
   - 与此一致，训练越久，中间层越"不可解码"：LayerSkip Fig.11 中层 PPL 升到数百；Pythia 中层相似度出现低谷；新一代模型早退潜力下降。
   - 所以"只在训练早期施加"刚好落在证据为正的区间。
3. **权重**：
   - 总量应归一。ESPnet 的约定是 (1−w)·L_final + w·mean_l L_l，w 取 0.3–0.5，不随被监督层数增长。
   - 层间分布应深度递增，或额外强调末层：LayerSkip、CALM、PABEE、SDN、RAFT、DEED 的 +L_N、MEViT 末层 ×2、BranchyNet。
   - 偏向浅层的配方有害：BitSkip、Ouro 几何先验、RRT 的 i/Σi（中间层过重）。
   - DAT 的消融给出了清楚的权衡：ω=n 对末层最好，但最浅层最差；ω=1/n 反过来；均匀权重整体最好。
4. **被监督的层数**：
   - 同标签监督的层越多，收益越快饱和：InterCTC 从 1 层到 3/7 层只再降约 0.4–0.5 WER；12 层模型上增加层数无增益；DeCRED 多头无额外收益。
   - 单个中后层往往就够了：DeCRED、ILO 的 9/12、REPA 的第 8 层、NITP 的约 20% 深度作目标。
   - 但 LLAL 显示接在 L5 救不到 L1；REPR-ALIGN 中全层对齐优于任何 1/3 子集。
   - 所以"每层都加"的价值在于**覆盖最底层**，而不在于层数本身。
5. **目标**：浅层用更软或更粗的目标更好。例子有 DTS、HC-CTC、DeepMIM hybrid、自蒸馏（BYOT、RRT KD、OISD），以及 Contrastive DS 用非任务目标。
6. **梯度记账**：
   - 第 i 层要接收 L−i+1 份辅助梯度。
   - 已有的补救：IMTA/ElasticBERT 的 gradient equilibrium；BoostNet 的 1/(N−n+1)；DAT 的 gradient scaling；RomeBERT 的冲突投影；ElasticBERT/LayerSkip/SDViT 每步只启用部分出口；BERxiT 奇偶步交替。

### 3.4 已报告的逐层权重与时间 schedule

| 方法 | 层间权重 | 时间 schedule | 备注 |
|---|---|---|---|
| Al-Rfou 2019 | 未给具体值（"lower layers weighted less and less as training progresses"） | 第 l 层在 l/(2n) 处移除，浅层先停，训练过半时全部移除 | 唯一"所有层 + 移除"的 LM 先例 |
| LLAL 2026 | 单层 λ=0.1 | 线性退火到 0，2k–40k 步（占总步数 1.5–44%） | 2k 与 10k 窗口效果相近 |
| LayerSkip 2024 | e(l)=e_scale·Σ_{i≤l} i，按深度二次增长；归一化 | rotational（每步 ⌈L/R⌉ 个出口）或 gradual（从深往浅每 T/2L 步启用一层） | 预训练 e_scale 0.2；微调 1.0 |
| CALM 2022 | ω_i = i/Σj | 常数 | |
| Depth-Adaptive T. 2020 | 均匀；ω=n 对末层最好；ω=1/n 对浅层最好 | 常数 | 可选 gradient scaling |
| EE-LLM 2024 | 0.25/0.5（1.3B），0.1/0.2（7B） | 常数（框架支持变权重） | |
| RRT 2025 | i/Σi 会过度强调中间层；aggressive 0.1 并对末层做 KD | 在后训练阶段加；co-train 有害 | |
| DEED 2024 | L_avg + L_N（末层等效为 1+1/N） | 常数 | |
| PABEE 2020 | Σ j·L_j / Σ j | 常数 | |
| SDN 2019 | τ_i ∝ 深度占比（0.15…0.9） | 从 0.01 线性 **warm-up** | 方向与 LLAL 相反 |
| RAFT 2020 | γ^(N−i)，γ=0.8 | 常数 | |
| InterCTC / SC-CTC | (1−w)·L_final + w·mean(中间层)，w=0.3/0.5 | 常数 | 共享 final LN + head |
| DeepFlow 2025 | 中间层 0.2，末层 1.0 | 常数 | |
| DeCRED 2025 | 中间层 β=0.4，末层 0.6 | 常数 | |
| Ouro 2025 | 学习 p(t\|x)，加熵正则，β 从 0.1 降到 0.05，均匀先验 | 分阶段 | 几何先验更差 |
| PonderNet 2021 | p_n，加 KL(Geom(λ_p)) | 常数 | |
| DSN 2015 | α_m，加 γ-hinge（达标即关闭） | α_m·0.1·(1−t/N) 衰减到 0 | |
| CNDS 2015 | 0.3 | α←α(1−t/N) | |
| HASTE 2025 | 0.5 | 在 τ 处硬停（可按梯度夹角确定 τ） | 停得太早或太晚都差 |
| HLD-LLM 2026 | — | 前 1–5% 预算后硬停 | |
| BERxiT 2021 | 等权 | 奇数步只训末层（50% 占空比） | |
| ElasticBERT / SDViT / LayerSkip | — | 每步只启用部分出口（分组轮转、随机 1 层、rotational） | 降低成本 |
| BitSkip 2025（反例） | 1/(l+1)，偏浅层 | 常数 | 伤末层 |

---

## 4. 对本实验的具体建议

以下是从文献中推出的设计，不是文献结论本身。

**4.1 基本形式**
- L = L_final + λ(t)·Σ_l w_l·L_l，其中 Σ_l w_l = 1，即对总量归一；L_l = CE(W_lm·RMSNorm_final(h_l), y)。
- λ(0)=0.1（沿用 LLAL），线性退火到 0 后，把辅助路径从计算图中移除。
- 窗口 T_w 扫 {≈1%, 3%, 10%} 的总步数。LLAL 的 2k/4k/10k/40k 对应 90k–135k 步里的 1.5%–44%。

**4.2 层间权重（主要消融轴）**
- (a) 均匀 w_l=1/L。
- (b) 深度递增 w_l ∝ l（CALM/PABEE），或 LayerSkip 的二次增长。
- (c) 只监督前 k 层，或前 ~20% 深度（NITP 与 REPA 都指向浅层）。
- (d) Al-Rfou 式错峰移除：浅层先停，第 l 层在 (l/L)·T_w 处停。也可以反过来深层先停，让最浅层被监督得最久，这更贴近 LLAL"救底层"的初衷。
- 对照：LLAL 单层版（第 1 层，或约 20% 深度的一层）。

**4.3 head 与 norm 的梯度路由（一个便宜但关键的开关）**
- A. 辅助梯度同时更新共享的 final RMSNorm γ 和 W_lm：LLAL、LayerSkip、InterCTC 都是这样。
- B. 辅助路径用 sg(γ)、sg(W_lm)，只塑造主干：SOLO 用只读拷贝，Liu 等人的 RNN-T 不更新 decoder。
- 如果模型是 tied embedding，A 还会改动输入 embedding，需要特别记录。

**4.4 目标（可选）**
- 窗口很短时，用硬 next-token 标签即可，因为早期末层分布本身也很差。
- 如果做更长窗口，可以试按深度的 label smoothing 或温度（DTS），或者对 detached 末层分布做 KL（RRT/OISD）。

**4.5 成本与显存**
- 每层都要算 B×T×V 的 logits，必须用分块或融合的 CE。
- 也可以每步只启用一部分层：LayerSkip 的 R、ElasticBERT 的分组、SDViT 每步随机选 1 层。
- 窗口很短时，总额外算力很小：LLAL 单层 <0.3%。

**4.6 诊断（逐层记录）**
- ① logit-lens loss 曲线：窗口前、窗口中、移除后，检查效果能否持续。LLAL 用专家范数和屏蔽实验判断持续性。
- ② cos(g_aux, g_main)：HASTE 的判据，由正转负就该停。
- ③ 每层更新范数和权重范数，与 LLAL 的 norm-collapse 分析对应。
- ④ 训练后看深度利用率：skip 某层后对输出的影响（Csordás et al.）。
- ⑤ 末层 val bpb 与种子噪声对比：130M 的 sd 约 0.0004，需要约 1e-3 的差异。至少用 2 个种子，并在移除后足够久再比较（HASTE 有暂时变差的阶段）。

**4.7 最强的公平对照**
- baseline（同 token、同 schedule）。
- LLAL 单层。
- 临时 layer dropout（Don't Drop Dropout 的递减 schedule，是另一种早期深度干预）。
- 可选：ProRes 残差 warmup。
- 预期：主效应最多约 1e-3 量级。

---

## 5. 开放问题

1. **线性退火、硬停、逐层错峰，哪个更好？** 没有文献直接比较过：LLAL 只测了线性退火，HASTE 与 HLDF 只测了硬停，Al-Rfou 只测了错峰。
2. **窗口长度如何随模型规模和总 token 缩放？** LLAL 从 40k 缩到 2k 效果相近；HASTE 中 τ 随模型大小而变（B 用 100K，L/XL 用 250K）。LLAL 作者也说"how the auxiliary-loss phase should scale with model size"尚未解决。
3. **dense 模型里有没有 LLAL 要解决的那种"浅层信号弱"？**
   - dense 首层 MLP 已经承载 bigram 子网络；Pythia 大模型的层在前 20% 训练内就稳定了。
   - 如果浅层梯度或更新范数并不弱，逐层 LM loss 的作用可能只剩正则。
   - 要先测再判断。
4. **共享 head 的辅助梯度会不会污染 head、final norm 或 tied embedding？** 没有文献做过 A/B（stop-grad 与否）的 LM 对比；SOLO 只在 local learning 中比较了"共享 vs 私有"。
5. **Muon / MuonH 下的相互作用？**
   - 没有文献。正交化更新会丢掉幅度信息，辅助梯度的方向占比可能比 λ 本身更重要。
   - 在弱信号的浅层，即使 λ=0.1，辅助梯度也可能主导更新方向。这一点有待测量。
6. **效应能否持续到训练结束？** DeepMIM 的增益随训练变长而缩小；Kubaty 显示大数据上 joint 有害；HASTE 停止后先降后升。
7. **对深度利用的影响方向？**
   - 逐层 LM loss 鼓励"早定型"（overthinking 视角），可能让后层更冗余，加剧 Curse of Depth 与 depth-efficiency 文献描述的现象。
   - 也可能反过来让每层都承担预测功能。
   - 目前没有证据，训练后需要测。
8. **与 layer dropout 的关系？**
   - LayerSkip 两者合用；Don't Drop Dropout 显示单用递减 layer dropout 就能降 val loss 并改善 early-exit。
   - 两者是互补还是互相替代，尚不清楚。
9. **目标的选择？** 硬 next-token、软化标签、对末层 KL，还是 NITP 式的表示级目标。

---

## 6. 无法核实或部分核实

以下内容**没有**作为正文事实使用，或已在正文中标明限定。

- **未能核实而未收录**：
  - *A comprehensive review on deep supervision in computer vision*（Neurocomputing 2025，ScienceDirect pii S0925231225028656）：只见到搜索结果。
  - *A Survey of Early Exit Deep Neural Networks in NLP*（arXiv 2501.07670）：arXiv API 限流，未能核实。
- **已收录但部分细节未核实**：
  - DTS（ICML 2026）：数字与目标公式未取得。OpenReview 有 bot 验证，也没有 arXiv 版本，只核实了 ICML 页面上的标题、作者和摘要。
  - Bengio 2006：结果表不可机读，只核实了摘要。
  - CPM：消融数字只在图中。
  - GoogLeNet：没有单独的 aux head 效果数字；二手资料里常见的"约 0.5%"在原文中找不到。
  - SDN：没有单列联合训练后的末层准确率。
  - Layer Pruning on Demand（2106.09216）、InterDecoder、Dynamic Encoder Transducer：方法细节只有摘要。
  - 2207.04177、2112.08778、2212.03480：只有"已投稿"信息，录用情况未确认。
  - BERxiT：各策略的末层数字只在图中。Right Tool 的末层下降未量化。ElasticBERT 与 DeepFlow 没有说明 head 是否共享。
  - HLD-LLM：C4 指标标注为 perplexity，但数值约 3.0，更像 log-ppl。
  - LLAL：辅助路径是否经过 final RMSNorm，博客没有说明，图中只有共享的 W_lm。
  - CALM：中间层是否施加 final LN，原文没有说明。
  - LayerSkip：从零预训练的末层逐任务数字只在 Fig.8，没有表格。
  - Ouro App. A：先验对比只有图，没有数字。
  - MoR：全文检索没有找到逐递归的 LM loss。这是"未发现"，不是作者的明确陈述。
  - Valade 2407.21082、LoPT、PGL、InterMPL、iREPA：只核实了摘要。
  - TRM 中引用 ARC Prize (2025a) 的"deep supervision 19%→39%"：我读到的 ARC Prize 博客正文只有"+13pp（0→1 步）"和"训练 16 步 >15pp"，没有找到 19→39 这组数字。
- **已核实但未收录**，因为是推理、综述或重复内容：
  - FlexEE（2609.17008）：只用 LayerSkip checkpoint 做推理。
  - Sangmin Bae 的博士论文（2509.05915）：与 FREE、RRT、MoR 重叠。
  - arXiv 2005.08081：没有辅助 loss。

---

## 附录 A：数据与文件

- 笔记：`survey_notes/verified_meta.txt`（arXiv API 核对记录）、`survey_notes/family_1_1b_8.md`、`family_3_5_7.md`、`family_4.md`（子代理的逐条英文笔记，字段更全）。
- 全文文本：`survey_notes/pdftxt/*.txt`（della-vis1 上用 PyMuPDF 提取），用于逐字核对 §0.3 中列出的关键数字。
- LLAL 原文：`llal1.txt`（主文）、`llal.txt`（后续文章）、`img1/aux_loss.png`（设计图）。
