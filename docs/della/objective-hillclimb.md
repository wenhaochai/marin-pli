# LLM objective hill-climb —— 研究日志（持续更新）

*分支 `objective-hillclimb`。台账（作业号、状态、时间线）在 `logs/objective_hillclimb_qwen3_h100x4.yaml`（gitignored）；本文只放结论级信息：思路从哪来、为什么值得试、判据是什么、结果如何、学到了什么。每个候选一有结果就更新，不等结题。*

**一句话现状（2026-09-19 02:00 EDT）**：在 Kaiyue 130m baseline 上，同数据、同步数下，NTP 加四种不同类型的辅助信号（twin / sr / pi / eos）在末层读出下全部比纯 NTP 差 0.0009–0.0041 c4_en bpb；MuonH 钉死 `hnn.Linear` 辅助头的实现伪影已修正，它解释了 sr 第 1 梯约 2/3 的损失，但自由头下 sr/pi/eos 仍负（+0.0007～+0.0033），每个信号都在"它自己所度量的维度上偏离 batch 多数"的域留下最重的伤；正在检验的是读出位置（R6，辅助头读中间层）与 ebm（能量式 NCE），尚无候选胜出。

---

## 1. 任务与约束

用户指令（2026-09-18，原话）：

> 我的要求是你不许改动数据，然后若无必要不需要改动架构和优化器。你的主要研究对象是 LLM 的 objective，就是找到一个比 next token prediction（或者在 NTP 上加别的 objective）更有 data efficiency 的 objective。限制是你还是老老实实跑完和 baseline 一样的 data 和训练步数策略什么的，compute 没有限制。你开始 design 吧，按照 kaiyue baseline 来 hill climbing，最后 eval 要更好。一直 keep 进步下去。

硬约束与后续澄清：

| 项 | 规定 |
|---|---|
| 数据 | 不许改动（fineweb-edu 10B 预分词缓存，同一 shuffle、同一 batch、同 4959 步） |
| 架构 / 优化器 | 非必要不改。允许新增 baseline 没有的参数（辅助头），但 baseline 的参数与其优化方式不动 |
| 判据 | **same data 下** `eval/paloma/c4_en/bpb` 对 baseline 1.16358；compute 不是判据（用户 2026-09-18 19:0x："same data 负了吗，我们不考虑 compute"） |
| eval 语义 | 训练时才加辅助项（`key is not None`），eval 永远是纯前向 NTP，保证 bpb 可比 |
| 否决 | 自蒸馏（"自蒸馏不好"）；一切 NTP 变体（teacher soft target、future-token label smoothing、token 重加权 / focal、其他 proper scoring rule）——"自蒸馏还是属于 NTP 的变体，你再详细想想别的" |
| 放宽 | "不是完全替代，如果能加辅助损失也可以"（2026-09-18 15:5x）：NTP 主损失 + 辅助损失是合法形式 |
| 文献 | "思路要开阔，要去寻找 2000 以前的 paper 去找找思路"；kexue.fm 作为参考库 |
| 算力通道 | 只用 H100：smoke 走 `pli-cp`，正式 run 走 `pli-short`；A100 不作为绕队列的手段 |
| 失败处理 | "失败了就详细观察 model 的表现，梯度，norm 啥的看看为什么变差，以及保证没有 bug，总结给提出新的 idea" |

## 2. 判据与 baseline

Kaiyue `muonh_qwen3_scaling` 130m：hidden 512、6 层、batch 128 × seq 4096、4959 步（2.6B token）、MuonH lr 0.02 / adam_lr 0.008、linear decay 0.8、max_grad_norm 1.0。4×H100 复现：

| 指标 | baseline |
|---|---|
| c4_en bpb（终值） | **1.16358**（loss 3.7812；原始 speedrun 1.16354，复现误差 +0.00004） |
| Paloma macro | 4.1798 |
| MFU / 吞吐 | 14.36% / ≈831k tok/s，57 min |
| 曲线（c4_en loss / macro，step 1000–4000） | 4.348/4.896 · 4.168/4.647 · 4.013/4.457 · 3.891/4.308 |

读数规则：日志里打印的是 `paloma/c4_en-marin-tokenizer loss`（nats），bpb 只在 W&B；相邻两次 eval 的抖动约 ±0.01–0.02 nats，**不用两个点判方向**，只用终值判胜负。判定阈值：优于 baseline 的幅度要超过 baseline 自身的复现带（≈0.0001 bpb 量级），且方向要在 w 的不同取值上一致。

## 3. 思路谱系（idea ledger）

原则：每一梯的候选必须是**一种不同类型的学习信号**，不是 NTP 的重塑；否决过的方向不再提。

| 梯 | 候选 | 信号类型 | 文献根（≤2000 优先） | 状态 |
|---|---|---|---|---|
| R1 | **twin**：反向模型的后见状态匹配 | 借另一个模型的 hindsight | Schuster & Paliwal 1997（BiRNN）；Twin Networks, Serdyuk et al. 2018 | w=0.1 负 +0.00148（钉死头）；自由头对照在跑 |
| R1 | **sr**：successor representation 的 TD 回归 | 折扣未来嵌入和 | Dayan 1993；Sutton 1988 TD | 全负：+0.00219（钉死）、+0.00085（0.03 自由）、+0.00069（0.1 自由）；伪影占约 2/3，残余对权重不敏感 |
| R2 | **pi**：过去-未来表示的互信息下界（InfoNCE） | 预测信息 | Becker & Hinton 1992（IMAX）；Bialek–Nemenman–Tishby 1999；CPC 2018 | 全负：+0.00414（钉死）、+0.00320（0.03 自由）、**+0.00494（0.1 自由，比钉死更差）**；主动伤害，末层 gain +23% |
| R3 | 权重判别（sr/pi w=0.03） | —— | —— | 完成，见 4.1：eos 成比例（竞争）、sr 残余小且对权重不敏感、pi 主动伤害 |
| R4 | **eos**：到文档结尾的距离（13 个 log2 桶的分类） | 篇章位置，NTP 不显式索取 | 用户提议 | 负：+0.00325（0.1）、+0.00101（0.03），成比例；逐子集预注册失败 |
| R5 | **ebm**：对自身一步去噪样本的因果 NCE | Boltzmann 负相 / 能量式扩散 | Hinton & Sejnowski 1983/1985；Gutmann & Hyvärinen 2010（NCE）；Deng et al. 2020（residual EBM）；EDLM, Xu et al. 2024 | 用户提议方向，排队（w=0.1 / 0.03） |
| R6 | **aux_layer=3**：辅助头改读第 3 层（共 6 层）残差流，顶部只归 NTP | 读出位置，不是新信号 | Caruana 1993/1997（多任务学习：共享隐层、输出专属）；Abu-Mostafa 1990（hints）；Suddarth & Kergosien 1990；deep supervision 2015 | sr / pi / eos / ebm 各一条 L3 arm（w=0.1）排队 |
| R7 | **aux_gate**：辅助权重按 NTP 损失水平退火（4.0→3.6 之间线性降到 0，约前半程有效） | 权重时间表，不是新信号 | 课程式辅助任务（Bengio et al. 2009 curriculum；Caruana 1997 §"when to stop the extra tasks"） | sr / eos 各一条 g4-3.6 arm（w=0.1）排队 |
| — | 自蒸馏；NTP 变体 | —— | —— | **否决** |
| — | 纯 diffusion LM 作为 objective | —— | —— | 不可受理：其 bpb 是 ELBO 上界，与 NTP bpb 不可比 |

待办池（尚未实现，按优先级）：

1. ~~辅助头读 pre-final-norm 或 unit-RMS 归一化状态~~ → 已升级为 R6 `aux_layer`（读中间层），见第 6 节。
2. **MTP 作为辅助**：用 D×D 投影复用 lm_head，而不是新建 Vocab×D 头（130m 上 26M 参数）；文献在 <3B 上对 NTP ppl 为负，故排后。
3. **能量的多步负样本**（ebm 的自然延伸）：用腐蚀→采样→再腐蚀的 Gibbs 式负样本，对应 contrastive divergence（Hinton 2002）；先看一步版本的判别器是否"看得见"。
4. **Predictability minimization**（Schmidhuber 1992）：用对抗预测器逼表示各维互不可预测（factorial code）——与 R1/2 的"拉近表示"相反的一类信号；顾虑是它作用在表示空间，可能重演跨域同质化。
5. **时间慢变 / trace rule**（Földiák 1991；SFA, Wiskott & Sejnowski 2002）：文档内表示慢变、边界处突变。同样是表示空间正则，先等 eos 的逐子集结果再决定。
6. ~~辅助权重随训练衰减到 0~~ → 已实现为 R7 `aux_gate`（NTP 损失水平门控，不需要 step），见第 6 节。
7. FBT 2×2 的结论（见 `closeout-2026-09-18.md`）建立在同样被钉死的 `hnn.Linear` 头上，引用前需在 free heads 下重跑。

## 4. 结果

### 4.1 已完成的 run（130m，same data / steps / batch）

| 变体 | w | 头 | c4_en bpb | Δ vs 1.16358 | macro Δ | MFU | 作业 |
|---|---|---|---|---|---|---|---|
| baseline | — | — | 1.16358 | — | — | 14.36% | — |
| twin | 0.1 | 钉死 | 1.16506 | **+0.00148** | +0.0133 | 14.40% | 14110500（1:50:28） |
| sr | 0.1 | 钉死 | 1.16577 | **+0.00219** | +0.0262 | 14.28% | 14112070（57 min） |
| pi | 0.1 | 钉死 | 1.16772 | **+0.00414** | +0.0246 | 14.27% | 14112397（57 min） |
| sr | 0.03 | 自由 | 1.16443 | **+0.00085** | +0.0079 | 14.13% | 14117922（57:52） |
| **sr** | **0.1** | **自由（归因 run）** | 1.16427 | **+0.00069** | +0.0179 | 14.20% | 14119259（57:08） |
| pi | 0.03 | 自由 | 1.16678 | **+0.00320** | +0.0303 | 14.30% | 14117923（57:20） |
| **pi** | **0.1** | **自由（归因 run）** | 1.16852 | **+0.00494** | +0.0263 | 14.23% | 14119260（57:13） |
| eos | 0.1 | 自由 | 1.16683 | **+0.00325** | +0.0260 | 14.35% | 14118118（57:20） |
| eos | 0.03 | 自由 | 1.16459 | **+0.00101** | +0.0165 | 14.39% | 14118120（57:35） |

**归因 run 的读数（2026-09-19 02:32）**：sr w=0.1 只把头从钉死改为自由，损失从 +0.00219 降到 +0.00069，macro 损伤从 +0.026 降到 +0.018。第 5 节的预注册两极都没命中：伪影是 sr 第 1 梯损失的**主要部分**（约 2/3），但信号本身仍花费约 +0.0007 bpb（复现带的 7 倍）且一无所得。机制上补偿消失了：末层 gain 61.45 对 baseline 61.80（钉死版 +9.2%），`sr_head` 范数 26.4（能动了），辅助 MSE 终值 0.85 与钉死版 0.88 相当——钉死版多出来的伤害来自主干为容量被钉死的头做补偿，不是拟合更差。

**更正**：01:15 写下的"sr 代价随权重成比例"把钉死的 w=0.1 与自由的 w=0.03 相比，被伪影混淆。自由头下两个权重给出 +0.00069（0.1）与 +0.00085（0.03），噪声内相同：**sr 的残余代价小而且对权重不敏感**。eos 的比例关系成立（两个 eos run 都是自由头）。**pi 的归因 run（03:30）反向**：自由头 w=0.1 得 +0.00494，比钉死的 +0.00414 更差——pi 是**主动伤害**、对权重不敏感（0.03 自由 +0.00320、0.1 自由 +0.00494），伪影不是它的原因，钉死的头反而是刹车。机制是到目前最尖锐的“共享顶部”症状：末层 norm gain 75.9 对 baseline 61.8（**+22.8%**；钉死版 +11.3%，sr-fh −0.6%），`pi_proj` 范数 22 → 130，主干总范数反而缩了 10；InfoNCE 终值 0.35 与钉死版 0.31 相当——目标被满足到同样程度，代价是 lm_head 输入尺度膨胀 23%。这正是 R6 要针对的失败：pi-L3（03:24 起跑）的预注册——若末层 gain 回到 baseline ±2% 且 c4_en 落在 +0.001 内，伤害就是顶部干扰；若 gain 仍膨胀或 c4_en 仍负 +0.003～0.005，pi 的几何压力经共享低层传播，pi 彻底退役。

**记分板（末层读出，同数据/步数/batch；\* = 钉死头，含伪影）**：twin0.1\* +0.00148 | sr0.1\* +0.00219 | sr0.1 +0.00069 | sr0.03 +0.00085 | pi0.1\* +0.00414 | pi0.1 +0.00494 | pi0.03 +0.00320 | eos0.1 +0.00325 | eos0.03 +0.00101。干净（自由头、末层读出）的 run 六个，全负，+0.0007～+0.0049。eos 在两个权重下学到的剩余长度信息一样多（交叉熵终值都是 1.8 nats），NTP 的代价随"objective 把信息压进共享表示的力度"变化，不随"学到多少"变化。

前三条都负，且幅度是 baseline 复现误差的几十倍。R3 的两条（2026-09-19）都仍负。sr 在 w=0.03 自由头下 +0.00085；pi 在 w=0.03 自由头下 +0.00320，几乎没比钉死的 w=0.1（+0.00414）好多少——像是**主动伤害**：InfoNCE 的 uniformity 项（512 个来自整个 batch 的负样本把所有状态互相推开）在任何权重下都便宜地重塑了 lm_head 所读的那个 h 的几何；但这一比较跨了钉死/自由两种头，最终归类等 pi-fh w=0.1（运行中）。自由头没有救回 pi w=0.03：`pi_proj` 范数自由后长到 119（损失对头的尺度不变，Adam 让范数漂移，本身无害），末层 gain 仍比 baseline 高 4.1%。sr w=0.03 的机制读数：`sr_head.weight` 39.48（钉死版全程 22.39），末层 gain 62.41 对 baseline 61.80（+1.0%，钉死版 +9.2%）。

### 4.2 逐子集指纹（W&B `eval/paloma/*/loss` 差值）

损伤不均匀：干净网页文本只略差（c4_en twin +0.0048 / sr +0.0071 / pi +0.0134 loss），损失集中在 **code**（dolma programming languages：+0.0926 / +0.1863 / +0.0760）和 **redpajama**（+0.0371 / +0.0885 / +0.0376），pi 还伤了 4chan（+0.0547）。sr 与 pi 在 15/16 子集上变差。这与同一天上午 FBT（untied）留下的指纹一致（code +0.033、4chan +0.034、redpajama +0.022）。

**sr w=0.03、自由头**（2026-09-19）：11/16 子集变差，code +0.0823、redpajama +0.0410、4chan +0.0220、c4_en +0.0028；twitterAAE −0.0238、gab −0.0156、ptb −0.0114 变好。同一枚指纹、约 0.45 倍幅度。**所以 code/redpajama 的损伤不只是钉死头的伪影**：自由的 sr 头依然最伤少数域。"表示空间辅助项施加多数域先验"的假说对逐子集模式重新成为主解释；它对 c4_en 终值的解释力仍然有限（c4_en 只差 +0.0028 loss）。

**pi w=0.03、自由头**：15/16 变差，code +0.1084、4chan +0.0863、redpajama +0.0540、ptb +0.0449；权重降到 0.3 倍，指纹幅度几乎没降。

**eos w=0.1、自由头**（2026-09-19 01:57）：16/16 变差。**预注册的逐子集检验失败**：预测是"eos 不出现 code/redpajama 塌陷"，实测 code +0.0767（最差子集）、redpajama +0.0461。按预注册的规定，"结构信号不会同质化表示"的强形式被否，改为：**任何辅助梯度都会伤害"在该信号所度量的维度上"统计不同于 batch 多数的域**。支持修订版的是指纹里新出现的部分：twitterAAE +0.0744、gab +0.0645 是第二、三差，而 sr/pi 在这两个域反而变好（sr w=0.03：−0.024 / −0.016）——它们正是短文档社交域，"到文档结尾的距离"的统计在这里与填满 batch 的长篇散文最不同；pi 伤得最重的 4chan（+0.086）在 eos 下几乎无伤（+0.004）。每个信号留下的指纹都长得像它自己的目标统计量。机制侧：`eos_head` 权重从 22.4 缩到 4.03，末层 gain 60.36（−2.3%，这次是向下动，共享顶部仍被推动）；训练损失差 1.80 nats = eos 交叉熵终值（随机水平 log 13 = 2.56），主干确实学到了剩余长度信息，NTP 为此付了账。

**推论（写在 eos/ebm 出结果之前）**：失败定位在**共享的顶部**——末层 h、末层 norm gain、lm_head 的工作点。下一个旋钮不是新信号，而是**辅助项读中间层**：让辅助头读某一中间层的残差流（无参数 RMS 归一化），顶部几层只归 NTP。文献根：Caruana 1993/1997 多任务学习（共享隐层、输出任务专属）；Abu-Mostafa 1990 "learning from hints"；Suddarth & Kergosien 1990 rule-injection hints；deep supervision（Lee et al. 2015）。实现为 `aux_layer`（默认 −1 = 现状），先 smoke，等 eos/ebm 落地后决定哪些 arm 上中间层。

### 4.3 在跑 / 排队

| run | 作业 | 状态（02:45） | 说明 |
|---|---|---|---|
| pi / twin w=0.1 **free-head 对照** | 14119260 / 61 | 02:33 起跑 | twin 另含反向权重修复；twin 约 1:50，若超 2h 手动续跑 |
| ebm w=0.1 / 0.03 | 14121505→14131136 / 14127836→14131545 | 02:42 / 02:54 起跑 | GPU smoke 02:37 通过：1.2 s/it ≈ 0.5× baseline 吞吐，40 步无发散；一次 run 约 1:50，两条都加了 resume 备用段 |
| **R6** sr / pi / eos / ebm **-L3**（w=0.1） | 14130181 / 82 / 83 / 84 | 排队 | GPU smoke 02:53 通过（130m eos-L3，40 步 1:03）：`scan_via` 在 FSDP + remat 下编译运行正常 |
| **R7** sr / eos **-g4-3.6**（w=0.1） | 14130605 / 06 | 02:10 入队 | 门控只是标量乘法，CPU smoke 已覆盖，不另做 GPU smoke |

## 5. 失败分析（2026-09-18 晚，按用户要求做在提新想法之前）

材料：baseline / twin / sr / pi 的 W&B 全史（495 行，step 10–4950）。

1. **梯度裁剪不是机制**：`grad/norm/total` 超过 1.0 的步数占 0.2% / 0.4% / 0.4% / 0.6%（base / twin / sr / pi）；均值 0.38–0.54。裁剪从未压制 NTP。
2. **辅助项量级正常**："名义 0.1、实际 1"的担心被否：sr 的 MSE 1.47→0.88，pi 的 InfoNCE 3.07→0.31（随机水平 log 513 = 6.24）。
3. **暴露问题的异常**：sr 的 `params/norm/total` 在每个 checkpoint 上与 baseline 四位有效数字相同（595.95 vs 594.76、…、2211.9 vs 2211.8），尽管多了一个 512×512 的头、梯度也不同。说明参数范数由优化器决定，不由梯度决定。`levanter/optim/muonh.py` 32–37 行：MuonH 把每个 `hnn.Linear` 权重严格保持在初始化 Frobenius 范数（`p_new = p_int/‖p_int‖·‖p‖`）；`create_mask` 把 Embedding→adam、lm_head→adamh、**任何 `hnn.Linear`→muonh**。四个辅助头全是 `hnn.Linear`。
4. **头从未改变尺度**：`sr_head.weight` 22.3867→22.3886，`pi_proj.weight` 22.331→22.332，`twin_proj.weight` 22.295→22.294。它们只能旋转不能缩放；sr 的 MSE 停在 0.88、头梯度 ~0.003，是一个够不到目标的头。twin_proj 的 bias（adam，自由）从 1.16 长到 3.69，是唯一能吸收压力的自由参数。
5. **主干被迫补偿**：末层 RMSNorm gain（adam，自由）在 step 4900：base 61.78、twin 65.11、sr 67.47、pi 68.79（+5%～+11%）。这个 gain 就是 lm_head 的输入尺度，而 lm_head 自身也被钉在 353.3（adamh）。从"钉死的头"到"NTP 损失"的每一环都在数据里。
6. **FBT 的 W_u / W_g 也是 `hnn.Linear`**，同样被钉死。这解释了为什么四个构造完全不同的信号留下同一枚逐子集指纹：它们共享一个实现伪影，而不是信号的共性。之前写下的"表示空间辅助项施加多数域先验"假说没有被否，但已不再必需，降为次要候选。
7. **代码审计另发现一个 bug**：twin 的反向帧 loss weight 是简单翻转，差一位（屏蔽了第一个真实目标、放开了最后一个回绕到垃圾的目标）。每条序列各一个位置，解释不了 +0.0015 bpb，但确是 bug，已修：`rev_weight = roll(flip(loss_weight), -1)`。sr / pi / eos 的掩码逐行复读无误。

**修正**（commit 4a2c2572f4）：辅助头改为普通 NamedArray 权重（`FreeLinear`，从 `hnn.Linear.init` 取数组，保证钉死 / 自由两版初始化相同），MuonH 的掩码将其标为 adam——范数自由，以 adam_lr 更新。优化器本身未改，只是 baseline 没有的参数被分到不同组，仍在"非必要不改优化器"之内。默认 `free_heads=True`，run id 加 `-fh`。

**本项目规则**：任何必须改变尺度的新参数，在 MuonH 下都不能建成 `hnn.Linear`。

**预注册读法**：free-head 对照若回到 ≤ baseline，说明 R1/2 的损失是头的伪影，三个信号重获一次真正的试验；若仍以同样幅度落后，伪影真实存在但不是原因，这些信号在此规模下确实无用。

## 6. 各候选的设计要点

所有变体在 `experiments/references/objective_qwen3.py`（`ObjectiveQwen3Config`，`VARIANT=twin | sr | twinsr | pi | eos | ebm`），共同点：损失 = NTP + w·aux；eval 只算 NTP；跨文档配对全部屏蔽（segment id 来自 EOS 分段，与 `block_cross_document_attention` 一致）。

**twin**（Twin Networks）。同形反向 Qwen3 读反转序列（segment id 同步反转），自有反向 NTP。前向 h_t（预测 x_{t+1}）经 Embed→Embed 仿射映射，被拉向 sg(反向状态 b_{t+2})——它读过 x_{≥t+2}、同样在预测 x_{t+1}，两者都没见过目标。loss = NTP_f + NTP_b + w·mean‖g(h_t) − sg(b_{t+2})‖²/D。反向模型在 eval 时丢弃。

**sr**（successor representation）。ψ_t = W h_t 回归 TD 目标 sg(e_{t+1} + γ·cont·ψ_{t+1})，e 为 RMS 归一化的 token 嵌入，γ=0.9，cont 要求 x_{t+2} 仍在同一文档。

**pi**（predictive information）。unit(W h_t) 要在 512 个从整个 batch 无放回抽取的 unit 状态里认出 unit(h_{t+k})，k=4，τ=0.1；两侧都有梯度。

**eos**（距文档末尾的距离，用户提议）。d_t = 第一个严格在 t 之后的 EOS 的位置 − t；13 路交叉熵，桶 = ceil(log2 d)（{1},{2},{3–4},…,{2049–4096}），能表达"要么很快结束要么很晚"这类双峰信念。屏蔽：窗口内没有后续 EOS 的位置（被截断的末文档，约 25% 位置）、EOS 位置本身。d=1 与 NTP 的 EOS logit 重合，d≥2 是新信息。EOS id 128001 在预分词缓存上经验验证（4M token 里 128001 出现 4067 次、BOS 128000 出现 4068 次，各文档末 token 均为 128001，平均文档 984 token）。

**ebm**（能量式 NCE，用户提议的"energy-based diffusion LM"方向，commit 3c95cb2551）。
- 约束判断：纯 diffusion LM 不能当候选（bpb 是 ELBO 上界，不可比；单遍 2.6B token 上 diffusion 在似然上落后 AR）。取的是 EDLM 的**训练信号**：真实数据与去噪器自身样本之间的 NCE。
- 腐蚀 = 以模型自身为噪声核的离散前向扩散：每条序列 ρ ~ U(0, 0.5)，每个位置以概率 ρ 被替换为从 p(x_t | x_<t) 采的样本（Gumbel-max，词表分 32 块扫描求 running max/argmax，精确等价 categorical，不物化 128k logits）。位置 0、EOS、采到 EOS、采到真 token 的位置不动。
- 第二次 trunk 前向读腐蚀序列（同一 mask），不加 NTP。
- 标量头（`FreeLinear`，adam 组）作用在 **unit-RMS 归一化**后的末层状态上，得到 s_t = log-odds "前缀是真的"（= −能量）——吸收第 5 节教训，头不再看到共享的 final-norm gain。
- 二分类 NCE：softplus(−s_clean) + softplus(s_noisy)，只在"同一文档内、t 之前至少有一次替换"的 informative 位置求平均（其余位置两侧前缀相同，是抛硬币）。NCE 恢复非归一化密度的推导见 kexue.fm/5617。
- 为什么是不同的信号：NTP 只见过干净前缀（teacher forcing）。这里让 trunk 在含自身样本的 off-manifold 前缀上工作，并要求它在替换很久之后仍能从残差流察觉"历史里混进了自己的生成"——Boltzmann 机负相 / residual-EBM 的全局一致性信号，局部 softmax 从不索取。
- 预注册：(a) same-data c4_en bpb 对 1.16358；(b) 逐子集若无 code/redpajama 塌陷，支持"R1/2 的塌陷是钉死头的伪影"；(c) NCE 值从 2 log 2 = 1.386 起步，若停在 1.3 附近说明判别器是瞎的、梯度只是噪声。
- CPU smoke（登录节点 3 min）：六个变体 eval 与原版 Qwen3 完全一致；采样器对 softmax 的最大 total variation 0.0497（4000 次抽样，32 个位置，含词表不能整除分块的填充路径）；保护位置从未被腐蚀；informative 掩码与逐文档暴力检查一致；平均替换率 0.198；NCE 初值 1.20；ebm_head 在 adam 组；梯度有限。每步成本约 2×（compute 不是判据）。

**aux_layer**（R6，读出位置，commit 随后）。`aux_layer=k` 时，用一次 `scan_via` 扫过 Stacked 层并同时返回每层输出；h_aux = 第 k 层之后的残差流，按位置做无参数 RMS 归一化；NTP、eval 与 ebm 的采样器仍用末层归一化后的 h。于是辅助梯度只到达 ≤k 层；>k 层、末层 norm、lm_head 只由 NTP 塑形。额外代价只有保存各层输出（130m 上每卡 0.8 GB）。
- 四条 L3 arm（k=3/6，w=0.1，自由头）各代表一类信号：纯竞争（sr）、主动伤害（pi）、结构（eos）、能量/NCE（ebm），所以这一梯测的是**放置位置**，跨四类信号。
- 预注册：(i) sr-L3 ≤ baseline 而 sr-fh 仍负 → 代价来自顶部干扰，放置规则进入之后所有候选；(ii) pi-L3 仍负约 +0.003 → pi 的伤害不在顶部（几何压力经共享的低层同样传到 NTP），pi 退役；(iii) eos-L3 / ebm-L3 对 eos-fh / ebm-fh，看在顶部中性或正向的信号下移后是否获益。四条都落在各自末层版本的噪声带内 → 位置无关，剩下的杠杆只有信号本身。
- CPU smoke：sr/eos/ebm/twin 在 aux_layer 1、2（共 2 层）下 eval 与原版一致；`forward_with_aux` 的 h 与 `activations` 一致；h_aux 单位 RMS；最后一层时 h_aux × 末层 gain 复现 h（误差 1e-3，索引与归一化正确）；L1/L2 读出不同；初始化时 aux_layer=2 与末层读出训练损失完全相同（gain=1），符合预期。

**aux_gate**（R7，权重时间表，commit 随后）。gate = clip((sg(L_ntp) − lo)/(hi − lo), 0, 1) 乘在每个辅助项上（twin 的反向 NTP 不受门控）；hi=4.0、lo=3.6 取自 baseline 自己的训练损失曲线（step 200 4.43、500 3.99、1000 3.85、1500 3.75、2000 ≈3.7、2500 3.59、3000 3.50、4000 3.38、末 3.25），即辅助项前约 500 步全开、到约 2400 步线性降到 0、后半程关闭；名义 w=0.1，早期推力与 R1 相同。门控用的是本 run 自己每个 batch 的 NTP 损失（stop-gradient），不需要 trainer 传 step。
- 直接检验 sr / eos 的"纯竞争"读法：辅助项在早期有没有买到能留存到关闭之后的东西。
- 预注册：(a) ≤ baseline（复现带 ±0.0001 内）→ 代价全是后期税，早期辅助免费但无用，平局、退役；(b) 赢 baseline > 0.0003 bpb → 早期塑形有留存价值，权重时间表成为一等旋钮（扫 lo、把门控加到 ebm）；(c) 仍负约 +0.001 → 连早期辅助梯度都要付账，伤害在前 2000 步造成且不可恢复——这是"NTP + 辅助"在此规模下最强的负结果，剩下的杠杆只有 R6 的放置位置。
- CPU smoke：gate=1 与无门控损失相同；gate=0 与纯 NTP 相同；gate=0.5 等于 NTP + aux/2；eos 头在 gate=0.5 下的梯度恰为无门控的一半（stop-gradient 成立，没有梯度从门漏出）。

## 7. 工程备忘

- 启动器：`experiments/references/della_muonh_qwen3_scaling.py`，env `VARIANT` 与 `TWIN_W/TWIN_OFF/SR_W/SR_GAMMA/PI_W/PI_K/PI_TAU/PI_NEG/EOS_W/EBM_W/EBM_RHO/FREE_HEADS/AUX_LAYER/AUX_GATE(hi:lo)`；W&B 项目 `marin-della`，group `muonh-qwen3-objective-della`；run id `muonh-qwen3-130m-della4xh100-<tag>[-g{hi}-{lo}][-L{k}][-fh]`。
- 正式 run：`sbatch --job-name=obj-130m-<v> --export=ALL,SIZE=130m,VARIANT=<v>[,EBM_W=0.03] scripts/della/muonh_qwen3_h100x4.sbatch`（pli-short，2h 段；≈2× 成本的 twin/ebm 视情况加 `--dependency=afterany` 备用段）。smoke：`scripts/della/muonh_qwen3_smoke.sbatch`（pli-cp，40 步）。
- CPU smoke：`scripts/della/objective_cpu_smoke.py`（`JAX_PLATFORMS=cpu PYTHONPATH=. .venv/bin/python …`），覆盖 eval 一致性、梯度、优化器分组、eos 目标精确检查、ebm 采样器与腐蚀统计。
- 队列经验：pli fairshare 已耗尽，起跑靠 backfill；每个 QoS 只有 10 个 pending 累积 age，故 pending 控制在 10 以内；sbatch 在作业开始时才 import 启动器，所以 python 侧修复会落到已排队的作业上（run id 随之变化，如 `-fh`）。
- 提交历史：d81d846dc9（twin+sr）→ 6efa140ec4（pi）→ 0f26a4d803（eos）→ 4a2c2572f4（free heads + twin 修复）→ 3c95cb2551（ebm）。

## 8. 更新日志

- **2026-09-18 13:5x** R1 twin/sr 实现、CPU smoke 通过、正式 run 入队。
- **15:0x** R2 pi 加入；用户放宽为"可加辅助损失"。
- **17:52** sr、pi 完成：+0.00219 / +0.00414 bpb，负。R3 w=0.03 判别 run 入队。
- **18:1x** R4 eos 按用户提议设计、EOS id 经验验证、入队。
- **18:43** twin 完成：+0.00148，负。用户澄清判据只看 same data，不考虑 compute。
- **18:5x** 逐子集分析：损伤集中在 code / redpajama，与 FBT 同指纹。
- **19:3x** 失败分析：查明 MuonH 钉死 `hnn.Linear` 头的伪影；twin 反向权重 bug 修复；`FreeLinear`，`-fh` 对照入队。
- **21:1x** 用户提议 energy-based diffusion LM 方向 → R5 ebm 设计、CPU smoke 通过、入队。
- **2026-09-19 00:17 / 00:37** sr-w003、pi-w003 起跑（均为 free head）；ebm w=0.03 补交。
- **01:01 / 01:08** eos w=0.1、w=0.03 起跑。
- **01:15** sr w=0.03（自由头）完成：+0.00085 bpb，仍负、缩小；code/redpajama 指纹仍在（0.45 倍）；`sr_head` 范数已能变化（39.5），末层 gain 补偿从 +9% 降到 +1%。
- **01:34** pi w=0.03（自由头）完成：+0.00320 bpb，代价不随权重缩小 → 主动伤害；15/16 子集变差。推论：失败在共享顶部，下一旋钮 `aux_layer`（辅助项读中间层）。
- **01:35** sr-fh w=0.1 归因对照起跑。
- **01:43** R6 `aux_layer` 实现（commit 90208d8ad4）、CPU smoke 通过；sr/pi/eos/ebm 四条 L3 arm 与 pli-cp smoke 入队。
- **01:57** eos w=0.1（自由头）完成：+0.00325 bpb，16/16 子集变差；预注册的逐子集检验失败，假说修订为"任何辅助梯度都伤害在该信号维度上偏离 batch 多数的域"（eos 的新指纹：twitterAAE、gab 短文档域）。待办第 6 条改为可行的"损失水平门控"衰减。
- **02:05** eos w=0.03 完成：+0.00101 bpb，与权重成比例（纯竞争，同 sr）。记分板：四类信号、七个 run、零胜。
- **02:10** R7 `aux_gate` 实现（commit d8b7d2e403）、CPU smoke 通过；sr / eos 的 g4-3.6 arm 入队（pli-short pending 回到 10）。
- **02:37 / 02:53** ebm 与 aux_layer 的 130m GPU smoke 先后通过（ebm 0.5× 吞吐）。
- **03:30** pi-fh w=0.1 归因 run 完成：+0.00494，比钉死版更差；末层 gain +22.8%。pi 定为主动伤害，只剩 pi-L3 一个翻案机会。
- **02:32** sr-fh w=0.1 归因 run 完成：+0.00069 bpb。伪影解释了 sr 第 1 梯约 2/3 的损失，残余 +0.0007 对权重不敏感；更正了 01:15 的"成比例"读法。pi-fh、twin-fh 起跑。
