# pls：kaiyue baseline 上的逐层 LM 监督（dense、每层、全程）——第一阶段 2026-10-01 结题；深度与残差结构的博客实验到 2026-11-01

**一句话**（2026-11-01 改写）：给每层加 next-token 损失总让最后一层变差（6 层 130m 全程开 +0.034–0.044 bpb，早关则无影响；48 层独立头时多数残差结构 +0.08–0.11 nats，4 条及以上流的 Hyper-Connections 只 +0.03–0.05 nats），而 HC 比 mHC 付得少不是因为它更多地使用读出看不到的各流之差——把这一部分打开给 mHC（wq w711/w712）后使用升到 4 倍，代价只从 0.102 降到 0.085 nats——而是因为两者的独立头模型一样好（2.99–3.03 nats），mHC 的普通模型却更好（2.913–2.926 对 HC 的 2.936–2.979 nats）。详见「因果检验结果」一节。

**第一阶段（2026-10-01 结题）**：在 kaiyue 的 dense MuonH baseline 上，给每一层加 next-token loss 要么伤最终层（全程开：
130m 三种读出 +0.034–0.044 bpb、300m +0.043，因为每层都成了预测器、需要多层配合的 in-context 电路形成被推迟、顶层闲置），
要么什么也不改变（在模型开始用上下文之前关掉——第 80 步、第 140 步、或 LLAL 式 0–150 步线性退火——130m 终值与 8 个
baseline run 的池差 ≤ 0.0006 bpb）；它在前 ~3% 训练里带来的 n-gram 加速从不转成持久收益，所以按动态指标调 schedule 在
这个规模上没有可收获的东西。

## 是什么

LLAL（Jin et al. 2026，"Mitigate Silent Expert Death in Ultra-Sparse MoE"，
https://mooler0410.github.io/puguJin/blog/llal/ ）在超稀疏 MoE 的第一个 MoE 层上加
λ·CE(W_lm h^(ℓ), y)，λ 从 0.1 在前 2k–40k 步线性退火到 0，只挂一层。这里按用户 2026-09-29 的要求改成 dense、
逐层、全程、权重 1 的版本。L 层模型：

    loss = Σ_{k=0}^{L-1} CE(lm_head(final_norm(h_k)), x_{t+1})

h_k 是第 k 层 block 输出的残差流；k = L−1 那一项就是原来的 NTP loss，所以 6 层模型一共 6 项（5 个 aux + final），
每项权重 1（`PLS_W` 是每个中间层的权重，默认 1）。final RMSNorm 与 lm_head 由所有读出共享、接收所有读出的梯度；
不加任何新参数，所以 MuonH 钉死新 `hnn.Linear` 头的坑（hill-climb 第 5 节）在这里不存在。数据、步数、batch、
优化器与 baseline 完全相同；eval（`key=None`）走 baseline 的 loss 代码，`eval/paloma/*` 与 baseline 池直接可比。

## 实现

- `experiments/references/per_layer_qwen3.py`：`PerLayerQwen3Config`（`pls_weight`、`pls_monitor_stride`、`pls_eval`）
  与模型。一次 `scan_via` 取出每层输出；训练返回每层 CE 作为 `train/pls/L{k}`（`train/loss` 在 pls1 下是总和，
  与 baseline 比较用 `train/pls/L{L-1}`）。
- `PLS_W=0`（pls0）：训练与 baseline 相同的计算（CPU 上 loss 与梯度逐位相同），每层读出只在每 16 个位置取 1 个、全程 stop-gradient，给出
  baseline 自己的 logit-lens 曲线。已实现并测过，但按用户决定（2026-09-29："baseline 我已经跑完了"）不跑。
- 逐层 eval：`ReadoutTaggedEvaluator` 一次前向解码所有层，统计量与 levanter `TaggedEvaluator` 完全一致，只多一个
  读出维度；在主 eval 的节奏下记 `eval/L{k}/{loss,macro_loss,bpb,macro_bpb,paloma/<ds>/{loss,bpb},...}`。
  `eval/L{L-1}/*` 必须等于主 eval（run 内自检；CPU f32 逐位相同，GPU bf16 端到端相对差 ≤ 2e-7，比较要带容差）。
- `lib/levanter/src/levanter/main/train_lm.py`：通用钩子，model config 若有 `extra_eval_callbacks` 就按主 eval 节奏挂上。
- 启动：`scripts/della/pls_h100x4.sbatch`（`--export=ALL,SIZE=130m,PLS_W=1`），`scripts/della/pls_env.sh` 让 worktree
  自己的 `lib/*/src` 排在共享 venv 的 editable 安装之前（那些 .pth 指向 speedrun 在用的 `project/marin`）。
- 测试：`scripts/della/pls_cpu_test.py`（假 4 卡 mesh）：初始化与 eval 等于 baseline；loss、每层指标与全部梯度等于
  逐层 Python 循环 + 稠密 logits 的独立参考（w = 1、0.3）；w = 0 的 loss 与梯度等于 baseline 训练步，监控值等于
  步长抽样的参考；读出逐位置等于参考，最后一层与 baseline eval 逐位相同；读出 evaluator 对每个读出等于单独跑的
  `TaggedEvaluator`（loss、macro、逐数据集 loss 与 bpb）。

## 成本

每个中间层多一次 lm_head（前向 + 反向）。130m 的 lm_head 约占训练 FLOPs 的 58%：pls1 每步约 3.9× baseline（300m 约
4.9×）。baseline 墙钟 130m ≈ 57 min、300m ≈ 4.6 h（4×H100），pls1 预计约 3.7 h 与 22 h（以 smoke 实测为准）。
pls0 的监控按代码记账约加 6%（130m）/ 8%（300m）训练 FLOPs（不跑）。逐层 eval 每次约为主 eval 的 4–5 倍（130m 主 eval
约 22 s）。显存（CPU 上按真实每卡形状编译、memory_analysis）：300m 训练 22.0 → 30.9 GiB，130m 13.2 → 15.9 GiB，300m eval
5.5 → 8.8 GiB，80 GB 卡上余量充足。

## 实验与判据

| arm（run 后缀） | 做法 | 130m | 300m |
|---|---|---|---|
| 共享头（`-pls1`） | 每层读出都用模型的 final norm + lm_head，它们接收所有层的梯度 | 1 run | 1 run |
| 只读共享头（`-pls1-dh`） | 同上，但中间层 loss 对 final norm / lm_head 是 stop-gradient，只训练主干 | 1 run | 取消（用户 2026-09-30） |
| 独立头（`-pls1-sep`） | 每个中间层自己的 RMSNorm + lm_head（只由该层 loss 训练）；主头只由最终 loss 训练 | 1 run | 看 130m 结果再定 |

- 最终层对照：130m 用已有 8 个 baseline run 的池（c4_en bpb 均值 1.16327、sd 0.00035；macro sd 0.007–0.010）；
  300m 用已有的 1 个 baseline（c4_en 1.05626、macro 3.80874）。
- 逐层：pls1 的 `eval/L{k}` 与 `train/pls/L{k}` 给出每层读出随训练的变化；baseline 没有逐层指标（不另跑 pls0）。
- 这是第一轮"看情况"的实验（n = 1）；大效应（> 0.003 bpb）单 run 可判，小效应要按 hill-climb 规则补到每臂 n ≥ 4。

## 审计（2026-09-29/30，用户要求"严格审计"）

独立审计 agent（报告 `tmp/pls/audit/AUDIT.md`）结论：无 BLOCKER / MAJOR。它在不经过 pls 代码的参考上复核了 loss、逐层
stats 与梯度（f32 逐位；bf16 策略下 loss/stats 逐位、梯度在 bf16 噪声内；2 个 microbatch 的梯度累积等于两份参考的平均），
监控的 roll-再抽样没有 off-by-one、stop-gradient 无泄漏，`train_lm.main` 端到端（bf16、microbatch、真实数据集类型、不满的
最后一个 eval batch、marin tokenizer 的 bpb）`eval/L{L-1}/*` 与主 eval 的键全部相等、同 step 触发，续跑正常；对原 CPU 测试
做 9 个定向变异，全部被杀掉。4 个 MINOR 已修：`scan_layers=False` 时拒绝（BlockSeq 的 scan_via 不按层切 key）；训练
sbatch 写死 `VARIANT=pls`、`SIZE` 必须显式给、清掉会改变 run 的 launcher 旋钮（TOTAL_STEPS、INIT_FROM、TIE、PRECISION、
EMA_BETA、SMOKE_STEPS…）；smoke 的 run id 带 `-j<jobid>` 且取自 launcher 的 dry run，并清理其临时 checkpoint；
`pls_report.py` 默认只读 `-pls1`。另：worktree 用自己的 venv 副本（`.venv`，复制自共享 venv），另一个会话的 `uv sync`
不会在 300m 续跑链中途换掉底层版本。

解读注意：pls1 的逐层梯度相加后，`clip_by_global_norm(1.0)` 会比 baseline 更常触发（`grad/norm/total` 与裁剪率要一起看），
这是处理的一部分，不是 bug。

## 跑之前写下的预测（2026-09-29）

1. pls1 的最终层变差，且明显超出噪声：130m c4_en bpb 比 baseline 池差 0.01–0.05，macro 同向。理由：每层权重 1 时
   总 loss 被浅层的高 CE 主导，共享的 final norm / lm_head 要同时服务所有层，残差流被迫在每一层都"可直接解码"。
2. pls1 的中间层读出逐层单调变好，浅层读出的 CE 远低于未受监督模型的 logit-lens 典型值。
3. 300m（12 层）上最终层的代价不小于 130m。
若 1 不成立（最终层持平或变好），那就是本项目要追的现象。

**独立头的预测（2026-09-30 21:5x，job 14769391 开跑后、第一个 eval 之前写下）**：独立头去不掉大部分伤害，130m 最终层
c4_en bpb 对池仍差 ≥ +0.02（最可能 +0.03–0.045，与共享头相当）。理由：只读共享头已排除"输出头被拉偏"。独立头去掉的是
"所有层共用最终头的坐标系"，去不掉"每个 block 的输出都要能直接读成预测"：第 j 个 block 收到 L−j 个 loss 的梯度，浅层
block 的容量主要服务 aux loss；而 baseline 的浅层本来在做别的事（它的 logit lens 到 L3 还是 7.41，预测在最后两层才
形成）。判据（单 run，伤害量级远大于噪声）：
- Δ ≤ +0.01：伤害主要来自"所有层被对齐到最终头的坐标系"（机制 2），独立头是值得上 300m、补 seed 的版本；
- Δ ≥ +0.03：伤害来自"每层都必须能预测"本身（机制 3），全程逐层监督在这个规模上本质有代价，下一步改成临时监督
  （LLAL 式退火）或很轻的权重；
- 介于两者：两种机制都有份。

## 结果

**130m（完成，2026-09-30 05:1x；W&B `muonh-qwen3-130m-della4xh100-pls1`）：最终层明显变差，且差距随训练扩大。**

| 指标（第 4958 步） | baseline 池（n=8） | pls1 | Δ |
|---|---|---|---|
| c4_en bpb | 1.1633 ± 0.0003 | 1.2042 | **+0.0409** |
| Paloma macro loss | 4.1865 ± 0.0058 | 4.3715 | **+0.1850** |
| c4_en loss | 3.7802 ± 0.0011 | 3.9130 | +0.1329 |

Δ 随训练（macro / c4_en bpb）：step 1000 +0.120 / +0.023，2000 +0.122 / +0.029，3000 +0.141 / +0.032，4000 +0.161 /
+0.037，终值 +0.185 / +0.041——不是早期的暂时代价。

逐层读出（终值 macro）：L0 5.621、L1 4.750、L2 4.479、L3 4.408、L4 4.381、L5 4.372。L3→L5 只好 0.036：上层几乎不再
贡献，模型的有效深度被压缩。run 内自检：`eval/L5/*` 与主 eval 40 个键最大差 1.1e-4（相对 ≤ 2e-5，正负随机，bf16 下
scan_via 与 fold 的融合差异）。曲线：`pls_figs/130m_*.png`。吞吐：MFU 20.3%，每步 1.75 s（baseline 约 0.64 s），墙钟 2.7×。

**300m 共享头（完成 2026-10-01 00:51；两段：14732139 到 step 11008 时限，14732140 续跑 436 步）**：终值 c4_en bpb
1.0996，对 1 个 baseline（1.0563）**+0.0434**；macro 4.0089，**+0.200**（130m 是 +0.041 / +0.185），差距一路扩大（step
2000 +0.019 → 8000 +0.039 → 终值 +0.043）。逐层终值 macro L0 5.521、L1 4.653、L2 4.331、L3 4.204、L4 4.110、L5 4.069、
L6 4.045、L7 4.027、L8 4.015、L9 4.016、L10 4.011、L11 4.009——L8→L11 只降 0.006，12 层里最后 4 层几乎不干活。

**baseline 自己的逐层读出（logit lens，同一个 final norm + lm_head）**：用 130m baseline（restore run）的最终 checkpoint
做逐层 eval（`scripts/della/pls_readout_eval.sbatch`，job 14767382）。日志 "Resuming training from step 4959"、train/loss 0，
没有训练任何一步；主 eval 与原 run 终值逐位相同（c4_en bpb 1.163219928741455，macro 4.181971549987793）。

| 层 | baseline 读出 macro | pls1 读出 macro |
|---|---|---|
| L0 | 9.329 | 5.621 |
| L1 | 8.784 | 4.750 |
| L2 | 7.898 | 4.479 |
| L3 | 7.406 | 4.408 |
| L4 | 5.351 | 4.381 |
| L5（最终层） | **4.182** | **4.372** |

baseline 的预测在最后两层才形成：L3→L5 降 3.22 nats，其中 L4→L5 一层就降 1.17。pls1 把预测提前到了浅层：L2 已经是 4.479
（比 baseline 的 L4 还好 0.87），但 L3→L5 只降 0.036，最终层反而比 baseline 差 0.19。逐层监督让每一层都成了近似的最终
预测器，代价是丢掉了 baseline 靠顶层完成的最后那段精修。300m 在 step 9000 同样如此：L8→L11 只降 0.005（macro 4.074 →
4.069）。

**只读共享头（`-pls1-dh`，130m，完成 2026-09-30 17:36）：切断浅层 loss 对输出头的更新，伤害没有减少。**

| 130m 终值 | c4_en bpb | Δ 对原模型池 | Paloma macro |
|---|---|---|---|
| 原模型（n=8） | 1.1633 ± 0.0003 | — | 4.1865 |
| 共享头 | 1.2042 | +0.0409 | 4.3715 |
| 只读共享头 | 1.2067 | +0.0435 | 4.3914 |

逐层读出（终值 macro）L0 5.680、L1 4.797、L2 4.507、L3 4.426、L4 4.401、L5 4.391，每层都比共享头略差，形态相同（L3→L5
只降 0.035）。Δ 随训练同样扩大（c4_en bpb：step 1000 +0.026 → 终值 +0.044）。两臂各 n=1，0.0025 的差不判方向；可以下的
结论是**伤害不来自共享输出头被浅层 loss 拉偏**（机制 1），而来自主干被逼成每层都能直接预测（机制 2/3）。独立头
（`-pls1-sep`）区分 2 与 3。

**独立头（`-pls1-sep`，130m，完成 2026-10-01 00:14）：去掉约六分之一的伤害，大部分还在。**

| c4_en bpb 对原模型池的 Δ | step 1000 | 2000 | 3000 | 4000 | 终值 4958 | 终值 macro Δ |
|---|---|---|---|---|---|---|
| 共享头 | +0.0226 | +0.0285 | +0.0319 | +0.0373 | **+0.0409** | +0.185 |
| 只读共享头 | +0.0256 | +0.0336 | +0.0369 | +0.0409 | **+0.0435** | +0.205 |
| 独立头 | +0.0161 | +0.0224 | +0.0261 | +0.0310 | **+0.0339** | +0.162 |

（池：终值 c4_en bpb 1.1633 ± 0.0003、macro 4.1865 ± 0.0058，n = 8。）独立头比共享头少 0.006–0.007，每个 eval 都是这个数，
不随训练变化；只读共享头每个 eval 都比共享头差 0.003–0.005（n = 1，方向一致，仍不下结论）。按开跑前写下的判据
（Δ ≥ +0.03），**伤害主要来自"每一层都必须能预测"本身（机制 3）**；"所有层共用最终头的坐标系"（机制 2）只占约 0.007。

| 层（终值 macro） | baseline logit lens | 共享头 | 只读共享头 | 独立头 |
|---|---|---|---|---|
| L0 | 9.329 | 5.621 | 5.680 | 5.641 |
| L1 | 8.784 | 4.750 | 4.797 | 4.750 |
| L2 | 7.898 | 4.479 | 4.507 | 4.437 |
| L3 | 7.406 | 4.408 | 4.426 | 4.380 |
| L4 | 5.351 | 4.381 | 4.401 | 4.360 |
| L5（最终层） | **4.182** | 4.372 | 4.391 | 4.348 |

独立头各层用自己的头读出：L2–L5 比共享头好 0.02–0.04，L1 持平，L0 差 0.02；L3→L5 只降 0.032，顶层照样几乎不干活。
主头只由最终 loss 训练，最终层仍比原模型池差 0.162 macro，说明受损的是主干，不是读出。step 2000 时最终层在 Paloma 上比 L4 还差
0.007（训练集上仍好 0.011），step 3000 起恢复单调。run 内自检：`eval/L5/*` 与主 eval 40 个键最大相对差 5.8e-5（ptb，
小数据集），正负混合，是 bf16 噪声。

**预测检验**：预测 1（最终层差 0.01–0.05 bpb）成立，落在上沿；预测 2（中间层读出远好于未受监督的 logit lens、且逐层
单调）成立，L0 好 3.7 nats；预测 3（300m 代价不小于 130m）成立：+0.043 对 +0.041。独立头的预测（≥ +0.02，最可能 +0.03–0.045）
成立：+0.034。

## 动态 weight schedule：分析（2026-10-01，用户："分析怎么通过一些动态指标监控来调整 weight schedule"）

**已有日志给出的事实**（`tmp/pls/dyn/`，wandb 每 10 步一行；pls 各臂与 baseline 主 run 同 init、同数据顺序，逐步配对）。
最终层 train CE 减去 baseline 主 run，按窗口平均：

| 步 | baseline 其他 seed（s1/s2/s3） | 共享头 | 只读共享头 | 独立头 |
|---|---|---|---|---|
| 0–50 | −0.015 / −0.071 / −0.042 | **−0.185** | **−0.139** | −0.045 |
| 50–100 | −0.067 / −0.065 / −0.048 | **−0.179** | **−0.131** | −0.065 |
| 100–150 | −0.030 / −0.045 / −0.025 | −0.053 | −0.026 | −0.005 |
| 150–200 | +0.017 / −0.044 / +0.005 | +0.054 | +0.078 | +0.087 |
| 200–300 | +0.010 / −0.018 / −0.001 | +0.067 | +0.081 | +0.083 |
| 2000–4960 | −0.006 / −0.004 / −0.004 | +0.114 | +0.128 | +0.094 |

- 只有用最终头坐标读出的两臂（共享头、只读共享头）有超出 seed 差异的早期优势：前 100 步低 0.13–0.18；约 140–160 步
  翻转，之后一路变差。独立头前 100 步的 −0.05 在 seed 差异内。解释：浅层被训练成用最终头的坐标写出 n-gram 预测，残差
  流把它直接带到最后一层，最终层白拿；独立头的浅层预测写在最终头读不到的坐标里。所以能"收获"的早期收益只存在于共享
  坐标的设计里。
- baseline 主 run 早期偏慢（其他 seed 前 150 步低 0.02–0.07），配对比较会夸大早期优势；上表按 seed 差异判读。
- 交叉点（约 3% 的训练）三臂同步；baseline 在 130–180 步下降得比各臂快（每 10 步多降 0.02–0.05），假设是多层电路
  （如 induction head）形成的时刻，逐层监督压制了它——待 probe 的 in-context 分数验证。
- 现有日志里没有指标能在线标出交叉点：L4−L5 读出差全程 0.001–0.005；L0−L5 平滑增长，无拐点；浅层 block 的梯度范数是
  baseline 的 4–8 倍，全程噪声大、无拐点；总梯度范数平。

**候选控制信号**（`PLS_PROBE=1`，`per_layer_qwen3._probe_stats`，stop-gradient，不改训练）：
- `pls_probe/cos_L{k}`：第 k 层 loss 与最终 loss 在共享主干（embedding + block 0..k）上的梯度余弦——一阶意义下"此刻第 k 层
  的 aux 梯度在帮还是在拖最终 loss"（Du et al. 2018 的辅助 loss 加权依据）。`gratio_L{k}` 是两者范数比，`cos_aux` 是 aux
  总梯度与最终梯度的余弦。控制律候选：w_k = clip(EMA(cos_k), 0, 1)，或 PCGrad 式只投影掉冲突分量。
- `pls_probe/icl_L{k}`：第 k 层读出的后半段位置 CE 减前段（位置 32–64）CE，检验交叉点是否是 in-context 能力形成的时刻。
- CPU 上随机初始化的小模型：共享头 cos_L0/L1 = +0.45/+0.75，独立头 ≈ +0.05——与"只有共享坐标有早期收益"一致。

**诊断 run**（130m，完整 schedule 的前 ~500 步，`scripts/della/pls_probe.sbatch`）：共享头 14808812、独立头 14808813、
baseline 14808814（baseline 的余弦是"如果开 aux 会怎样"的假想值）。判据：cos 是否在 130–160 步附近降到 0 附近或以下。

**决定性实验（待用户确认）**："神谕"切换——共享头在观察到的交叉点（约第 130 步）关掉 aux，之后纯 NTP，跑满，2 个 seed
（`PLS_OFF=130`）。它是任何动态 schedule 的上限：在最佳时刻关掉都留不下持久收益，监控指标再好也没用；留下了，才值得做
自动找关闭时刻的控制器。

**诊断结果（2026-10-01 13:4x；probe run 14808812/13/14，各跑完整 schedule 的前 456–582 步；图 `pls_figs/130m_dynamic_signals.png`）**

| 窗口（步） | 0–50 | 50–100 | 130–160 | 160–200 | 200–300 | 300–400 |
|---|---|---|---|---|---|---|
| cos(aux 总梯度, 最终梯度)：共享头 | 0.87 | 0.91 | 0.87 | 0.84 | 0.82 | 0.80 |
| 同上：独立头 | 0.70 | 0.86 | 0.84 | 0.83 | 0.81 | 0.82 |
| 同上：baseline（logit lens 损失若打开会怎样） | 0.30 | 0.12 | 0.09 | 0.10 | 0.11 | 0.13 |
| 最终读出 in-context 分数：baseline | −0.02 | −0.08 | −0.13 | **−0.18** | **−0.28** | −0.27 |
| 同上：共享头 / 独立头 | −0.02 | −0.09 | −0.12 | −0.13 | −0.25 | −0.26 |

1. **梯度余弦不能当开关信号。** 有监督的 run 里，aux 与最终 loss 的梯度余弦全程 0.8–0.9，交叉点前后只缓慢下降，没有变号、没有
   拐点；而 baseline 里同样的逐层 loss 与最终 loss 的梯度几乎正交（总体 ~0.1，L0 ~0.01）。在开着 aux 的 run 里量到的高对齐是
   aux 自己造成的：主干被改造成每层读出都与最终预测一致。按余弦加权（Du et al. 2018）会一直开着 aux，复现全程版的伤害。
2. **交叉点就是 baseline 开始使用上下文的时刻。** baseline 的 in-context 分数在 150–200 步从 −0.13 降到 −0.28；两个有监督的臂
   晚 30–50 步，约 290 步追平。配对 loss 差恰在这段达到峰值（约 220 步：共享头 +0.085、独立头 +0.12），追平 in-context 后回落
   到 +0.05 左右，之后随训练缓慢扩大（顶层闲置的长期代价）。baseline 浅层的 logit lens 读出（L0/L1）在转折处从 −0.12 变到
   −0.04——浅层不再做预测、改为给上层的 context 电路供特征；有监督的臂里浅层仍是预测器（−0.07 到 −0.17）。机制：逐层
   监督加速 n-gram 阶段，但推迟需要多层配合的 in-context 电路（如 induction head）形成。
3. **对动态 schedule 的含义**：开关信号应是 in-context 分数（每个读出两次子集 CE，开销可以全程开着），在它开始下降前关掉逐层
   监督；梯度余弦作废。只有共享坐标（共享头）的早期收益超出 seed 噪声，所以 schedule 实验用共享头。

**神谕切换实验的预测（2026-10-01 13:5x，提交前写下）**：共享头、130m、在交叉点前关掉逐层监督（`PLS_OFF=80`：优势峰值处切；
`PLS_OFF=140`：交叉点处切；`PLS_OFF=0:150`：LLAL 式从 1 线性退到 0）。预测：三者最终层都回到 baseline 池附近（|Δ c4_en bpb|
≤ 0.002），早期 n-gram 加速不留下持久收益——dense + MuonH 里没有 LLAL 要救的"死专家/塌缩浅层"，baseline 自己在 ~150 步已追平
n-gram。若某个臂最终层显著好于池（Δ ≤ −0.001 且补到 n ≥ 4 后仍成立），才值得做 in-context 触发的自动开关。

**神谕切换结果**

| 130m 终值（step 4958） | c4_en bpb | Δ 对池（1.16327 ± 0.00035，范围 1.16278–1.16370） | macro Δ |
|---|---|---|---|
| 全程（共享头） | 1.2042 | +0.0409 | +0.185 |
| 第 80 步关掉（`-pls1-off80`，14832726） | 1.16355 | **+0.0003**（同 seed 的 baseline 主 run 1.16358） | −0.005（池 sd 0.006） |
| 第 140 步关掉（`-pls1-off140`，14832727） | 1.16384 | **+0.0006**（略高于池的最大值） | +0.007 |
| 0–150 步从 1 线性退到 0（`-pls1-off0-150`，14832728，LLAL 式） | 1.16389 | **+0.0006** | −0.000 |

- 第 80 步关掉：伤害完全消失，也没有留下收益——与预测一致。过程中的优势是暂时的：c4_en bpb Δ 在 step 1000 +0.0010、2000
  **−0.0052**、3000 −0.0010、4000 +0.0005、终值 +0.0003。终值的逐层 logit lens L0 9.59、L1 8.91、L2 8.29、L3 7.55、L4 5.36、
  L5 4.18：浅层比 baseline（9.33/8.78/7.90/7.41）还难直接解码——关掉监督后浅层彻底转去给上层供特征。
- 第 140 步关掉：终值 +0.0006、macro +0.007、micro +0.012（2.2 sd），略差于池；越晚关，留下的伤害越多（n = 1，方向待定）。
- 0–150 步线性退火（LLAL 式）：终值 +0.0006、macro −0.000、micro +0.003，与池相同。
- **结论**：三种关法终值都在 baseline 池上（c4_en bpb +0.0003 / +0.0006 / +0.0006，macro −0.005 / +0.007 / −0.000），预测
  （|Δ| ≤ 0.002、无持久收益）成立。在最佳时刻关掉都只能回到 baseline，所以任何按指标调 weight schedule 的控制器在 130m 上
  的上限就是 baseline——没有值得做的控制器。三者 c4 都略正（均值 +0.0005，约 1.5 个池 sd），可能是早期监督的一点残余代价，
  但 macro 均值 ≈ 0，n = 1/臂，不下结论。

**没做的问题（用户 2026-10-01 决定结题，留作记录）**：
1. 更深的模型：LLAL 救的是 32 层 MoE 的第一层；6 层的 130m 可能太浅、浅层不缺信号。520m（24 层）上"在 in-context 起点前关掉"
   能否留下收益（一次约 18 h × 4 卡，需同配置 baseline）。
2. 换掉 MuonH：LLAL 的失效模式（AdamW eps 封顶的更新 + weight decay 把浅层范数压塌）在 MuonH 下结构上不可能；在会塌的
   AdamW dense baseline 上重做，检验 LLAL 的收益是否依赖这种塌缩。
3. 收工：把上面这句话作为本项目的结论。

## 结题（2026-10-01，用户："结题吧，把所有东西上传，ckpt删掉"）

- 代码、文档、过程材料全部在分支 `pls`，推到私有仓库 archive（wenhaochai/marin-private）；过程材料目录见
  `pls_archive/README.md`，每个训练 run 的评估记录与运行配置在 `pls_runs/`。
- 指标全部在 W&B `reself/marin-della`（group `muonh-qwen3-pls-della`；诊断 run 的 run id 带 `-probe-j<jobid>`）。
- checkpoint 已删除：`marin_store_big/speedrun/` 下 8 个 pls run 目录（约 21.6 GB）与临时 checkpoint 目录下 10 个 pls 目录
  （约 8.8 GB，其中 3 个是诊断 run 的残留——`pls_probe.sbatch` / `pls_smoke.sbatch` 的清理只查了临时目录的浅层，而 levanter
  把临时 checkpoint 存在镜像绝对路径的深层；两个脚本已改成直接删那条路径）。
- 原始基线的 checkpoint（不带 pls 的 run）未动。
- worktree `project/marin-pls`（含 12 GB 的 `.venv` 副本）和工作目录 `tmp/pls` 已删除；worktree 的 `logs/`（各 run 日志、slurm 输出、
  台账原件）打包为 `pls_archive/logs.tar.gz`。代码只在 archive 的 `pls` 分支（主仓库本地也留有 `pls` 分支）。

## 博客补充实验（2026-10-02/03）

为博客补了两组 130m 实验。图和脚本在 `docs/della/pls_archive/blog/`。最终值都取最后 50 步的平均训练损失。

1. **深度对照**（作业 14886384，`scripts/della/depth_control.sbatch`，`LAYERS=1..5`）：普通基线模型，宽度、数据、
   步数和 schedule 都和 6 层基线相同，只有层数是 1 到 5 层。d 层模型的骨干算力正好是 6 层基线的 d/6（记录的 FLOPs
   减去输出头，比值 1.000）。最终训练损失 d1..d5：3.971、3.543、3.417、3.343、3.294；6 层基线 3.260。
2. **只训独立头**（作业 14888104，`PLS_DETACH_BB=1`，run `-pls1-sep-bbfrozen`）：每层一个独立头，各自用自己的损失
   训练；骨干对这些头断开梯度，所以骨干的训练和基线完全相同。最后一层等于基线：训练损失 3.257 对 3.260，c4_en bpb
   1.1629 对 1.1636，Paloma macro 4.183 对 4.180。

同样骨干算力下，逐层最终训练损失（第 k 层读出对 k+1 层的普通模型）：

| 骨干层数 | 普通模型 | 共享头读出 | 只训独立头读出 |
|---|---|---|---|
| 2 | 3.543 | 3.669（+0.126） | 4.643（+1.100） |
| 3 | 3.417 | 3.505（+0.088） | 4.216（+0.799） |
| 4 | 3.343 | 3.437（+0.093） | 3.941（+0.598） |
| 5 | 3.294 | 3.406（+0.112） | 3.503（+0.209） |
| 6 | 3.260 | 3.392（+0.132） | 3.257（−0.003） |

结论：

1. 不加干扰的模型，中间层几乎不做下一词预测。在 2 层骨干处，读出比同算力的 2 层模型高 1.10 nats。预测主要在最后两层形成。
2. 逐层监督让每层都接近一个小模型的预测能力：中间层读出只比同算力的普通模型高 0.09–0.13 nats。代价是最后一层也高 0.13 nats。
3. 300m 只训独立头（作业 14888822，12 层）在排队，用来检查"预测在最后几层才形成"在更深的模型上是否成立。

### 页面换成网站的 Epoch 风格（2026-10-12）

网站换成了 Epoch 风格（Google 配色），owner 已批准（站点提交 85fceef）。

1. 页面的示意图由 `make_page_schematics.py` 和 `make_arch_schematics.py` 写入。这两个脚本写完后调用网站的
   `tools/exp-art/restyle_svg.py`，把旧封面风格的 SVG 改成新风格。在新页面上重跑两个脚本，输出和网站页面逐字节相同。
2. 封面动画用 `tools/exp-art/render_anim.py anim_plc.html OUT per-layer-contribution --restyle` 渲染
   （vis 节点，带 Playwright 的 Python，如 `project/frozen-critic/.venv-shot`）。
3. artifact 由网站页面构建（`build_blog_artifact.sh`，OUT_DIR 指向临时目录），不再有暗色主题。
4. 页面还没有 meta description 和 og:description。按网站规则，它要用 Takeaways 那句话，等 12/24/36 层普通模型的结果和
   Takeaways 写好后再加。

## Next: depth (2026-10-03, owner)

1. Queued: the 130m setup with 48 layers instead of 6, separate heads (job 14949463) and probes (job 14949464), 8 GPUs
   each, after an 8-GPU smoke test (job 14949462). Run ids `...-pls1-sep-d48` and `...-pls1-sep-bbfrozen-d48`.
2. 300m: shared head done, probes running (14888822), separate heads queued (14940604). The 300m shared head with
   stop-grad (14949465) was cancelled: the owner does not need it at 300m. From here on the blog follows separate heads
   only (owner, 2026-10-03), so 300m has no setup comparison and no ordinary models by depth (owner: cancelled).
3. To do, the owner asked that it not be forgotten: repeat the per-layer measurement under the setups of DepthBench
   (arXiv 2609.32534, "Measuring How Residual Connections Enable More Computational Depth"): model size and recipe fixed,
   width-depth ratio from shallow-wide to deep-narrow, 10 architectures including Pre-LN, LayerNorm Scaling, mHC and
   AttnRes. The question: how does each architecture change each layer's contribution?
   DepthBench's ten designs: Pre-LN, Sandwich-LN, LayerNorm Scaling, DeepNorm, KEEL, Hyper-Connections, mHC, AttnRes
   (Full), AttnRes (Block), MoDA; shapes at ~400M from (16 layers, width 1216) to (70, 640); 8B tokens, AdamW. The baseline's
   own design is Sandwich-LN (hybrid_norm: pre- and post-norm on both sublayers), not Pre-LN. Implemented as ARCH=<design>
   (experiments/references/depth_arch_qwen3.py, CPU tests scripts/della/depth_arch_cpu_test.py); the blog's Figure 6 shows
   separate heads against probes at 48 layers under the nine designs other than Sandwich-LN (Figure 6 is Sandwich-LN,
   Figure 7 the nine designs; numbering of 2026-10-11).
   Status 2026-10-11 (the -vs-d48-g4 runs: PARAM_MAP=vocab, 4 GPUs per run, two runs per 8-GPU pair job, 24 h chains):
   finished Sandwich-LN, Pre-LN, LNS, DeepNorm, KEEL (L48 train sep/probes 3.093/2.991, 3.073/2.983, 3.096/2.996,
   3.372/3.262, 3.128/3.027); training AttnRes, AttnRes (Block), MoDA; queued HC, mHC. Every finished design: separate
   heads much lower at the shallow and middle layers, higher at the last layers. Not done: the width-depth-ratio sweep
   of DepthBench (one shape only, 48 x 512).
   Status 2026-10-13: AttnRes (Block) finished (3.066/2.953) and AttnRes (Full) finished (3.023/3.045; eval
   3.765/3.738). AttnRes (Full) is the one design without the last-layer cost (train -0.02, eval +0.03 against +0.09..0.11
   and +0.14..0.18 for the others), and its separate-heads run has the lowest layer-48 loss of all designs; its ordinary
   model is weaker than AttnRes (Block)'s. Its probe readouts jump between 3.3 and 7.8 nats from layer 17 on (the mixes
   need not carry next-token information), so on the page it has its own y range in Figure 7 and stays out of Figures
   8-9. MoDA, HC, mHC training (MoDA resumes at step 3730 in 15341589).
   Hyper-Connections finished (2026-10-13 23:41, job 15341054): L48 train sep/probes 2.992/2.944 (+0.048, about half the
   others), eval 3.704/3.615 (+0.089); both models lowest of all designs at layer 48; probes fall steeply between layers
   11 and 17. Page Figures 7, 9, 10 and record (figure numbers moved up after the new Figure 8) updated.
   mHC finished (2026-10-14 02:40, job 15341056): L48 train sep/probes 3.024/2.922 (+0.102, the full Sandwich-LN cost,
   unlike HC's half), eval 3.749/3.576 (+0.173). Its ordinary model ends lowest of all designs at layer 48; HC's separate
   heads stay lowest with separate heads (2.992), so the HC "both lowest" line no longer holds. The mHC probes stay high
   (gap about -1.4) through layer 27 and fall steeply over the last twenty layers. Page Figures 7, 9, 10, record updated.
   Hidden-write theory (2026-10-14, owner: "find a theory that predicts the variants"). Separate heads cost the last
   layer because each layer's readout (own RMSNorm + lm_head; vocab 128k >> width 512, so the head has full column rank
   and no null space) sees the whole state the later layers read: a feature meant for later layers perturbs that layer's
   prediction, so the early-exit loss pushes it out (MSDNet, Huang et al. 2018: intermediate classifiers cut final ResNet
   accuracy by up to 7% and dense connectivity removes most of that). A design pays less only if a layer can write a
   feature that no readout sees. Checks against the 8 finished designs (L48 train gap sep - probes):
     single stream (Sandwich-LN .102, Pre-LN .089, LNS .100, DeepNorm .110, KEEL .101): no hidden write -> full cost,
       whatever the normalization (DeepNorm shifts both curves by +0.27 and keeps the gap).
     HC .048: readout = sum of 4 streams; writes 1 + learned offset are signed, so a zero-sum write is invisible -> less.
     mHC .102: writes 2 sigmoid > 0 and non-negative doubly stochastic mixing: every write shows in the sum -> full.
     AttnRes (Full) -.022: every sublayer output stored apart; a readout is one mix, an output can get ~0 weight in it
       -> no cost (confounded by its ordinary model's spikes; seed 1 pending).
     AttnRes (Block) .113: everything written joins the current block's running sum, which every readout sees -> full.
   MoDA (job 15341589, due ~11:00): one Pre-LN stream that every readout sees; its depth keys and values are projections
   of that visible state -> full cost, L48 gap 0.08-0.12. NOT a blind test: a partial MoDA gap (0.097 against Sandwich-LN
   0.089 at the same step) was seen on 2026-10-13. The theory is post hoc on all 8 designs; its blind tests are the
   mechanism measurement below and a causal run.
   MoDA result (15341589 COMPLETED 2026-10-14 10:16): L48 train separate heads 3.051, probes 2.948, gap .103 -> inside
   0.08-0.12, as predicted (not blind). Its gap curve has the single-stream shape; its probes model ends 3rd lowest at L48
   (mHC 2.922, HC 2.944, MoDA 2.948, AttnRes (Block) 2.953). Page v62 / site 44cfd0e: Figs 7, 9, 10 complete; MoDA bar
   (grey, one residual stream) added to Fig 11.
   Mechanism test on existing checkpoints (scripts/della/pls_hc_hidden_cpu.py): HC separate heads should hold more of its
   state in the zero-sum stream part than HC probes; mHC little in either, and its static writes all positive.
   Mechanism test result (job 15534022, 2 x 512 c4 tokens): the state-level prediction FAILS. Zero-sum share of the
   stream energy, mean over layers 1-47: HC separate heads 0.26, HC probes 0.59, mHC 0.35 / 0.36 (energy-weighted: every token's squared norms summed before the ratio; the token-averaged share of the later scripts is 0.41 for HC separate heads, the other three agree; audit 2026-10-31). HC's separate-heads model
   keeps LESS of its state out of the readouts' sight, not more (high in layers 1-12, then about 0.05-0.15 from layer 13);
   mHC is not low (unequal positive writes still put a zero-sum part in the streams). At the write level a narrower
   version fits: sublayers whose static write is >80% zero-sum (nearly invisible): HC separate heads 10/96 (layers
   4-14, 19), HC probes 1, mHC 0 and 0; negative write entries 14/96 vs 2/96, mHC 0 by construction. Post hoc again.
   Next (job 15534928): remove the zero-sum part after layer c and measure the layer-48 CE rise (how much later layers
   need what the readouts cannot see).
   Ablation result (job 15534928): every stream replaced by the stream mean after layer c (readout c unchanged), L48 CE
   rise at c = 4/8/12/16/20/28/36/44: HC sep +0.65/+1.12/+1.07/+0.45/+0.24/+0.07/+0.02/0; HC probes +2.67/+1.40/+0.56/
   +1.13/+2.24/+2.38/+0.10/+0.04; mHC sep +0.21/+0.29/+0.22/+0.14/+0.11/+0.06/+0.01/0; mHC probes +0.06/+0.12/+0.13/+0.18/
   +0.20/+0.23/+0.09/+0.10. HC really uses the invisible part as storage (about 5x mHC's dependence in both setups); under
   separate heads HC confines that use to layers 4-16, where its static zero-sum writes sit. mHC barely uses it. So the
   narrower claim survives: the design that both offers and uses readout-invisible storage pays half; the causal training
   test (HC with writes forced positive, mHC with signed writes) is the next step and needs the owner's approval.
   Causal test submitted (owner popup 2026-10-14): ARCH=hc_pos on branch pls-hcpos (worktree marin-pls-hcpos), HC with
   every write 2 sigmoid(offset + dynamic) > 0, identical to hc at init; 48-layer pair chain 15536266 -> 15536267.
   Prediction: L48 train gap 0.048 -> about 0.10, and the ablation dependence drops toward mHC's. Page: new question
   "Why does Hyper-Connections pay only half?" with Figure 11 (bars: layer-48 gap of the 5 single-stream designs, HC,
   mHC, HC with positive writes queued); figures 11-15 became 12-16.
   Same-depth reference (2026-10-14), all done. 36 layers: train 3.016 / eval 3.716, layer 36 of the separate-heads model
   +0.09 / +0.12 above it. Ordinary by depth 12/24/36/48: 3.141/3.057/3.016/2.991; the separate-heads model's layer 48
   (3.093) lies between the ordinary 12- and 24-layer models. 24 layers done: train 3.057 / eval 3.753, layer 24 of the separate-heads model
   +0.07 / +0.10 above it, and it ends below that model's layer 48 (3.093 / 3.827). The ordinary 12-layer model (15519767_12) ends at train 3.141 / eval 3.852; layer 12
   of the 48-layer Sandwich-LN separate-heads model at 3.239 / 3.964, 0.10 / 0.11 higher. So the 48-layer model's layer 12
   is a worse 12-layer model than a dedicated one, as every layer was at 6 layers. Figure 6 grey dots (d48ref.json).
   AttnRes (Full) instability (2026-10-13): of the 14 finished 48-layer runs, only AttnRes (Full) with probes (its
   ordinary model) has layer-48 loss spikes (11 events > 0.3 nats above the 101-step median, from step 630 on; all
   others 0). At step 2106 its last readout went from 3.41 to 8.32 in one step while the probe readouts below did not
   move: the head's depth softmax saturated (grad/norm/transformer.final_query 0.12 -> 3.8e-07, back by step 2300), and
   the run took ~1000 steps to recover. Not at a resume (resumes at 2308, 4632). Before it, the gap (sep - probes, L48
   train) was +0.00..+0.03 (other designs +0.03..+0.07 at the same steps) and the ordinary model was level with AttnRes
   (Block)'s (3.438 vs 3.452 at step 2000). The separate-heads run never spikes. So the missing last-layer cost is at
   least partly the ordinary run's spikes. Test (owner popup 2026-10-13): seed 1 (SEED=1, data order unchanged) pair
   scripts/della/pls_d48_g4_attnres_seed1.sh, jobs 15520124-26; page Figure 8 placeholder (fig-attnres-seed).
   Script: docs/della/pls_archive/status/attnres_params.py (param and grad norms from W&B).
   Declined by the owner (2026-10-13, via the terminal session): a 48-layer ordinary Sandwich-LN model without probes
   (the shared clip group check), a Sandwich-LN 48-layer seed 1 pair, an AttnRes (Full) probes run with a stabilized head
   softmax, and a second width-depth shape. Do not propose them again unless the results call for it.
   MTP and every layer (owner 2026-10-13, DeepSeek-V3 form; page question h-mtp, placeholder fig-mtp): branch pls-mtp
   (worktree marin-pls-mtp, 2cb07e35f4): PLS_MTP_W (one DeepSeek-V3 MTP module on the last layer predicting x_{t+2})
   and PLS_FUT (detached probes on x_{t+2} at every layer), CPU tests scripts/della/pls_mtp_cpu_test.py. Pair job
   15523669: 130m, NTP vs NTP + MTP 0.3, both with x_{t+1} probes (PLS_OPT_SPLIT=1) and x_{t+2} probes.
   MTP result (15523669 COMPLETED 2026-10-14 20:40; final = mean of last 50 steps, train, one seed each):
     x_{t+1} probes L1-L6  NTP 5.008 4.656 4.179 3.949 3.478 3.260 | NTP+MTP 4.983 4.666 4.193 3.939 3.558 3.262
     x_{t+2} probes L1-L6  NTP 6.582 6.321 6.119 5.945 5.659 5.575 | NTP+MTP 6.560 6.325 6.099 5.880 5.584 5.392
     MTP module loss 3.466. NTP's x_{t+1} probes repeat Fig 3's probes (L6 3.257). MTP does not move next-token
     information earlier (L5 +0.08 worse, L6 same); x_{t+2} information grows only in L4-L6 (-0.07, -0.08, -0.18).
     Page Fig 16 filled, question marked Answered. pls-mtp joined pls history by a tree-unchanged merge (5fccaa9692)
     so the scrub maps its commit for the record's Code link.
     Main model at 130m: eval loss 3.9762 / 3.9755, Paloma macro 4.1867 / 4.1880: MTP does not improve the next token.
   MTP across the ladder (owner popup 2026-10-14; page h-mtpl, placeholders fig-mtpl and fig-mtpl-depth): 300m, 520m,
   1_2b, NTP vs NTP + MTP 0.3, trained as the baseline (PLS_W=0, no per-layer eval) on 8 GPUs (NUM_GPUS=8), one seed;
   pls-mtp@5aea76151c, scripts/della/pls_mtp_ladder_submit.sh, pls_h100x8_single.sbatch. Probes after training only
   (owner: probing on the final checkpoint; in-training probes cost ~8-14x a baseline run past 130m because every probe
   is a full-vocabulary head): PROBE_FROM loads a frozen checkpoint strictly and trains fresh next-token heads (layers
   1..L-1) and x_{t+2} heads (layers 1..L); the last layer's next-token readout is the model's own head. 130m point and
   calibration: the Figure 16 checkpoints, 2000 probe steps (15591508-09), against their in-training probes.
   Limits: x_{t+2} probes have no held-out eval (train loss on batches the backbone saw; paired across arms); MTP arm
   costs a few % more FLOPs per token (comparison at equal tokens, as DeepSeek-V3).
   Readout definitions per design: the residual stream h after layer k for the residual designs and MoDA; the sum of the
   4 streams for HC and mHC; for AttnRes the mix that layer k+1's attention reads, built with layer k+1's pseudo-query,
   so in the separate-heads arm readout k's loss also trains that query (audit 2026-10-11).
   Probes at 48 layers: no gradient reaches the backbone, but the 47 probe heads share the output head's gradient
   clipping group (MuonH adamh group, clip 1.0): the output head's gradient is scaled by 0.6-0.8 on most steps; the loss
   effect is unmeasured (audit 2026-10-11; PLS_OPT_SPLIT=1 separates them).

## L6：优化器的实际设置（2026-10-22，Lean 跟进，不用 GPU）

脚本：`scripts/della/pls_opt_effective.py`（CPU，构建真实的优化器类，常数取自 -vs-d48-g4 run 的 W&B config）和
`docs/della/pls_archive/blog/opt_clip_groups.py`（读本地 W&B 文件里每 10 步记录的原始梯度范数）。

1. weight_decay = 0.1 写在 config 里，但没有任何组用到它：0.1 和 0.0 的一步更新逐位相同。用户弹窗 2026-10-26（TREC 会话提出）：只在启动脚本加注释，值保留，W&B config 不变（e284dfec16）。
2. AdamH 组（lm_head 和全部探针/独立头）用 MuonH 的学习率 0.02，不用 adam_lr 0.008。Adam 组（embedding、所有 norm
   增益、探针 norm）和 adam_new（HC/AttnRes 的路由参数）用 adam_lr。48 层：不 warmup，前 20% 恒定，后 80% 线性降到 0。
3. 每组按自己的全局范数裁剪（阈值 1.0）。探针和 lm_head 同在 AdamH 组，探针 norm 和最终 norm 同在 Adam 组。
   MuonH 层和路由参数不受探针梯度影响。eps = 1e-20 时，各步相同的裁剪系数在 Adam 里抵消，只有随步变化的系数起作用。
4. 48 层探针 run（-sep-bbfrozen）：AdamH 组在 86–97% 的记录步上裁剪，不算探针时只有 0.2–1.2%。探针把主 lm_head
   的裁剪系数乘上中位数 0.72–0.76（最小 0.06–0.11）。Adam、MuonH、adam_new 组几乎不变（≤ 1.8% 的步）。
   所以"普通模型 + 探针"的主干只在 lm_head 的步长权重上和普通模型不同。6 层上这个差别使最终 eval loss 从 3.9686
   变为 3.9704（+0.002 nats，单种子，在噪声内）。-os（PLS_OPT_SPLIT）去掉这个耦合；48 层探针 run 和新的
   arch2 run 都没有用 -os，所以它们之间可以直接比较。
5. 48 层独立头 run：HC 的路由参数（adam_new）每一步都裁剪（100%），探针 run 里 0%；Full AttnRes 72% 对 1.4%。
   独立头损失给路由参数的梯度比主损失大得多。

## A1：AttnRes 读法的锐度由 logit 尺度决定（2026-10-22，调研 A1，不用 GPU）

脚本：`docs/della/pls_archive/blog/query_entropy.py`（读 checkup/*.json），图 `query_entropy_plot.py` →
`a1_query_entropy.png`（给所有者的状态图，不上页面）。每个读取 = 一个子层的 attention 或 MLP 输入混合。
源 v 进入 softmax 的打分是 (v·w)/RMS(v) = s·cos(v, w)，其中 s = |w|·√H（w = query ⊙ key_gain；H 我们 512，K3 7168）。
归一化熵 = 每 token 熵的均值 / log(源数)。

1. 调研 C1/C2 比较的是 |w|，没有乘 √H。换成 s 以后，K3 的读法并不比我们软：K3 的 s 中位数 17.7，归一化熵 0.33；
   我们的四个模型 s 中位数 8.5–12.2，归一化熵 0.55–0.89。"K3 query 短，所以读法软"这个说法不成立。
2. 把块源模型（Block 探针、Block 独立头、K3，共 375 个读取）放在一起，s 和归一化熵的 Spearman 是 −0.80，
   |w| 只有 −0.28。s 在 [5,10)、[10,20)、[20,40)、≥40 时熵的中位数是 0.79、0.52、0.30、0.00。
   所以"锐度由打分尺度决定"在换算宽度后成立，K3 也在同一条趋势上。
3. s 不是全部原因：s 在 10–20 时，K3 熵中位数 0.40，我们的 Block 约 0.60。K3 的源方向更分散（推测）。
4. 独立头压低 s、让读法变平：Full AttnRes 探针模型有 22 个 s ≥ 40 的近二值读取（主要是第 15–30 层的 attention），
   独立头只剩 L0.mlp 一个。后三分之一层的 s 中位数 19.1 → 9.0，二值读取（最大权重 > 0.9）11.6% → 1.1%。
   Block AttnRes 的第一块（"死段"）两种都有高 s 读取（第一块里探针 8 个，独立头 9 个；全模型探针 25 个，独立头 12 个；审计 2026-10-31 改正）。
5. 对调研统一假说的影响：用 s 时"打分尺度 → 锐度"成立；"K3 与我们的差别来自权重衰减带来的 query 长度"不成立。
   B2（query 范数 Post Clip）仍可检验因果，但预测要改：它只会改变 s ≥ 40 的那些读取。

## L1：增益起点与步长各决定约一半的增益差距，第 48 层损失不变（2026-10-22，wq w15）

运行：Pre-LN，两个 pre-norm 增益从 LNS 的 1/√ℓ 开始（`GAIN_INIT=lns`），只加探针，48 层，8 GPU，全局 batch 与对照相同。
对照：`-sep-bbfrozen-apreln-vs-d48-g4`、`-sep-bbfrozen-alns-vs-d48-g4`。数据：`fetch_d48g4.py arch2:preln-gilns`、`fetch_gains.py`
（gains.json）、状态图 `l1_gain_plot.py`。

1. 第 48 层不变：探针最终训练损失 2.978（Pre-LN 2.983，LNS 2.996）；eval 3.666（Pre-LN 3.668，LNS 3.682）。差值在种子噪声内
   （AttnRes 两个种子第 48 层差 0.09）。
2. 中间层大部分移向 LNS：最终探针损失减 Pre-LN，第 20 层 −0.393（LNS −0.527），第 32 层 −0.260（LNS −0.319），第 44 层 −0.029
   （LNS −0.023）。
3. 训练后的平均 pre-norm 增益（相对 Pre-LN）：起点 1/√ℓ 的运行 0.45–0.92，第 42–48 层约 0.50；LNS 换算后 0.23–0.86，深层约 0.25。
   按对数计，起点解释差距的 0.33–0.83，第 25–48 层约一半。所以起点和步长各决定约一半。
4. 旧数字 “Pre-LN 的增益是 LNS 换算增益的 2.1–3.5 倍” 是逐层中位数（resnext-lean `ArchLean/PracLnsProfile.lean`）。逐层看，普通模型（探针）
   两个增益平均后的比值是 1.2–4.4（第 2 层最低，第 42 层最高）；独立头模型上 ln_1 是 1.7–5.9，ln_2 是 1.2–2.2。Fig 36 图注不再引用这个数字。

### L1 完整版：Pre-LN 精确等价于 LNS（2026-10-22，wq w61、w62；已完成，结果见「L1 完整版结果」）

起因：所有者说“所以没有完全回答对吧，你得做到和理论一样”。w15 只取了 LNS 的起点。Lean 的精确重参数化（`ArchLean.PracTrain`）还要求步长和
梯度裁剪也一致。条件：Pre-LN 的增益 g' = c_ℓ·g，c_ℓ = bf16(1/√ℓ)（LNS 前向中的值）。

1. 起点：g' 从 c_ℓ 开始（`GAIN_INIT=lns_bf16`）。
2. 梯度：进入 adam 组的 clip_by_global_norm 和 Adam 之前，增益梯度乘以 c_ℓ。
3. 更新：Adam 之后，增益更新再乘以 c_ℓ。Adam ε = 1e-20，所以 Adam 对每个坐标的缩放不变。weight decay 不用于增益。

代码：`LnsStepMuonHConfig`（`depth_arch_qwen3.py`，`GAIN_STEP=lns`），pls dabf9aaff6。CPU 测试 `scripts/della/lns_equiv_test.py`：4 层，宽 48，
40 步，max_grad_norm 0.05（每组都裁剪）。和 LNS 的最大差值：

| 运行 | float64 loss 差 | float64 参数相对差 | float32 loss 差 |
|---|---|---|---|
| 完全等价（起点 + 步长 + 裁剪） | 0 | 4.9e-14 | 2.3e-5 |
| 只取起点（w15 的设置） | 2.4e-2 | 1.5e-1 | 6.0e-2 |
| 只取步长和裁剪 | 4.4e-1 | 8.7e-1 | 4.5e-1 |
| 起点 + 更新乘 c_ℓ，裁剪看 Pre-LN 的梯度 | 3.6e-4 | 1.3e-3 | 1.5e-4 |

float32 下完全等价的运行在前 10 步差值为 0，之后舍入误差开始累积。第一次 float64 运行有部分参数仍是 float32，差值 1.9e-6；全部参数转为
float64 后差值为 0。裁剪的耦合很小：只差裁剪的运行偏离 3.6e-4，比只取起点小约 70 倍。

GPU 运行（8 GPU，20 h，`--partial`）：w61 `arch2-preln-lnsequiv`（`-apreln-gilnsbf-slns-vs-d48`），预测与 LNS 只差 bf16 舍入的影响，
即与 LNS 的差在种子噪声内；w62 `arch2-preln-slns`（`-apreln-slns-vs-d48`），直接测量步长的份额。Fig 36 已加占位（所有者弹窗 2026-10-22）。

## A2：Full AttnRes 最大的源（2026-10-22）

脚本：marin-pls-auto10 `scripts/della/attnres_massive.py`。数据：paloma c4_en validation，文本 A 16×1024 token（计分），文本 B 另 16×1024
token（求均值）。输出：`tmp/pls_status/attnres_massive/{full,full-s1,full-sep,full-sep-s1}.json`，状态图 `pls_archive/blog/a2_massive_plot.py`。
对每个模型取每 token RMS 均值最大的源（子层输出），在写入时替换它。

| 模型 | 最大的源 | 相对中位数 | 与均值的 cos | RMS 变异系数 | 换成文本 B 均值 | 置零 | 每 token 缩放到平均 RMS |
|---|---|---|---|---|---|---|---|
| 普通模型（探针）seed 0 | L44.mlp | 52,810× | 1.000 | 0.05 | +0.0000 | +8.95 | −0.0000 |
| 普通模型（探针）seed 1 | L47.mlp | 1,511× | 0.961 | 0.57 | +0.0014 | +0.0018 | +0.0008 |
| 独立头 seed 0 | L22.mlp | 26× | 0.234 | 2.43 | +0.0080 | +0.0083 | +0.585 |
| 独立头 seed 1 | L48.mlp | 33× | 0.910 | 11.5 | +0.0000 | +0.0000 | +0.0000 |

1. seed 0 的巨大源是一个常数偏置：每个 token 都写同一个向量，换成均值不改变损失，置零损失增加 8.95 nats。读它的是第 48 层 MLP，
   这一层的输出也是常数（与均值的 cos 1.000）；把 L48.mlp 输出换成均值，损失不变。
2. 这个源只属于 seed 0。seed 1 最大的源小 35 倍，由输出头读取，置零只增加 0.0018 nats。所以“巨大常数源”不是 Full AttnRes 的结构
   性质，是一个种子学到的偏置通路。
3. 独立头模型两个种子都没有巨大源（最大 26× 和 33×）；普通模型两个种子是 52,810× 和 1,511×。所以独立头损失让最大的源小 50 到 2,000 倍
   （每组两个种子）。seed 0 的最大源只在少数 token 上很大（p90 的 cos 0.99，中位数 0.09）：值本身不重要（置零 +0.008），但把每个
   token 缩放到平均大小的代价是 0.585 nats，信息在“哪些 token 写得大”里。seed 1 的最大源是 L48.mlp，只由输出头读取，任何替换都不改变
   损失。

## A3：结构变体与 HC 深度扫描（2026-10-23，wq w76–w83；已完成，结果见后面各节和「2026-10-31 审计补录」）

起因：所有者弹窗（2026-10-22）批准四个跟进实验，并要求本项目同时排队的作业不少于 4 个。代码 pls 436fdb2e3c，CPU 测试
`scripts/della/depth_arch_variant3_test.py`，作业脚本 `scripts/della/pls_arch3_fetchable.sbatch`（按作业名选设置；每个实验都跑
独立头 `-sep` 和普通模型加探针 `-probes` 两次，各 4 GPU，`--partial`）。独立头代价 = 独立头 run 最后一层最终训练损失减探针 run 的。

| 实验 | 作业 | 对照的代价 | 检验 |
|---|---|---|---|
| mHC，残差混合固定为 I（`MHC_RES=identity`，tag -mhcid） | w76 sep，w77 probes | mHC 0.102 | mHC 付全价是否因为双随机乘积随深度坍缩到低秩 |
| Full AttnRes，伪查询 Post Clip，s = ‖q⊙g‖√d ≤ 10（`AR_QCLIP=10`，tag -qclip10） | w78，w79 | seed 0 −0.021，seed 1 +0.061 | 近二值读取（s ≥ 40）是否只是温度效应；它是否影响代价 |
| HC 24 层 | w80，w81 | 48 层 0.048，6 层 0.125 | 半价从多深开始出现 |
| HC 12 层 | w82，w83 | 同上 | 同上 |

1. 批准的第五项“Full AttnRes 加空槽（Softmax₁）”在提交前去掉了（独立审查发现）。Softmax₁ 的权重等于普通 softmax 的权重乘以每个 token
   一个常数 c = S/(S + e^{−max})。AttnRes 的每个读取后面都接 RMSNorm（子层的 ln_1/ln_2，读出的 final norm），RMSNorm 把 c 消掉，所以模型
   计算的函数不变（只差 norm 的 eps）。源本身是分支输出，空槽碰不到它们。要让“什么都不读”起作用，空值必须改变归一化之后的结果（例如可学的
   空值向量，或读取加回一个不归一化的残差流）。
2. Post Clip（kexue.fm/archives/11729）：每步更新之后，若 s > τ，把 q 乘以 τ/s，方向不变，键增益 g 不裁剪。源的 logit = s·cos(v, q⊙g)，
   所以 τ 限制每个 logit 的绝对值。τ = 10：独立头模型后三分之一层 s 的中位数是 9.0，普通模型 19.1（A1）。预测：普通模型的二值读取
   （最大权重 > 0.9）从 11.6% 降到接近独立头的 1.1%，最后一层损失变化在种子噪声内。
3. mHC identity：残差 logit 的 phi 列、增益和偏置得到的梯度恰好为 0（测试 1），Adam（eps 1e-8）不更新它们。
4. HC 深度扫描用与 48 层相同的宽度、数据和步数（LAYERS=12/24，`-vs-d12`/`-vs-d24`）。比较用的其他结构只有 48 层的数据。
5. 网页 Fig 38（`fig-arch-var`）和 Fig 39（`fig-hc-depth`）加了占位（所有者弹窗 2026-10-22，“两张图”），已有的点已画出。
6. 裁剪检查（2026-10-23，`tmp/pls_qscale/pls_qscale.py`，读最新的临时检查点）：w78（独立头，step 299）96 个 query 的 s 最大
   10.0000，中位数 8.39，39 个在上限；读出头 s = 8.55。w79（普通模型，step 333）s 最大 10.0000，中位数 10.0，68 个在上限；读出头 s = 10.0。
   裁剪在真实训练里生效，而且普通模型的大多数读取顶在上限。每步 34.6 s，与不裁剪的 AttnRes 运行（34.7 s）相同。

## E1：本质是什么——读出是否看到后面各层用的全部状态（2026-10-23，wq w87–w90；已完成，结果见「E1 中途结果」及图 40）

起因：所有者说“一定要往本质思考，本质上到底是为什么这些设计上的差异产生了结果上的差异”（已写入 CLAUDE.md）。

第 48 层最终训练损失（d48g4.npz，单种子；代价 = 独立头 − 探针）：

| 结构 | 独立头 | 普通（探针） | 代价 | 普通模型第 36 层 |
|---|---|---|---|---|
| Sandwich / Pre-LN / LNS / KEEL | 3.07–3.13 | 2.98–3.03 | 0.089–0.102 | 3.22–3.40 |
| DeepNorm | 3.372 | 3.262 | 0.110 | 3.75 |
| HC / hc_pos | 2.992 / 2.989 | 2.944 / 2.954 | 0.048 / 0.035 | 3.007 / 3.021 |
| mHC | 3.024 | 2.922 | 0.102 | 3.61 |
| AttnRes (Full) s0 / s1 | 3.023 / 3.016 | 3.045 / 2.955 | −0.021 / +0.061 | 3.61 / 7.0 |
| AttnRes (Block) | 3.066 | 2.953 | 0.113 | 4.24 |
| MoDA | 3.051 | 2.948 | 0.103 | 3.31 |

1. 五种单流结构的归一化各不相同，代价却都是 0.09–0.11（DeepNorm 两条曲线一起高 0.27）。所以归一化不是本质；它们的共同点是只有一条流，
   读出看到全部状态。
2. 独立头模型的第 48 层落在普通 12 层与 24 层模型之间（Sandwich 3.093，对 3.141 / 3.057）：代价本质上是浪费的深度，一半的层只能在
   “答案空间”里计算。
3. 候选本质（10-14 的“隐藏写入”理论的一般形式）：逐层损失给每层“可见状态”加了一个信息瓶颈。词表 128k ≫ 宽度 512，读出头没有零空间，
   一个给后面层用、但还不是答案的特征会提高这一层的读出损失，所以被压掉。结构的代价取决于它有多少后面层能读、读出看不到的状态。
   HC 的普通模型在第 36 层已可解码（3.007），且大量使用流之差存储（消融 +2.2 到 +2.7 nats）：它本来就把“答案”和“草稿”分开，独立头几乎
   不用改它。
4. 还站不住的地方：hc_pos 的因果检验没有去掉看不到的状态（有符号混合仍产生流之差）；mHC 独立头模型很好（3.024），代价高是因为普通模型
   特别好；AttnRes 两个种子的普通模型差 0.09，独立头只差 0.007，代价的种子噪声可能和设计间的差别一样大；HC 在 6 层付全价（0.125）。
5. 直接检验（所有者弹窗 2026-10-23，pls 481766d849，`PerLayerQwen3Config.pls_visible_frac`，`PLS_VIS`，CPU 测试 `scripts/della/pls_vis_test.py`）：
   Pre-LN 48 层独立头，中间层读出只看前 f 的坐标（其余在 RMSNorm 前置 0，梯度恰好为 0），最后一层看全部；f = 1/2（w87）、1/4（w88）。
   对照为现有 Pre-LN 探针运行。预测：f 越小代价越低；若代价不变，此解释错误。噪声：Pre-LN seed 1 独立头（w89）和探针（w90）。
   网页 Fig 40（`fig-vis`）占位。所有者选择保留 query 裁剪（w78/w79）。

## L2 与 A3 中途结果：静态路由；AttnRes 普通模型不稳定（2026-10-24）

1. L2，HC 静态路由（wq w27 完成，普通模型加探针）：第 48 层最终训练损失 2.936，动态路由 2.944，Pre-LN 2.983。静态路由
   保留了 HC 相对 Pre-LN 的全部收益。第 36 层：静态 3.055，动态 3.007，所以动态项让中间层更早可读。网页 Fig 37 的 HC 面板已填。
2. 第 48 层损失尖峰（d48g4.npz，某步损失比前 51 步的中位数高 0.3 nats 以上）：
   | 运行 | 尖峰步数 | 最大超出 | 最终第 48 层 |
   |---|---|---|---|
   | AttnRes 普通模型 种子 0 | 38（第 630、881、1032、1240、1613、1828、2106–… 步） | 7.07 | 3.045 |
   | AttnRes 普通模型 种子 1 | 5（第 485、626、818、1024、2578 步） | 10.43 | 2.955 |
   | AttnRes 独立头 种子 0 / 1 | 0 / 0 | 0.10 | 3.023 / 3.016 |
   | Pre-LN、HC、mHC 普通模型 | 0 | 0.11 | 2.983、2.944、2.922 |
   AttnRes 普通模型的两个种子差 0.09，独立头只差 0.007：差别来自普通模型的尖峰（种子 0 在第 2,106 步附近的尖峰之后约 1,000 步才回到原来的水平）。
   所以 AttnRes 种子 0 的负代价（−0.021）来自不稳定的对照，不是独立头的好处。种子 1 的代价 +0.061 更接近其他结构。
   独立头的逐层损失让 AttnRes 训练稳定（A1：独立头的读取 s 中位数 9.0，普通模型 19.1）。
3. A3 query 裁剪（w79，普通模型，s ≤ 10）：第 456、520 步有尖峰，第 659 步发散：第 48 层损失从 3.75 跳到 11.0（第 660 步 8.29），第 720 步 7.62，
   探针（第 1、24 层）随后也上升。裁剪不能阻止尖峰，所以尖峰不是由尖锐读取（s ≥ 40）引起的。w79 在队列里，会从发散后的检查点接着训练。
   独立头的 w78 到第 1,386 步没有尖峰。

## A4：AttnRes 的源归一化（2026-10-24，wq w122 独立头、w123 普通模型加探针）

起因：所有者弹窗（2026-10-24）批准。A2 测过：普通模型最大的源是中位数的 52,810 倍（种子 0）和 1,511 倍（种子 1）；独立头模型
只有 26 倍和 33 倍。尖峰数目也按同样的顺序排列：种子 0 有 38 个，种子 1 有 5 个，独立头为 0。w79 已经说明尖锐读取不是原因
（所有者弹窗批准取消 w79，记录表标为已停止）。

解释：一个源比其他源大上万倍时，分给它的读取权重只要稍有变化，读取就被这个源占满。普通模型只有最后一个读取计入损失，所以能
容许这样大的源；独立头的每个读取都计入损失，所以不能容许。

检验只改一个性质：每个源在每次混合时归一化到单位 RMS（`AR_VNORM=1`，tag -vnorm，`depth_attnres_vnorm`），读取权重不变，
不加参数。代码 c8dd4cd25a。测试 `scripts/della/pls_vnorm_test.py`（扫描与循环两条路径一致，嵌入 RMS 0.02 时也一致；把一个源放大
1,000 倍，读取不变）。审查发现第 0 层的输入在扫描路径里没有归一化，已改正。

预测：普通模型没有尖峰（按上面同样的标准：比前 51 步中位数高 0.3 nats）；第 48 层最终训练损失不高于种子 1 的 2.955；独立头
代价在其他结构的 0.06 到 0.11 之间。注意（审查）：归一化去掉了源大小对读取的影响，源本身仍可以长大，所以要看的是尖峰，不是源的比值。
网页 Fig 41（`fig-ar-vnorm`）加了占位。

## 速度：无害的加速（2026-10-24，owner 经 terminal 转达；分支 pls-speed，worktree marin-pls-speed）

- 要求：找出能加速的设置，只上线无害的：数值不改变比较（逐位相同，或在匹配 A/B 里落在种子噪声内）；不就地改运行中和排队的作业；不加参数，不改配方。
- 搜索：ultracode workflow（9 个 agent：4 个找、4 个反驳、1 个写计划），计划和全部候选在 `tmp/pls_speed/plan.md`、`results.json`。
- 测得的成本（日志）：Pre-LN d48 每步 15.2 s，其中 47 个额外读出 12.3 s（81%）；读出受 f32 logits 的显存带宽限制。HC d48 在 microbatch 1 每步 25.4 s；AttnRes d48 34.5 s。jaxpr 和 HLO 文本日志每个程序 2–2.5 min（GPU 空闲）。
- 已发现的问题（不是速度）：
  1. resume 段的 smoke 和第一段用同一个 run id，W&B 丢掉第 0–29 步，所以 train-path 闸门在每次 resume 读的是第一段的 summary（w28 三段都是 7.4239）。已修：smoke 标签加上宿主作业 id（`684b48ead8`）。
  2. 读出 CE 的反向有两条路：microbatch 1 的设计（hc、mhc、attnres、moda）共享内存不够，退回 XLA 路径（f32 delta、TF32 矩阵乘、梯度在 bf16 累加）；PDP=4 的设计（preln、sandwich、lns 等）走 b-tiled 路径。同一设计的独立头和探针走同一路径，所以每个设计的代价不受影响；不同设计之间的代价比较多了一个 CE 梯度精度的差异。
- 候选（已实现为开关，默认关）：
  - A1 `LEVANTER_CE_BTILED_ONEHOT=1`：b-tiled 反向里标签项用 select 代替 scatter，让 XLA 把 exp、乘法、标签项和 bf16 转换融合成一个 kernel。CPU 上在真实宽度和词表（B 6000、V 128256，两个 b 块，有和无 soft cap）逐位相同。估计 PDP≥2 的 run 快约 10%。GPU A/B：pli-cp 16146325。
  - A4 `LOG_JAXPR=0`：不写 jaxpr 和 HLO 文本；编译出的程序相同。每段省约 4 min。没有脚本读这些文件。
  - A7 `EVAL_PDP=4`：microbatch 1 的设计用每卡 4 条序列做 eval，训练不变。A/B：pli-cp 16146461（同样 640 条序列）。
- 推迟：A3（截止前存 checkpoint，再把间隔改为 30 min，约 1.5–2%）、A5（一次 eval，约 0.9%）、A2（探针去掉无用的 dX 矩阵乘，约 6%，改数值，要 1100 步 A/B）。
- 不上（改数值或风险大）：hc/mhc 改 microbatch 4 加 vmm 分配器（约 30%，但换 CE 路径，和已完成的对照不可比）、AttnRes 快速 CE 反向、融合 Pallas CE、latency-hiding scheduler。
- 结束（owner 10-24，经 terminal：「tokenizer size和pls这两个调研完了就还是继续用老代码跑吧先不用做切换和测试，只是把经验都告诉nanomarin」）：pli-cp 上的 4 个测速作业在开始前取消（owner 弹窗同意）；pls 的作业继续用老代码，只保留 smoke 闸门修复。全部结论交给 nanomarin：`marin-pls-speed/docs/della/pls_speed/HANDOFF.md`，分支 pls-speed（archive）。

## A3：HC 12 层（2026-10-24，wq w82、w83 完成）

- HC 12 层：独立头最后一层训练损失 3.207，探针 3.085，代价 0.121。6 层是 0.125，48 层是 0.048。所以较低的代价在 12 层以后才出现；24 层等 w81（探针，已到 4.90k/4.96k 步，排队续跑）。Fig 39 已填 12 层的点，折线只连相邻的已测深度。

## L2：AttnRes 静态路由（2026-10-24，wq w28 完成）

- AttnRes (Full) 静态路由的普通模型（加探针）：第 48 层最终训练损失 2.913，0 个尖峰（最大超出 0.10 nats）。动态路由：种子 0 是 3.045（38 个尖峰），种子 1 是 2.955（5 个）；Pre-LN 2.983；HC 静态 2.936。
- 所以尖峰需要依赖 token 的路由；去掉依赖 token 的部分，AttnRes 从最不稳定变成这组普通模型里最低的。中间层探针（L12/L24/L36）静态比动态高（4.90/4.80/4.12 对 4.71/4.18/3.61）。
- 下一步（CPU，vis1）：静态模型有没有巨大的源（`tmp/pls_static_src/src_rms.py`，用模型自己的循环路径记录每个源的 RMS；同时重跑动态种子 1，必须得到 A2 的 1,511×）。有：尖峰需要“巨大的源”和“依赖 token 的读取权重”同时存在。没有：巨大的源本身是依赖 token 的路由造成的。
- 源的大小（2026-10-24，CPU，`scripts/della/attnres_src_rms.py`，结果在 `tmp/pls_static_src/{static,dynamic-s1}.json`）：用模型自己的循环路径记录每个源写入时的 RMS（Paloma c4_en，16×1024 token，同 A2）。方法检查：动态种子 1 得到和 A2 相同的源 RMS（L47.mlp 1161.0，L43.mlp 597.1；相对中位数 1,525×，A2 是 1,511×，只差在中位数）。
  | 模型 | 最大的源 | 相对中位数 | 尖峰 |
  |---|---|---|---|
  | 普通模型，动态路由，种子 0 | L44.mlp | 52,810× | 38 |
  | 普通模型，动态路由，种子 1 | L47.mlp | 1,525× | 5 |
  | 普通模型，静态路由（w28） | L5.mlp | 10× | 0 |
  | 独立头，动态路由，种子 0 / 1 | L22.mlp / L48.mlp | 26× / 33× | 0 |
- 所以巨大的源只在读取权重依赖 token 时出现。A2 的数据给出它被怎样读：种子 1 的巨大源（RMS 1161）由输出头以权重约 1.3e-5（p10–p90 0.85e-5 到 1.6e-5）读取，贡献约 0.015 RMS；种子 0 的巨大常数源（RMS 43,407）由第 48 层 MLP 以 0.68 读取。前者是一个几乎关上、开度随 token 变的门，后面是一个巨大的向量：logit 稍有上升，权重按指数放大，读取就被这个向量占满。这是尖峰的一个候选机制；A4（w122/w123，每个源以单位 RMS 进入混合）直接检验它。
- HC 24 层（w80/w81 完成）：独立头 3.095，探针 3.018，代价 0.077。HC 的代价按深度：6 层 0.125，12 层 0.121，24 层 0.077，48 层 0.048。到 12 层为止 HC 付全部代价；从 12 层到 48 层，代价随深度逐渐下降。其他结构在 48 层是 0.09 到 0.11。E1（w87/w88：Pre-LN 的读出只看一部分坐标）检验这个差别是不是来自读出看不到的状态。

## E1 中途结果：读出只看一半坐标（2026-10-24，wq w87 完成）

- Pre-LN 独立头，中间层读出只看前 1/2 的坐标：第 48 层最终训练损失 3.089；Pre-LN 独立头（看全部）3.073；Pre-LN 探针 2.983。代价 0.105，看全部时 0.089。所以读出只看一半坐标时代价没有降低。种子噪声要等 w89/w90，1/4 要等 w88。
- 如果 1/4 也不降，"读出看到全部状态"就不是代价的本质；HC 随深度下降的代价要另找原因（例如多条流本身，而不是读出看不到的部分）。
- 1/4（w88 完成）：第 48 层 3.113，代价 0.130。三个点：看全部 0.089，1/2 0.105，1/4 0.130。代价随读出看到的坐标减少而上升，和 E1 的预测相反（种子噪声等 w89/w90）。中间层读出的损失也同向上升（L12：3.24、3.291、3.377；L36：3.080、3.119、3.192）。
- 所以“读出看到全部状态”不是代价的来源。更可能的是：读出越难把这一层的状态解码成答案，它的损失越大，它对共享状态的推力越强，最后一层付得越多。HC 的读出只看 4 条流的和，按这个方向应当付得更多，但它在深层付得更少；HC 的原因要另找。
- 下一个检验（CPU，已训练的 48 层独立头检查点，不用 GPU）：在每层的残差流上，比较中间层损失之和的梯度与最后一层损失的梯度（范数比、cos），按结构和深度。如果代价来自两者的冲突，HC 深层的冲突应当更小，Pre-LN 看 1/4 的冲突应当更大。

## L1 完整版结果（2026-10-25，wq w61、w62 完成）

- w61（Pre-LN，取 LNS 的起点、步长和裁剪）：每层增益与 LNS 重合（相对 Pre-LN：L1/12/24/36/48 为 0.559/0.515/0.368/0.264/0.277，LNS 0.552/0.489/0.389/0.274/0.274），探针损失也重合；第 48 层 2.997，LNS 2.996。理论预测成立。
- w62（只取步长和裁剪，增益从 1 开始）：增益 0.71–1.37 倍 Pre-LN，没有向 LNS 移动；第 48 层 2.991。
- 所以只改步长不降低增益，只改起点走一半（w15），两者一起才到 LNS。第 48 层五次运行都在 2.978–2.997。Fig 36 已填。

## 梯度冲突（2026-10-25，`scripts/della/pls_conflict.py`，CPU 测试 + wq w145/w146 GPU）

- 同一个独立头检查点，用两个只差 `pls_detach_backbone` 的配置求骨干梯度：独立头得到 g_fin + g_int，探针得到 g_fin；g_int = 两者之差。
- Pre-LN 48 层（CPU，1×256 token）：|g_int|/|g_fin| = 41，cos = 0.957。每层的比值约等于这一层之后的读出个数（L1 43，L47 1），cos 0.94–0.999。中间层损失和最后一层损失方向几乎一致；它们主要决定更新的大小，使更新方向偏离最后一层损失的方向约 17°。
- w145（Pre-LN 看全部、1/2、1/4，种子 1）和 w146（HC 48/24/12 层，mHC）在 GPU 上用 8×1024 token 算同样的量。

## 新作业（所有者弹窗 2026-10-25，队列为 0 时）

- w147–w150：Pre-LN 12 层、24 层，独立头和探针（Fig 39 加 Pre-LN 折线）：代价随深度下降是不是 HC 独有。
- w151、w152：HC、AttnRes 静态路由的独立头（探针是 w27、w28；Fig 37 标签写两种路由的代价）。
  - HC 静态路由（w151 完成，2026-10-27 03:32）：独立头第 48 层 3.011，探针（w27）2.936，代价 0.074。动态路由的 HC 是 0.048，Pre-LN 0.089。去掉随 token 变化的路由，HC 的代价从 Pre-LN 的约一半升到约五分之四（0.83 倍）：低代价一部分来自随 token 变化的门，一部分来自多条流本身（静态路由仍有 4 条流和固定的读写）。Fig 37 的 HC 面板标签已更新（站点 76dbb28，artifact v110）。AttnRes 静态路由（w152）还在跑。
- 结果（w145/w146，2026-10-25；因脚本默认 JAX_PLATFORMS=cpu，实际在 CPU 上跑完，8×1024 token；已改为 cuda）。偏角 = 独立头更新方向 g_fin+g_int 与 g_fin 的夹角：

  | 检查点 | 代价 | \|g_int\|/\|g_fin\| | cos(g_int, g_fin) | 偏角 |
  |---|---|---|---|---|
  | Pre-LN 看全部，种子 0 / 1 | 0.089 / – | 41.5 / 40.5 | 0.949 / 0.958 | 18.0° / 16.3° |
  | Pre-LN 看 1/2 | 0.105 | 39.9 | 0.901 | 25.2° |
  | Pre-LN 看 1/4 | 0.130 | 39.9 | 0.854 | 30.7° |
  | mHC 48 | 0.102 | 36.9 | 0.932 | 20.7° |
  | HC 12 / 24 / 48 | 0.121 / 0.077 / 0.048 | 9.2 / 19.9 / 36.8 | 0.903 / 0.925 / 0.899 | 23.1° / 21.2° / 25.3° |

- 在 Pre-LN 里，偏角随读出看到的坐标减少而增大（18° → 25° → 31°，两个种子相差 1.7°），代价同向增大：读出看得越少，中间层损失把更新方向拉得越偏。跨结构不成立：HC 的偏角不比 Pre-LN 小（21°–25°），代价却随深度下降到 0.048。所以 HC 的低代价不在梯度方向，而在同样的偏离对最后一层损失的伤害更小（例如更新落在最后一层读出不敏感的方向）。下一步：沿 g_sep 方向走一小步，测最后一层损失的变化（一阶：-η g_fin·g_sep），比较 HC 与 Pre-LN。
- 注意：这些量都在训练结束的检查点上测，是终点状态，不是训练过程。
- 种子 1（w89/w90 完成，2026-10-25）：Pre-LN 看全部，独立头 3.075，探针 2.983，代价 0.092；种子 0 是 0.089。种子噪声约 0.003，远小于 1/2（+0.016）和 1/4（+0.040）的上升。E1 结论成立：读出看得越少，代价越高。Fig 40 完成。
- 新作业（所有者弹窗 2026-10-25 13:40）：w178/w179 HC 48 层种子 1（独立头、探针）；w180 Pre-LN 读出只看 1/8。
- 1/8（w180 完成，2026-10-26）：第 48 层 3.142，代价 0.159。四个点：看全部 0.089，1/2 0.105，1/4 0.130，1/8 0.159。代价随读出看到的坐标减少单调上升，每减半多 0.016、0.024、0.030 nats，越往后增得越多。中间层读出损失同向上升（L12：3.24、3.291、3.377、3.518；L36：3.080、3.119、3.192、3.330）。Fig 40 的第四个点已上网页（站点 101d172，artifact v109）。

## HC 的隐藏份额（2026-10-25，`scripts/della/pls_hc_hidden.py`，vis1 CPU，4×1024 token）

- 问题：HC 有 4 条残差流，读出只看 4 条流的和。HC 是不是把后面层要用的特征放在读出看不到的部分（每条流减去流均值），所以独立头的代价低？
- 量：每层每个 token，隐藏份额 = Σ_i |h_i − s/m|² / Σ_i |h_i|²（s 是流的和，m = 4）。0 表示 4 条流相同；0.75 表示流的和为 0。
- 预测：独立头模型的隐藏份额比普通模型（探针）大。

  | 检查点 | 独立头：平均 / 下半 / 上半 | 普通模型：平均 / 下半 / 上半 |
  |---|---|---|
  | HC 48 层 | 0.40 / 0.50 / 0.31 | 0.58 / 0.60 / 0.56 |
  | HC 24 层 | 0.34 / 0.46 / 0.21 | 0.60 / 0.54 / 0.66 |
  | HC 12 层 | 0.40 / 0.55 / 0.25 | 0.40 / 0.46 / 0.34 |

- 结果与预测相反：独立头让隐藏份额变小，主要在上半部分的层（48 层：0.56 → 0.31）。独立头要求每层的流的和能直接预测下一个 token，所以把内容从隐藏部分挤到读出看得到的部分。“把特征藏起来”不是 HC 代价低的原因。
- 另一个观察（d48g4.npz，普通模型的探针损失）：HC 普通模型在中间层已经能很好地读出（48 层的第 24 层探针 3.365，Pre-LN 4.154，mHC 4.380）。上半层探针与最后一层的平均差：HC 0.12，HC_pos 0.20，LNS 0.26，Sandwich 0.33，Pre-LN 0.48，mHC 0.70。HC 两个变体的差最小，代价也最小；其余结构的差从 0.26 到 1.25，代价都在 0.09–0.11，不随差变化。所以这不是一条通用规律，只说明 HC 的普通模型本来就接近“每层都能读出”。
- 优化器注记：MuonH 对每个矩阵的更新先做 Newton-Schulz 正交化，再按参数范数定步长，所以梯度的大小不影响矩阵参数的更新，只有每个矩阵内的方向起作用。|g_int|/|g_fin| = 41 的意义是：在下层的矩阵里，更新方向几乎完全由中间层损失决定。
- 新作业（所有者弹窗 2026-10-25 16:30，队列降到 3 时）：w194/w195 HC 48 层 1 条流（独立头、探针），w196/w197 HC 48 层 2 条流。1 条流的 HC = Pre-LN 加上 HC 随输入变化的标量门。代价回到约 0.09 → 低代价来自多条流；仍约 0.05 → 来自标量门。启动开关 `HC_STREAMS`（run id 标签 `-hcs{m}`），CPU 测试 1、2 条流的损失和梯度都是有限值。 结果放在新图 Fig 43（所有者选“新图”）。

## 速度：应上尽上（2026-10-25，所有者经 marin.cpp 和 terminal 转达）

- 所有者改了 10-24 的决定：「建议是修改，我纠正我的指令」「应上尽上，尽可能快！」，又说「可以再考虑底层，比如kernel」「你可以和另一个pretrain的session那边深度协同一下共同推进」。测试 job 改走 wq，优先级 10，不再用 pli-cp（terminal 转达，20:30）。
- 约束：研究 run 都要和已完成的老 run 比较，所以只有结果逐位相同的加速能用在研究 run 上。会改变结果的加速只能用在新的一组对照上（两边都用）。
- 分工（与 Tokenizer Size session）：pls 负责 CE；它负责 attention、remat、启动开销和 run 层面的 infra。每个改动一个 commit，可以单独 cherry-pick，两边互相 A/B。
- 已上线（计算不变；运行中的 job 在下一段开始时用上）：
  1. 合并 pls-speed 的开关，全部默认关（f40fafa15d）。
  2. A3：每段在训练时限前 150 s 存一个临时检查点（`LEVANTER_CKPT_DEADLINE`），临时检查点间隔 30 min（`CKPT_MIN=30`）；A4：不写 jaxpr/HLO 文本（`LOG_JAXPR=0`）（b7d1fe023d）。CPU 测试 `tmp/pls_speed/deadline_test.py`。
- 8 GPU：只适用单流结构。HC 和 Full AttnRes 在 PARAM_MAP=vocab 下用 8 GPU 会在第一个 AllToAll 卡住并中止（wq w16/w17，10-22）。HC 现在每步 25 s，因为它的 microbatch 是 1；早先 16 s 的 run 用 microbatch 4，第 257 步 OOM。
- CE 的账（Pre-LN d48，每个读出、每块 GPU、每个 8192 行的块）：前向的 f32 logits GEMM 写 4.2 GB、logsumexp 再读 4.2 GB；反向重算 bf16 logits、算 dlogits、两个 GEMM。共 4 个 logits 大小的 GEMM。隐藏宽度 512 时，每个 logit 元素的 GEMM 是 1024 FLOP，对应 2–18 B 的显存流量，所以 GEMM 的个数和流量一样重要。
- 新改动（默认关）：
  1. 前向算梯度（`LEVANTER_CE_GRAD_IN_FWD=1`，b-tiled 路径；`batched_xla._full_vocab_b_tiled_forward_with_grads`，`models/loss.py` 的载体）：均值损失的余切是 weight / sum(weight) 乘一个标量 c，所以前向就能算出梯度；每块用一个 logits GEMM 同时算损失和梯度，省掉反向的重算 GEMM。梯度经一个值为 0 的“载体”传回（每个样本一个 0，调用方求和，它的余切正好是 c），所以损失值不变，dW 的 reduce-scatter 也不变。c 是 2 的幂时（权重为 1 的 run 都是 c = 1）应逐位相同。不求导时（eval）走原来的前向。
  2. xla 路径（microbatch 1：hc、mhc、attnres）的标签项改成 select（`LEVANTER_CE_XLA_ONEHOT=1`），和 A1 同理。这条路径的 CE 约占 HC 每步 25 s 中的 14 s（反向的两个 GEMM 吃 f32 delta）。CPU 测试（`tmp/pls_speed/xla_onehot_test.py`，4096 token、宽度 512、词表 128256，有无 soft cap）：损失和两个梯度逐位相同。
  3. 手写 Triton 融合 CE 的估计：flash 式（不存 logits）要 4 个 GEMM，隐藏宽度 512 时效率低，不比 1 快；只在 1 之后的流量上还有约 10%，而且改变结果。先不做。
- GPU 测试（wq，优先级 10，4 GPU，每个 1.5 h，`scripts/della/pls_speed_arm.sbatch`）：w205/w206 Pre-LN 不开与开 A1（det：逐位比较；time：每步时间与 profile），w207/w208 HC、AttnRes 的 profile。
- CPU 测试的一个陷阱（`tmp/pls_speed/dot_round_test.py`）：XLA 默认允许多余精度，在 CPU 上把老路径反向里的 bf16 logits 舍入删掉了（bf16 输出的 dot 再转 f32，几乎每个元素都等于没有舍入的 f32 结果）。GPU 上 cuBLAS 确实写 bf16。所以前向算梯度在 CPU 上的对比要关掉多余精度（`--xla_allow_excess_precision=false`）才有意义；新实现用 `lax.reduce_precision` 做这个舍入，XLA 不会删它。GPU 上是否逐位相同由 det 轮决定（w217、w218）。
- 更多 GPU 测试（wq，优先级 10）：w217 Pre-LN 前向算梯度，w218 A1 加前向算梯度，w219 HC 不开（det 轮），w220 HC 开 xla select（det 和 time 两轮）。

## 补充 4 个 job（2026-10-25，所有者经 terminal：「让两个pretrain各补四个job吧」，这 4 个不需要弹窗）

- 两组“剂量-反应”实验，都检验各自对比所指的那个性质：
  1. w221/w222：HC 48 层 8 条流（独立头、普通模型加探针）。和 w194–w197（1、2 条流）、HC（4 条流，代价 0.048）一起，得到代价随流数的曲线。
  2. w223/w224：AttnRes Block 24 个块（独立头、普通模型加探针；开关 `AR_BLOCKS`，tag `-arb{n}`）。Block（8 个块）代价 0.113，Full（每个子层一个源）种子 1 是 0.061：代价是否随每次读取混合的源变多而下降。
- CPU 检查 `tmp/pls_fill/fill_test.py`：8 条流的 HC、3/6 个块的 AttnRes Block 损失和梯度有限，块数改变损失。dry run 的 run id 正确。代码 e1f3efbd9c。
- 占位（所有者弹窗 2026-10-25）：8 条流加到 Fig 43；AttnRes 按源的个数做新图 Fig 44（`fig-ar-sources`，已有 Block 8 个块和 Full 两个种子的点）。网页 b7d417e，artifact 版本 102。
- 这 4 个 job 都和老 run 比较，所以用老的 CE 路径；只有逐位相同的加速会用到它们。

## 中途结果（2026-10-26）

- Pre-LN 12 层（w147/w148 完成）：独立头最后一层训练损失 3.252，探针 3.139，代价 0.113。Pre-LN 48 层是 0.089，HC 12 层 0.121、48 层 0.048。所以 Pre-LN 的代价也随深度下降，但比 HC 降得少；24 层（w149/w150）还在跑。Fig 39 已填 12 层的点。
- mHC 残差混合固定为 I 的独立头（w76 完成）：第 48 层最终训练损失 3.026；Sinkhorn 混合的 mHC 独立头是 3.024。残差混合不影响独立头模型；代价要等探针 w77（第 2,820/4,959 步）。
- MuonH 更新效率（`tmp/pls_muon_eff/muon_eff.py`，Pre-LN 12 层，256 token，试跑）：独立头的更新得到最后一层损失最佳一阶下降（每个矩阵用 polar(g_fin)）的 95.6%；第 1 层 92.8%，最后一层 100%。48 层的 Pre-LN 和 HC（1024 token）在 vis1 上跑。

## 速度：GPU 结果（2026-10-26）

| 项 | 逐位相同（det 轮，40 步，50 个指标） | Pre-LN d48 每步（time 轮，中位数） | 结论 |
|---|---|---|---|
| 不开（w205） | — | 14.962 s | 基线 |
| A1，select 代替 scatter（w206） | 是 | 13.102 s（−12.4%） | 已对所有 pls job 打开（dc033b04a9） |
| 前向算梯度（w217） | 否：第 0 步起部分读出的前向损失就有差别（约 1e-7 相对） | 14.074 s（−5.9%） | 只用于全新的一组对照 |

- 前向算梯度不逐位：同一个形状的 f32 logits 矩阵乘，在新程序里有更多使用者，XLA 对其中一部分实例选了不同的融合和算法，累加顺序变了；CPU 上换成同一个矩阵乘就逐位相同，所以不是逻辑错误。
- HC、AttnRes 不开时（w207/w208）：每步 24.68 s、34.00 s。profile 会卡在 stop_trace（事件太多），已改为关掉 Python tracer、microbatch 1 的结构只记 1 步（03e6d9aaa2）。
- 离线 H100 编译（`tmp/pls_speed/ce_hlo.py`）：每个 8192 行块，现在约 36 GB 流量、4 个 logits 大小的矩阵乘；A1 约 19 GB；A1 加前向算梯度约 19 GB、3 个矩阵乘。xla 路径（microbatch 1）的 select 去掉每个词表块的 scatter 一遍。
- 一次读完 logits 的在线 logsumexp（XLA 的 variadic reduce）：CPU 上对 float64 真值的相对误差 1.4e-5，两遍法 5.9e-8；不上线。要无损地把 lse 并进矩阵乘需要 Triton kernel；Tokenizer Size session 不需要，pls 只有换新组合时才用得上。
- Tokenizer Size session 的结果（它的 300m，宽度 768）：FA4 attention 加 CUDA graph 循环每步快 50%（4096）到 77%（4480），结果在 bf16 噪声内但不逐位；它已用作默认。pls 已移植 FA4（`ATTN_BACKEND=fa4`，run tag -fa4，0ac7a45464）和 CUDA graph 循环（`XLA_CMD_BUFFER_WHILE=1`，38c771313a），都默认关；测速在排队（w237、w238、w242–w244）。

## MuonH 更新效率：不是 HC 代价低的原因（2026-10-26，`scripts/della/pls_muon_eff.py`，vis1 CPU，1×1024 token）

- 问题：MuonH 对每个矩阵用 polar(梯度) 定方向，步长与梯度大小无关。独立头的更新方向是 polar(g_fin + g_int)。它让最后一层损失下降多少，相对于只看最后一层的最佳方向 polar(g_fin)？HC 的代价低，是不是因为它的这个比例更高？
- 量：η = Σ_W |W| ⟨g_fin, polar(g_sep)⟩/|polar| ÷ Σ_W |W| ⟨g_fin, polar(g_fin)⟩/|polar|，对所有 Linear 权重（MuonH 更新的参数）求和；1 表示没有损失。一阶近似，在训练结束的独立头检查点上测。

  | 检查点 | 代价 | η | 第 1–12 层 | 13–24 | 25–36 | 37–48 |
  |---|---|---|---|---|---|---|
  | Pre-LN 48 层 | 0.089 | 0.966 | 0.961 | 0.979 | 0.992 | 0.998 |
  | HC 48 层 | 0.048 | 0.954 | 0.937 | 0.976 | 0.993 | 0.998 |

- 结果：两者每步只损失 3–5% 的最后一层下降，损失集中在下层；HC 的 η 反而更低。“HC 的更新浪费得少”不成立。到此排除的解释：梯度方向（偏角）、读出看不到的状态（隐藏份额）、MuonH 的更新效率。剩下：多条流本身或门控（w194–w197，Fig 43）。
- 注意：都在终点检查点上测；一阶近似；1024 个 token。

## 速度：GPU 结果（续，2026-10-26）

| 项 | 结果 | 结论 |
|---|---|---|
| A1 加前向算梯度（w218） | Pre-LN d48 12.13 s/步（只开 A1 13.10，基线 14.96）；对同一调优类的基线 w229，40 步内 \|d\| ≤ 1e-6，两次相同基线之间差到 0.076 | 已对所有 pls job 打开（c9176a2e8b） |
| 前向算梯度的回退分支（microbatch 1） | 02:25–02:50 开着时，回退分支在 shard_map 内建了带分片约束的数组，w195–w197、w221 的冒烟测试失败，宿主停取 job，w154 被取消后重排；已修（867c735932），GPU 上 HC 的回退与基线逐位相同（w255 = w219） | 已修，已重新打开 |
| `EVAL_PDP=4`（w225/w226） | HC 每次 eval 7.5 → 5.1 min，无 OOM | 已对所有 pls job 打开 |
| xla select（HC，w220） | 逐位相同（w220 = w230），但不快（24.75 对 24.68 s/步） | 不采用 |
| CUDA graph 循环（w237/w238） | Pre-LN −0.7%，HC −1.1%；开着时 profiler 的 stop_trace 挂住，GPU 闲置（已取消） | 不采用 |
| FA4 attention（w242–w244） | Pre-LN 14.83 → 12.44（−16.1%），HC 24.68 → 21.26（−13.9%），AttnRes 34.00 → 29.42（−13.5%） | 所有者弹窗：先用 Pre-LN 48 层一对验证（w261/w262），再用于新对照 |
| microbatch-1 改走 b-tiled CE（`LEVANTER_CE_BTILED_MIN_BATCH=4096`，w266–w269、w275） | HC、AttnRes 都 OOM（加前向算梯度 96–98 GiB，只开 A1 也要 74 GiB） | 不采用；HC/AttnRes 继续用 xla 流式 CE |
| Triton lse kernel（Tokenizer Size session 写，`LEVANTER_CE_LSE_KERNEL=1`） | 基准（w271）：8192×512 每块 2.46 ms，同样输出的 XLA 路径 6.84 ms；合并误差 5.8e-8（两遍法 5.6e-8）；f32 logits 与 cuBLAS 逐位相同 | 已接入（b3f40f216f，默认关）；GPU 检查 w278 |
| lse kernel v3（`LEVANTER_CE_LSE_MGPU=1`，Pallas Mosaic GPU，3 级流水，w385/w386/w440） | Pre-LN 10.06 → 9.83 s/步（−2.3%），HC b-tiled 10.12 → 9.86（−2.6%）；梯度与 Triton kernel 相对差 1.2e-6 到 1.5e-5，两者相对 f32 都是 2e-3 到 1.3e-2；重跑逐位相同 | 候选，等下一次弹窗 |

- profile（w239，Pre-LN d48，开 A1，关 command buffer）：每步 GPU 时间里读出头和 CE 占 68%（矩阵乘 4.09 s，lse 两次规约 1.70 s，A1 的 dlogits 一遍 1.02 s），attention 占 20%（2.05 s）。
- profile（w417，HC 新路径，每步 10.41 s）：读出头和 CE 52%（矩阵乘 2.16 s，Triton lse 2.02 s，logits 转换 1.17 s），collectives 16%（每个 microbatch 里每层权重梯度各做一次 bf16 all-reduce：HC 每步 32 个 microbatch，约 1,500 次），HC 的流混合 14%。
- profile（w418，AttnRes 新路径，每步 19.26 s）：读出头和 CE 29%；remat 重算 23%（深度注意力对全部 97 个来源槽算分数，反向又算两遍）；XLA 每层把整个来源缓冲区 bf16[97,1,4096,512]（407 MB）拷贝 4 次，一步 1.64 s；反向的 DUS scatter-add 1.10 s；collectives 10%。根本修法：反向用最终缓冲区加掩码代替每层保存的缓冲区（被掩码的槽权重正好是 0，前向逐位不变），需要改 AttnRes 的反向结构，记为待办。
- microbatch（HC，生产配置，流式 CE 不变，只改求和顺序）：每个 microbatch 1 条序列 24.78 s/步（w442），2 条 21.09（−14.9%，w443），4 条 19.37（−21.8%，w444）。4 条以前因临时缓冲区 32.9 GiB 在第 257 步碎片 OOM；内存测试 w459–w462 在排队。
- AttnRes 相反：1 条 33.99 s/步（w445），2 条 37.97（+11.7%，w446）。AttnRes 的开销是来源缓冲区的读写和整块拷贝，随 token 数增长，不是每个 microbatch 的固定开销；AttnRes 保持 1 条。
- HC 的训练步临时缓冲区（流式 CE）：1 条 17.27 GiB（w461），2 条 19.25（w459），4 条 21.67（w462）；状态 10.36 GiB。空闲池约 64 GiB 的一半是 32 GiB，4 条低于它。用户弹窗 2026-10-26 ~23:35：ARCH=hc 的 run 从下一段开始每个 microbatch 4 条（pls_h100x8_single.sbatch 的 HCMB，1d7b827079）；冒烟测试失败时自动退回 1 条；HC 新组合那一对和 8 条流的 run（等 w473）不改。
- UNNORM=2（移植 457d8dd438）：梯度测试 w448 通过。相对 f32 的误差，grad_x / grad_w：初始化附近 1.66e-3 / 1.67e-3，现在的路径 2.71e-3 / 2.84e-3；尖锐 logits 1.76e-3 / 1.95e-3，现在的路径 1.3e-2。不要输入梯度时，损失和 grad_w 逐位相同。Pre-LN 速度（w450 对 w449）：10.03 → 8.88 s/步（−11.5%，含 v3 kernel 和评估 kernel）。前 40 步的损失：第 39 步最后一层 7.0715，两次基线 7.0363 和 7.0441；48 个读出之和从第 20 步起比两次基线高 1.2–1.5（两次基线之间差 0.08–0.2）。梯度更准，所以这是早期轨迹的混沌差别，还是真实的早期偏差，要更长的对照才能分开。
- XLA 的 DUS 累加器零初始化消除（w451、w463、w464）：Pre-LN 0%，HC −0.1%，AttnRes −0.6%。不采用。
- UNNORM 加串行读出（PLS_SERIAL、GIF barrier，w466/w467）：Pre-LN 8.92 s/步（−11.0%）；训练步临时缓冲区 15.58 GiB，现在生产 28.11 GiB，不串行的 UNNORM 38.23 GiB。第 39 步损失低于两次基线（最后一层 7.0216 对 7.0363 / 7.0441），而不串行的那一臂（w450）高于基线：40 步内的差别是轨迹分叉，不是 UNNORM 的偏差。候选：新 run 用 fa4thd + v3 + UNNORM + 串行读出，等下一次新实验弹窗。
- 新 run 的候选组合（2026-10-27 凌晨，都过了第 20 步的评估和 checkpoint）：
  - Pre-LN 一类（b-tiled CE）：UNNORM + lse kernel v3 + 评估 kernel + 串行读出 + 每个 microbatch 8 条：10.03 → 8.32 s/步（−17%，w478），临时缓冲区 20.95 GiB（w479）。fa4thd 等验证对 w333/w334。
  - HC：新 CE 路径（microbatch 的 b-tiled、串行读出、GIF barrier、fa4thd）+ UNNORM + 每个 microbatch 4 条：7.34 s/步（w480），临时缓冲区 17.49 GiB（w481）。正在跑的旧路径 4 条是 19.37，原来的 1 条是 24.78。验证对 w387/w388 检验 HC 的代价是否依赖 CE 实现。
- DEPTH_INNER_REMAT=save（AttnRes/MoDA 两层 scan 的内层不再逐层 checkpoint）：CPU 上损失和梯度不变，但 AttnRes 新路径每步 17.35 → 25.07 s（+44.5%，w485）。逐层保存的中间结果里有整个 407 MB 来源缓冲区（和它的 f32 形式），存储流量超过省下的重算。不采用，默认关。真正的修法：反向不保存每层的缓冲区，用最终缓冲区加掩码（被掩码的槽权重正好是 0），需要给整个 AttnRes 层栈写 custom VJP。
- 每段开头的冒烟测试（30 步 + 评估 + 两道门）：HC 4 条 11–12 min（w196、w197），HC 1 条 14.5 min（w221），AttnRes 17–19 min（w122、w152、w223），一个 run 约 3 段，所以占 2–3% 的 GPU 时间。候选（未上线，等弹窗）：SINGLE_SMOKE_REUSE=1，续跑时如果代码（commit 加未提交的 diff）和开关与这个 run 上次通过冒烟时相同，就跳过冒烟；任何改动都照常冒烟。改好的脚本在 scripts/della/pls_h100x8_single.sbatch.smokereuse（未提交，未生效）。
- mHC 每个 microbatch 4 条（流式 CE，w487/w488/w489）：25.35 → 19.67 s/步（−22.4%），临时缓冲区 21.71 GiB。HCMB 只开了 HC；mHC 等弹窗。
- det 轮的教训：两次完全相同的 run 也不逐位一致（GEMM 自动调优按计时选算法，run 分成“调优类”）；判断一个开关要和两个基线比，看是否在两次基线的差别以内（速度计划的 G4），或固定调优（`--xla_gpu_autotune_level=0`）。

## 速度：新组合上线（2026-10-26）

| 项 | 结果 | 结论 |
|---|---|---|
| Triton lse kernel（`LEVANTER_CE_LSE_KERNEL=1`） | Pre-LN d48：12.13 → 10.06 s/步（w299）；梯度与 XLA 路径相对差 1.5e-6 到 1.6e-5，两者到 f32 参考都是 2e-3 到 1.3e-2，偏差约 1e-10（w306）；第 20 步 eval 和 checkpoint 之后正常训练（w299、w301） | 所有者弹窗：对所有 pls job 打开（e90ce97a02） |
| fa4thd（上游 FA4 的 SM90 varlen kernel） | Pre-LN：JAX flash 14.96、fa4 12.50、fa4thd 12.16 s/步（w288）；HC：24.69、21.30（fa4 + CUDA graph）、20.44（w289）；eval 走 segmented kernel（Paloma 一行最多约 290 个文档） | 新对照要先过验证对 |
| 新组合（生产设置 + kernel + fa4thd） | Pre-LN 7.39（w300），probes 模型 7.31（w301），HC 20.56（w302），AttnRes 28.50（w303）s/步；都过了第 20 步的 eval 和 checkpoint | 所有者弹窗：fa4 验证对 w261/w262 取消，改为新组合的一对 w333/w334；代价与 JAX flash 的 0.089/0.092 在种子噪声内一致后，新对照用新组合 |

- 上线前检查（控制会话 2026-10-26 提醒）：前向算梯度和 `REMAT=fdots` 都通过了短测试，却在真实 run 里出错；所以每个速度项上线前，测试要在训练中途做一次 eval 和 checkpoint，再继续训练（`SMOKE_MID=20`，4ab01211ab）。
- 固定调优（`--xla_gpu_autotune_level=0`）不能让 run 可重复：两个相同对照（w292、w294）第 0 步就不同。

## mHC 残差混合 = I（2026-10-26）

- w76/w77 完成：独立头 3.026，probes 2.926，第 48 层代价 0.0999；Sinkhorn mHC 是 0.1016，相差 0.0017，在种子噪声（约 0.003）内。
- 结论：mHC 的代价（约 0.10，与多数结构相同；HC 只有 0.048）不来自它的残差混合矩阵。HC 与 mHC 的差别要在其他部分找（读写权重的约束、归一化等）。
- 站点：图 38 填入，图注改为陈述结果（pretrain-practice 68f0b46，artifact v107）。

## 2026-10-26 下午：上线与新实验（所有者弹窗）

- 生产 job 脚本新开：`PLS_FOLD_NORM=1`（probes 模型折叠读出 RMSNorm 缩放，Pre-LN probes −16%，HC probes −14%；用于所有排队的 probes run）和 `LEVANTER_NORM_GUARD=1e-3`（范数保护，停下后以 99 退出，从上一个 checkpoint 恢复）。
- HC/AttnRes 的 CE 新路径（b-tiled + 串行读出 + GIF barrier）：HC 20.56 → 10.12 s/步（w370），AttnRes 28.50 → 18.17（w373），HC probes 加折叠 8.93（w378）；显存 24.9 GiB（w369，原路径 27.4）。数值：两条路径的梯度到 f32 都在 bf16 量级（w372）；40 步轨迹差别超出两次相同运行（w374–w376），旧路径在第 2 步最后几层损失上升，新路径不升。所以先跑一对验证（w387/w388，图 39 空心点）。
- 新实验：HC 读写 = sigmoid（w389/w390，图 38）：mHC 的残差混合已排除，检验它有界、为正的读写映射。

## Pre-LN 新组合验证对（2026-10-27，wq w333/w334 完成）

- 设置：Pre-LN 48 层，生产设置加 Triton lse kernel 和 fa4thd attention。w333 是独立头，w334 是只加探针。w334 的最后一段（10-26 19:39 起）开着 `PLS_FOLD_NORM=1`。这个开关只改 47 个探针头（它们的输入已切断梯度），不改第 48 层。
- 结果（第 48 层最终训练损失，最后 50 步的平均）：

  | 运行 | 独立头 | 探针 | 代价 | Paloma eval 代价 |
  |---|---|---|---|---|
  | JAX flash，种子 0 | 3.0725 | 2.9832 | 0.0893 | 0.150 |
  | JAX flash，种子 1 | 3.0747 | 2.9826 | 0.0921 | 0.156 |
  | 新组合（fa4thd + lse kernel） | 3.0719 | 2.9883 | 0.0837 | 0.135 |

- 新组合的代价比较低的种子低 0.0056。两个种子之间差 0.0028。差别来自探针模型（高 0.005）；独立头模型不变（低 0.001–0.003）。
- 判断：3 个 run 分不开 kernel 的系统偏差和轨迹噪声。如果是系统偏差，大小约 0.006，小于项目里要比较的效应（E1 每减半 0.016–0.030，HC 与 Pre-LN 差 0.042）。新组合只用于两边都用新组合的对比，这样的偏差会抵消。
- 新 run 用不用新组合，等弹窗。Pre-LN 完整新组合（fa4thd + v3 + UNNORM + 串行读出 + 每个 microbatch 8 条）的测速和显存在 w492–w494（独立头 time、mem，探针模型 time）。
- 逐步比较（同一种子、同一数据顺序，第 48 层训练损失减去 JAX flash 种子 0；`tmp/pls_speed/fa4_traj.py`，图 `tmp/pls_status/newstack_drift.png`）：
  - 独立头：新组合全程在 ±0.010 内（100 步滑动平均），结束时 −0.0006。种子 1 中途到 +0.016，结束时 +0.0021。
  - 探针模型：新组合在第 20–300 步高 0.02–0.05，第 300 步以后一直高 0.004–0.008，结束时 +0.0050。种子 1 中途在 −0.007 到 +0.004 之间换号，结束时 −0.0006。前 400 步没有新组合独有的尖峰（最大的上升在第 152、380 步，三条都有，来自数据）。
  - 所以探针模型的差别从训练早期开始，方向不变，不像种子噪声。独立头模型不受影响。
- HC 新组合那一对（w387/w388，还在跑，另外换了 CE 路径）与旧 HC 对比：独立头到第 3,086 步，早期低 0.01–0.04，第 1,000 步以后高 0.004–0.009；探针模型到第 3,809 步，全程高 0.01–0.05，第 2,000 步以后约 +0.010。
- 已排除：`PLS_FOLD_NORM`（w334 第一段没开，早期差别已经出现；它只改探针头）。GIF 在 40 步内与同调优类的基线差 ≤ 1e-6（w218/w229），不太可能是原因。Tokenizer Size session 的 attn_bench（w210/w283）：JAX flash 的输出和 dq 相对 fp32 误差约 0.00025 / 0.00055，fa4 / fa4thd 约 0.002 / 0.0028（输出舍到 bf16）；dk、dv 两者同级。
- 检验（wq w495–w498，优先级 10，`scripts/della/pls_trace_arm.sbatch`）：探针模型、种子 0、完整 4959 步学习率日程，各跑 70 min（约 300 步），设置为全关、只开 fa4thd、只开 CE 三项（A1、GIF、lse kernel）、全开（重复 w334 的第一段）。比较第 20–300 步相对种子 0 的偏移（`tmp/pls_speed/trace_compare.py`）。全开那条如果不重现偏移，w334 的差别就是偶然。

## 2026-10-27：短轨迹结果、HC 新组合、所有者弹窗

- 短轨迹（w495–w498 完成，`tmp/pls_speed/trace_compare.py`，图 `tmp/pls_status/trace_noise.png`）。第 48 层训练损失减去 JAX flash 种子 0，窗口平均：

  | 运行 | [20,60) | [60,100) | [100,150) | [150,200) | [200,300) | [300,400) |
  |---|---|---|---|---|---|---|
  | 全关 w495 | +0.021 | +0.015 | +0.020 | +0.012 | +0.008 | — |
  | 只开 fa4thd w496 | +0.012 | +0.008 | +0.005 | +0.008 | +0.009 | −0.0004 |
  | 只开 CE 三项 w497 | +0.016 | +0.002 | +0.006 | +0.006 | +0.004 | −0.0009 |
  | 全开 w498（w334 第一段的开关，到第 526 步） | +0.006 | +0.002 | +0.005 | +0.006 | +0.011 | −0.0015 |
  | w334 | +0.035 | +0.042 | +0.033 | +0.023 | +0.017 | 约 +0.005 |

  - 按事先写下的判据（全开那条不重现偏移，w334 的差别就是偶然）：w334 的早期偏移是那一次运行的轨迹，不是组合。四条重跑彼此相差最多 0.016 nats，第 300–400 步都回到种子 0 的水平。
  - 早期的运行间差异很大：同一代码重跑，第 1 步差 4e-4，第 7 步已到 ±0.1（训练早期的混沌）。所以 300 步的短轨迹只能看出 0.02 以上的效应。Tokenizer Size session 的观察相同：同一代码跑两次，第 120 步 eval 差 0.035。
  - 终点（+0.005）还要全长重跑：所有者弹窗（12:4x）选“重跑两条再定”：w500 JAX flash、w501 fa4thd，探针模型，种子 0，今天的 CE 侧，`PLS_FOLD_NORM=0`，run tag `-rerun`，优先级 10。结果出来前新研究对不开 fa4thd。网页图 45 占位。
- HC 新组合那一对（w387/w388 完成，换了 CE 路径并开 fa4thd）：独立头 3.0026、探针 2.9557，代价 0.0468；旧 HC 3.0026 对 2.9919、2.9557 对 2.9441，代价 0.0478。两个模型都高约 0.011，代价不变。所以 HC 的低代价不依赖实现；新旧路径的 HC run 不能混在同一个比较里。Pre-LN 只换 fa4thd 时没有这样的整体上移，所以这 0.011 更可能来自 HC 的 CE 路径（streaming → b-tiled、bf16 dlogits）。图 39 空心点已填。
- Full AttnRes 源取单位 RMS 的独立头（w122 完成）：第 48 层 3.018；Full 独立头种子 0 / 1 是 3.023 / 3.016。代价等探针 w123。
- Pre-LN 24 层（w149/w150）：独立头 3.153，探针 3.051，代价 0.102。Pre-LN 12 / 24 / 48 层：0.113 / 0.102 / 0.089；HC：0.121 / 0.077 / 0.048。
- 所有者弹窗（同一次）：mHC 每个 microbatch 4 条上线（HCMB 包括 ARCH=mhc）；恢复训练时，代码树（lib/、experiments/）、未提交改动、开关和导出变量都和上次通过的冒烟测试相同就跳过冒烟测试（`SINGLE_SMOKE_REUSE`，默认开；cd9c584d59）。
- 补队列（controller：队列 0.99 天；所有者弹窗）：独立头损失在训练中途停掉，Pre-LN 50%（w503，`PLS_OFF=2480`）、80%（w504，`PLS_OFF=3967`），HC 80%（w505）。问题：中间层损失停下以后，最后一层的代价留下还是消失。先跑预检 w502（`PLS_OFF=15`，40 步，第 20 步 eval 和 checkpoint），三条在它通过后开始。网页图 46 占位。
- 网页（pretrain-practice 462a05e）：图 39 空心点、Pre-LN 24 层；图 45、46 占位；记录补上 6 行的 Code 链接（marin-pli@7744825，泄漏扫描 111 个新 blob 干净）和 5 个新运行。

## Gated Residual 报告与 mHC 的路径分解（2026-10-27）

- 来源：所有者给的 Qwen3.8-Next 技术报告 §2.2（Gated Residual，GR）。GR = 4 条流，每条流先各自 RMSNorm，读是逐流逐通道的 sigmoid 门，写是每条流一个 2·sigmoid 标量，没有流间混合 H_res。他们的消融（25B-A3B）：只加宽（静态读写）损失 −0.021；读写随输入变化再 −0.002，但基准平均分 +1.98；sigmoid 比 tanh 好；读的粒度要细，写不需要；有了好的读写，H_res 没有用；28 层时 Full AttnRes 和 GR 都是 1.762。GatedNorm 在 3 倍学习率下把尖峰从每万步 32 次降到 3.2 次。每层只读门值最高的两条流：预训练指标不变，后训练质量下降。
- 和我们的对应：mHC 的 H_res 已排除（mhcid 代价 0.0999，Sinkhorn 0.1016）。HC（代价 0.048）和 mHC（0.10）剩下的差别主要是读写映射：HC 是线性的，可以取负；mHC 和 GR 是 sigmoid，只能取正。w389/w390 在检验这一点。GR 的训练对等 w389/w390 出结果再定（所有者弹窗 2026-10-27）。
- 新分析（所有者弹窗 2026-10-27，`scripts/della/pls_branch_paths.py`，vis1 CPU，8×1024 token）：mhcid 没有混合，所以第 v 个子层的输入精确拆成 a(u→v, c) = r_c(v) w_c(u) |f(u)|。用记录的写入量重建最后的流之和，相对误差：探针模型 5.6e-8，独立头模型 1.2e-7。图 47（fig-mhc-paths，站点 58d5a3c）。
- 结果（独立头 w76 对普通模型 w77）：
  - 份额按跳过的层数分组，独立头减普通模型（对 96 个读取方平均）：跳过 ≤1 层 −0.014，1–6 层 −0.127，6–24 层 +0.019，>24 层 +0.122，嵌入 −0.039。平均跳过层数 14.9 对 10.5。
  - 写入量（Σ_c w_c × |f|，每 8 层平均）：普通模型 4.4、3.2、5.6、9.9、21.3、40.5，随深度上升；独立头 25.6、33.2、18.3、11.1、9.0、7.7，前面的层写得最多，后面越来越少。
  - 读出（流之和）在第 48 层：来自最后 4 层的份额，独立头 0.023，普通模型 0.284。
  - 没有 GR 那样“一条远距离流”的分工：独立头模型的流 2 承担的远距离路径最多（跨度超过 24 个子层的路径里占 0.37，全部路径里占 0.39），流 3 占远距离路径的 0.29，流 1 用得最少（全部路径的 0.14）。
- 解释（待检验）：独立头要求每层的流之和都能直接预测下一个 token，所以模型在前面的层就写出大部分内容，后面的层只做小的修改，相邻读出才一致。最后一层的预测于是大多由早期层决定，后面的层几乎不再改进它，这可能就是代价的来源。下一步：在 Pre-LN 和 HC 上测同样的“每层写入量”，看代价低的 HC 是否在独立头下仍然保持较大的后期写入。
- 每层写入量（所有者弹窗 2026-10-27，`scripts/della/pls_write_size.py`，vis1 CPU，4×1024 token，图 48 fig-write-size，站点 a017d7f）。量：读出所读状态 s_l 的改变量 |s_l − s_{l−1}| 除以 |s_{l−1}|（占位图原定除以 |s_0|；HC 普通模型的流之和长到嵌入的 5×10⁴ 倍，所以改成除以改变前的状态，也就是归一化读出看到的量）。最后 8 层的平均（独立头 / 普通模型）：
  | 结构 | 代价 | 独立头 | 普通模型 |
  |---|---|---|---|
  | Pre-LN | 0.089 | 0.020 | 0.170 |
  | mHC | 0.102 | 0.021 | 0.227 |
  | mHC H_res=I | 0.100 | 0.026 | 0.208 |
  | AttnRes (Full) 种子 1 | 0.061 | 0.028 | 12.8 |
  | HC | 0.048 | 0.202 | 1.254 |
  - 在 Pre-LN 和 mHC 里，独立头让后期层几乎不再改变读出状态（约为普通模型的 1/10）。HC 的独立头模型直到最后一层仍有约 0.2 的改变，约为其他结构的独立头模型的 10 倍。代价最低的结构正是后期写入保持最大的结构，和假设一致。
  - AttnRes 的读出状态是混合，不是和，改变量不能横向比较；它的独立头模型后期改变也很小，代价却在中间（种子 1 0.061），这一点还没解释。
  - 下一步：w510/w511（mHC 线性读写，可取负）和 w389/w390（HC sigmoid 读写）检验读写的符号是否决定后期写入能否保持。如果 mHC 线性读写的代价降下来，它的后期改变也应该变大。

## AttnRes 源归一化（w122/w123 完成，2026-10-27 18:37）

- 独立头第 48 层 3.018，探针 2.920，代价 0.098。普通模型：步数 > 200 时，比前 100 步中位数高 0.1 以上的步数为 0（种子 0 是 66，种子 1 是 7），最终 2.920，低于种子 0 的 3.045 和种子 1 的 2.955。独立头模型三次都在 3.016–3.023。
- 结论：巨大的源引起 AttnRes 普通模型的尖峰；去掉尖峰以后，AttnRes 的代价（0.098）与 Pre-LN、mHC 等结构（约 0.09–0.10）相同。之前较小的代价（0.061、−0.021）来自普通模型被尖峰拖慢。图 41 已填，图 48 的图注补上这一点（站点见提交记录）。
- 这也补上了图 48 的缺口：AttnRes 独立头的后期改变很小，真实代价也在常见的 0.10 左右，和“后期写入小 → 代价高”一致。到目前为止只有 HC 例外。
- AttnRes 静态路由的独立头（w152 完成，2026-10-27 19:15）：第 48 层 3.024，探针（w28）2.913，代价 0.111。静态路由的普通模型没有尖峰，代价略高于 Pre-LN（0.089），和源归一化的动态 AttnRes（0.098）接近。AttnRes 两种路由去掉尖峰以后都付常见的约 0.10；HC 静态 0.074、动态 0.048，仍是唯一低于 Pre-LN 的结构。图 37 标签和图注已更新（站点 f32d559）。
- HC 读写 = sigmoid 的普通模型（w390 完成，2026-10-27 22:50）：第 48 层最终训练损失 2.976。HC 普通模型是 2.944（旧路径）/ 2.956（新 CE 路径），所以有界、为正的读写让普通模型变差约 0.02–0.03。代价要等独立头 w389（排队，从第 2724 步续跑）。
- fa4thd 重跑（w501 完成，2026-10-28 ~01:00）：Pre-LN 普通模型（种子 0，今天的代码，fa4thd）第 48 层 2.9884；w334（fa4thd）是 2.9883，JAX flash 两个种子 2.9832 / 2.9826。fa4thd 两次都比 JAX flash 高约 0.005，重复得很好。JAX flash 重跑（w500，第 ~3900 步）决定这 0.005 来自 fa4thd，还是来自今天代码的 CE 一侧。图 45 已加点（站点 d5199ee）。
- JAX flash 重跑（w500 完成，2026-10-28 04:09）：Pre-LN 普通模型（种子 0，今天的代码，JAX flash）第 48 层 2.9870。fa4thd 重跑 w501 是 2.9884，只高 0.0014；两个旧的 JAX flash 种子是 2.9832 / 2.9826，彼此差 0.0006。
  - 结论：w334 比旧运行高的 0.005 里，约 0.004 来自两次重跑共有的改动（今天的 CE 侧：A1、GIF、lse kernel，或别的代码变化），fa4thd 只占约 0.001。这与 HC 新 CE 路径整体上移 0.011（w387/w388）方向一致。
  - fa4thd 本身：Pre-LN 12.16 对 14.96 s/步（−19%，w288），损失 +0.0014。是否作为新任务对的默认，弹窗给所有者。
  - 图 45 说明改成结果（站点 9a4d838）。
- 所有者弹窗（2026-10-28 ~11:20）：以后新的任务对默认用 fa4thd。现在不交新任务；代码默认值等所有正在跑和排队的任务结束后再改（排队的任务启动时读仓库代码，现在改会改变它们的设置）。
- 2026-10-29 05:30：所有 pls 任务都已结束。启动器默认值改为 ATTN_BACKEND=fa4thd（fbc800111e）。新运行的 run id 带 -fa4thd；用 jax_flash 开始的旧运行要恢复时，必须写 ATTN_BACKEND=jax_flash（已用 dry run 检查两种 run id）。
- AttnRes Block 24 个块（w223 独立头 / w224 探针，完成）：第 48 层 3.044 / 2.956，代价 0.089。8 个块是 0.113，Pre-LN 0.089。探针模型除了所有运行都有的第 1391 步（数据）只有 2 个尖峰步（1235、2029；判据：比前 50 步中位数高 0.1 以上），Full 种子 0 有 49 个。所以下降不是尖峰造成的：每次读取混合的源越多，AttnRes 的代价越低，最低到 Pre-LN 的水平。图 44 说明改成结果（站点 cb5d0a9）。
- 训练到 50% 时停掉中间层损失（Pre-LN，w503 完成）：第 48 层 3.0265；探针模型 2.9832，全程独立头 3.0725。剩下的代价 0.043，约是全程代价（0.089）的一半：停掉以后最后一层在剩下 50% 的步数里恢复了约一半。停掉以后中间读出不再训练，第 1–47 层损失升到 27.6–71.9 nats（预期）。注意 w503 用今天的代码，比较对象是旧代码运行；按图 45，今天的代码约高 0.004。图 46 已加点，说明等 80% 两条（w504、w505）。
- 训练到 80% 时停掉中间层损失（Pre-LN，w504 完成，第二次重排后没有再 OOM）：第 48 层 3.054。剩下的代价：50% 时 0.043，80% 时 0.071，全程 0.089，约与带中间层损失训练的比例成正比。HC 80%（w505）还在跑。
- 发布审计（controller，2026-10-28）：7be7eaf 上的记录过时；scrub + 泄漏扫描通过，marin-pli 3bdc53a20；记录重建，站点 d22ff55。
- HC 读写 = sigmoid（GR 式的读写，w389 独立头 / w390 探针，完成）：第 48 层 3.020 / 2.976，代价 0.044。HC 新路径（w387/w388）0.047，旧路径 0.048。代价不变；两个模型的绝对损失都比新路径 HC 高约 0.02。所以 HC 的低代价不依赖读写用 tanh 还是 sigmoid。图 38 已加点，说明等 mHC 线性读写（w510/w511）。
- HC 8 条流（w221 独立头 2.999，w222 探针 2.950）：代价 0.049，与 4 条流（0.048）相同；2 条流 0.074。代价从 2 条流降到 4 条流，8 条流不再下降。1 条流（w194 独立头 3.097，w195 探针 2.972）：代价 0.125，高于 Pre-LN 的 0.089。门本身不降低代价；低代价来自多出的流，4 条流就够了。
- HC 48 层种子 1（w178 独立头 3.001，w179 探针 2.967）：代价 0.034，种子 0 为 0.048。HC 在 48 层的低代价在第二个种子上也成立。
- mHC 读写 = 线性（可取负、无界，w510 独立头 2.993 / w511 探针 2.913）：代价 0.080。mHC 0.102，HC 0.048。读写映射只解释 mHC 与 HC 差距的约 40%（0.022 / 0.054）；残差混合（单位矩阵）也不解释。剩下的差距来自 mHC 的其他部分。图 38 完成。
- HC 在 80% 处停掉中间层损失（w505）：第 48 层 2.982；探针 2.944，全程独立头 2.992。留下的代价 0.038，是全部代价 0.048 的 79%，与 Pre-LN 的 80% 相同。两种结构都不恢复；代价约与带中间层损失训练的比例成正比。

## 离线机理分析：HC 与 mHC 的代价差（2026-10-30，所有者弹窗「离线机理分析」，只用 CPU，不交 GPU 任务）

问题：mHC 的独立头代价（0.102）约为 HC（0.048）的两倍。读写映射（线性读写 0.080）只解释约 40%，残差混合改为单位矩阵（0.100）不解释。HC 只有 1 条流时代价 0.125，高于 Pre-LN 的 0.089。

方法：`scripts/della/pls_hidden_ablate.py`（1acef01565）。在第 l 层之后，把每条流换成各流的平均。这样流之和不变，读出 l 看到的状态不变，只删掉各流之差，也就是任何读出都看不到、只有后面子层能读的部分。然后照常跑完，比较第 48 层的交叉熵（Paloma c4_en，4×1024 个 token，vis1 CPU）。数据在 tmp/pls_paths/ha-*.json，图在 tmp/pls_paths/plot/hidden_vs_cost.png。

结果：
- 独立头模型在第 1–16 层删掉流之差时，第 48 层的交叉熵升高（平均值）：HC 1.58，HC 种子 1 1.37，HC sigmoid 读写 1.25，HC 8 条流 1.41，HC 2 条流 0.68，mHC 线性读写 0.82，mHC 单位混合 0.36，mHC 0.25；只有 1 条流的结构（Pre-LN、HC 1 条流）为 0。
- 它与代价的秩相关为 −0.84（10 个结构；加探针的普通模型为 −0.80）。在第 17–34 层只有 −0.47，第 35–47 层更弱：后期各结构都几乎不依赖流之差。
- 解释（2026-11-01：因果检验不支持，见下一节「因果检验结果」）：读出只看流之和。独立头要求每层的流之和都是好的预测状态，所以中间计算需要一个读出看不到的工作区。HC 能把早期和中期的计算放在各流之差里，所以付的代价小。mHC 的写入只取正值，双随机混合又把各流拉向平均，所以这个工作区用得少，代价接近单流结构。线性读写允许取负的写入，工作区用得多一些，代价也降了一些。
- 局限：这是相关，不是因果；除 HC 外每个结构只有一个种子；删除流之差也改变了每条流自身的大小。
- 对照（HA_MODE=shuffle，82a95afc6d，tmp/pls_paths/has-*.json）：第 l 层之后，把每个 token 的流之差换成另一个 token 的（4096 个 token 一个固定的随机置换）。流之和不变，每条流的典型大小也不变，只丢掉流之差里的信息。独立头模型第 1–16 层的平均升高：HC 2.68，HC 种子 1 2.67，HC sigmoid 读写 2.04，HC 8 条流 1.99，mHC 线性读写 1.89，HC 2 条流 0.91，mHC 单位混合 0.48，mHC 0.39。8 个结构都比取平均的升高大，所以升高来自流之差里的信息，不来自流的大小。与代价的秩相关为 −0.90（加 2 个单流结构仍为 −0.90；取平均为 −0.84）；探针模型为 −0.83。

可检验的预测（需要 GPU，所有者批准后才提交）：mHC 同时用单位混合和线性读写，既不把各流拉向平均，也允许取负的写入，代价应降到接近 HC（约 0.05）。反过来，把 HC 的写入限制为各流相同（不能写入流之差），代价应升到接近 Pre-LN。
- 所有者弹窗（2026-10-30）：「提交两对」。已提交 wq w711/w712（mHC 单位混合 + 线性读写，独立头 / 探针）和 w713/w714（HC 写入各流共用一个权重，HC_WRITE=shared）。代码 e4186600a8，CPU 测试 depth_arch_hcwrite_test.py 全部通过。新运行默认用 fa4thd（run id 带 -fa4thd）；代价是同一栈上两个模型之差，所以可以和旧的代价比较。
  - 更正：弹窗里 ② 写成「写入各流相同、混合不产生流之差」。这样各流永远相等，等于已经跑过的 HC 1 条流（0.125），所以实际提交的 ② 只限制写入（各流共用一个写入权重），混合保持 HC 的。

## 因果检验结果：打开流之差没有带来 HC 的代价；HC 与 mHC 的代价差在普通模型（2026-11-01，wq w711–w714 完成）

两个预测都不成立。第一对（mHC）提高了对流之差的使用，代价只降了一点。第二对（HC）没有关上流之差，所以检验不了这个解释。把代价拆成两个模型以后，HC 与 mHC 的差别在普通模型，不在独立头模型。

脚本与数据存档：`docs/della/pls_archive/analysis/causal_1101/`（`tmp/pls_audit2/` 中本节用到的脚本），消融结果在 `pls_archive/paths/`，图在 `pls_archive/paths/plot/`。网页：图 38、图 49 的说明已改写（pretrain-practice 845ad73）。

1. mHC 单位混合 + 线性读写（w711 独立头，w712 探针）
   - 第 48 层最终训练损失：独立头 3.003，探针 2.918，代价 0.085 nats。预测是接近 HC（约 0.05）。实际与 mHC 线性读写（0.080）和 Pre-LN（0.084–0.092）相当。
   - 流之差的使用（独立头检查点，第 1–16 层平均，`tmp/pls_paths/run_ha_causal.sh`）：取平均 0.997，置换 1.942。mHC 是 0.249 和 0.389，mHC 单位混合 0.357 和 0.483，mHC 线性读写 0.815 和 1.893，HC 1.582 和 2.684。探针检查点：0.775 和 2.634。
   - 这个改动让 mHC 的使用升到 4 倍，代价从 0.102 只降到 0.085。原 10 个结构的线性趋势在这个使用上给出 0.064（取平均）和 0.058（置换）。实际值高 0.021 和 0.027，是拟合残差标准差的 1.8 和 2.0 倍（`tmp/pls_audit2/fit_use.py`）。
   - 与 mHC 线性读写相比，单位混合只把使用从 0.815 提到 0.997，代价没有变（0.080 和 0.085）。让 mHC 代价降低的是线性读写，不是单位混合。
   - 跨结构的秩相关仍然成立（取平均：原 10 个结构 −0.839，加两个新结构 −0.830（12 个）；置换：−0.900 和 −0.830，`tmp/pls_audit2/spearman_use.py`）。但提高使用没有带来 HC 的代价，所以这个相关不能当作原因用。
   - 数据缺口：w711 第 3 次运行在检查点截止时间（00:16:23）刚跑完最后一步，保存了 step-4958，然后退出（rc 124）。W&B 没有收到第 4958 步的训练记录。第 4 次运行读入 step-4958，没有剩下的步，只做了最终评估。最终值取第 4908–4957 步的平均。`fetch_d48g4.py` 的 LOGGING_GAPS 已登记这一行。

2. HC 各流共用写入（w713 独立头，w714 探针）
   - 第 48 层最终训练损失：独立头 3.021，探针 2.979，代价 0.042 nats。预测是升到接近 Pre-LN（约 0.09）。实际与 HC（0.034–0.049）一样低。
   - 这个改动没有关上流之差。各流的写入相同，但 HC 的混合让每条流成为嵌入和之前各次写入的不同加权和，所以各流仍然不同。模型仍然使用流之差：独立头检查点取平均 1.205，置换 1.917（HC 1.582 和 2.684，HC sigmoid 读写 1.254 和 2.036）；探针检查点 1.786 和 3.564（HC 2.421 和 4.162）。
   - 所以这一对不能检验"工作区"的解释。它的点落在跨结构趋势上：使用 1.205 时趋势给出 0.055，实际 0.042，残差 −0.012（−1.0 倍标准差；置换 −0.017，−1.2 倍）。
   - 它说明另一件事：共用写入让 HC 的两个模型都变差（独立头 3.021，同一栈上的 HC 3.003；探针 2.979，HC 2.956），代价不变。HC 普通模型相对 Pre-LN 的收益大部分消失（2.979 对 2.988）。
   - 两次运行都完整（4959 步，没有缺行）。w713 第 4 次运行在 01:29:58 被 watchdog 停掉（训练日志 30 分钟没有输出），第 5 次从 step-4678 继续，03:47 rc 0 结束。

3. 把代价拆成两项（`tmp/pls_audit2/decompose.py`、`cost_terms_all.py`，图 `tmp/pls_paths/plot/sep_vs_probes.png`）
   - 独立头模型：4 条及以上流的结构都在 2.99–3.03（HC 2.992–3.003，HC 8 条流 2.999，HC 各流共用写入 3.021，mHC 线性读写 2.993，mHC 单位混合 + 线性读写 3.003，mHC 3.024）。1 条流的结构是 3.07–3.10（Pre-LN、HC 1 条流）。
   - 普通模型（探针）：4 个 mHC 运行都在 2.913–2.926，8 个 HC 运行在 2.936–2.979，Pre-LN 2.983–2.988。所有 mHC 运行都低于所有 HC 运行。
   - 所以 HC 比 mHC 付得少，是因为 HC 的普通模型较差，不是因为它的独立头模型更好。
   - 两个新结构给出最直接的对比。它们对流之差的使用几乎一样（置换 1.942 和 1.917，取平均 0.997 和 1.205），代价却差一倍（0.085 和 0.042）。mHC 结构的独立头模型反而更好（3.003 对 3.021），普通模型好得更多（2.918 对 2.979）。代价差 0.043 = 普通模型之差 0.061 − 独立头模型之差 0.018。
   - 在 10 个多流结构中，独立头检查点上的使用与独立头损失的秩相关为 −0.73（p 0.016），与普通模型损失为 +0.42（p 0.23，不显著）（`tmp/pls_audit2/use_vs_terms.py`）。新的 mHC 结构在独立头一侧落在趋势上（使用 1.00，损失 3.003）。它的代价高，是因为它的普通模型好（2.918）。
   - 全部 25 个结构（去掉 AttnRes 种子 0 和 DeepNorm 两个离群点）：代价与独立头损失的秩相关为 +0.73，与普通模型损失为 −0.02。所以跨所有结构，代价主要跟着独立头模型变（1 条流的结构独立头模型差）。只在 HC 与 mHC 之间，独立头模型一样，差别在普通模型。

4. 局限：除 HC 和 Pre-LN 外，每个结构只有一个种子。HC 普通模型在两个种子之间差 0.023（2.944 和 2.967），独立头模型差 0.009。新运行用较新的注意力和交叉熵栈（fa4thd）；HC 在两个栈上的代价是 0.048 和 0.047，Pre-LN 是 0.089 和 0.084。

5. 下一步只能经所有者弹窗决定。要回答的问题变了：不再是"HC 的独立头模型为什么好"，而是"mHC 的普通模型为什么比 HC 好"。4 个 mHC 变体的普通模型都在 2.913–2.926：混合（Sinkhorn 或单位矩阵）和读写映射（sigmoid 或线性）都不改变它。新 mHC 结构与 HC 还剩两个差别（`experiments/references/depth_arch_qwen3.py` 文档串）：(a) HC 的混合矩阵是学出来的（静态部分加每条流的 tanh 动态部分），新结构是单位矩阵；(b) 读写的动态部分，HC 由每条流各自算（s·tanh(RMS(H_s) W)），mHC 由所有流拼接、一起归一化后的一个投影算。HC 静态路由（所有路由权重都不随输入变化）的普通模型是 HC 中最好的（2.936），但只低于最好的 HC 运行 0.008，在种子差（0.023）之内。候选实验：HC 把混合固定为单位矩阵，只改 (a)。如果 (a) 让 HC 的普通模型变差，它的普通模型应接近 mHC（约 2.92），独立头模型仍约 3.00，代价应升到约 0.08。
   - 所有者弹窗（2026-11-01）：「HC 单位混合一对」，以及「加为图 50」。代码 33a3063f97：HC_RES=identity（depth_hc_res="identity"，H <- H + w ⊗ F，读写保持 HC 的）；CPU 测试 `scripts/della/depth_arch_hcres_test.py` 全部通过（初始化时与 HC 完全相同，res 和 w_res 的梯度为 0）；dry run 打印 depth_hc_res='identity'、48 层、4 条流。已提交 wq w810（独立头）和 w811（探针），2026-11-01 10:42 在 host 16541496 上开始，smoke 通过，W&B 配置是 depth_hc_res: identity。判据：普通模型降到约 2.92 则 HC 的可学习混合是普通模型较差的原因；仍在 2.94–2.98 则不是。
   - 网页（pretrain-practice 7062a1b）：图 50 画每个结构的普通模型和独立头模型（哑铃图，线段是代价）；图 38 加空柱、图 50 加空行作为 w810/w811 的占位（所有者弹窗批准）；记录表加 w810/w811 两行，图 50 的 16 个结构 32 个运行都标上图 50。`article-charts.js` 加了 text 标记和按行悬停（只增不改）。
   - 推送时遇到 `mmap failed: Permission denied`：两个新写的松散对象（69/3c67fa…、70/bde794…）能读不能 mmap。按原内容复制成新 inode 后正常（内容哈希不变）。

## 2026-10-31 审计补录：三个漏记的结果

所有者 2026-10-31：「我们现在开始整理和核对所有结果确保都已记录」。审计脚本在 `tmp/pls_audit2/`（`inventory.py`、`gaps.py`）。
它把 wq 里全部 179 个 pls 任务和 W&B 上 10-02 以后的 241 个 pls 运行（不含冒烟测试），与两个 YAML、本文档和网页记录逐条对照。
下面 3 个结果早已跑完，但本文档没有记录。

1. 中间层损失权重 1/47（wq w154，Pre-LN 48 层独立头，`PLS_W=0.0212766`，图 42）。
   - 问题：权重为 1 时，47 个中间层损失给骨干的梯度约是最后一层的 41 倍，余弦 0.96（见「梯度冲突」）。代价是否来自这个梯度的大小？
   - 方法：每个中间层损失的权重改为 1/47，使它们合起来与最后一层一样重。其他设置不变。参照是 Pre-LN 的探针运行，因为探针的权重不进入骨干。
   - 结果：第 48 层最终训练损失 3.025，探针 2.983，代价 0.042。权重为 1 时代价是 0.089。代价减半。
   - 中间层读出：第 1–47 层平均比权重为 1 时高 0.038 nats。第 7 层升得最多（+0.154）。第 36–47 层略低。它们仍远低于探针：与探针的平均差是 −0.825，权重为 1 时是 −0.863。
   - 漏记的原因：运行在 10-26 完成。取数脚本 `tmp/pls_blog/fetch_d48g4.py` 只会拼 `-pls1` 开头的运行名，所以没有取回，图 42 一直写着"排队中"。审计给脚本加了 `plsw<权重>:arch3:<中间名>:<层数>` 的写法，并取回了数据。
2. AttnRes (Full) 静态路由的独立头（wq w152，图 37）：第 48 层 3.024，探针（w28）2.913，代价 0.111。
   动态路由的 AttnRes：种子 0 是 −0.021，种子 1 是 +0.061（这两个值受普通模型尖峰影响）；源归一化后是 0.098（图 41）。
   Pre-LN 是 0.089，HC 静态路由是 0.074。所以去掉随 token 变化的路由后，AttnRes 的代价高于 Pre-LN。
3. AttnRes (Full) 伪查询尺度裁剪到 10 的独立头（wq w78，图 38）：第 48 层 3.015。不裁剪的 AttnRes 独立头：种子 0 是 3.023，种子 1 是 3.016。
   它的普通模型（w79）在第 659 步发散（第 48 层从 3.75 跳到 11.0），停在第 722 步，所以这一项没有代价。

同时整理了记录：
1. `logs/pls_depth_controls.yaml` 原来前半是一个映射、后半是一个列表，YAML 无法解析。现在用 `---` 分成两个文档。
2. 8 个条目的状态过时（running 或 queued），已按 wq 的状态改为 completed 或 running。
3. 4 个实验没有条目，已补上：w145/w146（梯度冲突）、w147–w150（Pre-LN 12/24 层）、w154（权重 1/47）、w495–w498（短轨迹）。
4. `logs/pls_speed.yaml` 的状态改为 completed，18 个写着 queued 的速度测试改为 done，6 个没有记录的速度测试补了一行。
5. 本文档 3 个写着"排队中"的标题已改为已完成。


数字核对（2026-10-31，Kaggle worker，报告 `tmp/pls_audit2/numbers_check.md`，重算脚本在同一目录）：
1. 范围：本文档「L6」一节到第 949 行、YAML 里 10-22 以后各条目的结果、记忆文件。共 790 个引用数字。
2. 737 个与数据一致。30 个不符，已全部改正（本文档 21 个，YAML 8 个，记忆文件 1 个），网页上有 1 个（图 38 的发散步数，站点 f91cb9f）。
3. 不符多数是最后一位的四舍五入。实质性的有 4 个：
   - 流之差消融与代价的秩相关是 −0.84，不是 −0.85。探针模型是 −0.80，不是 −0.82。第 17–34 层是 −0.47，不是 −0.49。
   - w79 在第 659 步发散，不是第 660 步。
   - mHC 路径分解里，流 3 占远距离路径的 0.29，不是"几乎不用"；用得最少的是流 1。结论"没有一条专门的远距离流"不变。
   - 四条短轨迹重跑彼此最多差 0.016 nats，不是 0.01 以内。
4. 23 个无法核对，原因是没有保存输出：
   - query 裁剪检查的 10 个数。`pls_qscale.py` 没有存输出，临时检查点已删除。
   - 记忆文件里 10-19 体检的 11 个数。worker 能解析的输出里没有它们的定义。
   - YAML 里体检的 2 个数。
5. 第 950 行以后的补录不在这次核对范围内。补录由另一个 worker 起草，我抽查了 6 个数字，全部一致。

## 补录：此前没有写进本文档的页面结果（2026-10-31 审计）

审计发现，网页上 49 张图里有 25 张的结果没有写进本文档，主要是 10-05 到 10-25 的实验。这些结果原来只在 YAML 和网页上。
下面各节由 Kaggle worker 根据 YAML、网页图数据和 JSON 起草（`tmp/pls_audit2/docs_gap_draft.md`）。我逐节核对，并抽查了数字。
审计期间找到的冲突都已解决，各节写明了改正了什么。来源标记：YAML = `logs/pls_depth_controls.yaml`；F:<图> = 网页图数据
（`assets/data/plc-figures.js`）；F:<图>/文字 = 网页图注；J:<文件>[键] = `tmp/pls_blog/` 下的 JSON（存档在 `docs/della/pls_archive/blog/`）；
"算" = 由来源相减或计数得到的数。

### 层损失权重与梯度范围（2026-10-05 至 10-07，作业 14998751–14998753、15064965、15064966）

1. 问题：独立头让最后一层的损失变高。原因是层损失的权重，还是层损失梯度的范围？
2. 方法：130m、6 层、独立头。R1：每层权重 0.2（`-pls0.2-sep`，14998751）。R2：每个头只训自己那一层（`-pls1-sep-local`，14998752）。R3：两者都用（`-pls0.2-sep-local`，14998753）。另有权重 0.1（`-pls0.1-sep`，15064965）和 0.05（`-pls0.05-sep`，15064966）（YAML）。`tmp/pls_blog/fetch_fix.py` 把数据取到 `fix.npz`。每个值是最后 50 步的平均训练损失。
3. 结果：
   1. 权重 0.2：L1–L6 为 4.199、3.747、3.555、3.420、3.357、3.322 nats（YAML；F:fig-fix 4.1994 … 3.322）。最后一层到基线的差距缩小 44%。第 1–3 层高 0.06 nats（YAML）。Paloma macro 4.287，c4_en bpb 1.1832（YAML）。
   2. 权重 0.1：L1–L6 为 4.264、3.787、3.622、3.465、3.356、3.298 nats。与同深度普通模型的差距平均 0.161 nats，最大 0.293 nats（YAML）。
   3. 权重 0.05：L1–L6 为 4.365、3.921、3.689、3.494、3.370、3.281 nats。差距平均 0.215 nats，最大 0.394 nats（YAML）。
   4. 权重 1、0.2、0.1、0.05 的平均差距：0.107、0.129、0.161、0.215 nats。最后一层的差距：0.110、0.062、0.038、0.021 nats（YAML）。
   5. R2（只训自己那层）：L1–L6 为 4.2884、3.8328、3.658、3.569、3.519、3.4909 nats（F:fig-fix）。YAML 原只记了第 4852 步的中途值 4.313 … 3.508，审计已补上最终值。
   6. R3（权重 0.2 + 只训自己那层）：L1–L6 为 4.365、3.965、3.715、3.536、3.412、3.354 nats。Paloma macro 4.350，c4_en bpb 1.1929（YAML；F:fig-fix 4.3652 … 3.3539）。
   7. 页面结论：权重每降一档，最后一层更接近基线，前几层离基线更远。把梯度停在头自己那层没有帮助，在权重 1 时让每一层都变差（F:fig-fix/文字）。YAML 写 "the weight is the lever, the gradient reach is not"。
4. 页面图：fig-fix。数据：`tmp/pls_blog/fix.npz`（`fetch_fix.py`）、`tmp/pls_blog/depth.json`（同深度普通模型）。

### 权重形状与梯度手术（2026-10-05 至 10-07，作业 15055614、15055616、15055618、15083130）

1. 问题：权重随深度增加，或去掉相互冲突的梯度分量，能否让每一层都低于权重 1 的独立头？
2. 方法：130m、6 层。A：权重 k/15（`-pls0.2-sep-shdepth`，15055614）。D：权重 0.2 加逐 token 梯度手术（`-pls0.2-sep-pcg`，15055616）。C：两者都用（`-pls0.2-sep-shdepth-pcg`，15055618）。E：权重 1 加梯度手术（`-pls1-sep-pcg`，15083130）（YAML）。冒烟测试 15055613（YAML）。判据：与同深度普通模型的逐层差距，看平均和最大（YAML）。
3. 结果：
   1. A：L1–L6 为 4.386、3.817、3.534、3.392、3.336、3.309 nats。差距 0.415、0.274、0.117、0.048、0.042、0.049，平均 0.158 nats，最大 0.415 nats（YAML）。
   2. D：L1–L6 为 4.213、3.728、3.550、3.424、3.358、3.323 nats。差距平均 0.128 nats，最大 0.242 nats。只用权重 0.2 的 R1 是 0.129（YAML）。
   3. C：差距平均 0.157 nats，最大 0.422 nats。A 是 0.158（YAML）。两对中，有手术和没有手术的逐层损失从第 1 步起相差 ≤ 0.01 nats（YAML）。
   4. E：L1–L6 为 4.139、3.668、3.496、3.418、3.384、3.367 nats。差距平均 0.107 nats，最大 0.168 nats。每层与权重 1 的独立头相差 ≤ 0.007 nats（YAML）。
   5. 2026-10-01 的探针运行（14808812/13）测得层损失与最终损失在骨干上的梯度余弦全程 0.8–0.9（YAML）。
   6. 页面结论：没有一种设置在每一层都低于独立头。随深度增加的权重让最后一层最低、前两层最差。梯度手术改变很小（F:fig-mix/文字）。
4. 页面图：fig-mix。数据：`tmp/pls_blog/fix.npz`。

### 晚开独立头（2026-10-07 至 10-08，作业 15168919）

1. 问题：先让头作为探针训练，到普通模型形成上下文电路之后再让梯度进入骨干。这样能否让每一层都低于从第 0 步开的独立头？
2. 方法：130m、6 层、权重 1。第 S 步之前骨干梯度为 0，第 S 到 S+100 步线性升到 1（`PLS_ON`）。S = 300（`-pls1-sep-on300-400`）和 S = 1500（`-pls1-sep-on1500-1600`），两者在一个 8-GPU 成对作业 15168919 里（YAML）。冒烟测试 15160666（YAML）。判据：每一层都低于从第 0 步开的独立头（YAML）。
3. 结果：
   1. 第 300 步开：L1–L6 为 4.162、3.665、3.496、3.415、3.379、3.362 nats。差距平均 0.109 nats，最大 0.191 nats（YAML）。
   2. 第 1500 步开：L1–L6 为 4.182、3.697、3.519、3.424、3.375、3.353 nats。差距平均 0.120 nats，最大 0.211 nats（YAML）。
   3. 从第 0 步开的独立头：差距 0.167、0.132、0.072、0.073、0.090、0.110，平均 0.107 nats（YAML）。
   4. 两种晚开都没有通过判据。第 300 步开：除第 1 层（+0.024 nats）外，与独立头相差 ≤ 0.01 nats。第 1500 步开：第 1–3 层高 0.02–0.04 nats，第 5–6 层低 0.01–0.02 nats（YAML）。
4. 页面图：fig-late。数据：`tmp/pls_blog/fix.npz`。

### 300m 独立头与探针（2026-10-03，作业 14940604、14888822）

1. 问题：130m 的逐层图形在 300m（12 层，宽 768）上是否仍成立？
2. 方法：300m 独立头 `-pls1-sep`（14940604）和探针 `-pls1-sep-bbfrozen`（14888822），各 11443 步（YAML）。
3. 结果：
   1. 独立头 L1–L12：4.047、3.560、3.355、3.231、3.143、3.093、3.064、3.046、3.034、3.026、3.020、3.016 nats（YAML；F:fig-traj-300m 4.0469 … 3.0156）。
   2. 最后一层比 300m 基线 2.9005 高 0.115 nats。130m 是 0.110 nats（YAML）。
   3. 探针第 12 层 2.8991，基线 2.8981，差 0.0009 nats（YAML）。探针 L1–L12：5.081 … 2.9015 nats（F:fig-heads-300m）。
   4. 共享头第 12 层 3.0399 nats（F:fig-heads-300m）。
4. 页面图：fig-traj-300m、fig-heads-300m。数据：`tmp/pls_blog/s300.npz`、`sep300.npz`（`fetch_sep300.py`）。

### 300m 头重训：差距在特征，不在头（2026-10-06，作业 15011512、15011513）

1. 问题：探针读早期层比独立头差。原因是头没训够，还是层里的特征不同？
2. 方法：冻结两个 300m 模型的骨干（第 11443 步），用新的优化器再训头 2000 步，warmup 100 步，cosine（YAML）。run id `-pls1-sep-headft2000`、`-pls1-sep-bbfrozen-headft2000`。`fetch_headft.py` 取 eval/L{k}/loss 和最后 50 步的训练损失。
3. 结果：
   1. 第 0 步的逐层损失与原运行的最终训练损失相差 ≤ 0.01 nats（YAML）。
   2. 训练损失：两个模型每层的变化都 ≤ 0.013 nats（YAML；J:headft.json[*.train]）。
   3. 独立头的 eval 每层都升高 0.017–0.072 nats（算：J:headft.json[-pls1-sep-headft2000.eval] 减 [-pls1-sep.eval]；YAML 写 +0.02..0.07）。
   4. 探针的 eval：第 1–8 层变化 ≤ 0.01 nats；第 9–11 层低 0.020–0.024 nats；第 12 层高 0.080 nats（算：J:headft.json；YAML）。
   5. 重训后第 1 层：独立头 5.378 nats，探针 6.696 nats（J:headft.json[*-headft2000.eval][0]）。
   6. 页面结论："The gap comes from the features of the layers"（F:fig-headft/文字）。页面原来还写 "No layer gets a better head"。探针第 9–11 层其实低 0.020–0.024 nats，所以审计把图注改为"重新训练几乎不改变任何一层"（站点 576cd37）。
4. 页面图：fig-headft。数据：`tmp/pls_blog/headft.json`。

### 降低最后一层代价的更多尝试（auto10，2026-10-16 至 10-19，作业 15698932、15698936、15699791、15700045、15700214、15699797、15772582、15805655、15755719）

1. 问题：除了降权重，有没有一种改动能让第 6 层到达普通模型，同时第 1 层留在独立头附近？
2. 方法：130m、6 层，分支 `pls-auto10`，脚本 `scripts/della/pls_auto10_pair.sbatch`（YAML）。14 种改动：1 单独裁剪（`-os`）；2 先归一化再混合 0.2（`-mix0.2`）；3 自蒸馏（`-kd`）；4 置信度门（`-conf`）；5 出口适配器 1024（`-ad1024`）；6 只更新 MLP（`-mlpo`）；7 只用前 256 个位置（`-first256`）；8 每步随机一层（`-randl`）；9 HC；10 AttnRes (Full)；11–13 只有第 1–n 层的读出进入骨干（`PLS_SEP_UPTO=3/2/1`）；14 n = 2 的种子 1（YAML）。kd 在第 3533 步停下，由 15755719 续跑（YAML）。量：最终训练损失减同深度普通模型（Sandwich-LN）。
3. 结果（第 1 层 / 第 6 层的差距，nats；权重 1 的独立头是 0.167 / 0.110）：
   1. 单独裁剪 +0.165 / +0.108；置信度门 +0.940 / +0.073；前 256 +0.318 / +0.165；随机单层 +0.379 / +0.130（YAML）。
   2. 混合 0.2 +0.870 / +0.155；适配器 +0.103 / +0.111；只更新 MLP +1.106 / +0.280（YAML）。
   3. 自蒸馏 0.578 / 0.698（YAML）。它从约第 100 步起落后于独立头：第 300 步 L6 7.02 对 4.21 nats。续跑处（第 3533 步）没有跳变（YAML）。
   4. HC 6 层：独立头 L6 3.339，探针 3.2135，自身代价 +0.125 nats。对 Sandwich 普通模型 +0.128 / +0.079（YAML）。
   5. AttnRes (Full) 6 层：独立头 L1–L6 4.134 … 3.367，探针 4.889 … 3.255。对基线结构的普通模型 +0.163 / +0.107。它自己的 6 层普通模型 3.255，基线 3.260（YAML）。
   6. 只有第 1–3 层进入骨干：差距 [0.105, 0.076, 0.088, 0.191, 0.196, 0.115]。第 1–2 层：[0.061, 0.084, 0.325, 0.382, 0.286, 0.082]（YAML）。
   7. 只有第 1 层：[0.024, 0.666, 0.730, 0.587, 0.272, 0.048]。第 1–2 层种子 1：第 1 层 0.070，第 6 层 0.080；种子 0 是 0.082（YAML）。
   8. 页面结论：没有一种改动通过判据。进入骨干的读出越少，最后一层代价越小，变成探针的那些层失去收益（F:fig-auto10/文字）。
4. 页面图：fig-auto10。数据：`tmp/pls_blog/auto10.json`（`fetch_auto10.py`）。

### Logit lens：最终头能读出各层多少（2026-10-17 至 10-19，作业 15720392–15720395、15787177、15795008）

1. 问题：探针和独立头显示一层知道什么。模型自己的输出头能否已经读出这一层？
2. 方法：`LENS_FROM` 读冻结的检查点，用最终 norm 和 lm_head 测每一层的留出损失（Paloma）（YAML）。脚本 `scripts/della/pls_lens_group.sbatch`，CPU 测试 `pls_lens_cpu_test.py`。作业 15720392（130m）、15720393（48 层 Sandwich-LN）、15720394（48 层其他结构）、15720395（300m）、15787177（130m 独立头的 8-GPU 重跑 15720165 的检查点）、15795008（lens-48c）（YAML）。`fetch_lens.py` 取 own 和 lens 两组损失。
3. 结果：
   1. 130m 独立头重跑 15720165 的最终 L1–L6 [4.1384, 3.6601, 3.4909, 3.4163, 3.3836, 3.3677]。原 4-GPU 运行 [4.1383, 3.6753, 3.4887, 3.4155, 3.3843, 3.3696] nats（YAML）。
   2. 130m lens 减自身读出：独立头 [1.471, 0.494, 0.244, 0.091, 0.026, 0.0]；探针 [2.861, 2.795, 2.246, 2.111, 0.592, 0.0] nats（YAML；J:lens.json[130m]）。
   3. 48 层 Sandwich-LN 独立头：lens 第 24 层比最后一层高 0.146 nats，第 36 层高 0.042 nats（算：J:lens.json[d48.sep.lens]）。YAML 原写 0.12 和 0.03，审计已改正。
   4. 48 层 Sandwich-LN 普通模型：lens 到第 40 层都在 5.3 nats 以上，最后 4 层下降（YAML；J:lens.json[d48.probes.lens] 第 40 层 5.313）。
   5. 300m 独立头：lens 第 9 层比最后一层高 0.099 nats，第 11 层高 0.035 nats（算：J:lens.json[300m.sep]）。YAML 原写"从第 9 层起约 0.04"，审计已改正。300m 普通模型：lens 第 11 层 4.146，最后一层 3.582 nats（J:lens.json[300m.probes]）。
   6. 12、24、36 层的普通模型（只有 lens）：倒数第 2 层 4.241、3.912、3.811 nats，最后一层 3.852、3.753、3.716 nats（J:lens.json[depth]）。
   7. 其他结构：HC、mHC 成功。AttnRes、AttnRes (Block)、MoDA 在 15720394 里显存不足（rc 1），因为 lens 不设 PDP，microbatch 是每卡的整个 batch（YAML）。重跑 15795008（PDP=1）的 6 个评测也全部失败：NCCL AllToAll 报 Cuda failure 2 out of memory（2026-10-19 10:52–11:18）。之后没有再跑。J:lens.json 里没有这三种结构。图 17 的三个面板现在写"显存不足；见图 34"，审计前写的是"排队中"。
   8. 页面结论：独立头把中间层放进输出头的坐标；普通模型只在最后四分之一进入。九种结构中，只有 HC 的普通模型在四分之三深度就进入（F:fig-lens/文字；F:fig-lens-arch/文字）。
   9. 15720392 的 d1 运行 rc 134，已去掉。它留下的 21 个 `-lensat` 目录（11.6 GB）已删除。15720393–15720395 的 `-lensat` 目录留给所有者删除（YAML）。
4. 页面图：fig-lens、fig-lens-arch。数据：`tmp/pls_blog/lens.json`（`fetch_lens.py`）。

### 多 token 预测在更大的模型上（MTP 阶梯，2026-10-15 至 10-21，作业 15591470–15591507、15714847、15907475、15907916）

1. 问题：130m 上 MTP 不改变下一个词的损失。300m、520m、1.2b 上是否不同？
2. 方法：每个尺度比较 NTP 和 NTP + 一个 DeepSeek-V3 MTP 模块（权重 0.3）。基线训练（`PLS_W=0`），8 GPU，一个种子，分支 `pls-mtp@5aea76151c`，脚本 `scripts/della/pls_mtp_ladder_submit.sh`（YAML）。训练后在冻结的最终检查点上训练新探针（`PROBE_FROM`）：130m 2000 步（15591508/09），300m 2000 步，1.2b 1000 步（YAML）。主模型的量是 Paloma eval/loss（`fetch_mtpl.py`）。
3. 结果：
   1. 主模型 eval 损失（NTP / MTP）：130m 3.97616 / 3.97545；300m 3.58374 / 3.58392；520m 3.33882 / 3.33409；1.2b 3.11632 / 3.11723 nats（J:mtpl.json[*.ntp, *.mtp]；YAML）。
   2. 520m 的 MTP 低 0.0047 nats。130m 和 300m 相差 ≤ 0.001 nats（YAML）。1.2b 相差 0.0009 nats（算）。页面把 520m 的差读作运行间噪声（F:fig-mtpl/文字）。
   3. 参数量（不含探针和 MTP 模块）：154,154,240、299,669,760、602,509,312、1,431,443,456（J:mtpl.json[*.params]）。
   4. 130m 的事后探针重现训练中的探针。MTP 减 NTP：第 5 层下一个词 +0.085 对 +0.080 nats，第 6 层后一个词 −0.185 对 −0.183 nats（`fetch_mtpl.py` 文档字符串）。
   5. 300m（12 层）后一个词探针：MTP 减 NTP 从约 60% 深度开始变负，最后一层 −0.262 nats。下一个词探针最大偏离 0.088 nats（算：J:mtpl.json[300m.probes]；F:fig-mtpl-depth）。
   6. 1.2b（16 层）后一个词探针：最后一层 −0.377 nats。下一个词探针：第 8 层 +0.152 nats，最后一层 +0.003 nats（算：J:mtpl.json[1_2b.probes]；F:fig-mtpl-depth）。
   7. 训练时间：300m NTP 2:52 h，MTP 3:40 h；520m NTP 11:41 h，MTP 14:34 h（YAML）。520m 和 1.2b 在每个 microbatch 32 条时 OOM；改为 520m 16 条、1.2b 8 条（YAML）。
   8. 1.2b 探针：第一次（PDP 2）冒烟 OOM，需要 109.30 GiB。改为 PDP 8 加 XLA slop 因子 70 后通过（YAML）。NTP 15907475 在 2026-10-21 10:12 rc 0，用了 16.5 h，因为宿主节点 della-k16g1 的 GPU 2 降频到 345 MHz（YAML）。
   9. 520m 事后探针（15716064 → 15738658，15699612 → 15738657）无效：冻结的最终头在训练路径上从第 2 步起读出 9.8 nats，eval 是 3.547 nats。原因与 48 层的 FSDP 问题相同。这两个运行已从图和 mtpl.json 去掉（YAML）。
4. 页面图：fig-mtpl、fig-mtpl-depth。数据：`tmp/pls_blog/mtpl.json`（`fetch_mtpl.py`）。

### 520m 探针重跑与保留权重的 130m 探针（2026-10-21 至 10-25，wq w12、w13、w14）

1. 问题：在 `PARAM_MAP=vocab` 下重跑 520m 事后探针（w12 NTP，w13 MTP）。再跑一次 130m NTP 探针并保留探针权重，用于写入方向检验（w14，L5）（YAML）。
2. 方法：`pls-mtp` 的 `scripts/della/pls_probe_fetchable.sbatch`，8 GPU（wq:w12–w14 spec）。run id `muonh-qwen3-520m-della8xh100-pls0-noev{,-mtp0.3}-vs-probes1000at9917`、`muonh-qwen3-130m-della8xh100-pls1-sep-bbfrozen-os-fut-probes2000at4958-keep`（`logs/slurm/*_w1[234].out` 第 14 行）。冒烟门：eval 的最后一个读出等于 eval loss；训练路径的最后一个读出 < eval + 0.5（YAML）。
3. 结果：
   1. 三个作业都完成，exit 0：w12 2026-10-24 07:53，w13 2026-10-25 05:17，w14 2026-10-25 01:37（wq:w12–w14 state）。YAML 原写 pending，审计已改正。
   2. w12 冒烟：eval loss 3.547，最后读出 3.5471，差 0.0001。训练路径最后读出 2.675（`logs/slurm/mtpl-probe-520m-ntp_w12.out` 第 10–11 行）。
   3. w13 冒烟：eval 3.534，最后读出 3.5335，差 0.0005。训练路径 2.676（`mtpl-probe-520m-mtp_w13.out` 第 10–11 行）。
   4. w14 冒烟：eval 4.223，最后读出 4.2227，差 0.0003。训练路径 3.250（`mtpl-probe-130m-ntp-keep_w14.out` 第 10–11 行）。
   5. 结果原来没有取回：`fetch_mtpl.py` 仍读第一次（无效）的运行名，图 21 还写 520m 探针"正在重跑"。审计改了运行名并取回（站点 2e9da4c）。重跑有效：冻结的最后一层在训练批上读 2.66 nats，eval 是 3.34 nats，与 130m、300m、1.2b 的训练–评测差相同。
   6. 520m（24 层）MTP 减 NTP：下下个词的探针从约一半深度开始变负（第 12 层 −0.050），第 24 层 −0.363 nats。下一个词的探针在中间几层最多 +0.165 nats（第 13 层），最后一层 +0.001 nats（J:mtpl.json[520m.probes]）。
   7. w14 保留了探针权重，检查点在 `marin_store_big/speedrun/muonh-qwen3-130m-della8xh100-pls1-sep-bbfrozen-os-fut-probes2000at4958-keep/2026.09.13/checkpoints/step-1999`。L5 的写入方向分析原来一直没有做；所有者弹窗 2026-10-31 批准后已做，结果见下一节，图 22 已填（站点 332a9f8）。
4. 页面图：fig-mtpl-depth、fig-mtp-dirs（占位）。数据：`tmp/pls_blog/mtpl.json`（`fetch_mtpl.py`）。

### L5：最后几层的写入落在哪里（图 22，2026-10-31，所有者弹窗「做图 22 的分析」，只用 CPU）

1. 问题：130m 普通模型里，下下个词探针从第 5 层到第 6 层只改善 0.09 nats，下一个词探针在最后一层仍改善 0.23 nats。最后一层是不是把写入避开了下下个词探针读取的方向？（2026-10-21 调研的预测：是。）
2. 方法：`marin-pls-mtp/scripts/della/pls_write_dirs.py`（pls-mtp b5fbe97269），读 w14 保留的探针（检查点 step-1999），Paloma c4_en 4×1,024 个 token，vis2 CPU，170 s。
   - 全词表探针的秩是 512，任何方向都会改变它的 logits，所以"探针读取的方向"按方向改变探针预测的程度加权：在探针输入处用它的 Fisher 度量 s_P(u) = Var_{y~p}(J u)_y，J 是 RMSNorm 加读出头的 Jacobian。
   - 比值 ρ = 写入方向的 s_P ÷ 随机方向的 s_P。ρ > 1 表示写进探针读取的方向，ρ < 1 表示避开。第二组随机方向给出估计噪声：0.996–1.005。
   - RMSNorm 去掉径向（沿状态的）分量，而写入的径向占比很高（第 1 层 0.93，第 2–5 层 0.45–0.60，第 6 层 0.35），所以只用写入的切向部分计算 ρ。两种探针的径向因子相同，所以两种探针之间的比较不受影响。
3. 结果（切向部分的 ρ，下一个词探针 / 下下个词探针）：第 1 层 1.06 / 1.09，第 2 层 1.09 / 1.01，第 3 层 1.34 / 0.92，第 4 层 1.99 / 2.72，第 5 层 3.08 / 3.15，第 6 层 2.50 / 4.82。
   - 第 1–3 层的写入和随机方向差不多。从第 4 层起，写入落在两种探针读取的方向上。
   - 最后一层的写入最偏向下下个词探针（4.82），高于下一个词探针（2.50）。预测被推翻：最后一层没有避开下下个词探针。
   - 同一段文本上，下下个词探针第 5 层 6.009、第 6 层 5.969 nats，只改善 0.04 nats。所以这点小改善不是因为写入避开了探针。
   - 因果读法（用第 k 层的探针读 h_{k−1}）：第 6 层的下下个词探针从 5.97 升到 7.40 nats。这个读法混入了探针与输入分布不匹配的影响，只能说明去掉最后一层的写入不会让这个探针变好，也就是没有"主动压制"的迹象。
4. 页面图：fig-mtp-dirs（图 22）。数据：`tmp/pls_mtpdirs/write_dirs.json`（存档 `docs/della/pls_archive/blog/write_dirs.json`），日志 `tmp/pls_mtpdirs/run2.log`。

### Kimi K3 的 logit lens（2026-10-17 至 10-19，作业 15726408、15736820）

1. 问题：AttnRes 的读数在层间跳动。在 2.8T 参数、93 层、Block AttnRes 的 Kimi K3 上是否也这样？
2. 方法：`scripts/della/k3_lens.py`（`k3_lens.sbatch`，分支 `pls-auto10@e84a10ccc5`）。64 × 1,024 个 WikiText-103 测试 token。每层三种读法：下一层读的混合（next）、在此处停下时的输出混合（exit）、各块的直接和（sum）；另记 AttnRes 权重（YAML）。冒烟 15726408（pli-cp，1 GPU，4 层）；全量 15736820（8 GPU）。
3. 结果：
   1. 15736820 在 2026-10-19 00:55–02:26 完成，8 个 rank，90.3 min（YAML；sacct）。
   2. 经最终头的读数到第 78 层都在 10.2–13.9 nats（YAML）。
   3. next / exit：第 84 层 6.2–6.6 nats，第 91 层 1.45 nats，第 93 层 0.41 nats（YAML；J:k3lens.json[layers][90].next_ce 1.451，[92].next_ce 0.406）。
   4. 第 93 层 top-1 准确率 0.90（J:k3lens.json[layers][92].next_acc 0.9027）。YAML 认为 WikiText 可能在 K3 的训练数据里。
   5. AttnRes 混合跳动：next 有 11 次升高 > 0.3 nats，最大 1.39 nats。各块的直接和平滑，最大升高 0.21 nats，最后 6.0 nats（YAML；J:k3lens.json[layers][92].sum_ce 6.008）。
   6. 权重文件 `models/Kimi-K3`（1.5 TB）等待删除。工具的权限检查拒绝了 rm（YAML）。
4. 页面图：fig-k3lens。数据：`tmp/pls_blog/k3lens.json`。

### 项目暂停与目标改变（2026-10-18、2026-10-19，所有者）

1. 2026-10-18：所有者在 K3 lens 之后暂停项目（「做完K3的探针就先停一停吧」）。已排队的 15736820、15805655、15755719、15795008、15738657/8、15760472/3、15779949/50 跑完；填入结果；不交新实验（YAML）。
2. 2026-10-19（/goal）：放弃"让独立头的最后一层追上普通模型"这个目标。新目标：先对 K3 和我们的 AttnRes 做全面体检，再对十种 48 层 DepthBench 结构做内部检查，比较普通模型和独立头（YAML）。页面图 23–29 用 `checkup/`，图 30–34 用 `depthmed/`（YAML）。20 个 48 层检查点保留到检查结束（YAML）。
3. 2026-10-18 磁盘清理：删除 65.6 GB（冒烟副本、1.2b 的 step-20000 检查点、`tmp/pls_blog/compare`）（YAML）。

### AttnRes 全面体检：Kimi K3 和我们的模型（2026-10-19，作业 15844650、15855315）

1. 问题：在同一段未见过的文本上，K3 和我们的 48 层 AttnRes (Block)、AttnRes (Full)（普通模型和独立头）在读数、混合权重、查询和写入上有什么不同？
2. 方法：K3 用 `marin-pls-auto10/scripts/della/k3_checkup_v2.py`（作业 15844650，2026-10-19 14:01–14:15）和 `k3_checkup_v3.py`（作业 15855315，16:19–16:32）（sacct）。第一次的作业 15817869 被取消（它的日志）。我们的模型用 `attnres_checkup.py`（只读，读 `-vs-d48-g4` 的 step-4958）。文本：Paloma c4_en validation；K3 130,944 个 token，我们的模型 16,320 个 token（J:checkup/*.json[tokens]）。YAML 原来没有这些作业号，审计已补上一个条目。
3. 结果：
   1. 最后一层的 next 读数：K3 2.045，Block 普通模型 3.411，Full 普通模型 3.491，Block 独立头 3.516，Full 独立头 3.480 nats（J:checkup/*.json[ablations.baseline.ce]）。
   2. 输出混合在嵌入上的权重：Block 普通模型第 1–6 层 0.88–0.92，第 48 层 0.09（J:checkup/block.json[layers][k].exit_probs[0]）。第 48 层输出混合的熵：Full 普通模型 3.72 nats，Block 普通模型 1.31 nats（J:checkup/*.json[layers][47].exit_entropy）。
   3. 独立头把输出权重移到较老的块：从第 24 层起占 0.61–0.83（Block）和 0.73–0.91（Full）。K3 的输出头平均把 0.82 放在当前块，最后一层 0.84；每块的最后一层降到 0.07–0.37（F:fig-ck-mix/文字）。
   4. Attention 读取：Block 普通模型平均给当前块 0.41、嵌入 0.15。独立头从第 24 层起给较老的块 0.70（Block）和 0.79（Full）。K3 平均给当前块 0.73；每块第一层读刚结束的块 0.30–0.91（F:fig-ck-read/文字）。MLP 读取：Block 普通模型当前块 0.43；独立头较老的块 0.51（Block）和 0.53（Full）；K3 当前块 0.78（F:fig-ck-read-mlp/文字）。
   5. 子层查询与输出查询的平均余弦：普通模型 0.105（Block）、0.145（Full）；独立头 0.486（Block）、0.583（Full）；K3 0.193（算：J:checkup/*.json[queries.cos_out] 去掉最后一项后取平均）。独立头从第 15 层（Block）或第 18 层（Full）起，每个 attention 查询的余弦都 > 0.9（F:fig-ck-query/文字；记忆文件）。
   6. 写入：Block 普通模型每块最后一层的写入是其他层的 1.43 倍，独立头是 0.71 倍。Full 普通模型第 44 层有一个写入是中位数的 52 倍。K3 每块最后一层的写入是其他层的 2.2 倍（attention）和 3.35 倍（MLP），与输出查询的余弦是 −0.15 和 −0.14（F:fig-ck-write/文字）。
   7. 每个混合都用等权重时的最终损失：K3 9.62，Block 普通模型 10.92，Full 普通模型 17.01，Block 独立头 5.05，Full 独立头 6.60 nats（J:checkup/*.json[ablations.uniform_all.ce]）。去掉嵌入：K3 16.16 nats，Block 普通模型 15.08 nats。只留最近两块：K3 13.96 nats，我们的四个模型 11.89–15.91 nats（同文件 no_embedding、recent_only）。
   8. 只在第 1 块用等权重：K3 2.425 nats（+0.38），Block 普通模型 8.199 nats（+4.79）（算：J:checkup/*.json[ablations.uniform_block_1.ce] 减 baseline）。页面写 K3 的嵌入代价全部在第 1 块的读取里（F:fig-ck-ablate/文字）。
4. 页面图：fig-ck-lens、fig-ck-mix、fig-ck-read、fig-ck-read-mlp、fig-ck-query、fig-ck-write、fig-ck-ablate。数据：`tmp/pls_blog/checkup/{k3,block,full,block-sep,full-sep}.json`。

### 十种结构的内部：独立头改变了什么（2026-10-19，作业 15844436、15849760、15855981）

1. 问题：在十种 48 层 DepthBench 结构上，独立头怎样改变删层代价、写入大小、直接贡献、logit lens 和状态的形状？
2. 方法：`marin-pls-auto10/scripts/della/depthmed.py`（只读，float32）。文本：Paloma c4_en validation 16 × 1,024 个 token；删层用前 8 条序列（`depthmed.py` 文档字符串）。作业 15844436（八种结构完成；AttnRes (Full) 的两个模型显存不足失败，所以 sacct 记 FAILED，日志最后仍打印 "all done"）、15849760（AttnRes）、15855981（`--delete`，真删层）（sacct；日志）。作业写到 `tmp/pls_status/depthmed/`，页面读 `tmp/pls_blog/depthmed/`。
3. 结果：
   1. 删掉第 48 层的代价（`skip_ce_delete − base_ce`）。普通模型：Pre-LN 0.495、LNS 0.324、DeepNorm 0.440、KEEL 0.416、mHC 0.448、MoDA 0.394、Sandwich 0.073、AttnRes (Block) 0.048、HC 0.003、AttnRes (Full) 0.000 nats。独立头：十种都 ≤ 0.0021 nats（算：J:depthmed/*.json[layers][47].skip_ce_delete 减 [base_ce]）。记忆文件写 "0.32-0.49 nats in six ord designs"，与前六种一致。
   2. 删掉后代价低于阈值的层数，普通模型 / 独立头（算：J:depthmed/*.json[layers][k].skip_ce_delete 减 [base_ce]）。阈值 0.01 nats：Sandwich 0 / 30，Pre-LN 0 / 31，LNS 1 / 31，DeepNorm 0 / 13，KEEL 0 / 28，HC 12 / 21，mHC 8 / 27，AttnRes (Full) 5 / 25，AttnRes (Block) 0 / 3，MoDA 0 / 31。阈值 0.005 nats 时，独立头是 3–28 层。记忆文件原写"13–28 层"，没有写阈值，任何一个阈值都对不上，审计已改为这里的数。
   3. HC 普通模型第 1 层：真删（混合也变为单位矩阵）的代价 5.688 nats；只把写入置零、保留混合的代价 0.616 nats（算：J:depthmed/hc-ord.json[layers][0].skip_ce_delete 8.942、skip_ce 3.870，base_ce 3.254）。
   4. 写入比：AttnRes (Full) 普通模型第 44 层 MLP 1,339 倍，第 47 层约 1,006 倍。它的独立头模型最大是第 22 层 42 倍（J:depthmed/attnres-*.json[layers][k].mlp_write_ratio；F:fig-dm-write/文字）。
   5. 最后一个 MLP 对正确 token 的直接贡献（`mlp_dla_w`）。普通模型：Pre-LN −1.93、LNS −1.57、KEEL −2.79、mHC −2.44、MoDA −1.66 logits。这五种的独立头在 ±0.04 logits 内（J:depthmed/*.json[layers][47].mlp_dla_w）。
   6. 第 24 层：普通模型的 lens 是 5.05（HC）到 10.18（AttnRes (Full)）nats，自身探针 3.81–5.16 nats；独立头的 lens 是 3.61–4.32 nats（J:depthmed/*.json[layers][23].lens_ce、own_ce；F:fig-dm-lens/文字）。
   7. 第 48 层的有效秩：多数模型 330–400；HC 独立头 63.6（J:depthmed/*.json[layers][47].eff_rank）。HC 独立头的最终损失 3.299 nats，是十个独立头模型中最低的（J:depthmed/hc-sep.json[base_ce]）。最大的单个激活在 AttnRes (Block) 普通模型第 44 层，是 RMS 的 637 倍（F:fig-dm-shape/文字）。
4. 页面图：fig-dm-skip、fig-dm-write、fig-dm-dla、fig-dm-lens、fig-dm-shape。数据：`tmp/pls_blog/depthmed/<结构>-{ord,sep}.json`。

### hc_pos 检查点的 CPU 分析（2026-10-17，作业 15536267 之后）

这一节补 fig-hc-write 的缺口。docs 的 E1 表已有 hc_pos 的代价 0.035。
1. 结果：在 step-4958 上，hc_pos 的写入全部为正，但流之差的份额仍然高。删掉流之差后的依赖与 HC 一样大（独立头在 c = 8 时 +4.19 nats）（YAML）。
2. YAML 的结论："the causal test does not remove the storage"。页面写这个检验没有确定原因（F:fig-hc-write/文字）。
3. 页面图：fig-hc-write。数据：YAML（我没有找到对应的 JSON 文件）。

### 脚本与分析目录索引

审计日期：2026-10-31。只读。docs = `docs/della/pls.md`（974 行），YAML = `logs/pls_depth_controls.yaml`（554 行）。行号按这个版本。

| 名称 | 是什么 | docs 是否记录 | 位置 |
|---|---|---|---|
| pls_d48_eval_path.py | 诊断（10-08）：48 层探针 checkpoint 走训练 run 自己的 eval 函数和 ReadoutTaggedEvaluator，按 run 的设备数跑 | 否（docs 只写 PARAM_MAP=vocab，不写原因） | docs「## Next: depth」第 336 行只提开关；YAML 88、92、94、97、98、100、108、143、146、149、151 |
| pls_d48_optcoupling_cpu.py | 诊断（10-07）：48 层、小宽度、MuonH 分组裁剪时，探针模型的骨干是否与普通模型相同 | 部分（耦合与 -os 有记录，模拟结果没有） | docs「## L6：优化器的实际设置」第 4 条、「## Next: depth」第 440–442 行；YAML 80 |
| pls_d48_paths_cpu.py | 诊断（10-07）：同一权重上，逐层前向（scan_via）与普通前向（fold）在 48 层是否相同 | 否 | YAML 82、83、85 |
| pls_d48_paths_probe_ckpt.py | pls_d48_paths_cpu.py 的变体（10-08）：加载 48 层探针 checkpoint（step 4958），只用 JAX_FLASH | 否 | YAML 85、87、88 |
| pls_gradmix_cpu_test.py | CPU 测试：PLS_SHAPE（权重随深度）与 PLS_PCGRAD（逐 token 梯度投影） | 开关门控测试 | YAML 54、70 |
| pls_hc_hidden_ablate_cpu.py | 10-14：HC/mHC 终点 checkpoint，第 c 层后把每条流换成流均值，测第 48 层 CE 升高 | 是（记 job 15534928，不记脚本名） | docs「## Next: depth」第 383–391 行；YAML 224 |
| pls_hc_mixing.py | 10-30：HC 48 层 checkpoint 的残差混合 A_v，记 sum_gain、leak、diff_gain 和静态读写 | 否 | 无（输出在 tmp/pls_paths/mix-*） |
| pls_heads_only_cpu_test.py | CPU 测试：HEADS_FROM / pls_heads_only（冻结骨干，只训头） | 开关门控测试 | YAML 40、61、65 |
| pls_lateon_cpu_test.py | CPU 测试：PLS_ON（晚开：骨干梯度从 0 线性升到全量） | 开关门控测试 | YAML 70 |
| pls_local_cpu_test.py | CPU 测试：PLS_LOCAL（每层损失只训自己的头和自己的层） | 开关门控测试 | 无 YAML 行名脚本 |
| pls_optsplit_cpu_test.py | CPU 测试：PLS_OPT_SPLIT=1（辅助头和辅助 norm 进入自己的优化器组） | 开关门控测试（开关在 docs 有记录） | docs「## L6：优化器的实际设置」第 4 条；YAML 80 |
| depth_arch_hcrw_test.py | CPU 测试：HC_RW=sigmoid（HC 用 mHC 的读写映射） | 开关门控测试 | YAML 400（只写开关） |
| depth_arch_mhcrw_test.py | CPU 测试：MHC_RW=linear（mHC 用 HC 的读写映射） | 开关门控测试 | YAML 445 |
| depth_arch_nan_probe.py | CPU 探针：48 层 hc/mhc/preln 的前向损失、每层最大幅度和几步 MuonH，找 NaN 出现的位置 | 否 | YAML 29、32 |
| depth_arch_variant_test.py | CPU 测试：ROUTE=static（hc、attnres 静态路由）与 GAIN_INIT=lns | 开关门控测试 | 无 YAML 行名脚本 |
| depth_inner_remat_test.py | CPU 测试：DEPTH_INNER_REMAT=save（两层 scan 的内层保存中间量） | 是（记 CPU 结果，不记脚本名） | docs「## 速度：GPU 结果（续，2026-10-26）」第 815 行 |
| tmp/pls_conflict | conflict.py（scripts/della/pls_conflict.py 的前身）；test-preln.json（CPU，256 token）；gpu/*.json（w145/w146，8×1024 token） | 是 | docs「## 梯度冲突」、「## 新作业（所有者弹窗 2026-10-25，队列为 0 时）」第 696–707 行；YAML 509–515 |
| tmp/pls_diag | train_eval_readout.py（训练与 eval 读出对比）；moda_bench.py、prof/、hlo/、grad_moda*.npy（MoDA 速度基准与布局修复）；prof_top.py | 否 | YAML 51、188、189 |
| tmp/pls_fill | fill_test.py（8 条流 HC、3/6 块 AttnRes Block 的 CPU 检查） | 是 | docs「## 补充 4 个 job」第 752 行 |
| tmp/pls_gain | lsck.py（列 checkpoint 键）；sep.py（独立头模型 Pre-LN/LNS 增益比）；equiv*.log（LNS 精确等价 CPU 测试输出） | 是 | docs「## L1：增益起点与步长各决定约一半的增益差距…」第 4 条、「### L1 完整版：Pre-LN 精确等价于 LNS」表；YAML 307 |
| tmp/pls_hidden | hc_hidden.py（隐藏份额）；hc1_test.py（1、2 条流 HC 的 CPU 检查）；9 个 json/log | 部分（mHC 两行没有记录；一处冲突） | docs「## HC 的隐藏份额」；YAML 348 |
| tmp/pls_muon_eff | muon_eff.py；Pre-LN、HC 48 层 json/log；Pre-LN 12 层试跑 | 是 | docs「## MuonH 更新效率」、「## 中途结果（2026-10-26）」第 760 行 |
| tmp/pls_qscale | pls_qscale.py（AttnRes 伪查询尺度 s）；snap0.txt（内容是 wq 状态和冒烟测试行，不是 pls_qscale 的输出） | 是 | docs「## A3：结构变体与 HC 深度扫描」第 6 条 |
| tmp/pls_static_src | src_rms.py、run.sh、static.json、dynamic-s1.json、run.log（AttnRes 每个源的 RMS） | 是 | docs「## L2：AttnRes 静态路由」第 661–667 行 |
| tmp/pls_paths | paths-*（pls_branch_paths.py）、ws-*（pls_write_size.py）、ha-*/has-*（pls_hidden_ablate.py）、mix-*（pls_hc_mixing.py）、plot/（hidden_vs_cost*、causal_gap_live*）、run_*.sh、summarize.py、tab_has.py | 部分（ws-* 的 6 个 HC/mHC 变体、mix-*、causal_gap_live 没有记录） | docs「## Gated Residual 报告与 mHC 的路径分解」、「## 离线机理分析」；YAML 452–461 |

#### pls_d48_eval_path.py

1. 问题：48 层探针 run 的逐层 eval 与主 eval 不一致。run 在同一权重上记录 eval/L47 c4_en 6.479，主 eval c4_en 4.231 [YAML 108]。
2. CPU，1 个设备，T=256：grug eval batch 与 with_segment_ids batch 的损失都是 6.1936。ReadoutTaggedEvaluator 累加的 L47 也是 6.1936 [tmp/optsplit/evalpath_ev1.log；YAML 94]。CPU，8 个设备（15220720），T=128，B=16：主 eval = 读出 L47 = 6.3277 [YAML 98]。
3. GPU，参数不分片（15235521、15220612）：8 卡 4.2039，4 卡 4.2529，1 卡 3.7848。每个程序的读出 L47 都等于主 eval [YAML 108、149]。GPU，4 卡，FSDP 分片（15255032、15264999）：48 个读出的程序给出 L47 4.2378 和 4.2737，主 eval 是 4.2529 [YAML 143、151]。rep 和 vocab 布局下每个程序都是 4.2529 [YAML 151]。
4. YAML 记录的原因：堆叠层权重的 FSDP 分片。修复：PARAM_MAP=vocab [YAML 151]。600 步验证（15265000）：eval/loss 4.550，eval/L47 4.5501 [YAML 146]。

#### pls_d48_paths_cpu.py

1. 问题：T3（15140369，48 层探针）结束时，普通路径 eval/loss 5.725，读出路径 eval/L47/loss 7.570，权重相同 [YAML 82]。
2. CPU，T2 checkpoint（15140368_4，step 599），随机序列：fold 与 scan_via 的最终隐状态差在 fp32 下是 0，在 bf16 下是 2.861e-06。两条路径的损失在 fp32 下都是 12.6096，在 bf16 下都是 12.6083。VANILLA 和 JAX_FLASH 结果相同 [tmp/optsplit/paths_cpu_T4096.log]。
3. YAML 的结论：CPU 上 fold 与 scan_via 完全一致，逐层循环在舍入误差内 [YAML 82]。

#### pls_d48_paths_probe_ckpt.py

1. CPU，探针 checkpoint（-pls1-sep-bbfrozen-d48 step 4958），Paloma c4_en 前 8192 个 token，带文档隔离：普通路径 3.5847，读出 L47 3.5847。读出 L0 7.0166，L23 4.9773，L46 3.5809 [tmp/optsplit/probe_real_seg_cpu.log]。YAML 写作 3.585 [YAML 87]。
2. 没有文档隔离时，这个模型在 CPU 上得 8.29 [YAML 87]。
3. GPU，1 张 H100（15207626）：普通路径 = 读出 L47 = 3.785（前 16384 个 token），4.253（前 65536 个 token） [YAML 88]。

#### pls_d48_optcoupling_cpu.py

1. 设置：宽 64，48 层，100 步，骨干初始化相同 [logs/optcoupling_d48.txt]。
2. 裁剪阈值 1.0：探针与普通模型的骨干最大差在第 99 步是 1.628e-01。最后 20 步平均最终损失：普通模型 1.8435，探针 1.8434 [logs/optcoupling_d48.txt]。
3. 不裁剪：骨干最大差在第 99 步是 1.481e-04。平均最终损失都是 1.8776 [logs/optcoupling_d48.txt]。
4. YAML 把耦合记为候选原因，没有证明。修复开关是 PLS_OPT_SPLIT=1 [YAML 80]。后一条记录写：优化器拆分不是主因 [YAML 82]。

#### pls_hc_mixing.py

1. 输入：6 对 HC 48 层 checkpoint（step 4958），每对独立头和探针：4 条流、4 条流种子 1、1/2/8 条流、sigmoid 读写 [tmp/pls_paths/run_mix.sh]。
2. 每个子层 v（96 个）记录 sum_gain、leak、diff_gain、scale_rr、scale_w 和静态读写 [tmp/pls_paths/mix-*.json，键 rows]。
3. 汇总（独立头 / 探针）。来源：tmp/pls_paths/mix-<tag>.log 最后一行。例外：mix-sep-ahc-vs-d48-g4.log 不存在，这一格由本审计用脚本最后一行的同一公式从 mix-sep-ahc-vs-d48-g4.json 算出。

   | checkpoint | 平均 leak | 平均 diff_gain | v ≥ 80 平均 sum_gain | log10 Π sum_gain |
   |---|---|---|---|---|
   | HC 4 条流 | 0.160 / 0.184 | 1.064 / 1.061 | 1.038 / 1.103 | −1.03 / −3.68 |
   | HC 4 条流，种子 1 | 0.119 / 0.438 | 1.075 / 1.188 | 1.065 / 3.735 | −1.21 / 4.90 |
   | HC 1 条流 | 0 / 0 | 0 / 0 | 1.489 / 0.985 | 2.51 / −0.76 |
   | HC 2 条流 | 0.059 / 0.079 | 0.971 / 0.932 | 1.132 / 1.138 | −0.13 / −0.08 |
   | HC 8 条流 | 0.384 / 0.267 | 1.244 / 1.174 | 1.242 / 0.987 | −1.75 / −12.38 |
   | HC sigmoid 读写 | 0.184 / 0.212 | 1.114 / 1.090 | 1.084 / 0.954 | −1.05 / −6.54 |

4. 结论（2026-10-30）：静态混合的乘积噪声大，不能解释代价。HC 1 条流的独立头模型在后期子层把残差每层放大约 1.5 倍（v ≥ 80 的 sum_gain 1.489）。

#### depth_arch_nan_probe.py

1. 48 层冒烟测试 14969857：hc 在第一步出现 NaN，mhc 在第 11–30 步内出现 NaN [YAML 29]。
2. CPU 探针（48 层，小宽度）：前向和 6 步 MuonH 都是有限值 [YAML 29]。
3. 审计找到原因：hc/mhc 对称初始化上的 Adam eps 1e-20，以及朴素 Sinkhorn。提交 8e6240cb56 修复 [YAML 32]。

#### tmp/pls_diag

1. train_eval_readout.py：AttnRes Block 冒烟测试中 eval L1 是 10.6，训练是 7.36。CPU 检查同一个 batch：训练读出与 eval 读出在 f32 下差 0，在 bf16 下差 ≤ 0.002 [YAML 51]。
2. moda_bench.py：MoDA 在 microbatch 1 时每步 104 s [YAML 188]。
3. 布局修复（604cdc18c7），1 张 A100，只算骨干，seq 4096，48 层前向加反向：旧 4830 ms，修复后 1647 ms，去掉深度项的 MoDA 789 ms，Pre-LN 285 ms。新旧 f32 梯度相对差 4e-4 [YAML 189]。

#### tmp/pls_hidden（补 mHC 两行；隐藏份额的两种定义）

1. mHC 48 层隐藏份额，平均 / 下半 / 上半：独立头 0.35 / 0.38 / 0.32，探针 0.36 / 0.40 / 0.32 [tmp/pls_hidden/130m-della4xh100-pls1-sep-amhc-vs-d48-g4.json 与 …-sep-bbfrozen-amhc-vs-d48-g4.json，键 hidden_share，本审计取平均]。
2. HC 48 层独立头的隐藏份额有两个值，两者都对，定义不同。
   - 0.26（第 1–47 层平均，job 15534022，2×512 个 token）[YAML 224；docs「## Next: depth」第 377 行]。
   - 0.40（48 层平均），第 1–47 层平均 0.407（4×1024 个 token）[docs「## HC 的隐藏份额」表；tmp/pls_hidden/130m-della4xh100-pls1-sep-ahc-vs-d48-g4.json，键 hidden_share]。
   - 同一 checkpoint 的另外三个值一致：HC 探针 0.59 对 0.589，mHC 0.35 / 0.36 对 0.355 / 0.365 [YAML 224；上面两个 json 和 …-sep-bbfrozen-ahc-vs-d48-g4.json，第 1–47 层平均]。
   - 原因（审计核对代码）：10-14 的 `scripts/della/pls_hc_hidden_cpu.py` 先把所有 token 的平方范数加总再相除，按能量加权；10-25 的 `tmp/pls_hidden/hc_hidden.py` 和 10-30 的 `pls_hidden_ablate.py` 先算每个 token 的比例再取平均。范数大的少数 token 主导第一种算法。同一检查点按 token 平均三次都在 0.40 左右（4×1024 个 token 0.407，1×1024 个 token 0.403，`ha-sep-ahc-vs-d48-g4.json` 0.407）。

#### tmp/pls_paths（只补没有记录的部分）

1. 每层写入量，6 个 HC/mHC 变体（tmp/pls_paths/run_ws2.sh，2026-10-30）。量：最后 8 层 rel_write 的平均，独立头 / 普通模型。来源：tmp/pls_paths/ws-<tag>.json，键 rel_write，本审计算平均。这个算法对 docs 表中的 5 个结构给出相同的值（例如 HC 0.202 / 1.254）。
   - HC 1 条流：2.855 / 0.145
   - HC 2 条流：0.708 / 0.353
   - HC 8 条流：0.839 / 0.665
   - HC 种子 1：0.272 / 95.097
   - HC sigmoid 读写：0.239 / 0.527
   - mHC 线性读写：0.025 / 0.233
   - 结论（2026-10-30）：在这几个 HC 变体之间，最后几层的相对写入大小与代价不同步。HC 种子 1 的普通模型 95.1 是一个离群值。
2. mix-*：见上面「pls_hc_mixing.py」。
3. plot/causal_gap_live.py：从 W&B 读 w711–w714（还在训练）的实时代价，画 causal_gap_live.png。没有数值输出文件，所以这里没有数字可记。

#### 开关门控测试（每个一行）

1. pls_gradmix_cpu_test.py：门控 PLS_SHAPE 和 PLS_PCGRAD。通过记录：YAML 54（「CPU tests 20/20」）、YAML 70；tmp/optsplit/pls_gradmix_cpu_test.log（「ALL PASS」）；docs/della/pls_audit_2026-10-07.md 第 5 行。它门控的 run（-shdepth、-pcg，YAML 66–68、77）的结果在 docs 中没有记录。
2. pls_heads_only_cpu_test.py：门控 HEADS_FROM / pls_heads_only。通过记录：tmp/optsplit/pls_heads_only_cpu_test.log（「ALL HEADS-ONLY CPU TESTS PASSED」）；提交 69bc662575。300m 头重训的结果（YAML 65）在 docs 中没有记录。
3. pls_lateon_cpu_test.py：门控 PLS_ON。通过记录：tmp/optsplit/pls_lateon_cpu_test.log（「30 passed, 0 failed」）；YAML 70；docs/della/pls_audit_2026-10-07.md 第 7 行。晚开 run 的结果（YAML 115）在 docs 中没有记录。
4. pls_local_cpu_test.py：门控 PLS_LOCAL。通过记录：tmp/optsplit/pls_local_cpu_test.log（「all pls_local tests passed」）；提交 a4fe04941a、a23157fb62。YAML 57、58 记录了「own-layer heads」的 R2、R3 结果，logs/ 下有 -pls1-sep-local 和 -pls0.2-sep-local 两个 run 目录；docs 中这两个结果没有记录。
5. pls_optsplit_cpu_test.py：门控 PLS_OPT_SPLIT=1。通过记录：YAML 80（「probes backbone == baseline (diff 0) at L=6 and 48」）；提交 d69ff2d433 的说明。
6. depth_arch_hcrw_test.py：门控 HC_RW=sigmoid。通过记录：提交 825326986c 的说明（初始化时与 hc 的相对差 1.8e-7）。YAML 400 只写开关，不写测试。
7. depth_arch_mhcrw_test.py：门控 MHC_RW=linear。通过记录：YAML 445（「ALL OK」）。
8. depth_arch_variant_test.py：门控 ROUTE=static 和 GAIN_INIT=lns。通过记录：提交 30da580d69 的说明（「depth_arch_variant_test.py all OK」）。
