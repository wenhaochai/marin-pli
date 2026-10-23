# pls：kaiyue baseline 上的逐层 LM 监督（dense、每层、全程）——已结题（2026-10-01）

**一句话**：在 kaiyue 的 dense MuonH baseline 上，给每一层加 next-token loss 要么伤最终层（全程开：
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
   stream energy, mean over layers 1-47: HC separate heads 0.26, HC probes 0.59, mHC 0.35 / 0.36. HC's separate-heads model
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

1. weight_decay = 0.1 写在 config 里，但没有任何组用到它：0.1 和 0.0 的一步更新逐位相同。
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
   Block AttnRes 的第一块（"死段"）两种都有高 s 读取（探针 25 个，独立头 12 个）。
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

### L1 完整版：Pre-LN 精确等价于 LNS（2026-10-22，wq w61、w62，排队中）

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
token（求均值）。输出：`tmp/pls_status/attnres_massive/{full,full-s1,full-sep}.json`，状态图 `pls_archive/blog/a2_massive_plot.py`。
对每个模型取每 token RMS 均值最大的源（子层输出），在写入时替换它。

| 模型 | 最大的源 | 相对中位数 | 与均值的 cos | RMS 变异系数 | 换成文本 B 均值 | 置零 | 每 token 缩放到平均 RMS |
|---|---|---|---|---|---|---|---|
| 普通模型（探针）seed 0 | L44.mlp | 52,810× | 1.000 | 0.05 | +0.0000 | +8.95 | −0.0000 |
| 普通模型（探针）seed 1 | L47.mlp | 1,511× | 0.961 | 0.57 | +0.0014 | +0.0018 | +0.0008 |
| 独立头 seed 0 | L22.mlp | 26× | 0.234 | 2.43 | +0.0080 | +0.0083 | +0.585 |

1. seed 0 的巨大源是一个常数偏置：每个 token 都写同一个向量，换成均值不改变损失，置零损失增加 8.95 nats。读它的是第 48 层 MLP，
   这一层的输出也是常数（与均值的 cos 1.000）；把 L48.mlp 输出换成均值，损失不变。
2. 这个源只属于 seed 0。seed 1 最大的源小 35 倍，由输出头读取，置零只增加 0.0018 nats。所以“巨大常数源”不是 Full AttnRes 的结构
   性质，是一个种子学到的偏置通路。
3. 独立头模型没有巨大源（最大 26×）。它的最大源只在少数 token 上很大（p90 的 cos 0.99，中位数 0.09）：值本身不重要（置零 +0.008），
   但把每个 token 缩放到平均大小的代价是 0.585 nats，信息在“哪些 token 写得大”里。只有一个独立头种子，所以“独立头去掉巨大源”还不成立。
