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

## A3：结构变体与 HC 深度扫描（2026-10-23，wq w76–w83，排队中）

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

## E1：本质是什么——读出是否看到后面各层用的全部状态（2026-10-23，wq w87–w90，排队中）

起因：所有者说“一定要往本质思考，本质上到底是为什么这些设计上的差异产生了结果上的差异”（已写入 CLAUDE.md）。

第 48 层最终训练损失（d48g4.npz，单种子；代价 = 独立头 − 探针）：

| 结构 | 独立头 | 普通（探针） | 代价 | 普通模型第 36 层 |
|---|---|---|---|---|
| Sandwich / Pre-LN / LNS / KEEL | 3.07–3.13 | 2.98–3.03 | 0.089–0.102 | 3.22–3.40 |
| DeepNorm | 3.372 | 3.262 | 0.110 | 3.75 |
| HC / hc_pos | 2.992 / 2.989 | 2.944 / 2.954 | 0.048 / 0.035 | 3.007 / 3.021 |
| mHC | 3.024 | 2.922 | 0.102 | 3.61 |
| AttnRes (Full) s0 / s1 | 3.023 / 3.016 | 3.045 / 2.955 | −0.021 / +0.061 | 3.61 / 7.0 |
| AttnRes (Block) | 3.066 | 2.953 | 0.113 | 4.25 |
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
3. A3 query 裁剪（w79，普通模型，s ≤ 10）：第 456、520 步有尖峰，第 660 步发散：第 48 层损失从 3.76 跳到 8.29，第 720 步 7.62，
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
  - HC 静态路由（w151 完成，2026-10-27 03:32）：独立头第 48 层 3.011，探针（w27）2.936，代价 0.074。动态路由的 HC 是 0.048，Pre-LN 0.089。去掉随 token 变化的路由，HC 的代价从约一半升到约四分之三：低代价一部分来自随 token 变化的门，一部分来自多条流本身（静态路由仍有 4 条流和固定的读写）。Fig 37 的 HC 面板标签已更新（站点 76dbb28，artifact v110）。AttnRes 静态路由（w152）还在跑。
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
- 1/8（w180 完成，2026-10-26）：第 48 层 3.142，代价 0.159。四个点：看全部 0.089，1/2 0.105，1/4 0.130，1/8 0.159。代价随读出看到的坐标减少单调上升，每减半多 0.016、0.025、0.029 nats，越往后增得越多。中间层读出损失同向上升（L12：3.24、3.291、3.377、3.518；L36：3.080、3.119、3.192、3.330）。Fig 40 的第四个点已上网页（站点 101d172，artifact v109）。

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
- 另一个观察（d48g4.npz，普通模型的探针损失）：HC 普通模型在中间层已经能很好地读出（48 层的第 24 层探针 3.365，Pre-LN 4.154，mHC 4.380）。上半层探针与最后一层的平均差：HC 0.12，HC_pos 0.20，LNS 0.26，Sandwich 0.33，Pre-LN 0.48，mHC 0.70。HC 两个变体的差最小，代价也最小；其余结构的差从 0.26 到 1.26，代价都在 0.09–0.11，不随差变化。所以这不是一条通用规律，只说明 HC 的普通模型本来就接近“每层都能读出”。
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
- 判断：3 个 run 分不开 kernel 的系统偏差和轨迹噪声。如果是系统偏差，大小约 0.006，小于项目里要比较的效应（E1 每减半 0.016–0.029，HC 与 Pre-LN 差 0.041）。新组合只用于两边都用新组合的对比，这样的偏差会抵消。
- 新 run 用不用新组合，等弹窗。Pre-LN 完整新组合（fa4thd + v3 + UNNORM + 串行读出 + 每个 microbatch 8 条）的测速和显存在 w492–w494（独立头 time、mem，探针模型 time）。
- 逐步比较（同一种子、同一数据顺序，第 48 层训练损失减去 JAX flash 种子 0；`tmp/pls_speed/fa4_traj.py`，图 `tmp/pls_status/newstack_drift.png`）：
  - 独立头：新组合全程在 ±0.008 内，结束时 −0.0006。种子 1 中途到 +0.016，结束时 +0.0021。
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
  | w334 | +0.035 | +0.042 | +0.033 | +0.023 | +0.020 | 约 +0.005 |

  - 按事先写下的判据（全开那条不重现偏移，w334 的差别就是偶然）：w334 的早期偏移是那一次运行的轨迹，不是组合。四条重跑彼此在 ±0.01 内，第 300–400 步都回到种子 0 的水平。
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
- 新分析（所有者弹窗 2026-10-27，`scripts/della/pls_branch_paths.py`，vis1 CPU，8×1024 token）：mhcid 没有混合，所以第 v 个子层的输入精确拆成 a(u→v, c) = r_c(v) w_c(u) |f(u)|。用记录的写入量重建最后的流之和，相对误差 6e-8。图 47（fig-mhc-paths，站点 58d5a3c）。
- 结果（独立头 w76 对普通模型 w77）：
  - 份额按跳过的层数分组，独立头减普通模型（对 96 个读取方平均）：跳过 ≤1 层 −0.014，1–6 层 −0.127，6–24 层 +0.019，>24 层 +0.122，嵌入 −0.039。平均跳过层数 14.9 对 10.5。
  - 写入量（Σ_c w_c × |f|，每 8 层平均）：普通模型 4.4、3.2、5.6、9.9、21.3、40.5，随深度上升；独立头 25.6、33.2、18.3、11.1、9.0、7.7，前面的层写得最多，后面越来越少。
  - 读出（流之和）在第 48 层：来自最后 4 层的份额，独立头 0.023，普通模型 0.284。
  - 没有 GR 那样“一条远距离流”的分工：独立头模型的流 2 承担大部分远距离路径（份额 0.39），流 3 几乎不用。
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
- 结论：巨大的源引起 AttnRes 普通模型的尖峰；去掉尖峰以后，AttnRes 的代价（0.098）与 Pre-LN、mHC 等结构（约 0.09–0.10）相同。之前较小的代价（0.061、−0.022）来自普通模型被尖峰拖慢。图 41 已填，图 48 的图注补上这一点（站点见提交记录）。
- 这也补上了图 48 的缺口：AttnRes 独立头的后期改变很小，真实代价也在常见的 0.10 左右，和“后期写入小 → 代价高”一致。到目前为止只有 HC 例外。
- AttnRes 静态路由的独立头（w152 完成，2026-10-27 19:15）：第 48 层 3.024，探针（w28）2.913，代价 0.111。静态路由的普通模型没有尖峰，代价略高于 Pre-LN（0.089），和源归一化的动态 AttnRes（0.098）接近。AttnRes 两种路由去掉尖峰以后都付常见的约 0.10；HC 静态 0.074、动态 0.048，仍是唯一低于 Pre-LN 的结构。图 37 标签和图注已更新（站点 f32d559）。
- HC 读写 = sigmoid 的普通模型（w390 完成，2026-10-27 22:50）：第 48 层最终训练损失 2.976。HC 普通模型是 2.944（旧路径）/ 2.956（新 CE 路径），所以有界、为正的读写让普通模型变差约 0.02–0.03。代价要等独立头 w389（排队，从第 2724 步续跑）。
- fa4thd 重跑（w501 完成，2026-10-28 ~01:00）：Pre-LN 普通模型（种子 0，今天的代码，fa4thd）第 48 层 2.9884；w334（fa4thd）是 2.9883，JAX flash 两个种子 2.9832 / 2.9826。fa4thd 两次都比 JAX flash 高约 0.005，重复得很好。JAX flash 重跑（w500，第 ~3900 步）决定这 0.005 来自 fa4thd，还是来自今天代码的 CE 一侧。图 45 已加点（站点 d5199ee）。
- JAX flash 重跑（w500 完成，2026-10-28 04:09）：Pre-LN 普通模型（种子 0，今天的代码，JAX flash）第 48 层 2.9870。fa4thd 重跑 w501 是 2.9884，只高 0.0014；两个旧的 JAX flash 种子是 2.9832 / 2.9826，彼此差 0.0006。
  - 结论：w334 比旧运行高的 0.005 里，约 0.004 来自两次重跑共有的改动（今天的 CE 侧：A1、GIF、lse kernel，或别的代码变化），fa4thd 只占约 0.001。这与 HC 新 CE 路径整体上移 0.011（w387/w388）方向一致。
  - fa4thd 本身：Pre-LN 12.16 对 14.96 s/步（−19%，w288），损失 +0.0014。是否作为新任务对的默认，弹窗给所有者。
  - 图 45 说明改成结果（站点 9a4d838）。
- 所有者弹窗（2026-10-28 ~11:20）：以后新的任务对默认用 fa4thd。现在不交新任务；代码默认值等所有正在跑和排队的任务结束后再改（排队的任务启动时读仓库代码，现在改会改变它们的设置）。
- AttnRes Block 24 个块（w223 独立头 / w224 探针，完成）：第 48 层 3.044 / 2.956，代价 0.089。8 个块是 0.114，Pre-LN 0.089。探针模型除了所有运行都有的第 1391 步（数据）只有 2 个尖峰步（1235、2029；判据：比前 50 步中位数高 0.1 以上），Full 种子 0 有 49 个。所以下降不是尖峰造成的：每次读取混合的源越多，AttnRes 的代价越低，最低到 Pre-LN 的水平。图 44 说明改成结果（站点 cb5d0a9）。
- 训练到 50% 时停掉中间层损失（Pre-LN，w503 完成）：第 48 层 3.0265；探针模型 2.9832，全程独立头 3.0725。剩下的代价 0.043，约是全程代价（0.089）的一半：停掉以后最后一层在剩下 50% 的步数里恢复了约一半。停掉以后中间读出不再训练，第 1–47 层损失升到 27–62 nats（预期）。注意 w503 用今天的代码，比较对象是旧代码运行；按图 45，今天的代码约高 0.004。图 46 已加点，说明等 80% 两条（w504、w505）。
- 训练到 80% 时停掉中间层损失（Pre-LN，w504 完成，第二次重排后没有再 OOM）：第 48 层 3.054。剩下的代价：50% 时 0.043，80% 时 0.071，全程 0.089，约与带中间层损失训练的比例成正比。HC 80%（w505）还在跑。
- 发布审计（controller，2026-10-28）：7be7eaf 上的记录过时；scrub + 泄漏扫描通过，marin-pli 3bdc53a20；记录重建，站点 d22ff55。
- HC 读写 = sigmoid（GR 式的读写，w389 独立头 / w390 探针，完成）：第 48 层 3.020 / 2.976，代价 0.044。HC 新路径（w387/w388）0.047，旧路径 0.048。代价不变；两个模型的绝对损失都比新路径 HC 高约 0.02。所以 HC 的低代价不依赖读写用 tanh 还是 sigmoid。图 38 已加点，说明等 mHC 线性读写（w510/w511）。
- HC 8 条流独立头（w221 完成）：第 48 层 2.999；代价等探针 w222。
