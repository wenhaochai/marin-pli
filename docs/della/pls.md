# pls：kaiyue baseline 上的逐层 LM 监督（dense、每层、全程）

**状态（2026-10-01 01:0x）**：每层权重 1、全程开的逐层 LM 监督让最终层 c4_en bpb 变差 0.034–0.044，130m 三种读出都一样
（对 8 个 baseline run 的池，约 100 个 sd）：共享输出头 +0.041、只读共享头 +0.044、每层独立的头 +0.034；300m 共享头
+0.043，代价不随规模变小。代价主要来自每一层都被逼成预测器、顶层不再精修；共用一个读出坐标系只占约 0.007。共享坐标的
两臂在前 ~130 步（约 3%）有超出 seed 噪声的早期优势，之后翻转——下一步研究按动态指标调整 weight schedule。

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
