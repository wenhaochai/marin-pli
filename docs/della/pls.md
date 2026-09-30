# pls：kaiyue baseline 上的逐层 LM 监督（dense、每层、全程）

**状态（2026-09-30 13:3x）**：130m 完成——每层权重 1、全程开的逐层 LM 监督让最终层 c4_en bpb 变差 0.041（对 8 个
baseline run 的池，约 130 个 sd）、Paloma macro 变差 0.185，差距随训练扩大，上层几乎不再贡献；300m 同向（step 8000：
+0.039 bpb），预计 17:10 完成。

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

| arm | 130m | 300m |
|---|---|---|
| pls1（每层权重 1，全程） | 1 run | 1 run |

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

**300m（进行中，截至 step 8000/11444）**：c4_en bpb +0.0385、macro +0.173（对 1 个 baseline），同样随训练扩大（step 2000
+0.019 → 8000 +0.039）；逐层 macro L8 4.136、L9–L11 4.134——最后 4 层合计贡献约 0.002 nats。

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

**预测检验**：预测 1（最终层差 0.01–0.05 bpb）成立，落在上沿；预测 2（中间层读出远好于未受监督的 logit lens、且逐层
单调）成立，L0 好 3.7 nats；预测 3（300m 代价不小于 130m）到目前为止成立。
