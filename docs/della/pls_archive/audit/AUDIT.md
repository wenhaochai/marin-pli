# pls 严格审计（per-layer LM supervision，VARIANT=pls）

审计对象：worktree `/scratch/gpfs/GROUP/USER/project/marin-pls`，分支 `pls`。主审 `1fed9c9c29`（相对 `d0cb3cbdb8`）；
审计期间分支前进到 `fe7d082ed4`（`aa7d1ee9f7` 只加 `docs/della/pls.md`；`fe7d082ed4` 改 smoke 默认 CONFIGS 为 `130m:1 300m:1`、
加入 `scripts/della/pls_report.py`）。`experiments/` 与 `lib/` 在这两个提交里没有变化，本报告覆盖 HEAD `fe7d082ed4`。
审计全程没有改任何 tracked 文件、没有提交、没有提交 Slurm 作业；所有计算在 della-vis1（CPU，`nice -n 19`）上跑。
审计脚本与日志都在 `/scratch/gpfs/GROUP/USER/tmp/pls/audit/`。

## 结论

没有 BLOCKER，也没有 MAJOR。loss、stop-gradient 监控、逐层 eval、`train_lm` 钩子都与约定的 spec 一致，已用一个
**只走 baseline 代码路径**的独立参考、真实 trainer 的 bf16 混合精度 loss wrapper、levanter 的 microbatch 梯度累加、
以及 `train_lm.main` 端到端（含 checkpoint 续跑）逐项验证。发现 4 个 MINOR（1 个潜在代码路径崩溃、3 个脚本健壮性）
和若干 NOTE（成本、解读、实验设计）。正在排队的 smoke（14728893，CONFIGS `130m:1 300m:1`）不受这些 MINOR 影响，可以照跑。

## 发现

### 1. MINOR：`scan_layers=False` 时训练直接崩溃（潜在路径，launcher 不会走到）

- 位置：`experiments/references/per_layer_qwen3.py:115`
  `_, outs = tr.layers.scan_via(step)(x, mask=attn_mask, key=keys, pos_ids=None)`。
- 原因：`scan_layers=False` 时 `tr.layers` 是 `BlockSeq`；haliax 的 `BlockSeq.scan_via`（`lib/haliax/src/haliax/nn/scan.py` 约 298–314 行）
  把 kwargs 原样传给每一层，不像 `BlockSeq.fold` 那样按层切片，于是每层都拿到整个 `[L, 2]` 的 key 数组。
- 复现（`audit_misc.py` D1）：`ValueError: split accepts a single key, but was given a key array of shape (3, 2) != (). Use jax.vmap for batching.`
  `key=None`（eval / 读出）不受影响；`scan_layers=True`（默认，launcher 用的就是它）正常。
- 修法：在 `__post_init__` 里 `assert self.scan_layers`；或者对 `BlockSeq` 手写循环 `for i, layer in enumerate(tr.layers.unstacked())`，传 `keys[i]`。

### 2. MINOR：smoke 的 run id 写死，重跑时会续上旧的 W&B run 或旧 checkpoint，SUMMARY 可能读到上一次的值

- 位置：`scripts/della/pls_smoke.sbatch:31–52`。与 `over_vocab_smoke.sbatch` 不同，这里没有 `RUN_TAG=-j${SLURM_JOB_ID}`，
  run id 靠字符串拼出来（第 41 行 Python 与第 52 行 `rm -rf`），不是从 launcher 的 `DRY_RUN=1` 输出取的。
- 失败场景：
  - (a) 同一 config 第二次 smoke。W&B tracker 用的是 `resume="allow"`（`lib/levanter/src/levanter/tracker/wandb.py:347`），
    会续上同一个 W&B run，summary 会保留上一次写入的键。上一次如果被 Slurm 杀掉、`rm -rf` 没来得及跑，或者留下了
    temporary checkpoint，trainer 就会从旧 checkpoint 续跑（它会同时搜 permanent 和 temporary 两个根目录，`TrainerConfig.checkpoint_search_paths`，`trainer.py:936–946`）。
    temporary checkpoint 在 `$MARIN_PREFIX/tmp/checkpoints-temp/$MARIN_PREFIX/speedrun/<run_id>/…`，第 52 行的 `rm -rf` 根本不碰这个目录；
    300m smoke 超过 10 分钟就会写 temporary checkpoint，正常结束时的 permanent save 会把它清掉，但被 `timeout` 杀掉时不会。
    如果续上的 checkpoint 已经在第 40 步，trainer 走 "Training already complete. Running final hooks only"，一步训练都不跑，
    而 SUMMARY 里的 `mfu`、`tok_s`、`train/*` 是上一次的旧值。rc=0，看起来是通过的。
  - (b) PLS_W 不是 launcher `:g` 的写法（例如 `130m:1.0`，launcher 生成的 id 是 `-pls1-`），或者提交 shell 里漏进了会改 id 的变量
    （SEED、EMA_BETA、TIE、PRECISION、RUN_TAG、DEVICE_TAG、INIT_FROM）。这时 W&B 查询和 `rm -rf` 都会落空，产出目录留在盘上，下一次 smoke 就会触发 (a)。
  - (c) 提交环境里有 `SMOKE_STEPS=0` 时，`${SMOKE_STEPS:-40}` 保留 0，launcher 把它当成**正式 run**：用正式 run id，14 分钟后被杀，
    `rm -rf …-smoke0` 什么也删不掉，正式 run 的 W&B 与 checkpoint 目录被一次半截的 smoke 污染。
- 安全性：`rm -rf` 的目标由写死的 `MARIN_PREFIX` 加字面量 `-smoke<N>` 组成，**删不到正式 run 的产出**。
- 当前 14728893 不受影响：两个 id（`…-130m-…-pls1-smoke40`、`…-300m-…-pls1-smoke40`）已在 W&B 与盘上确认不存在。
- 修法：照 OV smoke 的写法。`export RUN_TAG=-j${SLURM_JOB_ID}`；`run_id=$(DRY_RUN=1 $PY -m … | sed -n 's#^output: speedrun/\([^/]*\)/.*#\1#p')`；
  W&B 查询和 `rm -rf` 都用这个 `run_id`，同时删掉 `$MARIN_PREFIX/tmp/checkpoints-temp$MARIN_PREFIX/speedrun/$run_id`；
  开头加 `[[ $SMOKE_STEPS -gt 0 ]] || exit 1`。

### 3. MINOR：`pls_h100x4.sbatch` 会继承提交环境里的 VARIANT

- 位置：`scripts/della/pls_h100x4.sbatch:25`，`export VARIANT=${VARIANT:-pls}`。提交方式是 `--export=ALL,SIZE=..,PLS_W=..`，
  提交 shell 里要是还留着 `VARIANT=ov`，这个作业就会在 pls worktree 里训练 OV，而且 run id 是 `muonh-qwen3-130m-della4xh100-ov12.8m`。
  那个 run 现在正在跑（W&B state=running）。结果是两个作业续写同一个 checkpoint 目录。
- 同类问题：漏进来的 `TOTAL_STEPS` 会悄悄改 schedule 长度，run id 却不变（baseline 的 sbatch 也有这个暴露面）。
- 修法：`export VARIANT=pls`（smoke 已经这样写）；开头把 `SIZE`、`PLS_W`、`SMOKE_STEPS`、`TOTAL_STEPS`、`INIT_FROM`、`RUN_TAG` 打印出来，或者直接 unset 不该出现的变量。

### 4. MINOR：`pls_report.py` 默认 `--suffixes=-pls1,-pls0`，pls0 已经不跑了

- 位置：`scripts/della/pls_report.py`，argparse 默认值。`api.run(f"{PROJECT}/muonh-qwen3-130m-della4xh100-pls0")` 会抛异常，按默认参数运行直接崩。
- 修法：默认值改成 `-pls1`。

### 5. NOTE：成本（与 docs 的估计一致）

- 按代码自己的 FLOP 记账（`audit_misc.py` D3，V=128256，T=4096），pls1 每 token 前向 FLOPs：
  - 130m：baseline 的 3.88×（lm_head 占 baseline 的 58%）；
  - 300m：baseline 的 4.90×（lm_head 占 35%）。
- baseline 墙钟（W&B）：130m 3395 s，0.63 s/step；300m 17960 s，1.44 s/step。
- 300m pls1 估计约 20 h 以上，也就是要 10 个以上 2 h 的 pli-short 段。整条依赖链都算进"最多 16 个 pending GPU 作业"的额度；每一段还要重新编译 train、主 eval、读出 eval。
- `pls_h100x4.sbatch` 只是单段作业，不会自动续链。

### 6. NOTE：显存估计，OOM 风险低

在 CPU 后端只编译不执行（`audit_memory.py`），用真实的每卡形状：32×4096，V=128256，bf16 policy，launcher 的 mesh，
FSDP 参数切分，XLA memory_analysis 的 temp 大小如下：

| | baseline | pls1 | pls0 |
|---|---|---|---|
| 130m train value+grad | 13.2 GiB | 15.9 GiB | 14.7 GiB |
| 300m train value+grad | 22.0 GiB | 30.9 GiB | 26.5 GiB |
| 300m eval loss_fn | 5.5 GiB | 8.8 GiB | 8.8 GiB |

GPU 的 buffer 分配和 CPU 不同，这里应该只看增量：300m 多约 9 GiB，离 0.9×80 GB 还远。smoke 在真实形状下跑 train 和
eval（`max_eval_batches=1` 时 batch 形状与完整 eval 相同），可以作为最终确认。
另外，preallocation 打开时 nvidia-smi 看到的峰值恒在约 75.7 GiB（OV smoke 的 `peak_mem_MiB=75751` 就是这个），拿它判断余量没有意义。

### 7. NOTE：`eval/L{L-1}` 与主 eval 相等是"浮点意义上的相等"，不是逐位相等

- f32 下逐位置的 loss 逐位相同（原 CPU 测试第 4 节）。
- 端到端的 bf16 pipeline（`audit_e2e_train_lm.py`）里，12 个聚合键的最大相对差是 1.6e-7，来自 `[R,b,t]` 与 `[b,t]` 的求和顺序不同。
- 所以 run 内自检必须带容差（约 1e-5）。`pls_report.py` 打印 max|diff|，没问题。`docs/della/pls.md` 里"逐位相同 / 必须等于"的措辞应改成"相差约 1e-7"。

### 8. NOTE：pls0 的"逐位相同"只在 CPU 上成立；docs 里监控成本"约 2%"偏低

- CPU 上，bf16 与 f32、有无 microbatch，pls0 的 loss 和全部梯度都与 baseline 逐位相同（A3、B2）。
- GPU 上 pls0 编译出的 HLO 不同（scan 带 stacked 输出，外加监控），不保证逐位相同，只保证数学上等价。
- 按代码的 FLOP 记账，监控在 130m 多 6.0%、300m 多 8.1% 的训练 FLOPs；显存多 1.5 GiB（130m）和 4.5 GiB（300m）。docs 写的"约 2%"偏低。pls0 不跑，只影响文档。

### 9. NOTE：指标解读

- pls1 的 `train/loss` 是 L 个 CE 之和，约为 NTP 的 L 倍。和 baseline 比 NTP 要用 `train/pls/L{L-1}`（代码和 docs 都已写明）。
- `throughput/mfu` 把读出的 FLOPs 也算进去了，只能看硬件效率。算成本看 `tokens_per_second`，它会掉到 1/3 到 1/5。
- 逐层梯度加起来会让每个优化器组内的梯度范数变大，`clip_by_global_norm(1.0)` 会更常起作用（MuonH 对 muonh、adamh、adam 三组各自裁剪）。Adam、AdamH、Muon 对梯度尺度基本不敏感，这不是 bug，但 baseline 与 pls1 的差别里包含了这个效应。

### 10. NOTE：EMA、z-loss、progress 事件（当前计划都不触发）

- 读出 eval 只评估 `step.model`。`EMA_BETA>0` 时主 eval 还会记 `eval/ema/*`，但没有逐层的 EMA 读出。
- `_monitor_ce`（`per_layer_qwen3.py:149`）的 `logsumexp_weight=None`，而 `train/pls/L{L-1}` 在 `z_loss_weight>0` 时包含 z-loss，两者口径不一致。launcher 设的是 0。
- 读出 eval 不发 `EVALUATION_STARTED/FINISHED` 事件（主 eval 会发）。这个配置里 progress watchdog 的超时全是 None，所以没有影响。

### 11. NOTE（实验设计）：pls0 不跑以后，pls1 的逐层指标没有 baseline 可比

- baseline run 里没有 `eval/L{k}`。docs 的预测 2 提到"未受监督模型的 logit-lens 典型值"，这个值在当前计划里没人测。
- 130m baseline 的最终 checkpoint 还在盘上：`marin_store_big/speedrun/muonh-qwen3-130m-della4xh100-restore/2026.09.13/checkpoints/step-4958`
  （W&B finished，group `muonh-qwen3-della`，c4_en bpb 1.16322）。它的参数树与 `PerLayerQwen3LMHeadModel` 完全相同，
  可以用 `ReadoutTaggedEvaluator` 事后跑一遍逐层 eval，只得到最终一步的 logit-lens。
- 300m 盘上没有 baseline checkpoint。要拿到逐层对照，只能跑 pls0（+8% FLOPs）或者重跑 baseline。

### 12. NOTE：和共享 venv 的环境耦合

- 运行时 jax、jaxlib 以及 native wheel（`dupekit_native`、`iris_native`）都来自 `project/marin/.venv`。
- 另一个会话如果在两段之间 `uv sync`，一条可续跑的作业链中途就换了底层版本。
- sbatch 每段都会打印 jax 版本和 `levanter.__file__`，保留这一行；最好再记一行 `pip freeze | md5sum`。

## 已验证无误（做了什么检查）

1. **loss 语义**：spec 公式；用模型自己的 final RMSNorm 与 lm_head，二者接收所有读出的梯度；k=L-1 就是 NTP；z-loss 参数透传；
   按权重的均值；loss_weight 掩码；默认 gradient checkpointing；RNG key 与 baseline 一致。
   - 原测试 `scripts/della/pls_cpu_test.py` 在我这里 6 节全过（42 s，`orig_cpu_test.log`）。
   - `audit_loss_test.py` 另建了一个**完全不经过 per_layer_qwen3 代码**的参考：对每个 k，把 stacked 权重切出前 k+1 层，
     构造 baseline 的 `Qwen3LMHeadModel`，走 baseline 自己的 `compute_next_token_loss`（fold + final norm + fused CE），
     再经过 levanter 的 `WrappedLossFunction`，在 launcher 的 4 卡 mesh 上比较：
     - f32：loss 和逐层 stats 相对差为 0，梯度 6e-7；
     - bf16 policy（`p=f32,c=bfloat16`）：loss 和 stats 逐位相同；梯度差在 bf16 噪声以内（`audit_bf16_yardstick.log`：
       pls、参考、baseline 三者相对 f32 的 bf16 误差同为 0.6–4%）；
     - microbatch 2×4（`levanter.grad_accum.microbatched`）加 bf16：loss 和 stats 等于两个 microbatch 各自参考值的平均
       （相对差 3.6e-8），和 baseline 的 loss 一样是 microbatch 均值的等权平均。
2. **监控路径（w=0）**：
   - 采样位置 0、s、2s…；目标 x_{t+1} 是先 roll 再抽样，没有 off-by-one；权重用的是 next-token 权重。
   - bf16 加 microbatch 下，监控值等于抽样位置上的 baseline 参考（相对差 3.6e-8）。
   - loss 与**全部梯度**跟 baseline 逐位相同（有无 microbatch 都是），stop-gradient 没有漏到 trunk、final norm 或 lm_head。
   - batch 维在 4 卡上切分；position 维不切分。
3. **ReadoutTaggedEvaluator**：
   - 逐行对照 `TaggedEvaluator`：RunningMean 在 R 维上的广播、逐 tag 掩码、bpb、macro 与 micro、out_sharding、state 初始化与切片、`construct_log_dict` 的键名。
   - 端到端测试用真实的 `NamedLmDataset`（LmExample batch）、不满的最后一个 eval batch、层级 tag（`paloma/a`、`paloma/b`），
     以及 marin tokenizer 在 128256 词表上的 bytes。三次 eval 中，`eval/L2/*` 与 `eval/*` 的 12 个键全部相等（相对差 ≤1.6e-7）；
     `eval/L{0,1}/*` 键齐全，也没有多出来的键。
4. **train_lm 钩子**：
   - 参数顺序与 `cb_tagged_lm_evaluate` 一致；只有 pls 的 config 定义了这个方法（全仓库 grep），其他 config 上 getattr 返回 None，不受影响。
   - `train_lm.main` 端到端（`audit_e2e_train_lm.py`，bf16 policy，2 个 microbatch，steps_per_eval=2）：读出 eval 与主 eval
     在同样的 step 触发，包括训练结束时的强制 eval；从 checkpoint 续跑后步数接着走；`train/pls/L{k}` 每步都记，
     `train/loss` = L_{L-1} + w·ΣL_k。w=0 时 `train/loss` = `train/pls/L{L-1}`。
5. **launcher**：env 解析、tag（`-pls1`）、W&B group `muonh-qwen3-pls-della`、model 构造（与 baseline 相同的 kwargs 加上 pls 字段）、SMOKE 模式。
   - reself/marin-della 里 8 个 pls 与 pls-smoke 的 id 都不存在（`check_wandb_ids.py`；用已存在的 baseline 与 OV id 验证过这个查询是有效的）；盘上也没有同名目录。
   - draccus 编码出 `type: qwen3_pls` 和 pls 字段，W&B 的 config 记录正常。
6. **pls_env.sh**（`check_imports.py`，实测）：
   - levanter、haliax、marin、fray、rigging、iris、zephyr、finelog、finestore、ducky、dupekit、tasktrove_verify 和 `experiments.*` 全部解析到 worktree；
     **没有一个源码模块**来自 `project/marin`，只有编译好的 wheel（`dupekit_native`、`iris_native`）来自 venv。worktree 里没有同名包，所以不存在遮蔽问题。
   - 排在最前面的 cutlass `dsl_packages` 里只有 `cutlass` 和 `iket`。
   - `experiments.references` 是 namespace package，但它的父包 `experiments` 是 worktree 里的普通包，所以不会漏到另一个 checkout。
   - `set -u` 下 `${PYTHONPATH:+…}` 是安全的。
   - 两个 checkout 目前在同一个 base commit 上，文件列表只差 `per_layer_qwen3.py`。
   - 训练在 fray LocalClient 的线程里跑，也就是同一个进程，PYTHONPATH 一直有效。
7. **CPU 测试的检出能力**：用 9 个变异体做 mutation testing（`mutate.py`，把 scratch 目录放在 PYTHONPATH 最前面遮蔽模块，tracked 文件没动）：
   监控目标不 roll、stride 偏移、跳过 final norm、per-tag 权重、读出取层输入、final 层也乘 w、bpb 分母、监控丢掉权重、读出顺序反转。**9/9 全部被原测试杀掉**。
   原测试没覆盖、由我补上的：bf16 policy、microbatch、`train_lm` 端到端与续跑、LmExample 形式的 eval batch。
   仍然只能靠 H100 smoke 覆盖的：GPU 上的 `batched_xla` fused CE（OV 与 SS 的 H100 日志显示训练选中的就是它）、JAX_FLASH、真实 Paloma 数据。
8. **hook 节奏**：`run_hooks` 的触发条件是 `info.step > 1 and step % every == 0`，或者 force；主 eval 与读出 eval 用的是同一个条件。
9. **smoke 的 `rm -rf`**：只可能删到 `…-smoke<N>` 的产出，删不到正式 run。

## 复现

```
ssh della-vis1 'cd /scratch/gpfs/GROUP/USER/project/marin-pls && source scripts/della/pls_env.sh && export JAX_PLATFORMS=cpu && nice -n 19 $PY /scratch/gpfs/GROUP/USER/tmp/pls/audit/<script>.py'
```

| 脚本 | 内容 | 耗时 |
|---|---|---|
| `audit_loss_test.py` | 截断 baseline 参考、bf16、microbatch | 约 1 min |
| `audit_e2e_train_lm.py` | `train_lm.main` 端到端加续跑 | 约 4 min |
| `audit_misc.py` | scan_layers=False、draccus、FLOPs | |
| `audit_memory.py 130m\|300m` | 只编译的显存分析 | 约 1.5 min |
| `mutate.py` | mutation testing | 约 6 min |
| `check_imports.py` | 模块解析 | |
| `check_wandb_*.py` | W&B 查询 | |

对应日志：`*.log`。e2e 测试在 worktree 的 gitignored `logs/` 下写过 `pls-audit-w{0,1}*`，已经删掉。
