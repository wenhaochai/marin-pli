# Speed investigation of the 300m vocab runs (workflow wf_cb88f9ae-72b, 2026-10-24)

One sentence: the charwin runs lose about 40% of their step time to a 128-token flash block, every 300m run spends about
half of its step in a pure-JAX flash attention that full per-layer remat runs twice, and three candidate fixes are built on
branches but none is verified on H100.

Owner decision (10-24, via the terminal session): the vocab and pls studies keep running on the current code; nothing
below is switched on or A/B-tested; the findings go to the nanomarin session. Worktrees stay in
`/scratch/gpfs/GROUP/USER/tmp/speed_wt/`. Every speed number is from an A40/A100 on a shared vis node or an estimate
from logs, never an H100 measurement. Raw agent outputs: the workflow journal
(`~/.claude/projects/-home-USER/c574d901-7815-4be8-b851-ba33c8717449/subagents/workflows/wf_cb88f9ae-72b/journal.jsonl`).

**Update 2026-10-25 (owner, via the marin.cpp session and a popup: 「应上尽上，尽可能快」, then 「全部都上」).** The 10-24
decision is reversed: every candidate goes in after one H100 check. Branch `speed-all` (worktree
`tmp/speed_wt/all`) = vocab-overfitting + the three branches above + the flash-block rule (largest divisor of
SEQ_LEN <= 1024 and a multiple of 16) + REMAT=none + PROFILE=<start>:<steps> + the pls session's CE knobs and checkpoint
deadline (LEVANTER_CKPT_DEADLINE = SLURM_JOB_END_TIME - 150 s in the launcher). H100 checks in wq at prio 10:
w203/w204/w212 A/B (old checkout vs new arms, step time and per-step loss), w227/w228 CE time and determinism, w231
per-family profile (scripts/della/speed_all/trace_families.py), w210 attention kernels (JAX flash, FA4, Pallas
Triton, cuDNN packed). Live already: A1, the CE label term as a select (bitwise equal on H100 in the pls det round),
on by default in the launcher since 0f2a3941b0. First H100 number (w202): 300m 128K at 4 GPUs 1.233 s/step with
REMAT=flash and CUDA-graph loops, against 1.51-1.56 s on the old code in recent runs; per-GPU throughput is flat
from 1 to 4 GPUs within 2% (130m 232-243K, 300m 106-108K tokens/s/GPU).

**H100 A/B, 2026-10-25 (w203, w204; 300m seed 0, 60 steps at the real batch, 4 H100s, median step after step 3).**

| Config | old checkout | JAX flash + REMAT=flash + CUDA-graph loops | FA4 + CUDA-graph loops |
|---|---|---|---|
| 128K, SEQ_LEN 4096 | 1434 ms | 1228 ms (-14%) | 719 ms (-50%) |
| 32K, SEQ_LEN 4480 (charwin, block 128 -> 896) | 2609 ms | 1196 ms (-54%) | 607 ms (-77%) |

Per-step train loss against the old checkout over the 60 steps: max abs difference 0.021-0.025, mean -0.0015 to
+0.0014, last-10 mean -0.002 to +0.006, so within bf16 noise for all arms (no same-code repeat in these jobs). FA4's sm90
backward ran here for the first time and trained normally. **Switch (2026-10-26, 0:30):** the live checkout
(vocab-overfitting, fast-forwarded to speed-all a2e3335945) defaults to ATTN_BACKEND=fa4 and XLA_CMD_BUFFER_WHILE=1;
ATTN_BACKEND=jax_flash and XLA_CMD_BUFFER_WHILE=0 give the old path. Every queued vocab plan dry-runs with an unchanged
run id, so resumed runs keep their checkpoints. The owner chose numerics-changing items for the queued jobs too
(popup 「全部都上」), so runs from here on differ from earlier ones at the rounding level. Still pending: w210 (kernel
errors against fp32 with document masks), w212 (REMAT=none), w227/w228 (CE gradient in the forward), w231 (profile).

**Results since (2026-10-26, H100).** Kernel benchmark w210 (one GPU, 32 x 4096 x 12 x 64, document masks; fwd / fwd+bwd ms;
rel. error of dq/dk/dv vs fp32): JAX flash 44.4 / 106 (0.5-2.2e-3, up to 9e-3 at block 128); FA4 2.4 / 7.3 (2.5-2.8e-3);
cuDNN packed 1.35 / 8.8 (2.5-2.8e-3); Pallas Triton 3.2 / 28.5. FA4 stays (cuDNN is close; faster with one document per
row). REMAT=none does not fit (179 GiB, w212); REMAT=dots saves 60 ms of recompute but adds 40 ms of layer-scan
copies and 27 GiB: -2.5% (w248), not adopted. CE knobs under FA4 (w227): A1 -4.9%, A1+GIF -9.3%; det round w245 (JAX flash):
A1 bitwise, GIF within 1.9e-6; run-to-run GEMM autotuning alone moves losses by up to 0.076 (pls w205 vs w229). A1 on
by default. GIF went on at 01:50 and off again at 02:50: its fallback crashes under shard_map for any shard that skips the
b-tiled CE path, and every vocabulary below 65536 does (pls bug report; no truncated-vocab main run started with it);
back on after the pls fix and an 8K check. Profile w249 (FA4, 2 GPUs): CE + head 37%, FA4 16%, MLP 15%, remat
recompute 14% (FA4 forward 5%), projections 8%, norms 5%, collectives 2%, embedding 2%, optimizer 1%; GPU busy 93%
with command buffers off.

**Defaults since 2026-10-26 04:30 (H100, 2 GPUs at 32 sequences per device, 60 steps).** GIF fix: w256 at 8K (det flags,
JAX flash) gives GIF on bitwise equal to off, per step and in the final train and eval loss, so GIF is on again; below
65536 the flag does nothing, above it it saves one logits GEMM. REMAT=flash under FA4 (the FA4 forward now names its
output and logsumexp): 1350 ms against 1402 ms (-3.7%, w253), loss within bf16 noise, so REMAT=flash is the code
default (REMAT=full gives the old policy; A/B and profile arms that name no REMAT pin full). FA4 at SEQ_LEN 5504 (charwin
v8k, w254): 1372 ms against the old 3971 ms (-65%), loss within bf16 noise. The launcher's speed settings now live in
`scripts/della/speed_env.sh`, sourced by the launcher and by `over_vocab_smoke.sbatch`: until now the smokes ran python
without them, so a smoke could pass while its main run crashed (how the GIF bug slipped past the vocab smokes).

## Measured baseline (logs, 4x H100)

| Run | SEQ_LEN | flash block | s/step |
|---|---|---|---|
| 8K (w107) | 4096 | 1024 | 1.2-1.29 |
| 64K (w58) | 4096 | 1024 | 1.4 |
| 128K (w20 etc.) | 4096 | 1024 | 1.51-1.56 |
| bytes (w29) | 4096 | 1024 | 1.25 |
| charwin v64k / v32k / v16k / v8k | 4224 / 4480 / 4992 / 5504 | 128 | 2.5 / 2.6 / 3.1 / 3.7 |
| charwin bytes | 19456 | 1024 | 18.5 |

Peak device memory 12.3-13.0 GiB of 71 GiB on every run. Non-step overhead (eval, checkpoints, compile) 3.5-5.5% of wall
time; Paloma eval alone 3.5% at 8K. Head + cross-entropy about 17-19% of the step at 128K, 11% at 64K, 2% at 8K
(step-time differences across vocabularies). Collectives about 1% (estimate from the compiled HLO's collective volume).

## Candidates

| Candidate | Change (where) | Expected gain | Numerics | State |
|---|---|---|---|---|
| Flash block = largest divisor of SEQ_LEN <= 1024 and a multiple of 32 | `experiments/references/della_muonh_qwen3_scaling.py:112` (FLASH_BLOCK rule); e.g. 4224->704, 4480->896, 4992->832, 5504->688 | charwin attention per layer -33 to -38% (A40, B=32); est. charwin step 2.5->~1.45 s | reordering-level (same flash path, other chunking); out diff 0.023, same error vs f32 as block 128 | not built (one line), not on H100 |
| FA4/CuTe segmented attention backend | branch `speed-fa4-segmented-attention-backend` 6497ef5720: new AttentionBackend case in `lib/levanter/src/levanter/layers/attention.py` routing to `lib/levanter/src/levanter/grug/attention/_fa4_cute.py` | fwd 6-12x faster than JAX flash (A40); full 300m fwd 2.0x at 4096 (A40); est. -40% step at 4096, -60% charwin | precision-level, not bitwise; error vs f32 equal or smaller than JAX flash on sm80; no cross-document leak in 30 probes | review: not harmless yet. sm90 backward never ran (head_dim 64 goes through the generic launcher); random-init checks are weak; A/B plan unsound (afterok on a raw sbatch id, no seed-noise reference, covers only 20% of the schedule) |
| Save flash residuals in remat (REMAT=flash) | branch `speed-save-flash-residuals-remat` 444ea137bf: names flash_o/flash_ell in the flash forward, save them in the per-layer checkpoint policy | -15% at 4096, -26% at 4224 (A40, batch 2); est. 12% on H100; +2.5 GiB/device at 300m | same math, bf16 grads differ 1.1-1.7% rel L2 (same distance to f32 as full remat; 1.8e-6 in f32) | review: not harmless yet (no H100 A/B; cleanup and retry hazards in the A/B scripts; weak acceptance test) |
| CUDA-graph while loops (XLA_CMD_BUFFER_WHILE=1) | branch `speed-command-buffer-while` fe5a177339: launcher adds `--xla_gpu_enable_command_buffer=+WHILE,+CONDITIONAL` | est. up to 30% for charwin (the 128-block loop is launch-bound); A40 timing was noise | same kernels; step-0 loss bitwise, 158/159 grad leaves bitwise on A40 | review: not harmless yet (H100 and multi-GPU FSDP unverified; A/B rerun hazard with fixed run ids) |
| cuDNN attention with a dense segment mask | `jax.nn.dot_product_attention(..., implementation='cudnn')` in levanter `dot_product_attention` | fwd+bwd 2x vs JAX flash at 4096 (A40); est. 20% | precision-level | not built; fallback if FA4 fails on H100 |
| XLA CE fast backward / balanced vocab blocks | levanter fused CE (`maybe_fused_next_token_loss`) | 0.5-3% | rounding-level; the current 128K H100 path has grad rel error 2e-2 vs f32, 5-20x the XLA path | not built |
| Collective overlap flags | XLA latency-hiding scheduler flags | ~0.7% | none | not built |
| SIGTERM checkpoint, charwin task limits, per-step host sync | trainer / plans | 0.5-1.5% | none | not built |
| Rejected | pure data parallel (0%), no remat + microbatching (-5%) | | | |

## Not tested anywhere

H100 (sm90) timing and numerics of every candidate; multi-GPU FSDP numerics; a same-seed full-schedule A/B (the planned
A/Bs covered only the first ~20% of the schedule at high LR); 520m/1_2b memory with saved residuals.
