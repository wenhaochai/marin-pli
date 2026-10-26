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

**Real runs and the next round (2026-10-26 05:00).** The first vocab runs on the new stack train 2.1x (128K, w46: 0.716
s/step against 1.506, a whole run 2:18 against 4:47) and 2.3x (8K, w216: 0.571 against 1.291) faster than the old code,
eval and checkpoints included, with full remat. Profiles of the defaults (w263 128K, w264 8K; 2 GPUs, command buffers
off, summed kernel time 1213 / 914 ms): 128K = output head and CE 31%, attention 19%, MLP 18%, remat recompute 10%,
projections 9%, norms 6.5%; 8K = attention 27%, MLP 23%, remat 12%, projections 11%, CE 9%, norms 9%. Items:
- REMAT=fdots (saved matmul outputs on top of flash): -3.3% at 4096 (w265, 1233 -> 1192 ms), -3.6% at 5504 (w274, 1276
  -> 1230 ms, fits), loss within noise: the default at 130m/300m.
- XLA collective pipelining + latency-hiding scheduler: 1210 against 1207 ms (w279), no gain; not adopted.
- FP8 Linear layers: no speedup at hidden 512-1024 and +0.0026 bpb at 130m (logs/kaiyue_fp8_qwen3_h100x4.yaml,
  2026-09-14); dropped.
- CE logsumexp kernel (branch ce-lse-kernel, for the pls session): a Triton logits GEMM whose epilogue writes the bf16
  logits, the f32 label logit and per-tile logsumexp partials (w271, 8192 x 768 x 128256): 3.28 ms against 7.33 ms for
  the XLA ops with the same outputs; lse within 5.8e-8 of float64 (two-pass logsumexp 5.6e-8); f32 logits bitwise equal
  to cuBLAS; GEMM part 1.47x the XLA bf16 GEMM. pls integrated it (b3f40f216f, LEVANTER_CE_LSE_KERNEL=1, default off;
  here 57f176bcb8); vocab A/B w280.
- Attention: on H100 the segmented FA4/CuTe kernel is Ampere-style (forward) and the 64x64 sm120 path (backward).
  Upstream FA4's native SM90 varlen kernels (flash_attn.cute, wgmma + TMA) at 32 x 4096 x 12 x 64 with packed documents:
  1.19 ms forward, 3.49 ms forward + backward, against 2.39 / 7.31 (torch bench w272; errors 2.0-2.5e-3 against fp32,
  as the segmented kernel); one document per row 2.07 / 6.80 against 5.45 / 17.68. Zero-length padding of cu_seqlens
  costs (w273): 256 sequences 3.58 ms, 512 3.94, 1024 5.37, 2048 11.0; fineweb-edu rows hold 4-5 documents (max 17 over
  ~2.4M windows; 32 rows 163 +- 14, max 240 in 200K draws; tmp_plots/vocab/docs_per_row.py), so 10 per row is safe.
  Backend ATTN_BACKEND=fa4thd / AttentionBackend.FA4_THD (branch fa4-sm90, merged into speed-all 3fd2d8f254). Its first
  H100 call aborted in C++ (std::bad_variant_access, w276/w277): levanter's THD wrapper passed q/k/v/out as static
  layouts, and upstream's SM90 varlen forward builds a ragged TMA tensor for O on top of them (quack
  create_ragged_tensor_for_tma, sentinel extents near 2^31), which cute.make_tensor cannot form statically; the old
  head_dim 128 GQA path aborted the same way (w281). Dynamic layouts fix it (386de97318; w282: forward and backward at
  head_dim 128 GQA and 64 MHA). Through levanter at 32 x T x 12 x 64 (w283; fwd / fwd+bwd ms, fa4 against fa4thd):
  T 4096 packed 2.42 / 7.41 against 1.31 / 3.94, one document per row 5.52 / 17.68 against 2.27 / 6.84; T 5504 packed
  3.46 / 10.38 against 1.86 / 5.48, one per row 9.54 / 31.26 against 3.62 / 11.70; errors against fp32 equal to fa4's
  in all digits shown (out 2.0e-3, dq/dk/dv 2.5-2.8e-3). Training A/B w284 (300m 128K, 4096): 1206 -> 1107 ms (-8.2%),
  loss within noise, but eval loss NaN: eval batches carry document segment ids too, and Paloma rows hold far more
  documents than the bound (per 4096-token row twitterAAE mean 244, gab 100, subreddits 21; the densest 32 rows up to
  8680; tmp_plots/vocab/docs_per_row_paloma.py). Fix 8ae15eec59: one custom_vjp per device whose primal (no gradient,
  so evaluation) is the segmented kernel, exact for any document count and equal to today's eval, and whose fwd/bwd
  (training) are the THD kernels with the NaN guard. w290: the two paths agree to bf16 rounding (rel L2 9.7e-4), and past
  the bound only the gradient path holds the NaN; w291: 1206 -> 1107 ms (-8.3%) with finite eval (7.2705 / 7.2666);
  w285 at 5504: 1269 -> 1073 ms (-15.4%). ATTN_BACKEND=fa4thd is the default at 130m/300m since 2026-10-26 05:45
  (e50e2b3dfb; larger sizes keep fa4: their GQA at head_dim 64 is not checked on this path). Open: under FA4 the
  CUDA-graph capture of while/conditional loops shows 70 ms/step of memcpy/memset and 8% idle in the production-flag
  profile w286. The full stack with and without that capture (fa4thd, fdots, CE A1 + GIF + lse kernel): 991 / 989 ms at
  4096 (w295), 1010 / 1020 ms at 5504 (w296; the old code took 3971 ms there, w254): the capture stays on.

**fdots out again; checks past an eval (2026-10-26 05:40-05:50).** REMAT=fdots was the default at 130m/300m from 05:15 to
05:41. The first production run on it (w160, 300m 8K unigram smoothing) ran out of memory 14 min in: after the first
eval and checkpoint the train step could not get one 40 GiB block (RESOURCE_EXHAUSTED), and again on its retry; the
60-step A/Bs evaluate only at their last step, so they never ran a train step after an eval. The default is REMAT=flash
again (31fe5e3c79); w160 was requeued. New check before any default change: SMOKE_EVAL_EVERY / SMOKE_CKPT_SECONDS (smoke
runs only) and ab_arms2 configs 4096post / 5504post (80 steps, an eval every 20 and a checkpoint every 20 s). The live
defaults (fa4thd, flash, CE A1 + GIF + lse kernel, CUDA-graph loops) pass it: w297 (128K, 4096) 1034 ms, evals at
20/40/60/80 and checkpoints at steps 2/21/41/61; w298 (8K, 5504) 1075 ms; no OOM, eval losses finite.

**Where the live stack stands (2026-10-26 06:20).** First production run on it: w44 (300m 128K, 4 GPUs) trains at 1.8
it/s, ~0.556 s/step (old code 1.506; the w46 stack 0.716), smoke 538 ms/step; a whole 128K run ~1:45 against 4:47.
Profiles of the live defaults (w304 128K, w305 8K; 2 GPUs, graphs off; summed kernel time 1027 / 780 ms): 128K = CE 29%
(the lse kernel 110 ms), MLP 21%, remat recompute 12%, attention 11%, projections 11%, norms 8%, collectives 3%,
embedding 2%; 8K = MLP 27%, remat 15%, attention 15%, projections 14%, CE 11%, norms 10%. GPU busy 89-91%. The host
leaves the GPUs idle between steps: Trainer.train_step reads result.loss.item() right after dispatch, then runs the
hooks, logs and loads the next batch before dispatching it; production W&B gives wall minus step duration = 3.1-3.2% of a
step (w216 0.546 vs 0.529 s, w46 0.687 vs 0.666 s; tmp/speed_wt profile w286 shows ~65 ms per boundary under the
profiler). Branch async-hooks (91e23bf466): LEVANTER_ASYNC_STEP=1 dispatches step N+1 before finishing step N (hooks see
each step's own state, which the train step then keeps instead of donating); levanter test_train_lm passes in both modes;
4-GPU check w307 (live vs liveasync, evals and checkpoints mid-run, wall_ms from W&B runtime).

**The host path between steps (2026-10-26 07:00-09:00; 4 GPUs, 300m, wall per step over steps 20-190 from W&B
runtime).** The async loop as first written gained nothing: w309 128K 558.8 ms sync against 562.9 async, w310 / w311 the
same with the loop capture or all CUDA graphs off. Profiles w313 / w315 (4 GPUs, graphs on, Python tracer;
scripts/della/speed_all/trace_step_timeline.py prints host frames and GPU idle on one timeline): GPU busy 503 of 600 ms
per step, idle 84 ms at the step boundary. Between the loss read and the next dispatch the host runs, in order:
jnp.isnan / jnp.isinf on the host loss (each a device computation read back), checkpoint_hook's saveable_state (an eqx
filter, 13 ms), Checkpointer.on_step's save decision as a jnp array read back through broadcast_one_to_all every step,
~120 scalar metrics read one at a time by the wandb tracker (0.14 ms each), the watch hook every 10 steps (~45 ms), then
the named_jit dispatch (~80 ms under the tracer). Under the async loop the two device computations queue behind step
N+1, so the host waited for all of N+1 and the loop stayed synchronous. Fixes on branch async-hooks (533848cd1a,
e95273ab99): math.isnan / math.isinf, a numpy array for the save decision, is_inside_jit without its probe array when
no transformation is active, and tracker.log starting every metric array's device-to-host copy before a tracker reads
one; levanter test_train_lm, test_jax_utils, test_checkpoint* and test_tracker pass in both loop modes. w317 (128K, 200
steps): sync 557.4 ms (the fixes alone change little: 557.8 / 558.8 before), async 523.9 ms (-6.0%, losses within
noise); profile w318: GPU idle 11.2% of the trace sync, 1.7% async. Allocator (w316, 8K, 200 steps): on the default BFC
allocator REMAT=fdots ran out of memory at step 8 asking for one 40.28 GiB block (w160's failure: its largest free block
was 40.34 GiB, split by small buffers); XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async (CUDA's stream-ordered pool, no
contiguous block needed) runs flash at 439.3 ms against 445.2 on BFC (-1.3%) and fdots at 409.6 ms (-8.0%). Queued:
w319 XLA dot merger (--xla_gpu_dot_merger_threshold_mb=4096; q/k/v and gate/up as one GEMM each), w320 main code vs the
fixed sync loop vs async, w321 fdots + cuda_async at 128K, w322 / w323 the same with full Paloma evals and checkpoints
every 20 steps (128K, 8K unismooth), w324 / w325 the combined stack (async + fdots + cuda_async), w326 the lse kernel in
evaluation (LEVANTER_CE_LSE_KERNEL_EVAL=1; evaluation's forward-only CE still takes the XLA path, 7.33 against 3.28 ms
per 8192-row block at 128K, and evaluations are ~4.4% of a 128K run).

**Results of that round (2026-10-26 09:00-10:20; 4 GPUs, 300m, 200 steps, wall per step over steps 20-190; each
comparison within one job).** Async host loop: 128K 555.8 -> 523.8 ms (-5.8%, w324), 558.5 -> 523.7 (-6.2%, w331); 8K
441.2 -> 410.1 (-7.0%, w328), 437.6 -> 408.2 (-6.7%, w332); losses within noise; with full Paloma evals and checkpoints
every 20 steps (w325) 6 evals and 7 checkpoints, no OOM, median step 525 ms. The host fixes alone in the sync loop: 560.1
-> 558.1 (w320). fdots + cuda_async in the sync loop: 128K -3.9% (w321), and it passes full evals and checkpoints at
128K (w322, median 537 -> 512) and 8K unismooth (w323, 420 -> 394). Async + fdots + cuda_async: 128K out of memory at
step 28 in both w324 and w325 (cuMemAllocAsync of 37.18 GiB failed with 3.96 GiB in use: the driver still held the
previous step's buffer), 489 ms per step until then; at 8K it ran 200 steps at 380.0 ms (-13.9%, w328). Not adopted:
unexplained memory failure at 128K. REMAT=fproj (flash plus the q/k/v/o projection outputs, 14d5455732): nothing
(128K async 522.7 -> 521.9, w329; 8K 410.3 -> 409.6, w330), so fdots' gain is the MLP recompute. XLA dot merger: +1.7%
slower (w319). Lse kernel in evaluation: each full eval 23-24 s -> 20-22 s (w326), ~0.6% of a 128K run. Input-embedding
gradient: FSDP shards the table along embed, so XLA lays the gradient out column-major and the backward is a bf16
scatter-add of rows 128256 elements apart; one H100 (w327, 131072 Zipf ids): bf16 scatter 93-465 ms with 17-20% relative
error on the 16 most frequent rows against an f32 sum, f32 scatter 0.7 ms, sort + segment sum 0.25-0.64 ms.
LEVANTER_EMBED_GRAD_F32=1 (591359b069, the gradient summed in f32 and cast once): 128K 558.5 -> 548.8 (-1.7%), with the
async loop 515.4 (-7.7% from live); 8K 437.6 -> 431.8 (-1.3%), with async 402.9 (-7.9%); it changes training: the train
loss over the first 200 steps is 0.002 nats lower at 128K and 0.0013 at 8K (async alone: 0.0001-0.0003 either way).
Figure: tmp_plots/vocab/flip/speed_async.png.

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
