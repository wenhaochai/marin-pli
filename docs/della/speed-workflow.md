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

**Adopted (owner popup 2026-10-26 ~12:15).** "设为默认" for the async host loop with the host-path fixes, "改，不重跑" for the
f32 embedding gradient: both default in scripts/della/speed_env.sh (LEVANTER_ASYNC_STEP=1, LEVANTER_EMBED_GRAD_F32=1;
merge cc5727a36c into vocab-overfitting, aaed00bef7), for every job that starts after 12:17; check_plans passes on the
queued plans. Runs that started earlier (w156-w158) keep the old code. The f32 gradient changes numerics, so a run that
starts after 12:17 is not bitwise comparable to one before (~0.002 nats lower train loss early); the owner chose no
baseline reruns. Not adopted: REMAT=fdots and the cuda_async allocator (out of memory beside the async loop at 128K),
REMAT=fproj (no gain), the XLA dot merger (slower), the lse kernel in evaluation (~0.6% of a run; its numerics were not
checked on one checkpoint). Told the pls and terminal sessions.

**MLP outputs saved, a silent corruption, a guard (2026-10-26 12:20-13:10).** On the new defaults (async + f32 embedding
gradient), saving part of the MLP fits beside the async loop where fdots did not: REMAT=fgate (flash + the MLP gate
output, ~8.5 GB) and fmlp (+ gate and up, ~17 GB; LlamaMlp names them mlp_gate / mlp_up, no-ops for other policies).
200 steps: 128K 522.4 -> 514.7 / 509.9 ms (w335), 8K 402.4 -> 393.6 / 388.3 (w336). Full Paloma evals and checkpoints
every 20 steps (median): 128K 516 -> 511 / 505 (w339), 8K unismooth 399 -> 392 / 386 (w340); SEQ_LEN 5504 (charwin
v8k shape): fgate 530 -> 522, fmlp out of memory (one 38.89 GiB block, w341). Silent corruption: the vocab run w158
(unismooth 128K 0.02, the stack of 10:17, sync loop) collapsed at step 3171 to near-uniform predictions (loss 3.556 ->
11.597, final bpb 3.24) because its lm_head norm, which AdamH recomputes every step and keeps constant, went 353.33 ->
3.556 in one update: the value of that step's train loss. Not at a save, eval, resume or fetch; only lm_head moved (the
next of 92 constant-norm keys 5.6e-7); no NaN. A scan of every reself/marin-della run since 10-20 (375 runs,
tmp_plots/vocab/flip/norm_jumps.py; pls ran its own over 25 runs) finds no other jump: healthy constant norms move ~1e-6
between watch steps. The bad update ran in the hooks variant of the step (3170 is a watch step); its compiled HLO (w345,
the w158 config) packs 317 scalars into one in-place async all-reduce (all-reduce-start.54, output aliased to input) whose
element 0 is a jnp.linalg.norm sum (AdamH's lm_head path) and elements 1-2 the forward pass's loss sums, so a mix-up or an
early read in that buffer would give exactly what w158 shows; one event in ~400 runs does not separate the capture, the
async collectives and the kernels. LEVANTER_NORM_GUARD=<rel> (watch.py, d394c5f26e): the watch hook raises when a
constant-norm weight moves by more than <rel> between watch steps, so the job's retry resumes from its last checkpoint;
w343 with 1e-3: 516 ms on and off, no false stop over 6 evals and 7 checkpoints. **Adopted (owner popup ~13:07):** REMAT
code default at 300m fmlp up to SEQ_LEN 4096, fgate above (81bdd4f5a8), and LEVANTER_NORM_GUARD=1e-3 in speed_env.sh
(5ab7a4e9a5), for jobs that start after 13:09; w158 reruns as w342 (RUN_TAG=-rerun).

**Where the time goes now, and the next round (2026-10-26 14:00-17:00).** A whole run on the live defaults (w117, 128K,
11444 steps, 1:42:20 including its smoke) splits into the smoke 90 s, startup 49 s (first step 11 s with the compile cache),
12 full Paloma evals of 24 s each (288 s, 4.6%), checkpoints ~2 s each and the train steps (0.496 s each). Profiles of the
live stack before fmlp (w350 128K, w351 8K; 2 traces, graphs off): 128K summed kernels 498 ms = CE 31% (lse kernel 55 ms,
the two gradient GEMMs 33 + 33 ms, the dlogits pass 25 ms, the f32 grad_w add 5 ms), MLP 20% (GEMMs at ~750 TFLOP/s; the
SwiGLU elementwise passes 31 ms), attention projections 11%, attention 10%, remat recompute 8%, norms 8%, collectives
4.8% (24 ms; summed kernels 498 ms against 487 ms busy, so they barely overlap), optimizer 3%, layer-scan bookkeeping 3%.
The layer scan copies each layer's saved gate and up outputs from the GEMM's buffer into the stacked buffers (2 x 5.7
ms) and fills both stacked buffers with zeros first (5.3 ms; jax's scan lowering broadcasts a zero scalar). The async
host loop allocates the next step's temp buffer while the running step holds its own, so every GiB saved per layer
costs twice: fmlp's 28 GiB temp fits twice in 71 GiB, fdots' 37 GiB does not. Lse kernel v3 (lse_logits_mgpu.py,
Pallas Mosaic GPU, warp-specialized TMA + wgmma with the logsumexp partials in the epilogue): per 8192-row block at K
768 2.92 ms against 3.20 for the Triton kernel and 2.16 for cuBLAS (w371); of its 0.47 ms over the plain Pallas GEMM,
0.27 ms is the exp epilogue (MUFU-bound: 16 exp per clock per SM) and 0.19 ms the label gather-dot (w379); in training
-1.0% (w368), and past full evals and checkpoints every 20 steps 499 -> 494 ms median, eval loss within noise (w384).
Training gathers the head physically as [H, V] (its all-gather output has layout {0,1}), so a row gather of w.T may stay
strided (631b5a3dac; w396 bench, w397 A/B). B-tiled CE below 65536 entries (LEVANTER_CE_BTILED_MIN_VOCAB=4096; the
v-tiled path's GEMMs ran at 4.7-13.3 ms against ~2.5 for their FLOPs, w351): past full evals and checkpoints 8K 389 ->
383 ms (w394), 32K at SEQ_LEN 4480 474 -> 419 ms (-11.6%, w395). Its backward reads bf16 logits, while the v-tiled path's
bf16 dot feeds an f32 convert that XLA may compute in f32, so 8K/32K gradients may lose precision against every earlier
8K/32K run; LEVANTER_CE_BTILED_F32_LOGITS=1 keeps f32 logits (d49d9708ed; precision by path w409, speed w410/w411).
Not adopted: LEVANTER_SCAN_EMPTY (lax.empty's AllocateBuffer instead of the zero fill): XLA's SPMD partitioner does not
shard AllocateBuffer, so each GPU asked for the unsharded buffers, 65.5 GiB (w402). Queued: plain data parallelism
instead of FSDP (MESH=dp) and XLA dynamic-slice fusion (w398), the f32 grad_w GEMM with its sum folded into cuBLAS
(LEVANTER_CE_GRADW_F32, asked by pls; w403), XLA's DUS-accumulator zero-init elimination, programmatic dependent launch
and NCCL user buffers (w408), and the three CE items together past full evals and checkpoints (w405-w407).

**The CE backward from unnormalized probabilities (LEVANTER_CE_UNNORM, async-hooks 90a6f7c73f..1bf8f97776, 2026-10-26
17:10-17:40).** At 128K a GIF block (8192 rows) is the lse kernel 3.5 ms, the dlogits pass 1.5 ms (read the bf16
logits, write the bf16 dlogits), two gradient GEMMs 2.1 ms each and the f32 grad_w add 0.3 ms. Now lse kernel v3 takes
each row's label logit c (an f32 gather-dot before the kernel) as a row offset and stores P = bf16(exp(z - c)), clamped
at 60, with the f32 row sums S of the same exps (probs_and_sum_mgpu); the loss is log S. Since P at the label is exp(0) =
1 and S >= 1, nothing overflows unless a token's loss passes 60 nats, and nothing underflows. The gradients are plain
cuBLAS GEMMs on P: dlogits = (g / S) P minus g at the label, with g / S moved onto x for grad_w and onto the output rows
for grad_x. Mode 1 folds the label term into P (1 - S at the label); mode 2 adds it apart in f32 (-g w_label to grad_x,
g x summed into a [V, H] f32 table subtracted from grad_w after the loop). The dlogits pass goes away, and P carries
bf16's relative rounding of a probability instead of exp of a bf16-rounded logit. Checks: the gradient algebra on CPU
with an XLA stand-in for the kernel (V 66000, f64 reference); the kernel at 2.80 ms per block (v3 2.95, cuBLAS GEMM
2.21), row sums within 1.7e-6, P within bf16 rounding (w426). One H100, 65536 rows at 128K, against an f32 reference
(w427): the live path 76.7 ms with gradient error 3.2e-3 (grad_x) / 3.6e-3 (grad_w); mode 1 62.9 ms, 2.4e-3 / 3.3e-3;
mode 2 64.6 ms, 1.7e-3 / 1.7e-3 (the bf16 gradient outputs alone set ~1.1e-3); f32 logits 96 ms, 1.9e-3 / 2.5e-3.
One GPU, 200 steps of 4 microbatches: 1904 -> 1790 ms per step with mode 1 (-6.0%, w425). Queued: 4 GPUs past full
evals and checkpoints with the CE items together (w428), mode 2 on one GPU (w429).

**Results and adoption (2026-10-26 17:30-19:45).** One GPU (4 microbatches per step, 200 steps; w421-w433): the SwiGLU
custom_vjp alone +0.8% (hybrid_norm's post-MLP norm reads the MLP output, so the remat still recomputes the product);
saving the MLP output too (REMAT=fmlpo) with it -2.0%, with the attention output (fmlpoa) -2.6%; LEVANTER_CE_UNNORM
mode 1 -6.0%, mode 2 -5.4%; XLA's DUS-accumulator zero-init elimination -3.1% to -3.5% (dynamic-slice fusion, with or
without legacy cuBLAS: no change); everything together -11.1% to -11.2%, fmlpo and fmlpoa alike, and without the
zero-init flag the combination ran out of memory on one GPU. 4 GPUs past full evaluations and checkpoints every 20 steps
(median ms per step, 8 checkpoints and 6 evaluations per arm, losses within noise): 128K 503 -> 457 (-9.1%, w435; a full
evaluation 24 -> 20 s), 8K 388 -> 336 (-13.4%, w436), 32K at SEQ_LEN 4480 with fgate 477 -> 423 (-11.3%, w437).
**Adopted (owner popup ~19:15, "全部设为默认，不重跑"; c6886d9b2b, jobs that start after 19:45):** LEVANTER_CE_UNNORM=2,
LEVANTER_CE_BTILED_MIN_VOCAB=4096 (f32 logits below 65536), LEVANTER_CE_LSE_MGPU=1 with LEVANTER_CE_LSE_KERNEL_EVAL=1,
LEVANTER_SWIGLU_VJP=1 and the zero-init flag in speed_env.sh, and REMAT code default fmlpo up to SEQ_LEN 4096. The two CE
items change gradient numerics, both toward f32; the rest is the same math. Not in it: LEVANTER_SCAN_EMPTY (out of memory,
w402); still queued on 4 GPUs: plain data parallelism and the dynamic-slice fusion (w398), the f32 grad_w GEMM (w403;
+0.7% on pls, w404), the XLA flag set (w408), Triton GEMM fusions (w413), the remat options alone (w420).

**The 4-GPU tests that were still queued (ended 2026-10-26 evening; 300m, 4 H100s, 200 steps, median ms per step against
the same job's baseline arm, the pre-bundle live stack).** Plain data parallelism instead of FSDP (MESH=dp) 505 -> 501
(w398; the baseline arm moves by 2-5 ms between jobs, so this is noise); dynamic-slice fusion 505 -> 505 (w398) and with
legacy cuBLAS 504 -> 505 (w413); the f32 grad_w GEMM 502 -> 500 (w403); the DUS-accumulator zero-init flag 500 -> 493
(-1.4%, w408; in the bundle); programmatic dependent launch 501 and NCCL user buffers 502 against 500 (w408); Triton
GEMM fusions 504 and 505 against 504 (w413); REMAT fmlpo, fmlpos and fmlpoas 491 against 505 (-2.8%, w420; fmlpo is in
the bundle); the b-tiled CE at 8K 386 -> 357 (-7.5%, w415) and at 32K 475 -> 430 (-9.5%, w416), and mode 2 of the
unnormalized CE at 128K 503 -> 477 (-5.2%, w428; both in the bundle). Nothing new to adopt.

**One step in flight (idea, 2026-10-26 ~21:00; not built).** The async host loop keeps two steps' temporary buffers
(PJRT frees a step's temporaries when it ends, and step N+1's are allocated at its dispatch), which is what ruled out
fdots at 128K (w324). A loop that dispatches N+1 as soon as N's loss is on the host, then runs N's hooks while N+1
computes, would keep one. A draft passed levanter's test_train_lm on CPU in both loop modes, but the memory it frees has
no use: the default fmlpo already saves the MLP's gate, up and output, the projection outputs gained nothing on top of
flash (fproj, w329/w330), fdots' gain over flash was the MLP recompute, and REMAT=none needs 179 GiB. The draft would
also leave the dispatch on the critical path between steps, so it was dropped.

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
