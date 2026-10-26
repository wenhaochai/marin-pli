# pls speed investigation: handoff to nanomarin (2026-10-24)

Status: investigation done; nothing switched (owner 10-24: pls keeps running on the old code; nanomarin takes the design lessons).
All GPU timing smokes and A/Bs were cancelled before they started, so every GPU number below comes from production logs,
not from a controlled A/B. CPU numerics tests were run on della-vis1.

Files: `plan.md` (ranked plan with A/B protocols, from a 9-agent search: 4 finders, 4 skeptics, 1 planner),
`results.json` (every candidate, the skeptic verdicts, and each finder's measured facts with log paths).
Branch `pls-speed` (worktree /scratch/gpfs/GROUP/USER/project/marin-pls-speed): knobs A1, A4, A7 and PROFILE are
implemented there, default off. The only change that went into the live `pls` branch is the smoke-gate fix (684b48ead8).

## Where the time goes (4 x H100, 130m width 512, 48 layers, vocab 128256, batch 128 x 4096)

| Run | s/step | Note |
|---|---|---|
| Baseline d48, no readouts, PDP=4 | 2.9 | |
| preln d48, 48 readouts, PDP=4 | 15.0-15.2 | 47 extra readouts = 12.3 s = 81% |
| hc d48 / mhc d48, PDP=1 | 24.3-25.4 | forced to microbatch 1 by memory |
| hc d48 PDP=4 (before OOM at step 257) | 16.0-16.3 | BFC fragmentation, not pool size |
| attnres d48, PDP=1 | 30.8-34.6 | mb4 gives 32.5: its cost is its own source mixing |

- Readouts are ~89-93% of FLOPs and memory-bound on f32 logits traffic (roofline predicts 15 s/step for preln, measured 15.2).
- Wall budget of a finished hc d48 run (37 h): pure steps 90.1%, checkpoints 3.1% (blocking host staging, 20 s per save),
  evals 3.9%, smoke 1.6%, jaxpr/HLO debug logging + startup ~1%.
- Per-step host overhead is negligible (0.2-0.6%); step times are tight (p90/p50 < 1.03).

## Candidates

| id | change | saving (estimated) | numerics | where | tested |
|---|---|---|---|---|---|
| A1 | b-tiled CE backward: label term as a select, not a scatter (fuses exp, mul, label, bf16 convert; -8.4 GB HBM per 8192-row block) | ~10% of PDP>=2 runs | bitwise on CPU (B 6000, V 128256, 2 blocks, with and without soft cap) | lib/levanter/.../fused_cross_entropy_loss/batched_xla.py `_backward_b_tiled_from_lse`, knob LEVANTER_CE_BTILED_ONEHOT | CPU only; GPU fusion and timing untested |
| A2 | probes: drop the dX GEMM (custom_vjp returns only dW and the norm-scale grad) | ~6% of probe runs | dg rounding changes (changes-math, last bits) | experiments/references/per_layer_qwen3.py:454-476 | not implemented |
| A3 | save a checkpoint just before the known time limit, then a 30-min interval | 1.5-2% | none | launcher + Checkpointer.on_step (checkpoint.py:578, request_checkpoint :653) | not implemented |
| A4 | log_jaxprs / log_xla_hlo off | ~4 min per segment (0.4-0.5%) | none (same executable) | launcher, knob LOG_JAXPR | implemented; nothing reads the dumps |
| A5 | one eval pass: alias readout L-1 as the main eval | ~0.9% | eval values at summation-order level | train_lm.py:362, per_layer_qwen3.py:775 | not implemented |
| A6 | static causal loops in JAX_FLASH attention (data-dependent while loops) | ~3%, speculative | bitwise in intent | lib/levanter/src/levanter/models/flash_attention.py:278-282, 416-419 | needs a profile |
| A7 | eval at 4 sequences/device for PDP=1 designs | ~0.6-1.3% | eval at summation-order level | launcher, knob EVAL_PDP | implemented; memory untested |
| A8 | skip the resume smoke when the code hash is unchanged | 0.3-0.5% | none | sbatch | not implemented; removes a gate |
| B1 | hc/mhc at microbatch 4 with a non-fragmenting allocator (vmm / cuda_async) | ~30% of hc/mhc runs | changes CE path (see confound) | env + PARAM_MAP_PDP | not shipped: breaks comparability with finished controls |
| B2 | fast CE backward (LEVANTER_CE_XLA_FAST_BWD) for PDP=1 designs | <=6% for attnres | changes-math | xla.py:35/714 | not shipped |

Rejected (reasons in plan.md): vocab-parallel CE, fused Pallas CE (Pallas GPU forwards measured 8-9x slower than XLA),
latency-hiding scheduler, 8-GPU rendezvous timeout, AttnRes buffer slicing, pinned checkpoint staging.

## Findings that matter for a rewrite

1. Microbatch silently picks the CE kernel. PDP=1 (4096 tokens/device) fails the batched_xla shared-memory guard and falls
   back to the `xla` streaming CE (f32 delta, TF32 GEMMs, bf16 grad accumulation over vocab blocks); PDP>=2 uses b-tiled
   (bf16 dlogits, f32 accumulation). On CPU vs f32: grad_x rel err 5.3e-3 vs 7.6e-4. So designs at different microbatch
   train with different gradient numerics, and only a RuntimeWarning says so. A rewrite should make the CE path explicit,
   logged, and independent of microbatch.
2. Memory is the binding constraint, and it is per-readout copies: each microbatch all-gathers the 47 bf16 heads (5.8 GiB)
   and makes full f32 head grads (11.5 GiB) before the reduce-scatter. At b=8192 with a 1-trip 8192 tile the temp is
   64 GiB (8 readouts, CPU compile); with a 4096 tile 8 GiB. Microbatch 2 OOMs everywhere for this reason.
3. BFC fragmentation killed hc d48 at mb4 at step 257 (a small live chunk split the free pool). Allocator choice matters.
4. Checkpoint save blocks the loop for 20 s (host staging at 1.65-2.7 GiB/s), every 10 min; no SIGTERM handler saves at
   a time limit, so ~6 min of work is redone per segment end.
5. jaxpr/HLO debug dumps default on and cost 35-213 s of GPU idle per process start.
6. The compile cache works only because ~/.bashrc sets JAX_COMPILATION_CACHE_DIR; nothing in the job scripts does.
7. Resume smoke gate bug: same smoke run id across segments, W&B resumes it and keeps segment 1's summary (fixed 684b48ead8).
