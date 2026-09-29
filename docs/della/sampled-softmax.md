# Sampled softmax on the Kaiyue muonh_qwen3 baseline

**Status (2026-09-28):** implemented (`VARIANT=ss`, branch `sampled-softmax`); CPU tests pass; on H100 the candidate
CE is 5.2x cheaper than the full-vocabulary CE at P = 24,576 (41 vs 215 ms per device-step); the 60-step 130m smoke
runs every stage cleanly, 1.36x faster per step than the baseline on the same node in the first stage (~1.29x over
the whole schedule). Full 130m runs (4 ss seeds + 2 restore baselines, jobs 14658326-8) queued; no quality verdict yet.

## What it is

The training loss of nanoGPT speedrun record #92 (ANVIL2, 2026-08-30, KellerJordan/modded-nanogpt PR #360), the largest
single item of that record's ablation (+8.8 s of its 34 s): for most of training each device's cross-entropy normalises
over a shared candidate set instead of the whole vocabulary. Our port, `experiments/references/sampled_softmax_qwen3.py`:

* **Candidate set, per device and step:** every class that is a target somewhere in the device's microbatch, plus
  negatives up to P from a golden-ratio stride sweep of the vocabulary (a permutation), read from an offset drawn from
  the step key and staggered per device (record #92: stride 20011 over 50,304 from a per-rank offset; same idea).
* **Kernel:** the lm_head rows of the set are gathered and the CE runs on the `[Embed, P]` head with targets remapped
  to positions in the set, through the row-tiled fwd/bwd pair that levanter's H100 dispatch uses for the full 128K
  vocabulary (for a head narrower than 65,536 the dispatch picks its vocab-streaming kernel, 18-27% slower here). The
  row block grows as P shrinks to keep the full path's logits footprint. The lm_head gradient is dense over the
  vocabulary, zero off the set. The full-softmax stage calls exactly the baseline's kernel.
* **Schedule:** stages by step through `levanter.trainer.current_train_step()` (a context var that
  `Trainer._train_step` sets to the traced `state.step`; the loss function has no step argument). One compiled step
  holds every stage (`lax.switch`), so stage changes cost no recompilation. The run ends on the full softmax.
* **Safety:** a device whose microbatch has more distinct targets than P uses the full softmax that step (logged as
  `train/ss/overflow_frac`); no target is ever dropped. Evaluation (`key=None`) is the baseline code path, bit for bit.
* **Logged:** `train/ss/{candidates,present_max,present_mean,overflow_frac}`. `train/loss` is the sampled loss while a
  stage is active (below the full-softmax loss, stepping up at each stage change); `eval/*` is always full softmax.
  MFU keeps the full-vocabulary FLOP count, so it reads as baseline-equivalent throughput.

## Choosing P (measured, not assumed)

fineweb-edu-10B device-batches (32 x 4096 tokens, the 130m/300m per-device microbatch), 3,000 random draws over four
regions of the cache (`distinct_targets.py`, 2026-09-28): **16,950 +- 309 distinct targets, max 17,883**; a whole
global batch (128 windows) holds ~32.8K. So a per-device set is half the size of a global one, and P = 24,576 never
overflowed (31% of P left for negatives). Default `SS_SCHEDULE` = record #92's schedule as fractions of the vocabulary
and of training:

| record #92 (V = 50,304, 1194 steps) | ours (V = 128,256; 130m: 4959 steps) |
|---|---|
| P 10,240 (20% of V), steps 0-680 (57%) | P 24,576 (19%) until 57% (step 2827) |
| P 14,336 (28%), to 964 (81%) | P 36,864 (29%) until 81% (step 4017) |
| P 24,576 (49%), to 1106 (93%) | P 65,536 (51%) until 93% (step 4612) |
| full softmax from 1107 | full softmax from step 4612 |

Why it can pay here: at 130m the lm_head (512 x 128,256 = 65.7M) is about three times the whole transformer body
(22.8M), so the output layer dominates the matmul FLOPs.

## Verification

1. `scripts/della/sampled_softmax_cpu_test.py` (fake 4-device mesh built from the launcher's MeshConfig) -- PASSED
   2026-09-28: candidate sets equal the specification; per-device loss and every parameter gradient equal an independent
   reference (dense logits, numpy candidate sets per device block) at each stage; one compilation served steps in all
   three stages; the full stage, the all-device and mixed overflow fallbacks and evaluation reproduce the baseline or the
   reference; a keyed call outside the train step raises.
2. `scripts/della/sampled_softmax_gpu_check.py` (H100, one real device-batch: 131,072 targets, 17,125 distinct) --
   PASSED (job 14652357): every stage within 3e-6 of dense f32 on the loss, 3e-3 relative on dx (bf16 matmuls), and
   dW exactly zero off the set. CE fwd+bwd per device-step (ms; D = 512 is 130m, 768 is 300m):

   | head | D=512 dispatch | D=512 row-tiled | D=768 dispatch | D=768 row-tiled |
   |---|---|---|---|---|
   | full 128,256 (baseline) | 215.7 | (same) | 261.9 | (same) |
   | P = 24,576 | 51.6 | **42.4** (5.1x) | 67.5 | **51.2** (5.1x) |
   | P = 36,864 | 86.7 | **63.3** (3.4x) | 113.8 | **76.5** (3.4x) |
   | P = 65,536 | 112.0 | 111.7 (1.9x) | 135.0 | 135.3 (1.9x) |

   The candidate build costs 0.05 ms. Commit 314113fcdd moved the candidate heads to the row-tiled pair.
3. `scripts/della/sampled_softmax_smoke.sbatch`: 60-step 130m smokes of ss and baseline on one node -- PASSED
   (job 14653414; the first attempt, 14652357, died in the executor because the Paloma caches were gone, see Data
   restore). Stages switched at steps 34 / 49 / 56 as scheduled, no overflow (max 17,860 targets per device). Mean
   step time (W&B throughput/duration, steps >= 3):

   | | step time | vs baseline |
   |---|---|---|
   | baseline (full softmax) | 645 ms | 1.00x |
   | ss, P = 24,576 (57% of steps) | 473 ms | 1.36x |
   | ss, P = 36,864 (24%) | 515 ms | 1.25x |
   | ss, P = 65,536 (12%) | 526 ms | 1.23x |
   | ss, full softmax (7%) | 628 ms | 1.03x |

   Schedule-weighted, ~502 ms vs 645 ms per step, ~1.29x. At step 60 (train loss on the full softmax in both) ss
   6.117 vs baseline 6.146, eval loss 7.182 vs 7.227, Paloma macro 7.253 vs 7.294: no sign of breakage, and 60 steps
   says nothing about the final quality.
4. Full 130m runs, packed two per 8-GPU node (`muonh_qwen3_h100x8_pair.sbatch`; the molt runs hold 9 of the 10
   pli-short node slots), everything else as the pool's SubmitLine (`--export=ALL,SIZE=130m,...`):
   * 14658326 pair1: baseline SEED=0 RUN_TAG=-restore | ss SEED=0 (same node: a paired control for speed and quality,
     and a check that the restored data reproduce the pool)
   * 14658327 pair2: ss SEED=1 | ss SEED=2
   * 14658328 pair3: ss SEED=3 | baseline SEED=1 RUN_TAG=-restore
   Readout: `objective_compare.py --pool=,-s1,-s2,-s3,-ema0.999,-ema0.999-s1,-ema0.999-s2,-ema0.999-s3
   --cell=-ss24k.57-36k.81-64k.93,...-s1,-s2,-s3` (Paloma macro and per domain, Welch) and `sampled_softmax_timing.py`.
   Pending.

## Data restore (2026-09-28)

The 09-24 store cleanup had kept only `marin_store_big/{tokenized,tokenizers}`, which removed both inputs of this
baseline. Restored on della-vis1:

* `fineweb-edu-10B/2026.06.28`: re-downloaded from `marin-community/fineweb-edu-pretokenized-10B` (1m39s, 20 GB);
  10,000,000,738 tokens / 9,966,814 documents, identical to the counts recorded in objective-hillclimb.md.
* Paloma with the marin tokenizer (`paloma/<domain>-marin-tokenizer/2026.06.28`, what the pool evaluated on): raw
  `allenai/paloma@65cd6fc` val files copied from the HF hub cache to `raw/paloma-fc6827/65cd6fc` (571 files), then
  tokenized. All 16 domains are byte-identical (tokens and document offsets, sha256) to the surviving
  Llama-3.1-tokenized caches in `tokenized/paloma/`, so the eval set is exactly the pool's.

## Changelog

* 2026-09-29 01:50: user call -- no more 130m seeds (pair2/pair3 cancelled); run the size ladder with ss seed 0 against
  the existing baselines: 300m chain 14666494-6 (2h segments), 520m chain 14666513-19 (3h), 1_2b chain 14666546-8
  (23h55). 520m/1_2b run two microbatches per step, the first use of the sampled loss under microbatching.
* 2026-09-29 01:06: pair1 done. Seed 0, n=1 vs the 8-run pool: ss macro 4.1802 vs pool 4.1865 (-0.0063, t -1.02,
  p 0.34), c4_en bpb +0.00006 (t 0.16); the step-1000 gap (+0.12 macro) closed by the end. Restore baseline 4.1820,
  inside the pool (t -0.73): the data restore reproduces the pool. Same node, equal steps: ss 40.6 min vs 52.8 min of
  training steps, 1.30x (465 / 486 / 535 / 639 ms per stage vs 639 ms). Nominal-only domain moves (no Bonferroni):
  social domains -0.03, code +0.07 (t 1.94). Seeds 1-3 pending (pair2, pair3).
* 2026-09-28 22:55: smoke passed (14653414); full runs 14658326-8 submitted.
* 2026-09-28: fineweb-edu-10B cache re-downloaded (it was deleted with the 09-24 store cleanup; 10,000,000,738 tokens,
  matches the recorded count). Implementation + CPU tests, commit 209e8d1a9d. Smoke 14652357 submitted.
