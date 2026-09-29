# Sampled softmax on the Kaiyue muonh_qwen3 baseline

**Status (2026-09-28):** implemented (`VARIANT=ss`, branch `sampled-softmax`); CPU tests pass; H100 smoke (job
14652357) queued. No verdict yet on speed or quality.

## What it is

The training loss of nanoGPT speedrun record #92 (ANVIL2, 2026-08-30, KellerJordan/modded-nanogpt PR #360), the largest
single item of that record's ablation (+8.8 s of its 34 s): for most of training each device's cross-entropy normalises
over a shared candidate set instead of the whole vocabulary. Our port, `experiments/references/sampled_softmax_qwen3.py`:

* **Candidate set, per device and step:** every class that is a target somewhere in the device's microbatch, plus
  negatives up to P from a golden-ratio stride sweep of the vocabulary (a permutation), read from an offset drawn from
  the step key and staggered per device (record #92: stride 20011 over 50,304 from a per-rank offset; same idea).
* **Kernel:** the lm_head rows of the set are gathered and the baseline's fused CE kernel runs on the `[Embed, P]` head
  with targets remapped to positions in the set. The lm_head gradient is dense over the vocabulary, zero off the set.
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
2. `scripts/della/sampled_softmax_gpu_check.py` (H100, real tokens): kernel parity and fwd+bwd timing per P -- pending.
3. `scripts/della/sampled_softmax_smoke.sbatch`: 60-step 130m smokes of ss and baseline on one node -- pending.
4. Full 130m runs vs the 8-run baseline pool with `objective_compare.py` -- pending.

## Changelog

* 2026-09-28: fineweb-edu-10B cache re-downloaded (it was deleted with the 09-24 store cleanup; 10,000,000,738 tokens,
  matches the recorded count). Implementation + CPU tests, commit 209e8d1a9d. Smoke 14652357 submitted.
