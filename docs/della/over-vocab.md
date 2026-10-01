# Over-vocabulary (OV) on the Kaiyue muonh_qwen3 baseline

**Status (2026-10-01):** OV (over-encoding + over-decoding) beats the baseline at 130m, 300m and 520m on the same data
and steps, and with the sampled softmax on both heads it does so at about baseline speed: Paloma macro -0.066 at 130m
(t -10.8 vs the 8-run pool), -0.038 at 300m (t -5.3 vs the 4-run pool) and -0.029 at 520m (one baseline run, all 16
domains lower); 1.04x / 1.03x faster than the baseline at 130m / 300m, and 1.13x faster than OV alone at 520m (8 GPUs,
no 8-GPU baseline). n = 1 per arm. 1.2B is running.

## What it is

Over-Tokenized Transformer (Huang et al. 2025, arXiv 2501.16975), both halves, on the unchanged baseline recipe
(`experiments/references/over_vocab_qwen3.py`):

* **Over-encoding (OE-12.8M):** input embedding = token embedding + hashed 2-gram and 3-gram embeddings (12.8M rows
  per table, `(x_t + x_{t-1} V + x_{t-2} V^2) mod m`, exact in int32), each projected to d_model, the sum divided by
  1 + k(n-1). k from d_model/(n k) ~ 256 (k = 1 here; table width 170 at 130m since 512/3 is not an integer, 256 at
  300m). Zero tokens before the window and across documents. Per-table moduli m + 4t (the paper's "m + 2" trick).
  Tables are row-sharded over the data axis for parameters and compute; their rows are padded to a multiple of 64
  (`ROW_ALIGN`, since 2026-09-30) so any device count up to 64 divides them. Rows >= m are never indexed, so the
  padding leaves the hash and the model unchanged; the 130m/300m runs predate it (rows = m on 4 GPUs).
* **Over-decoding (n = 2):** the paper's product decomposition of a 2-gram output vocabulary, L = CE(h E_1, x_{t+1}) +
  lambda_2 CE(W_2 h E_2, x_{t+2}); E_1 = lm_head, E_2 = new `od_lm_head`, W_2 d x d, lambda_2 = 0.1. Not MTP-DS (user
  call, 2026-09-29): the paper's final OT model uses MTP-DS, this uses its OD formulation.
* **Training setup = baseline:** data, steps, batch, schedule, MuonH with its own labels for the new parameters
  (tables adam like the token embedding, no weight decay; W's muonh; od_lm_head adamh like lm_head). Eval = main head,
  full softmax, so eval/paloma is comparable with the baseline pool.
* **ovss:** OV with the sampled softmax on both output heads (docs/della/sampled-softmax.md).

## Verification

* `scripts/della/over_vocab_cpu_test.py` (fake 4-device mesh, sharded tables) -- PASSED: hash == int64 formula for the
  real V and m; boundary zeros; OE embedding == direct computation; loss == NTP + 0.1 OD and every gradient == an
  independent reference; eval == NTP; ovss == ov in the full-softmax stage and == the per-head candidate-set reference
  in sampled stages; optimizer labels.
* `scripts/della/over_vocab_smoke.sbatch` (job 14697957, 60 steps, H100) -- PASSED, rc 0 for all four:

  | | step time | vs baseline | step-60 eval loss |
  |---|---|---|---|
  | 130m baseline (smoke 14653414) | 645 ms | | 7.227 |
  | 130m ov | 887 ms | 1.38x slower | 7.131 |
  | 130m ovss | 595 ms | 1.08x faster | 7.096 |
  | 300m baseline (pool) | 1450 ms | | |
  | 300m ov | 1736 ms | 1.20x slower | 7.430 |
  | 300m ovss | 1406 ms | 1.03x faster | 7.297 |

  OV's cost is mostly the second full-vocabulary output head; the sampled softmax on both heads more than pays for
  OE + OD. Peak memory is not measurable from nvidia-smi (JAX preallocates 90%); no OOM at either size.

## Results

**130m ov (14711157), n=1 vs the 8-run pool, 4959 steps:** macro_loss 4.13839 vs 4.18646 (-0.04807, t -7.85, p 1e-4),
macro_bpb -0.01615 (t -7.08), c4_en bpb 1.14378 (-0.01948, t -55.6). 13 of 16 domains lower; 7 clear the Bonferroni
bar (c4_100_domains -0.070, c4_en -0.063, falcon -0.069, m2d2_wikipedia -0.084, m2d2_s2orc -0.041, mc4 -0.052,
wikitext_103 -0.057, gab -0.073); code +0.031 (t 0.85, ns). The gap is already there at step 1000 (4.840 vs the
pool's ~4.884) and widens to the end. Step time 890 ms vs 640 (1.38x): 73.9 min of training steps vs 53. Final train
losses: ntp 3.129, od (x_{t+2}) 4.925. Not yet separated: how much comes from OE and how much from OD.

**All arms (seed 0).** Training time = summed throughput/duration over all steps; deltas vs the same-seed baseline
(130m: the restore baseline, same node as ss) and, in brackets, vs the pool mean with the one-vs-sample t:

| size | arm | training time | vs baseline | Paloma macro | c4_en bpb |
|---|---|---|---|---|---|
| 130m | baseline | 52.8 min | 1.00x | 4.1820 | 1.1632 |
| 130m | ss | 40.6 min | 1.30x faster | -0.0018 (-0.006, t -1.0) | +0.0001 |
| 130m | ov | 74.1 min | 1.40x slower | -0.0436 (-0.048, t -7.9) | -0.0194 |
| 130m | ovss | 50.6 min | 1.04x faster | -0.0616 (-0.066, t -10.8) | -0.0196 |
| 300m | baseline | 276.7 min | 1.00x | 3.8087 | 1.0563 |
| 300m | ss | 243.8 min | 1.13x faster | -0.0055 (-0.010, t -1.4) | +0.0004 |
| 300m | ov | 336.8 min | 1.22x slower | -0.0360 (-0.040, t -5.7) | -0.0114 |
| 300m | ovss | 269.7 min | 1.03x faster | -0.0335 (-0.038, t -5.3) | -0.0103 |
| 520m | baseline (4 GPU) | 1115.8 min | 1.00x | 3.5726 | 0.9883 |
| 520m | ss (4 GPU) | 1042.5 min | 1.07x faster | -0.0043 | -0.0003 |
| 520m | ov (8 GPU) | 630.5 min | (8 GPU) | -0.0361 | -0.0093 |
| 520m | ovss (8 GPU) | 557.7 min | 1.13x faster than ov | -0.0287 | -0.0083 |

300m per domain: both OV arms lower all 16 domains; 6 clear the Bonferroni bar (c4_100_domains, c4_en, falcon,
subreddits, m2d2_wikipedia, mc4). The gain shrinks from 130m to 300m (macro -0.066 -> -0.038, c4_en -0.020 -> -0.010),
as the paper's log-linear law would put it for a fixed table against a growing model. At 520m (one baseline run, so no
t statistic; the 300m pool's seed sd was ~0.006) both OV arms again lower all 16 domains, ov by -0.036 macro (the same
as at 300m) and ovss by -0.029; the ovss - ov gap (+0.007) is within one seed sd. The 520m OV arms ran on 8 GPUs, so
their training time is comparable only with each other: a perfectly scaling 8-GPU baseline would take ~558 min, about
ovss's 557.7. Open: OE vs OD attribution; 1.2B.

## Runs (seed 0, compared with the existing baseline pools)

* 130m: ov 14711157 (COMPLETED), ovss 14711158 (COMPLETED)
* 300m: ov 14711159 -> 14711160 (COMPLETED), ovss 14711162 -> 14711163 (COMPLETED)

### 520m and 1.2B on 8 GPUs

User call (2026-09-30): both sizes run OV and ovss on 8xH100. At 4 GPUs the OE tables alone would take ~35 GB (520m)
and ~70 GB (1.2B) per device in fp32 parameters + Adam state + gradients. Quality is compared with the 4-GPU
baselines (same data, steps and global batch; only the data-parallel layout differs); speed is compared only among the
8-GPU runs, since there is no 8-GPU baseline. Launcher overrides: `NUM_GPUS=8` (run ids `della8xh100`) and, at 1.2B,
`PDP=16` sequences per device per microbatch, which keeps the baseline's 128 sequences per microbatch.

* First 8-GPU smoke (14768821, 520m): `IndivisibleError`, a 12,800,004-row table over 8 devices; fixed by the row
  padding above (1604abd2e2), CPU tests re-passed.
* 520m smoke (14775297, 60 steps): ov 3779 ms/step, ovss 3350 ms/step, rc 0 for both.
* 1.2B smoke (PDP=16, 14793357 on pli-cp after 14775298 sat 5 h on pli-short): ov 4998 ms/step, ovss 4313 ms/step
  (a 60-step smoke passes through all four ss stages in the run's proportions, so this is the run average), rc 0 for
  both; the 210 GiB final checkpoint saved in 122 s.
* 520m runs: ov 14779674 (13:30 h), ovss 14779675 (12:30 h), single resumable jobs on pli-short.
* 1.2B runs: ov 14794354 -> 14812270 (23:55 h + 15:00 h), ovss 14794356 -> 14794357 (23:55 h + 9:30 h), pli-short
  chains submitted before the smoke result so they accrue queue age; second segments sized from the smoke (~33.6 h
  and ~29.9 h of work including evals and hourly checkpoint stalls).
* **1.2B OOMs under the default BFC allocator (2026-10-01).** Persistent state is ~27 GiB per device (tables,
  their Adam state, the model) and every train step allocates one contiguous temp buffer of 30.18 GiB (ov) or
  25.08 GiB (ovss), most of it the tables' gradient and its microbatch accumulator. ovss died at step ~130 and ov at
  step ~1,290 with ~44 GiB free but the largest free block ~0.7 GiB short of the temp: a long-lived small block had
  landed beside the temp region. Allocator diagnosis 14824921 (1.2B ovss, 300 steps, `scripts/della/ov_alloc_diag.sbatch`):

  | allocator | outcome | stage-1 ms/step | over the run |
  |---|---|---|---|
  | BFC, fraction 0.9 (default) | OOM at step 119, as in production | 4253 | |
  | cuda_async 0.9, no preallocation | 300 steps | 5509 (+30%) | 5688 ms |
  | **cuda_async 0.75, preallocated** | **300 steps** | **4312 (+1.4%)** | **4491 ms (+4%)** |
  | vmm 0.9 | 300 steps | 4985 (+17%) | 5138 ms |

  Since 83270354c5 the launcher gives 1.2B OV runs cuda_async with a preallocated pool unless the job chose an allocator:
  0.8 for ov (30.2 GiB temp; 63.3 GiB pool), 0.75 for ovss (25.1 GiB temp; 59.4 GiB pool). The cuda_async process
  can abort at exit (rc 134 after the step succeeded); the run itself is unaffected. Runs: ovss 14824631 -> 14824632
  (started 10-01 16:43); ov's queued second segment 14812270 resumes from step 678 and picks the allocator up at start.
* Checkpoint policy for OV variants (50dfe21fc9): an OV checkpoint carries the n-gram tables with their Adam state
  (79 GB at 300m, 112 GB at 520m, ~227 GB at 1.2B). No step-interval permanent checkpoints (the GROUP fileset was
  96% full), and the resume checkpoint is saved hourly instead of every 10 minutes, because a save stalls training
  while it stages to host: at 300m ov, ~70 s per save, ~11% of wall-clock. The reported speeds are summed step times,
  which exclude checkpoint stalls for every arm, so the 130m/300m OV wall-clock was ~11% longer than its step-time
  total. Training itself is unchanged.
