# Over-vocabulary (OV) on the Kaiyue muonh_qwen3 baseline

**Status (2026-09-30 00:10):** at 130m (n=1, same data/steps/optimizer) OV lowers Paloma macro by 0.048 (4.1384 vs the
8-run pool's 4.1865, t -7.9) and c4_en bpb by 0.0195 (t -56), at 1.38x the step time; 300m and the ovss arms are running.

## What it is

Over-Tokenized Transformer (Huang et al. 2025, arXiv 2501.16975), both halves, on the unchanged baseline recipe
(`experiments/references/over_vocab_qwen3.py`):

* **Over-encoding (OE-12.8M):** input embedding = token embedding + hashed 2-gram and 3-gram embeddings (12.8M rows
  per table, `(x_t + x_{t-1} V + x_{t-2} V^2) mod m`, exact in int32), each projected to d_model, the sum divided by
  1 + k(n-1). k from d_model/(n k) ~ 256 (k = 1 here; table width 170 at 130m since 512/3 is not an integer, 256 at
  300m). Zero tokens before the window and across documents. Per-table moduli m + 4t (the paper's "m + 2" trick,
  kept divisible by 4 for row sharding). Tables are row-sharded over the 4 GPUs for parameters and compute.
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

## Runs (seed 0, compared with the existing baseline pools)

* 130m: ov 14711157, ovss 14711158
* 300m: ov 14711159 -> 14711160 (4 h segments), ovss 14711162 -> 14711163 (3 h segments)
