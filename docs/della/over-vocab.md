# Over-vocabulary (OV) on the Kaiyue muonh_qwen3 baseline

**Status (2026-10-04):** 300m on a 2.5B-token subset repeated 2.4 times: OT +0.153 vs the baseline (-0.036 on the full data); repetition alone costs OT 0.21 and the baseline 0.02. **Earlier status (2026-10-03):** 1.2B is done: OV +0.122 and OV + ss +0.132 macro (worse on all 16 domains); 1.2B is the only size that repeats fineweb-edu-10B (2.4 passes), see "1.2B: OV loses under repeated data". Earlier status (2026-10-01): OV (over-encoding + over-decoding) beats the baseline at 130m, 300m and 520m on the same data
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
ovss's 557.7. Not planned (user, 2026-10-01): extra seeds, 8-GPU baselines, the OE vs OD split. 1.2B is the last
run; the blog follows it.

## Review notes (2026-10-03)

- 130m deltas name their baseline: the older numbers in this doc (OV -0.044, OV + ss -0.062, focal 2x2) use the `restore`
  rerun (4.1820) or the 8-run pool mean (4.1865); the blog page uses the seed-0 run `della4xh100` (4.1798), against which
  ss is +0.0004, OV -0.041 and OV + ss -0.060. All differences between these references are within the pool sd (0.006).
- Per-GPU speed at 520m mixes microbatching: the 8-GPU OV runs take 1 microbatch of 32 sequences per device, the 4-GPU
  baseline 2. At 1.2B both take 2 (8 x 16 vs 4 x 32).
- "Repeats its data": 520m makes 1.04 passes, so it repeats 4% of its data; 1.2B is the only size that repeats most of it.
- W&B refuses steps below a run's highest logged step, so after each resume the overlap is kept from the earlier segment:
  the 1.2B OV eval at step 10k (-0.065) comes from segment 2 before its timeout; segment 3 retrained steps 9790-10171 from
  the same checkpoint and data order without logging them.

## 1.2B: OV loses under repeated data (2026-10-03, both final)

1.2B trains 22,888 steps x 256 x 4096 = 24.0B tokens on fineweb-edu-10B, which holds 10,000,000,738 tokens (cache
offsets), so it runs 2.4 epochs; epochs 2 and 3 start at steps 9537 and 19073. 520m runs 1.04 epochs, 300m and 130m
less than one. Paloma macro vs the 4-GPU baseline: OV -0.065 at step 10k, then rising to +0.118 at 21k; OV + ss +0.014
at 10k and +0.132 final (3.4929 vs 3.3609, all 16 domains worse). Both OV arms jump up right after each epoch start
(OV 3.7376 -> 3.7347 from 10k to 11k while the baseline falls 0.048; OV 3.4541 -> 3.4873 from 19k to 20k while the baseline falls
0.016), and training loss keeps falling smoothly. Final (seed 0, vs the 4-GPU baseline 3.3609): OV 3.4824 (+0.1215, c4_en bpb +0.031, worse on 16/16 domains, 2103.6 min of training steps on 8 GPUs); OV + ss 3.4929 (+0.1321, 1805.4 min, 1.17x faster than OV). Both final checkpoints (step-22887, 211 GB each) kept, temp leftovers removed; with the two 520m ones these are the only OV checkpoints kept. Reading: the 12.8M-row hashed n-gram tables memorize repeated
n-grams, so OV overfits once data repeats. The timing evidence is correlational. The isolating test is the same model
on the same data with and without repetition (`tmp_plots/ss130/epoch_check.py`, `epoch_1_2b.py`).

### Seen vs unseen training data (2026-10-04, eval only)

`experiments/references/ov_memorization_eval.py` rebuilds each run's shuffled fineweb-edu-10B order (checked against the
run's own mixture) and evaluates 16 evenly spaced slices of 256 x 4096 tokens with known pass counts, plus Paloma c4_en
(jobs 14953407, 14961897; figure `tmp_plots/ss130/ov_memorization.png`). Slices that straddle a pass boundary (one per
run) are labelled mixed and left out (audit 2026-10-04). OT minus ss loss (ss stands in for the baseline):

| run | seen 3x | seen 2x | seen 1x | never seen | c4_en |
|---|---|---|---|---|---|
| 300m (0.6 passes) | | | -0.097 | -0.064 | -0.038 |
| 1.2B (2.4 passes) | -0.170 | -0.163 | | | +0.098 |

ss itself has a seen-unseen gap of 0.011 at 300m; OT has 0.044, so OT already fits the data it saw better than fresh data
of the same distribution after one pass. At 1.2B OT is 0.17 better on its training data and 0.10 worse on c4_en: it
overfits. Scrambling the n-gram indices costs OT more on seen data (300m: 0.562 seen, 0.520 unseen, 0.417 c4_en; 1.2B:
1.11-1.12 train, 0.74 c4_en): the tables help most where they were trained, which fits memorization but does not by
itself prove it; the seen-unseen loss gap carries that claim. OT+ss matches OT within 0.004 everywhere. Size and repetition are still confounded at 1.2B; the
300m 2.4-pass runs (14940537, 14952678) separate them.

### 300m with repeated data: repetition alone flips OT (2026-10-04, job 14940537)

Same 300m recipe, trained on a random 2.5B-token subset of fineweb-edu-10B repeated 2.4 times (DATA_EPOCHS=2.4,
max_train_batches 4768, passes 2 and 3 start at steps 4768 and 9536). Paloma macro, final step 11443:

| | full data (0.6 passes) | subset, 2.4 passes | cost of repetition |
|---|---|---|---|
| baseline | 3.8087 | 3.8299 | +0.021 |
| OT | 3.7728 | 3.9825 | +0.210 |
| OT minus baseline | -0.036 | **+0.153** | |

Before the subset repeats the two settings match (OT minus baseline -0.04 to -0.05 at steps 1k-5k). After pass 2 starts
the gap goes +0.014 (6k), +0.059 (7k), +0.076 (10k), and after pass 3 starts +0.151 (11k). So repetition, not size,
causes the 1.2B loss: repeating the data costs the baseline 0.02 and OT 0.21. Figure `tmp_plots/ss130/rep300_gap.png`.
Variants (14952678, final): 10x smaller tables (1.28M rows) +0.012, no second head (OV_OD_W=0) +0.179, so the input
tables cause it and their size sets it. Seen/unseen eval of the final checkpoints (14975608), loss minus the baseline's:
seen twice -0.216, three times -0.202, never seen (same distribution) +0.061, c4_en +0.132. The baseline's own
seen-unseen gap is 0.05-0.06, OT's 0.32: OT fits the repeated subset far better and generalizes worse, textbook overfitting. Scrambling the n-gram
indices costs OT 1.06-1.08 on seen data, 0.76 on unseen and 0.62 on c4_en. The same eval on the two variants
(14975609_3, 14997698_4; columns seen 3x / 2x / never / c4_en, minus the baseline): no second head -0.211 / -0.230 /
+0.074 / +0.150, the same overfitting as OT with both heads, so the second head plays no part; 1.28M-row tables -0.055 /
-0.060 / -0.013 / +0.004, a seen-unseen gap of 0.04-0.05 beyond the baseline's against 0.26-0.28 for 12.8M rows (0.29-0.30 without the second head). Literature: the recsys "one-epoch phenomenon"
(Zhang et al. 2022, arXiv 2209.06053; MEDA 2305.19531; AdamAR 2511.06374) and MoE under repetition (Xue et al. 2023,
2305.13230; Jha et al. 2026, 2609.11917): total, not active, parameters set repetition damage; 300m OT tables hold 6.5B
parameters against 2.5B repeated tokens. Next: tables frozen at the pass-2 start (14983240) separates "tables keep
fitting" (MEDA) from "backbone adapts to trained rows" (Zhang).

### Real 2-gram output vocabulary (od_mode=hashed), 130m (2026-10-04, job 14952023)

Paloma macro minus the seed-0 baseline 4.1798 at the final step: real 2-gram output vocabulary -0.0623, OT (product
decomposition) -0.0414, OT + ss -0.0595 (pool sd 0.006, one seed each). So predicting (x_{t+1}, x_{t+2}) jointly over
12.8M hashed classes with a sampled softmax beats the factorised second head by 0.021 (about 2.5 sd of a two-run difference) at the same time per step
(0.906 vs 0.890 s). Caveat: od_proj's gradient norm spiked to 2.3-2.9 at steps 600-1500 (peak LR) and was clipped; it
settled after step 1600. 300m queued (14988894). Page: Figure 7.

Fair comparison (owner, 2026-10-04): OT-2's head always trains by a sampled softmax, and at 130m OT-H + ss (-0.0594)
already ends close to OT-2 (-0.0622): 0.003 apart, about 0.3 sd of a two-run difference. So the 0.021 gain over OT-H may
come from the sampled softmax, not from the real 2-gram vocabulary. At 300m ss does not help OT-H (+0.002), so the 300m
OT-2 run tests the vocabulary more cleanly. VARIANT=ovgramss (a15a909f24) adds SS_SCHEDULE to OT-2's main head and
completes {OT-H, OT-2} x {main head full, sampled} at 130m and 300m (jobs 15009779/15009780 after smoke 15009778).
Page Figure 8 now shows OT-2, OT-2 + ss and OT-H + ss, each minus OT-H. Note: OT-H + ss trails OT-H by 0.08-0.10
through most of training (sampled stages, full-softmax eval) and catches up only after the full-softmax stage.

### Does a smaller vocabulary make repeated data hurt less? (2026-10-04, jobs 15005159-62)

Owner's question after the 300m result above. The Marin tokenizer is Llama 3 byte-level BPE whose ids 0..127999 are
BPE ranks, so keeping the K lowest-ranked tokens and the merges among them is the tokenizer BPE training would have
produced at K (`experiments/references/small_vocab_tokenizer.py`; K = 64K/32K/16K/8K, specials moved to K..K+255).
Pretokenized data converts without the text: inside a pre-token, every token of id >= K re-encodes on its own, so a
fixed table expand[t] maps the stored ids (validated exact against re-encoding decoded text). Converted caches
(`convert_small_vocab_cache.py`, job 15003340): fineweb-edu-10B and the 16 Paloma subsets, each checked for document
count, identical decoded text, and equality with re-encoding where the source ids are canonical (Paloma was tokenized
in newline chunks, so only some documents are). Tokens per text, fineweb-edu-10B: 64K x1.0313, 32K x1.1025,
16K x1.2043, 8K x1.3513.

Design (owner's picks): all four K; fixed text and passes, so steps scale by the token ratio and every K sees the
same bytes. Per K one 8-GPU pair at 300m baseline: full data (0.6 passes) and the 2.5B-token-text subset (2.4
passes); the 128K pair is rep-300m (14940537, run b) with baseline 300m s0. Launcher: `VOCAB_K=K` (run tag -v{K}k).
Metric: Paloma bits per byte. Levanter's byte count overcounts partial UTF-8 tokens (U+FFFD counts 3 bytes), more so
at small K; Levanter/exact bytes, mean over subsets: 128K 1.0019, 64K 1.0035, 32K 1.0048, 16K 1.0064, 8K 1.0090
(`tmp_plots/ss130/bpb_bytes_check.py`). Multiply each logged subset bpb by its factor before any difference.
Confound to state with the result: at a smaller K one pass over the text is more steps, so the model takes more
updates per repeated byte.

Power check before the runs started: with the Marin tokenizer, 2.4 passes cost the 300m baseline only +0.0078 bits
per byte at the end (repeated minus full, exact bytes), and the gap wanders by up to 0.013 during passes 1 and 2. The
300m baseline seed sd is 0.0024 bpb (s0-s3: 1.3867, 1.3862, 1.3915, 1.3880), so the difference of two repetition costs
has sd ~0.0048 with one seed per run: even a K with no harm at all would differ by only ~1.6 sd. 2.4 passes can show a
K that hurts much more, not one that hurts less. Owner's call: run both 2.4 and 8 passes. The 8-pass arm uses a
0.75B-token part (round(steps / 8) batches) for every K including 128K (jobs 15006934-36); Muennighoff et al. 2023
find repetition nearly free up to about 4 epochs, so 8 passes should cost clearly more.

Results (2026-10-06; final Paloma bits per byte, exact bytes, one seed each; cost = repeated minus full):

| K | full (0.6 passes) | 2.4 passes | cost 2.4 | 8 passes | cost 8 |
|---|---|---|---|---|---|
| 128K | 1.3867 | 1.3945 | +0.0078 | 1.4695 | +0.0828 |
| 64K | 1.3939 | 1.3992 | +0.0053 | 1.4690 | +0.0751 |
| 32K | 1.4058 | 1.4223 | +0.0165 | queued | |
| 16K | 1.4267 | 1.4399 | +0.0132 | queued | |
| 8K | 1.4552 | 1.4539 | -0.0013 | 1.4814 | +0.0262 |

At 2.4 passes the costs scatter around the 128K one (-0.009 to +0.009, within ~2 sd of 0.0048) with no trend in K, as
the power check predicted. At 8 passes 8K pays 0.057 less than 128K (~12 sd): a small vocabulary does make heavy
repetition hurt less, but 128K still ends 0.012 lower in absolute terms (the full-data gap of 0.069 shrinks to 0.012).
The 32K and 16K 8-pass pair (15006935) decides whether the cost falls steadily with K. Job 15006934 ended FAILED (rc 134,
"terminate called without an active exception" on exit) after both runs logged "succeeded" and W&B finished at the last
step; the record counts it as completed.

## Focal loss + OV (2026-10-02, done: negative, stopped at 130m)

User call: focal loss on the baseline's one head (VARIANT=focal) and on both of OV's heads (VARIANT=ovfocal), gamma 0.5
and 1, at 130m, seed 0, same data/steps/batch as the baseline. Per-token cross-entropy l becomes (1 - e^-l)^gamma * l
with the full-softmax p (gradient through both factors), reduced as the baseline's loss, so gamma 0 is the baseline
exactly; eval stays plain cross-entropy (`experiments/references/focal_qwen3.py`, CPU tests
`scripts/della/focal_cpu_test.py`). The 2 x 2 {baseline, OV} x {CE, focal} separates focal's own effect from its
interaction with OV. Prior (MiLe, arXiv 2310.19531): gamma >= 1 raises Pile perplexity, gamma 0.5 about neutral, so the
expected Paloma result is at best neutral. Go to 300m only if ovfocal beats OV by more than 0.012 macro (~2 seed sd).
Jobs: smoke 14885494, ovfocal pair 14885495, focal pair 14885496.

**Result (130m, n = 1 per cell, Paloma macro vs the 8-run pool 4.18646 +- 0.00577 sd; t is one-vs-sample):**

| gamma | baseline + focal | OV + focal | focal effect on OV |
|---|---|---|---|
| 0 (CE) | 0 (pool) | -0.0481 (t -7.9) | |
| 0.5 | +0.0074 (t +1.2), lower on 4/16 domains | -0.0528 (t -8.6), 16/16 | -0.0047 |
| 1 | +0.0134 (t +2.2), lower on 4/16 domains | -0.0570 (t -9.3), 16/16 | -0.0089 |

Focal raises the baseline's loss (as MiLe found) and lowers OV's by less than the gate at both gammas, so both 300m
pairs (14888823, 14888825) were cancelled. The interaction (OV effect minus baseline effect) is -0.012 at gamma 0.5 and
-0.022 at gamma 1, about 1.5-2.7 single-run sd, one seed: a hint that OV's extra capacity makes the down-weighting
of easy tokens useful, not a result. Scripts: `tmp_plots/ss130/focal_2x2.py`, `focal_2x2_fig.py`.

## Run monitoring (P0, training-monitor skill, 2026-10-02)

Levanter already logs train/loss, grad/norm/total (and per parameter), throughput/duration, MFU and the Paloma evals; W&B
keeps train metrics every 10 steps. `scripts/della/p0_check.py` reads the recent points of each running run every
5 minutes and alerts once per event on: non-finite loss or grad norm; z(log loss) > 8; z(log loss) > 6 and z(log grad
norm) > 6 at the same step (co-spike); more than 5% of the last 300 points above max_grad_norm; median step time of the
last 30 points above 1.25x the run's own median. z is the robust spike score over a trailing 100-point window. The
thresholds come from six healthy runs (130m-1.2B: OV, OV+ss, baseline, ss): z(log loss) max 4.5, no co-spike at 6,
clip rate <= 0.5%, step-time p99 <= 1.28x median; z(log grad norm) alone reaches 25 at sampled-softmax stage changes,
so it only alerts together with the loss. Peak device memory, which the trainer does not log, comes from the launcher's
MEMSTATS line (default every 600 s, max over local devices) and alerts above 95% of the allocator limit. Not covered:
the output absolute maximum (needs a model hook) and worst-rank loss (one JAX process per run).

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
