# TREC: training re-evaluation curves of existing MuonH checkpoints

**Result (2026-10-27, extended 2026-10-28).** Re-scored with their final weights, two MuonH runs fit best the batches they read about four
fifths of the way through training (0.81 at 300m with cosine, 0.89 at 130m with a linear decay; 0.069 and 0.088 nats
below unread text) and have forgotten the first quarter (0.001 +- 0.001 nats below unread text). Bergsma et al.'s
formula 1 - sqrt(c) t^m follows this shape only with a fitted exponent (m 3.4-4.8, Pearson r 0.96-0.99), and then a
control without any averaging (c = learning rate) fits nearly as well (r 0.963 against 0.971 for the momentum reading
at 300m, 0.988 for both at 130m); with the paper's m* for AdamW every reading misplaces the low point by 0.04-0.15 of
training. More steps over the same text (8K, bytes) move the low point later than the averaging predicts, with
either training seed. On 24 more checkpoints (2026-10-28), a larger learning rate, a long EMA and bytes move the low point by
0.07-0.11 of training, the averaging predicts the direction and a quarter to three fifths of the size, depth, batch and
seed do not move it; runs that reread their data fall on one TREC when each batch is placed at its last read.
On 2026-10-29: equal weights c (the 1/sqrt(t) schedule) keep 0.003 nats more of the first quarter and cost 0.03-0.11 nats;
early batches are learned (0.07-0.12 nats) and then forgotten (1-4% left), at a rate set by sum(lr^1.3).

Owner request 2026-10-26: a new Experiments page that applies Bergsma, Dey and Hestness (2025, arXiv 2509.25380) to
our finished pretraining checkpoints. Page: `exps/trec.html` on the website branch `pretrain-practice`
(drafted in the detached worktree `tmp/trec_site_wt`).

## Scope (owner popups 2026-10-26 and 27)
- Runs: the pls depth control `muonh-qwen3-130m-della4xh100-d12` (130m width, 12 layers, lr 0.02 held for the first
  fifth then linear to zero, no warmup, 4959 steps) and the Tokenizer Size 128K unique run
  `muonh-qwen3-300m-della4xh100-s1-rerun` (300m, 1000-step warmup then cosine, lr 0.01, 11444 steps).
- Every training batch scored, plus 512 unread batches at stream indices 11444..18598 (shared by both).
- Page split (owner, via Tokenizer Size): data properties and repetition (8-pass vs unique, token frequency) belong to
  the vocab-overfitting page; this page has the method, TREC shapes, schedules, the MuonH prediction and placement.
- Ask the owner before launching any experiment, CPU checks included.
- 2026-10-27 popup: add the learning-rate-only control to Figures 2 and 4 (and a fourth setup card); the paper's m*
  low points stay in the text; run the layer-0 CPU check.

## Facts
- Data order: one fineweb-edu-10B cache under the Marin tokenizer, linear permutation with data seed 42, then
  MixtureDataset's permutation inside every 2048-sequence block (even for one component); batch t = sequences
  128t..128t+127 of LmDataConfig.train_set; the 130m run reads the first 4959 batches of the 300m stream.
- Replay check (smoke w447): the last training batch matches W&B train/loss to 0.0001 nats at both sizes.
- MuonH reads no weight decay: the 0.1 in the config reaches no group (comment-only fix chosen by the owner in both
  sessions). Matrices (MuonH) and lm_head (AdamH) keep a fixed Frobenius norm; embeddings and vectors use plain Adam.
- Fixed-norm forgetting (vocab session's note tmp/trec/muonh_prediction.md): every step W <- a_t (W - eta R O_t),
  a_t = (1 + 2 eta r + eta^2)^(-1/2); momentum makes steps correlated, r = kappa eta + r0, alpha ~ A eta^2 / 2 with
  A = 73 measured on 300m checkpoints (momentum 0.98), A = 23 from the toy at momentum 0.95 (130m).
- TREC numbers (experiments/trec/analyze.py on w465): unread level 3.1422 / 2.8983 nats (+-0.0011); low point
  t = 0.886 / 0.815 (100-step mean; 0.894 / 0.805 with 500 steps); quarter means minus unread: 130m -0.0010,
  -0.0073, -0.0274, -0.0669; 300m -0.0010, -0.0060, -0.0313, -0.0406 (+-0.0013); last 1% of 300m -0.0007 +- 0.0025.
- Fits (m by Pearson r on the 100-step mean): 130m momentum 4.2 / 0.9879, independent 4.8 / 0.9880, adamw-0.1
  3.9 / 0.9892, lr-only 4.8 / 0.9880; 300m 3.5 / 0.9712, 4.2 / 0.9628, 3.4 / 0.9708, 4.3 / 0.9626. Paper's m*
  (TPP 14.7 / 20.0): 130m 8.93, 2.17, 6.11 (low 0.948, 0.814, 0.929); 300m 12.30, 1.78, 6.85 (low 0.925, 0.669, 0.878).
- Figure 4 draws the measured curve as (s - min) / (unread - min); min-max over the noisy maximum had put its forgotten
  part near 0.85. Half-depth width at 300m: measured 0.253, predictions 0.349-0.374.
- Init retention (init_retention.py): cos(W_T, W_0) median 0.019 / 0.024 in layers 1-11; layer-0 attention up to
  0.197 / 0.344; predictions 0.63 / 0.81 (A = 1) and 2e-5 / 1.5e-7 (momentum).
- Layer-0 check (init_step.py, 2026-10-27): the next step NS(M) of the final checkpoints overlaps W_0 at chance in every
  matrix: q_O / (1/sqrt(n)) has rms 0.91 (130m) and 1.15 (300m) over 84 matrices, layer-0 attention -2.7..+0.4. No
  update adds the initialization back at the end; the layer-0 excess comes from earlier in training (only final
  checkpoints exist for these runs).
- w438 (vocab session): unique minus 8-pass, low point at 0.783 / 0.830 / 0.856 of training (128K / 8K / bytes);
  the momentum reading with m = 3.3 fitted at 128K predicts 0.789 / 0.792 / 0.807. Second seed (w457, owner approval
  2026-10-27): 8K seed 0 at 0.814 (-0.0157 bpb), bytes seed 1 at 0.872 (-0.0144); 128K seed 0 has no unique-token run.
  Figure 5 draws one line per seed; each predicted ring sits on the seeds' mean curve.

## Code (branch `trec`, worktree `project/marin-trec`, private remote `archive`, public `marin-pli` branch `trec`)
- `experiments/references/trec_score.py`: chunked, resumable scoring; `scripts/della/trec{,-smoke}.sbatch`.
- `experiments/trec/predict.py` (c_i for three readings and the lr-only control, 1 - sqrt(c) t^m, m by Pearson r),
  `analyze.py`, `init_retention.py`, `init_step.py`, `toy_sphere.py`, `fetch_wandb.py`, `build_site_data.py`,
  `build_record.py`, `setups_svg.py`, `diff_spec.json`.
- Outputs: `project/trec/out/<run>/<first index>.npz`, `out/init/`, `out/init_step/`; W&B cache
  `project/trec/cache/wandb/`; job records `project/trec/logs/*.yaml`.

## Jobs
- w439 smoke: check failed (replay order, fixed). w441: cancelled before start. w447 smoke: passed.
- w465 main: done 2026-10-26 23:46 - 2026-10-27 01:27 on 4 GPUs, 6.7 GPU-h.

## Page (exps/trec.html)
- Q1 answered (Fig 1), Q2 answered (Figs 2-4), Q3 answered (Fig 5, w438). Standfirst = Takeaways = meta description.
- Before publishing (2026-10-27): anti-autoresearch, deterministic-only (no codex reviewer in the session), L0 on the
  page text: CLEAN_GIVEN_EVIDENCE (tmp/trec_audit/REPORT.md). A manual check of every hand-written number fixed two
  captions (the unread band is +-2 SE, not 2 SE wide; the low points are four fifths to nine tenths, not about four
  fifths). Public code: wenhaochai/marin-pli branch trec at 3b44498, then 09e8580 with the second seed, then 2347279 with Figures 6-7 (scrubbed; secrets 0,
  cluster user and group 0).
  Website: pretrain-practice e2d3fef; the main merge waits for the owner's weekly approval.
  2026-10-28: Figures 6-7, Q4/Q5 answered; pretrain-practice 7be7eaf, released to main as d22ff55 (owner approval via the
  controller; record audit: 37 rows done, main 37.0 H100-h, other 0.4).

## Other checkpoints (owner popups 2026-10-27; done 2026-10-28)
- 24 more final MuonH checkpoints, one setting changed per group: depth (130m d1-d5, d24, d36), size (pls0-noev 300m,
  520m, 1.2b), learning rate (300m v8k, lrx2/3/4 from step T/8), EMA (ema0.999, ema0.9998; raw and averaged weights),
  training length (130m tpp200, two seeds), batch (b96), tokenizer (v16k, v32k, v64k, bytes) and seed (v8k-s1).
  1 in 8 training batches (blocks of 8 steps every 64) plus 512 unread ones.
- Jobs: w512 smoke 21:50-22:09 (1 GPU; all 26 last batches within 0.0001 nats of W&B), w513 23:23-02:32 and w514
  22:21-02:47 (4 GPUs each), 30.3 GPU-h. Controller 2026-10-27: wq jobs as whole 4-GPU blocks from now on.
- Low point = minimum of a moving mean over 8 sampled blocks (512 steps). Noise: the 7 other block offsets of the
  fully scored d12 / s1-rerun move it by sd 0.009 / 0.011; windows of 4 or 16 blocks by up to 0.034. Runs that share a
  data stream share batch difficulty, so seeds agree to 0.001 (128K 0.797 / 0.797, 8K 0.822 / 0.822).
- Figure 6 (measured / momentum prediction, m 4.2 at momentum 0.95, 3.5 at 0.98): depth 1-36 all 0.898 / 0.898;
  lr x1-x4 0.822, 0.842, 0.867, 0.896 / 0.798, 0.817, 0.832, 0.844; EMA weights 0.999 0.797 / 0.782, 0.9998 0.685 /
  0.750 (raw 0.797 / 0.796; EMA prediction averages c over the EMA weights, predict.run_coefficients); batch 96 0.812 /
  0.798; 8K 0.822, 16K 0.806, 32K 0.799, 64K 0.806, 128K 0.797, bytes 0.863 / 0.797-0.812. The prediction gets each
  direction and 0.24-0.62 of the large shifts (lr x4, EMA 0.9998, bytes).
- The fineweb-edu-10B stream has 2,441,406 sequences: 520m (1.04 passes), 1.2b (2.40) and 130m tpp200 (2.60) read it
  more than once. Figure 7 (owner popup 2026-10-28: both x axes): against t the 1.2b TREC repeats every pass (0.417);
  against the last read of each batch's data it is one curve, low 0.81 (prediction 0.80); 520m 0.79 (0.80); tpp200
  0.957 in both seeds (0.92), 0.03 nats deep. At the read-count edge (1.2b 0.833, tpp200 0.769) the loss changes by
  <= 0.003 nats (within 1.4 SE): an earlier read leaves no measurable trace.

## 1/sqrt(t) schedule (owner request and popup 2026-10-28)
- Source: Su, kexue.fm/archives/11459 (EMA view of weight decay): every batch keeps the same weight in the final
  weights when e^(lambda_j eta_j) eta_j = eta_(j-1); with lambda proportional to eta, eta_s = eta_max / sqrt(1 + 2 lambda_max eta_max s).
  MuonH's fixed norm shrinks the weights by a ~ 1 - A eta^2 / 2 per step, a weight decay lambda = A eta / 2 proportional
  to eta, so eta_s = eta_max / sqrt(1 + A eta_max^2 s) (within 4% of the exact recursion with r0). On 130m d12 the weight
  c of the first quarter goes from 0.004 to 1.0 (300m: 0.000 to 1.0).
- Arms (popup: 130m, both): d12 trained again from its commit 4530d3bb (branch trec-rsqrt, worktree
  project/marin-trec-rsqrt, LR_SCHEDULE knob) with rsqrt (lr 0.02 / sqrt(1 + 0.0092 t), ends at 0.0028) and rsqrtcd
  (the same to step 3967, then linear to 0); CPU check: the built schedules equal the formula, and without the knob
  the code builds d12's schedule. Job scripts/della/trec_rsqrt.sbatch: 30-step smoke, training, TREC (MODEL=130m-d12).
  wq w590 (rsqrt), w591 (rsqrtcd), 4 GPUs, about 8 GPU-h each. Record logs/trec-rsqrt.yaml.
- Page: Question 6 and Figure 8 placeholder (popup approved): learning rates now, TREC queued.
- Result (w590 17:43-19:38, w591 17:54-19:49, exit 0; 15.4 H100-h; tools/rsqrt_trec.py, rsqrt_eval.py): logged lr
  within 0.9% of the formula; no NaN. Quarter means minus unread (d12 / rsqrt / rsqrtcd): Q1 -0.0010 / -0.0037 /
  -0.0042, Q2 -0.0073 / -0.0116 / -0.0144, Q3 -0.0274 / -0.0214 / -0.0316, Q4 -0.0669 / -0.0701 / -0.0656 (+-0.0014;
  paired against d12, same batches: Q1 -0.0027 / -0.0032 +- 0.0001). Unread 3.1422 / 3.2390 / 3.1702; Paloma eval/loss
  3.8519 / 3.9591 / 3.8828. Low point (100-step) 0.886 / 0.998 / 0.886 (predicted 0.898 / 1.000 / 0.896).
  Conclusion: c 0.004 -> 1.0 keeps 0.003 nats more of the first quarter and costs 0.03 (cooldown) to 0.11 nats of final
  loss; the share c of an update does not set how well the final model fits its batch.
- Page: Q6 answered, Figure 8 filled, standfirst = takeaways = meta extended; deterministic anti-autoresearch L0:
  CLEAN_GIVEN_EVIDENCE. Not done: public code for trec-rsqrt (based on the pls commit 4530d3bb; the scrub covers the
  trec branch only), so the two record rows carry no Code link.

## Imprint and decay (owner via the controller 2026-10-29: sessions add experiments; popup "提交两个", placeholder approved)
- Question 7: were the early batches learned little, or learned and then forgotten? d12 and rsqrtcd trained again
  with a checkpoint every 248 steps (KEEP_EVERY=248, tag -ck248, commit 31a2bb26e0); each checkpoint scores 1 in 16
  training batches (blocks of 8 every 128) and 256 unread ones, the final one every batch (scripts/della/trec_ck.sbatch;
  the smoke keeps checkpoints every 10 steps and must leave step-10 and step-20). wq w642, w643 (--partial, 4 GPUs).
  Figure 9: one line per checkpoint (steps 992-4958), unread level minus batch loss against when the batch was read.
- Record logs/trec-ck.yaml. Scratch file quota: 97.2M of 100M files (2026-10-29); a checkpoint is 24 files.
- Result (w642 04:31-07:30, w643 -07:55, exit 0; tools/imprint.py): imprint E = (U_s - L_s(b)) - (U_p - L_p(b)), p the
  last checkpoint before the read. d12: first-quarter batches 0.073 nats right after the read, 0.0007 at the end (1%);
  0.75-0.95: 0.102 -> 0.072 (71%); batches of the first half lose 81% within ~500 steps. rsqrtcd: first quarter 0.120 ->
  0.0044 (4%). Decay of the two runs lines up best against the sum of lr since the read (relative spread 0.24; steps
  0.56, sum lr^2 0.57; tools/imprint_collapse.py). Q7 answered: learned, then forgotten.
- Q8 (owner popup 2026-10-29): branches from d12-ck248 step 2480 (INIT_FROM: step, optimizer, data position) at a
  constant lr with lr x N = 5: 0.01 x 500, 0.005 x 1000, 0.0025 x 2000 (sum lr^2 0.05, 0.025, 0.0125); checkpoints every
  124 steps score batches 2232-2479 + 256 unread (scripts/della/trec_branch.sbatch, commit dc5c92b4c9; smoke must start
  from the checkpoint, loss < 4). wq w651-w653. Figure 10: fraction of the imprint left against the three measures.

- Branch jobs w651-w653 failed their smoke gate (the smokes were good: resumed at 2481, eval loss 4.25-4.28; a 10-step
  run prints no train-loss postfix). Gate fixed (resume line + W&B eval/loss < 5, c316659b4d); rerun as w658-w660.
- Figure 9 at 300m (owner popup 2026-10-29): the s1-rerun config (its W&B config equals what trec-rsqrt builds for
  SIZE=300m SEED=1, tools/cfg_diff_300m.py) trained again with a checkpoint every 572 steps (run
  muonh-qwen3-300m-della4xh100-ck572-s1, scripts/della/trec_ck300.sbatch, 2aaa9d3507); wq w667 (--partial, ~40 GPU-h).

- Q8 result (w658-w660 exit 0; tools/branch_fit.py): imprint at the branch 0.157 nats. Fraction left after 248 steps
  0.33 / 0.57 / 0.77 (lr 0.01 / 0.005 / 0.0025), half-life 124 / 339 / 784 steps; at sum lr 5: 0.19 / 0.23 / 0.26. The
  arms meet against sum(lr^p) with p = 1.32 (1.26-1.39 within 2x the best gap): mean squared log gap 7.0e-4, against
  1.5e-2 (p=1), 6.7e-2 (p=2, the fixed-norm shrink), 0.13 (steps). Q8 answered; takeaway extended.

## Open (not approved)
- When does layer 0 keep its initialization? Only the vocab session's bytes-s1 run has intermediate checkpoints
  (step 10000-54247): cos(W_t, W_0) and the step overlap per layer over training would show it.
