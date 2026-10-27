# TREC: training re-evaluation curves of existing MuonH checkpoints

**Result (2026-10-27).** Re-scored with their final weights, two MuonH runs fit best the batches they read about four
fifths of the way through training (0.81 at 300m with cosine, 0.89 at 130m with a linear decay; 0.069 and 0.088 nats
below unread text) and have forgotten the first quarter (0.001 +- 0.001 nats below unread text). Bergsma et al.'s
formula 1 - sqrt(c) t^m follows this shape only with a fitted exponent (m 3.4-4.8, Pearson r 0.96-0.99), and then a
control without any averaging (c = learning rate) fits nearly as well (r 0.963 against 0.971 for the momentum reading
at 300m, 0.988 for both at 130m); with the paper's m* for AdamW every reading misplaces the low point by 0.04-0.15 of
training. More steps over the same text (8K, bytes) move the low point later than the averaging predicts.

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
  the momentum reading with m = 3.3 fitted at 128K predicts 0.789 / 0.792 / 0.807.

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

## Open (not approved)
- When does layer 0 keep its initialization? Only the vocab session's bytes-s1 run has intermediate checkpoints
  (step 10000-54247): cos(W_t, W_0) and the step overlap per layer over training would show it.
