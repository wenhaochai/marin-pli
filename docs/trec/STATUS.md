# TREC: training re-evaluation curves of existing MuonH checkpoints

Owner request 2026-10-26: a new Experiments page that applies Bergsma, Dey and Hestness (2025, arXiv 2509.25380) to
our finished pretraining checkpoints. Page: `exps/trec.html` on the website branch `pretrain-practice`
(drafted in the detached worktree `tmp/trec_site_wt`).

## Scope (owner popups 2026-10-26)
- Runs: the pls depth control `muonh-qwen3-130m-della4xh100-d12` (130m width, 12 layers, WSD, lr 0.02, 4959 steps) and
  the Tokenizer Size 128K unique run `muonh-qwen3-300m-della4xh100-s1-rerun` (300m, cosine, lr 0.01, 11444 steps).
- Every training batch scored, plus 512 unread batches at stream indices 11444..18598 (shared by both).
- Page split (owner, via Tokenizer Size): data properties and repetition (8-pass vs unique, token frequency) belong to
  the vocab-overfitting page; this page has the method, TREC shapes, schedules, the MuonH prediction and placement.
- Ask the owner before launching any experiment (smoke then main was approved: "smoke 通过就直接跑").

## Facts
- Data order: one fineweb-edu-10B cache under the Marin tokenizer, linear permutation with data seed 42, batch t =
  sequences 128t..128t+127; the 130m run reads the first 4959 batches of the 300m stream (same cache, seed, batch).
- MuonH reads no weight decay: the 0.1 in the config reaches no group (comment-only fix chosen by the owner in both
  sessions). Matrices (MuonH) and lm_head (AdamH) keep a fixed Frobenius norm; embeddings and vectors use plain Adam.
- Fixed-norm forgetting (vocab session's note tmp/trec/muonh_prediction.md, checked 2026-10-26): every step
  W <- a_t (W - eta R O_t), a_t = (1 + 2 eta r + eta^2)^(-1/2); momentum makes steps correlated, r = kappa eta + r0,
  alpha ~ A eta^2 / 2 with A = 73 measured on 300m checkpoints (momentum 0.98), A = 23 from the toy at momentum 0.95
  (130m). My first reading (A = 1, eta^2/2) was wrong by ~70x. tau_eff = 2/(A eta^2 T) = 0.024 (300m).
- Init retention (CPU, init_retention.py): cos(W_T, W_0) is 0.006-0.057 in layers 1-11 of both runs, layer 0 attention
  0.07-0.34; predictions 0.63 / 0.81 (A = 1) and ~1e-5 / 1e-7 (momentum). Norms match per slice (rebuild checked).
- w438 (vocab session): unique minus 8-pass, low point at 0.783 / 0.830 / 0.856 of training (128K / 8K / bytes);
  the momentum reading with m = 3.3 fitted at 128K predicts 0.789 / 0.792 / 0.807.
- Replay order: MixtureDataset permutes inside every 2048-sequence block even for one component; replay through
  LmDataConfig.train_set (trec_score.training_batches). The component-only replay failed the W&B check (smoke w439).

## Code (branch `trec`, worktree `project/marin-trec`, private remote `archive`)
- `experiments/references/trec_score.py`: chunked, resumable scoring; `scripts/della/trec{,-smoke}.sbatch`.
- `experiments/trec/predict.py` (c_i for three readings, 1 - sqrt(c) t^m, m by Pearson r), `fetch_wandb.py`,
  `build_site_data.py`, `build_record.py`, `diff_spec.json`.
- Outputs: `project/trec/out/<run>/<first index>.npz`; W&B cache `project/trec/cache/wandb/`; job records
  `project/trec/logs/*.yaml`.

## Jobs
- w439 smoke: done, check failed (replay order, fixed). w441: cancelled before start. w447 smoke (fixed): queued.
- Main (4 GPUs, --partial, 4 h): submit after the smoke passes (owner: "smoke 通过就直接跑").

## Page (exps/trec.html, owner popups 2026-10-26)
- Fig 1 TRECs of 130m and 300m (queued); Fig 2 c(t) for momentum / independent / weight decay 0.1; Fig 3 init kept;
  Fig 4 predicted vs measured shapes (planned); Fig 5 three tokenizers (w438).

## Open
- Cover animation, setup schematics.
- Optional follow-up (vocab session's idea, not approved): why layer 0 attention keeps more init: cos(G, W_0) of the
  gradient or momentum buffer per slice (embedding anchor hypothesis).
