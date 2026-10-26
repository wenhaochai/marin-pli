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
- Fixed-norm forgetting: an orthogonal update of relative size eta shrinks old weights by 1/sqrt(1+eta^2), so
  alpha = eta^2/2 per step (2e4 steps at lr 0.01). Sum over the run: 0.21 (300m), 0.46 (130m).

## Code (branch `trec`, worktree `project/marin-trec`, private remote `archive`)
- `experiments/references/trec_score.py`: chunked, resumable scoring; `scripts/della/trec{,-smoke}.sbatch`.
- `experiments/trec/predict.py` (c_i for three readings, 1 - sqrt(c) t^m, m by Pearson r), `fetch_wandb.py`,
  `build_site_data.py`, `build_record.py`, `diff_spec.json`.
- Outputs: `project/trec/out/<run>/<first index>.npz`; W&B cache `project/trec/cache/wandb/`; job records
  `project/trec/logs/*.yaml`.

## Jobs
- w439 smoke (1 GPU, prio 10): queued 2026-10-26.
- Main (4 GPUs): submit after the smoke passes (last batches match W&B train/loss).

## Open
- Cover animation, setup schematics.
- Possible follow-up to ask the owner: cos(W_T, W_0) per matrix against exp(-sum eta^2/2) (CPU, init regenerated
  from the seed), a direct test of the fixed-norm forgetting rate without the training-fraction term.
