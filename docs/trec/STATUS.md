# TREC: training re-evaluation curves of existing MuonH checkpoints

**In one sentence (2026-10-31).** The final weights of MuonH runs fit best the batches read four fifths to nine
tenths into training: every batch is learned (0.07-0.12 nats) and then forgotten at a rate set by the sum of lr^p
with p = 1.2-1.4; how much of the final weights its update keeps does not set the fit (equal shares under a
1/sqrt(t) schedule keep 0.003 nats more of the first quarter), and how far the weights move does not set the rate.
Open: whether MuonH's fixed-norm shrink sets p (momentum-0.98 branches w727-w729, queued).

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
On 2026-10-30: p = 1.22, 1.32, 1.40 at branch points 1240, 2480, 3720; at 300m nothing of the first quarter is
left. On 2026-10-30/31 (CPU): neither the weight displacement since the branch nor the reversal of the steps before
it lines the branches up as well as sum(lr^p).

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

## Reruns (checked on W&B 2026-10-31)
- The checkpointed runs are reruns, not copies: same config, GPU nondeterminism. Train loss agrees at step 2 (4e-4)
  and drifts by up to 0.026 nats mid-run (d12, step 1000). Final train loss: d12-ck248 3.1170 vs d12 3.1151,
  rsqrtcd-ck248 3.1429 vs 3.1434, ck572-s1 2.86534 vs s1-rerun 2.86532; Paloma eval/loss 3.8557 vs 3.8519, 3.8852 vs
  3.8828, 3.5909 vs 3.5983. Figures 9 and 10 compare checkpoints of one run only, so the drift does not enter them.

## Number check (2026-10-31, owner: 「整理和核对所有结果」)
- Kaggle worker, detect-only: every hand-written number of the page against its data (out/verify/numbers.md, recompute
  scripts in out/verify/work/). 110 numbers: 103 match, 2 rounding (1.3 for 1.32; "a quarter" for 0.239), 3 wrong,
  2 without a source. Rechecked here: Fig 10 "6 to 25" -> 6 to 24 and "only a little better" -> at most twice as
  well; Fig 9 peak 0.78 -> 0.79; Fig 6 window 0.034 -> 0.026 (all four fixed in EN and ZH). The two without a source
  hold: 0.0013 at the 130m edge is tools/boundary.py's largest |diff|; the 6 of "12 layers instead of 6" is SIZES['130m']
  in della_muonh_qwen3_scaling.py. Worker note, not rechecked: with Figure 2's r0 = 0.004 the first-quarter c under
  1/sqrt(t) is 0.93-0.94, not 1.0 (the page's own model, without r0, gives 1.0).

## Code (branch `trec`, worktree `project/marin-trec`, private remote `archive`, public `marin-pli` branch `trec`)
- `experiments/references/trec_score.py`: chunked, resumable scoring; `scripts/della/trec{,-smoke}.sbatch`.
- `experiments/trec/predict.py` (c_i for three readings and the lr-only control, 1 - sqrt(c) t^m, m by Pearson r),
  `analyze.py`, `init_retention.py`, `init_step.py`, `toy_sphere.py`, `fetch_wandb.py`, `build_site_data.py`,
  `build_record.py`, `setups_svg.py`, `diff_spec.json`.
- Analysis tools that produce page numbers: `experiments/trec/tools/` (17 scripts, moved 2026-10-31; symlinked from
  `project/trec/tools/`, so the commands in this file still work).
- Outputs: `project/trec/out/<run>/<first index>.npz`, `out/init/`, `out/init_step/`; W&B cache
  `project/trec/cache/wandb/`; job records `project/trec/logs/*.yaml`.

## Jobs (every wq job of the page; H100-h = GPUs x wall time of every segment; checked against wq/jobs 2026-10-31)
| wq | name | what | state | H100-h | log |
|---|---|---|---|---|---|
| w439 | trec-smoke | Smoke of trec_score.py; replay check failed (mixture permutation) | done | 0.02 | trec-smoke.yaml |
| w441 | trec-smoke | Smoke rerun; cancelled before start (code change) | cancelled | 0.00 | trec-smoke2.yaml |
| w447 | trec-smoke | Smoke of trec_score.py; passed | done | 0.02 | trec-smoke2.yaml |
| w465 | trec | TREC of d12 and 300m s1-rerun, every batch (Figs 1-6) | done | 6.71 | trec.yaml |
| w512 | trec-runs-smoke | Smoke of the 24 other checkpoints | done | 0.32 | trec-runs.yaml |
| w513 | trec-runs-a | TREC of the other checkpoints, part a (Figs 6-7) | done | 12.59 | trec-runs.yaml |
| w514 | trec-runs-b | TREC of the other checkpoints, part b (Figs 6-7) | done | 17.73 | trec-runs.yaml |
| w590 | trec-rsqrt | d12 retrained with 1/sqrt(t), TREC (Fig 8) | done | 7.69 | trec-rsqrt.yaml |
| w591 | trec-rsqrtcd | d12 retrained with 1/sqrt(t) + cooldown, TREC (Fig 8) | done | 7.69 | trec-rsqrt.yaml |
| w642 | trec-ck-d12 | d12 retrained, checkpoint every 248 steps, TREC of each (Fig 9) | done | 11.92 | trec-ck.yaml |
| w643 | trec-ck-rsqrtcd | rsqrtcd retrained, checkpoint every 248 steps (Fig 9) | done | 11.90 | trec-ck.yaml |
| w651 | trec-br-a | Branch 2480 arm a; smoke gate bug, failed | failed, exit 3 | 0.09 | trec-branch.yaml |
| w652 | trec-br-b | Branch 2480 arm b; smoke gate bug, failed | failed, exit 3 | 0.08 | trec-branch.yaml |
| w653 | trec-br-c | Branch 2480 arm c; smoke gate bug, failed | failed, exit 3 | 0.08 | trec-branch.yaml |
| w658 | trec-br-a | Branch 2480, lr 0.01 x 500 (Fig 10) | done | 2.12 | trec-branch.yaml |
| w659 | trec-br-b | Branch 2480, lr 0.005 x 1000 (Fig 10) | done | 3.12 | trec-branch.yaml |
| w660 | trec-br-c | Branch 2480, lr 0.0025 x 2000 (Fig 10) | done | 5.91 | trec-branch.yaml |
| w667 | trec-ck-300m | 300m retrained, checkpoint every 572 steps (Fig 9) | done, 4 segments | 34.72 | trec-ck300.yaml |
| w671 | trec-br1240-a | Branch 1240, lr 0.01 x 500 (Q8 text) | done | 2.14 | trec-branch.yaml |
| w672 | trec-br1240-b | Branch 1240, lr 0.005 x 1000 (Q8 text) | done | 3.03 | trec-branch.yaml |
| w673 | trec-br1240-c | Branch 1240, lr 0.0025 x 2000 (Q8 text) | done | 5.75 | trec-branch.yaml |
| w674 | trec-br3720-a | Branch 3720, lr 0.01 x 500 (Q8 text) | done | 2.08 | trec-branch.yaml |
| w675 | trec-br3720-b | Branch 3720, lr 0.005 x 1000 (Q8 text) | done | 3.04 | trec-branch.yaml |
| w676 | trec-br3720-c | Branch 3720, lr 0.0025 x 2000 (Q8 text) | done | 5.76 | trec-branch.yaml |
| w727 | trec-br2480m98-a | Branch 2480 at momentum 0.98, lr 0.01 x 500 (Fig 10 panel 5) | queued | 0.00 | trec-branch-m98.yaml |
| w728 | trec-br2480m98-b | Branch 2480 at momentum 0.98, lr 0.005 x 1000 (Fig 10 panel 5) | queued | 0.00 | trec-branch-m98.yaml |
| w729 | trec-br2480m98-c | Branch 2480 at momentum 0.98, lr 0.0025 x 2000 (Fig 10 panel 5) | queued | 0.00 | trec-branch-m98.yaml |

Total 144.5 H100-h. Record table: main 143.9 (17 jobs), other 0.6 (6 smokes and failed gates); w441 never ran.
The vocab session's w438 and w457 scored the Figure 5 runs; their cost is on that page.
Every training run logged to W&B live (reself/marin-della); all 14 TREC training runs are finished there (2026-10-31).

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
  fully scored d12 / s1-rerun move it by sd 0.009 / 0.011; windows of 4 or 16 blocks by up to 0.026 at the figure's blocks (0.039 at any
  of the 8 choices; recomputed 2026-10-31 with tools/settings_look.py functions: the page's earlier 0.034 had no source). Runs that share a
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
- Figure 7 check (owner question 2026-10-29, "multiepoch你是不是画错了这个图"; tools/multipass_exact.py, outputs
  out/multipass_exact_<run>.npz; edge numbers above from tools/boundary.py): each scored sequence's stream position
  is mapped to its data index exactly as MixtureDataset does (block permutation fold_in(mix_key, block)). Sequences
  scored in two passes get identical final-model losses (11,148 / 388,322 / 259,386 pairs in 520m / 1.2b / tpp200,
  |diff| 0), so the mapping is exact. The batch-level last read the figure uses is within 0.0014 of training of the
  exact per-sequence one (mean <= 0.0001); its read count is wrong for 0.08-0.13% of sequences. The low batch-level
  correlation across passes comes from the block permutation scattering a batch's sequences, not from a wrong map.
  Only fix: the Chinese title of the right panel.

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
  CLEAN_GIVEN_EVIDENCE. Not done then: public code for trec-rsqrt (based on the pls commit 4530d3bb; the scrub covered the
  trec branch only), so the two record rows carried no Code link. Done 2026-10-30 (see Public code below).

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
  0.33 / 0.57 / 0.77 (lr 0.01 / 0.005 / 0.0025), half-life 124 / 339 / 784 steps (superseded: 121 / 339 / 784 in Results 2026-10-30, the page's numbers); at sum lr 5: 0.19 / 0.23 / 0.26. The
  arms meet against sum(lr^p) with p = 1.32 (1.26-1.39 within 2x the best gap): mean squared log gap 7.0e-4, against
  1.5e-2 (p=1), 6.7e-2 (p=2, the fixed-norm shrink), 0.13 (steps). Q8 answered; takeaway extended.

- Q8 at other branch points (owner popup 2026-10-29): the same three arms from d12-ck248 steps 1240 and 3720, scoring
  the 248 batches read before each (trec_branch.sbatch, job names trec-br<step>-a|b|c, 8970c90296; smoke gate eval
  loss < 6, since the step-1240 checkpoint sits higher). wq w671-w676.

- Public code (owner 2026-10-30, "都要"): marin-pli trec fast-forwarded 2347279 -> 4a20d7cd9c, new branch trec-rsqrt at
  efa55ee76b (tmp/trec_scan2/scrub.sh <branch>; scan: secrets 0, cluster user and group 0; reself and the pli account as
  before). Record: Figures 1-7 rows link trec, Figures 8-10 rows link trec-rsqrt. Figure 10 gets a fourth panel (sum of
  lr^1.32). The Kaggle-worker permission rule was blocked as self-modification: the owner adds it.

- Results 2026-10-30: branch points (tools/branch_fit.py, branch_curves(b)): p = 1.22 (1.14-1.30) at 1240, 1.32
  (1.26-1.39) at 2480, 1.40 (1.33-1.48) at 3720; half-lives at lr 0.01/0.005/0.0025: 146/383/814, 121/339/784,
  102/235/578 steps (halving lr: x2.1-2.8). 300m (w667, tools/imprint.py run 11444): first quarter 0.092 -> -0.0012
  (nothing left), 0.75-0.95: 0.072 -> 0.049 (69%); its last checkpoint peaks near 0.79 (cosine to zero; the plotted
  3-block mean peaks at 0.794, the unsmoothed blocks at 0.783; the page said 0.78 until 2026-10-31). Takeaway:
  power between 1.2 and 1.4. Record cost: main 143.9 H100-h (17 jobs), other 0.6.

- Displacement test (2026-10-30, Kaggle worker; tools/branch_drift.py, out/ck/drift/drift.json): the mean angle D of the
  85 fixed-norm matrices to the branch point does not line the arms up (spread 5.7e-3 / 7.4e-3 / 2.7e-2 at S = 1240 /
  2480 / 3720, against 9.6e-4 / 7.0e-4 / 1.1e-3 for the fitted p and 7.6e-3 / 1.5e-2 / 2.8e-2 for p = 1). D ~ lr^0.63
  N^0.43 (1/alpha 2.2-2.5, near a random walk); at the same D the smaller-lr arm keeps less. D's spread is 0.50-0.97 of
  p = 1's and 5.9 / 10.6 / 23.9 times the fitted p's (the page said "6 to 25" and "a little better" until 2026-10-31). Next candidate: the part of
  W_s - W_S along the update directions before S, not the total displacement.

- Projection test (2026-10-31, owner popup "CPU 测试 + 动量分叉"; Kaggle worker, tools/branch_proj.py,
  out/ck/proj/{proj.json,SUMMARY.txt}): r = -<W_s - W_S, u>/<u,u>, u = tangential part of W_S - W_{S-248}. The later
  movement undoes only R = 0.14-0.26 of u by the end of each arm while 1 - f = 0.74-0.86, and 97-99% of |W_s - W_S| is
  orthogonal to u. Best R axis (attention-only R_pool) spread 4.2e-3 / 2.2e-3 / 4.4e-3, 3.2-4.4x worse than the fitted
  steps*lr^p (R_mean 3.3-7.5x, R_pool 6-53x); R_mean grows as N^0.41-0.48 lr^0.55-0.64 (q = 1.34/1.27/1.34, does not rise with S as p does). Caveat:
  u is the whole 248-step displacement, mostly not what those batches taught; a batch-specific u (gradient of their
  loss at W_S) would be cleaner.

- Momentum test (owner popup 2026-10-31): the 2480 arms at momentum 0.98 (A 23 -> 73: fixed-norm shrink about 3x).
  If the shrink sets p, p rises toward 2. trec_branch.sbatch job names trec-br2480m98-a|b|c, MOMENTUM env
  (trec-rsqrt 464b25a824). wq w727-w729. Figure 10 fifth panel (placeholder).

## Open (not approved)
- When does layer 0 keep its initialization? Only the vocab session's bytes-s1 run has intermediate checkpoints
  (step 10000-54247): cos(W_t, W_0) and the step overlap per layer over training would show it.
