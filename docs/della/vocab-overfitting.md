# How Vocabulary Size Shapes Overfitting (page blogs/vocab-overfitting.html)

**Status (2026-10-08):** at 300m and 8 passes, each halving of the vocabulary lowers the cost of repetition (128K +0.083 to
8K +0.026 bits per byte, one seed). The rare tokens' own input rows carry more than half of that gap: with every input
row the mean of its 8K pieces' rows (-cvin8k) the cost is +0.047, the full-data model is worse (1.4127 vs 1.3867) and
the 8-pass model is better (1.4601 vs 1.4695, and 8K 1.4814). A window of the 8K truncation's bytes (+0.083) and a third
more steps (+0.081) do not move the cost. Q5's first head (mean of pieces) capped a token's logit at its largest piece's
(full data 1.767): redo with the sum queued (why-cvoutsum). 520m 128K costs +0.095. At 130m the cost also falls with every halving (128K +0.110 to 8K +0.018). The 128K second seed costs +0.078 (seed 0 +0.083). 8K seed 1, 520m 8K, bytes queued. Split from the sampled-softmax
/ OT project (docs/della/over-vocab.md), whose Q5 this was.

## Questions and runs

| Q | Question | Runs (300m unless noted) | Plan / job |
|---|---|---|---|
| Q1 | Does a smaller vocabulary make repeated data hurt less? | 128K, 64K, 32K, 16K, 8K x (full, 2.4, 8 passes), done | sampled-softmax project |
| Q2 | Other sizes? | 130m: 64K-8K pairs + 128K 8 passes; 520m: 128K 8 passes, 8K pair | logs/plans/vo130.txt, vo520-v8k-seg{1,2}.txt, 15162227 |
| Q3 | Is the trend real? | byte check (done: 8K/128K bytes 1.0004); seed 1 for 128K (-s1-rerun, -rep8-s1) and 8K | why-seed128k, why-seed8k |
| Q4 | Input rows of rare tokens? | grouped analysis by input-token frequency; 128K with input rows = mean of 8K pieces (-cvin8k) | why-cvin |
| Q5 | Output rows of rare tokens? | grouped by target frequency; -cvout8k (mean: capped, dropped), -cvout8k-sum (redo) | why-cvout, why-cvoutsum |
| Q6 | Fewer bytes per window? | grouped by context bytes; 128K at SEQ_LEN 3072, batch 172 (-sl3072-b172) | why-sl3072 |
| Q7 | More optimizer steps? | 128K at batch 96 (1.33x steps, same text; warmup stays 1000 steps, as in the 8K runs) | why-b96 |

| Bytes | Does the trend hold at the byte extreme? (owner, 2026-10-08) | K=256 (the 256 byte tokens, no merges; 4.74x the 128K tokens, so 4.74x the steps): 300m and 130m, window 4096, full + 8 passes; window 16384 at batch 32 (same tokens per step) at 130m and 300m; Q4-Q6 grouped analyses on the 300m pair. Added to fig-vocab-final, fig-vocab, fig-sizes, fig-freq-in, fig-freq-out, fig-window (popup-approved captions) | bytes-300m-seg{1,2}, bytes-130m(-seg2), bytes-300m-sl16384-seg{1..4}; ~600-720 H100h |

Not done (owner): reasons 4 (Adam on rare rows), 6 (shorter token strings), 7 (weaker model), 8 (composing words), the
parameter-count control (low-rank tables).

## Spec checklist (audit 2026-10-07): decision -> implementation -> check

1. New page, title "How Vocabulary Size Shapes Overfitting", slug vocab-overfitting -> website blogs/vocab-overfitting.html
   (tmp_plots/vocab/gen_vocab_page.py) -> vo_shot.py (renders, 11 charts, no overflow at 1280/420/375/360, EN and ZH).
2. Move the softmax page's Q5 here; delete it there -> website 7980b15; ssov builders without fig-vocab -> ssov record
   rebuilt byte-identical; page has 8 charts.
3. No parameter-count control -> no low-rank run in any plan -> check_plans.py lists every training task.
4. Sizes: 130m all five K, 520m 128K and 8K -> logs/plans/vo130.txt, vo520-v8k-seg*.txt, 15162227 -> check_plans.py.
5. Q3-Q7, one sub-question each, designs approved in popups -> page sections h-q3..h-q7 with placeholders; plans
   why-*.txt -> check_plans.py (dry-run run id of every task equals the declared run).
6. cv: full row = mean of the 8K pieces; small tables start as the baseline's own rows; f32 compose; MuonH labels kept ->
   experiments/references/composed_vocab_qwen3.py:77 (compose), :87 (f32), :92/:120 (own rows) ->
   scripts/della/composed_vocab_cpu_test.py (composition, losses, logits, gradients, init, bf16, MuonH: ALL PASS).
7. SEQ_LEN / BATCH keep the tokens: della_muonh_qwen3_scaling.py:99, :280 -> dry runs (11355 and 15259 steps);
   review: tokens per 8-pass part 749.7M / 749.8M / 749.9M.
8. Packing rules (controller, 2026-10-07): steps <= 24 h in one job; each independent job its own smoke copy; two 4-GPU
   runs side by side; every task through tools/packed_task.sh (ledger) -> scripts/della/packed_job.sbatch (+ pgroup.sh
   so a limit stops the whole process group; per-task logs; wait -n; one retry for a failed par task) -> dummy-task
   tests on della-vis1 (env isolation, '+' values, tabs/comments, timeout kills python, smoke gate, retry, missing
   script rc 127, unknown mode exit 2).
9. Record: packed jobs split by the ledger, two lines sum to sacct -> tmp_plots/ss130/ssov_cost.py:78 (read_ledger),
   :94 (ledger_main_hours), :196 (sum assert) -> tmp_plots/ss130/test_ledger.py (13 hostile/spec cases) + both records
   rebuilt: unchanged inputs give identical outputs.
10. Per-token analysis exactly as the evals -> experiments/references/vocab_token_losses.py -> 16 subset means of v8k and
    v8k-rep8 equal W&B eval/paloma loss within 1e-4.
11. Cover animation explains only the token cutting (owner) -> tmp_plots/vocab/anim_vocab.html (real splits checked with
    the tokenizers); card = its last frame (card_svg.py).
12. New page card first in the lists (owner) -> index.html, blogs/experiments.html (85e5945).
13. Standalone artifact of the page only, every file republished -> website build_vocab_artifact.sh.
14. Never push the website; marin only to the private archive remote (checked PRIVATE).
15. Plan durations: timeout(1) takes one number and one unit (390m, 6.5h); "6h30m" exits 125 at once (found by the
    review; would have killed why-b96 and both vo520 segments) -> plans fixed; check_plans.py regex.
16. Knobs never inherited from the submitting shell (--export=ALL) -> packed_job.sbatch unsets them; every training task
    trains only if its dry-run run id equals the plan's RUN (run_checked.sh, exit 3, no retry) -> dummy test with
    VOCAB_K/DATA_EPOCHS exported (scrubbed; a wrong RUN refused).
17. A failed run must not idle its GPUs: lanes (lane=<name>) run their tasks in turn, so a failed run frees its GPUs for
    the lane's next task; vo130 runs as two lanes -> dummy test (failed and timed-out tasks, lane goes on).
18. Cost lines: timed-out main segments (124/137) count as main, as plain TIMEOUT jobs; a packed job counts by its ledger
    even when no run id reached its logs -> test_ledger.py (14 cases); record hours per run from W&B _runtime or the
    ledger, not the whole packed job.
19. Grouped analysis: macro over the 16 subsets (as the bars); seed-matched 128K/8K pairs; seeded tie-break -> invariance:
    a checkpoint against itself gives 0 in every group; the 8K pair's macro total 0.0262 equals the page's 8K cost.

21. Q5 redo (owner, 2026-10-08): output rows = SUM of the pieces' rows (CV_OUT_REDUCE=sum, tag -sum), input rows stay
    means -> composed_vocab_qwen3.py compose(reduce) -> composed_vocab_cpu_test.py (out/both with the sum: composition,
    losses, logits, pulled-back gradients, MuonH labels, init; a rare token's logit exceeds every piece's at 20-24% of
    (position, token) with the sum and never with the mean: ALL PASS); the per-position eval reads the reduce from the run
    id (a sum checkpoint loads as a mean one without error, review MAJOR 2); review MAJOR 1: the sum inits repeated-piece
    tokens (up to 64 pieces) at up to 45x a piece's logit (logsumexp ~50 nats at init) -> gate by hand: smoke step-60 eval
    loss near cvin's 7.49 (the mean smoke was 8.49), W&B steps 200-500 near the baseline, else cancel.

20. Bytes (owner, 2026-10-08): K=256 tokenizer = small_vocab_tokenizer build 256 (0 merges, specials 256..511) -> review:
    all 128,000 expansions equal the token's bytes; HF and Levanter load 512 ids, BOS 256, EOS 257. Data: conversion job
    15211747 (decode-equality check on sampled documents; the canonical check is implied for bytes). Run tag -bytes;
    eval every round(ratio) x 1000 steps (5000 for bytes; 1000 for every existing K, so queued runs are unchanged:
    check_plans.py). Every chained segment carries its own smoke copy (review: afterany would train after a failed
    segment-1 smoke). Levanter's bpb counts each non-ASCII byte token as 3 bytes: bpb_bytes_check.py with KS=256 before
    any figure (per-subset factor; residual bias up to +0.6% on code, ~+0.07% macro, mostly cancelling in rep8 - full).

## Open risks

- Bytes keep the 1000-step warmup of every run here: 1/54 of training at 300m against 1/11 at 128K. Consistent with the
  truncated tokenizers; Q7 (more steps) speaks to the step count, not to the warmup share.
- Bytes frequency deciles (Q4/Q5 analysis) over 256 types: frequent bytes straddle decile edges and are split by the
  seeded tie-break; read the byte panels as coarse.
- Analysis scripts still need K=256 entries (vocab_cost_groups.py TOK/TRAIN/PAIRS) and SEQ_LEN=16384 in the token-loss
  specs (vocab_token_losses.py defaults to 4096); figure/record builders need a bytes point (RATIO/VOC/KTICKS, rid()).

- why-* jobs hold one run per lane: a run that fails twice (deterministic crash) leaves its 4 GPUs idle; Della cancels the
  job after 90 idle minutes, taking the healthy partner with it. The retry covers transient crashes only. Decision
  (2026-10-07, delegated by the owner): keep the 8-GPU jobs. Each job first runs its own smoke copy of the same setting,
  which catches a deterministic crash before either full run starts; an 8-GPU job starts sooner than two 4-GPU jobs, and
  resubmitting would reset the queue age.
- tools/packed_task.sh writes a ledger line only when timeout returns: a job ended by Slurm (wall, idle cancel, scancel)
  loses the running tasks' lines, and their hours count as other compute (tool owned by the controller).
- The vo130 job's last step (128K, 8 passes) runs on 4 GPUs for about an hour while the other 4 idle (under 90 min).
