# How Vocabulary Size Shapes Overfitting (page blogs/vocab-overfitting.html)

**Status (2026-10-08):** at 300m and 8 passes, each halving of the vocabulary lowers the cost of repetition (128K +0.083 to
8K +0.026 bits per byte, one seed). The rare tokens' own input rows carry more than half of that gap: with every input
row the mean of its 8K pieces' rows (-cvin8k) the cost is +0.047, the full-data model is worse (1.4127 vs 1.3867) and
the 8-pass model is better (1.4601 vs 1.4695, and 8K 1.4814). A window of the 8K truncation's bytes (+0.083) and a third
more steps (+0.081) do not move the cost. Q5's first head (mean of pieces) capped a token's logit at its largest piece's
(full data 1.767); the redo with the sum (why-cvoutsum 15228167) failed its gate on 2026-10-09 and was cancelled: smoke eval 8.43 (mean 8.49, cvin 7.49), train loss 4.36 at step 890 and 4.10 at 1580 (baseline 3.73, 3.53; mean 4.48, 4.25). Q5 (owner, 2026-10-09): answered by the target-frequency grouping (correlational); fig-cvout dropped, the composed-output attempt told in prose. 520m 8K (2026-10-10): full 1.3607, rep8 1.3997, cost +0.0390 vs 128K +0.0951 (130m +0.0177 vs +0.1104, 300m +0.0262 vs +0.0828): the saving holds at every size; it is largest at 130m (0.093) and about the same at 300m (0.057) and 520m (0.056); audit 2026-10-11 corrected the page's "shrinks as the model grows". Q2 answered on the page ("So far, yes"); 520m bytes running since 2026-10-11 (15283034). 130m bytes: cost +0.0066, the lowest of all K at 130m; window 16384 (batch 32) +0.0143. 300m bytes (2026-10-09, 54247 steps, 4.74x the 128K steps): cost +0.0041, full 1.3699 below every other K (128K 1.3867), rep8 1.3740; its gap stays within +-0.015 of zero at every evaluation. Per-token losses of the byte pair: vo-toklosses 15293174. Grouped analyses (seed-matched seed-1 pairs, macro totals equal the W&B costs): the 128K cost is far highest in the rarest input decile (0.219 vs 0.029-0.090) and rarest target decile (0.210); 8K is flat and low; cvin cuts the rarest input decile to 0.093; the cost does not grow with context bytes (Q6 negative again). 520m 128K costs +0.095. At 130m the cost also falls with every halving (128K +0.110 to 8K +0.018). Second seeds (Q3 answered): 128K +0.078 (seed 0 +0.083), 8K +0.020 (seed 0 +0.026). Split from the sampled-softmax
/ OT project (docs/della/over-vocab.md), whose Q5 this was.

## Questions and runs

| Q | Question | Runs (300m unless noted) | Plan / job |
|---|---|---|---|
| Q1 | Does a smaller vocabulary make repeated data hurt less? | 128K, 64K, 32K, 16K, 8K x (full, 2.4, 8 passes), done | sampled-softmax project |
| Q2 | Other sizes? | 130m: 64K-8K pairs + 128K 8 passes; 520m: 128K 8 passes, 8K pair (done 2026-10-10, cost +0.0390 vs +0.0951) | logs/plans/vo130.txt, vo520-v8k-seg{1,2}.txt, 15162227 |
| Q3 | Is the trend real? | byte check (done: 8K/128K bytes 1.0004); seed 1 for 128K (-s1-rerun, -rep8-s1) and 8K | why-seed128k, why-seed8k |
| Q4 | Input rows of rare tokens? | grouped analysis by input-token frequency; 128K with input rows = mean of 8K pieces (-cvin8k) | why-cvin |
| Q5 | Output rows of rare tokens? | grouped by target frequency (the answer); -cvout8k (mean) and -cvout8k-sum (gate failed, cancelled): composed heads cannot fit the data | why-cvout, why-cvoutsum |
| Q6 | Fewer bytes per window? | grouped by context bytes; 128K at SEQ_LEN 3072, batch 172 (-sl3072-b172) | why-sl3072 |
| Q7 | More optimizer steps? | 128K at batch 96 (1.33x steps, same text; warmup stays 1000 steps, as in the 8K runs) | why-b96 |
| Q8 | Overtrained regime (owner, 2026-10-09)? | 130m at TPP=200 (TPP knob: 10x the recipe's steps on the same fineweb-edu-10B, no new data, so all data = 2.6 passes), every vocabulary (128K, 64K-8K, bytes), full + 8 passes; TPP=20 anchors at 2.6 passes, so the cost is 8 minus 2.6 passes at both TPPs; fig-tpp (popup-approved) | tpp200-{128k,64k,32k,16k,8k}, tpp200-bytes-seg{1,2}, tpp20-anchors (15299908-15); ~640 H100h RESULT (2026-10-13): TPP=200 costs (8 minus 2.6 passes) 128K +0.024, 64K +0.025, 32K +0.014, 16K -0.002, 8K -0.020, bytes -0.029 (TPP=20: +0.100 ... +0.014), monotone in K; second seeds (init only, 15421594/5): 128K +0.039, 8K +0.044 (full 1.5892 vs 1.5884, rep8 1.6332 vs 1.5688). The 8K seed spread 0.064 exceeds the whole range across K (0.053): cannot rank; 2-seed means 128K 0.032, 8K 0.012. Answered "could not tell". |
| Q9 | Only the parameter count? (owner, 2026-10-11) | 128K with rank-48 input and output tables (VARIANT=lr, -lr48; ~115M parameters, as 8K's 115M; 128K has 300M), full + 8 passes; fig-params (popup-approved). Existing cross-size runs already argue against a pure count: 520m 8K (357M parameters) +0.039 against 300m 128K (300M) +0.082. RESULT (2026-10-12): full 1.5350, 8 passes 1.5711, cost +0.036 (128K +0.082, 8K +0.026); gate failed at step 1000 (loss 4.15 vs 3.72) and the full-data fit ends worse than 8K (1.535 vs 1.451), so a low cost cannot separate fewer parameters from a weaker fit: inconclusive | why-lr48 (15378166) COMPLETED |
| Q5b | Output rows, causal (owner, 2026-10-11; redesigned 2026-10-12) | (1) 128K with the output rows of tokens 8000-127999 fixed at their initial values (VARIANT=fo, -fo8k): gate failed 2026-10-12, train loss 4.65 at step 1000 and 4.24 at 2450 vs baseline 3.72, 3.41; cancelled. (2) The baseline model whose rare output rows learn during pass 1 and then stop: MuonH (freeze_rows_muonh.py) zeroes their gradient and update from update 1430 = round(11444/8), in the full-data run too, so only the repeated passes are withheld from them (VARIANT=fop, -fop8kp2); bit-identical to the baseline before the freeze (CPU test); fig-frozout caption popup-approved 2026-10-12 | why-fo8k (15378165, cancelled), why-fop8k (15429947) RESULT why-fop8k (2026-10-13): full 1.4806, rep8 1.5665, cost 0.086 (128K 1.3828/1.4651/0.082) but CONFOUNDED: AdamH sizes the head step by the whole table, so once the frozen rows' Adam ratios fade the 8256 kept rows take the whole step, ~4x their baseline rate for passes 2-8 (review). Rerun why-fop8kb 15521703 with FOP_BLOCK_NORM=1 (kept-block step, commit 714921febd); fig-frozout draws the rerun. |
| Q4b | Second seed of the composed input (owner, 2026-10-11) | -cvin8k-s1 / -cvin8k-rep8-s1, seed-matched to the grouped figures' 128K and 8K seed-1 pairs; per-token losses afterwards. RESULT (2026-10-13): full 1.4098, 8 passes 1.4674, cost +0.058 (seed 0: 1.4086 / 1.4557, +0.047; 128K +0.082 / +0.077, 8K +0.026 / +0.020); removes 44% of the 128K-8K gap (seed 0 63%), so about half holds; at 8 passes seed 1 is 0.002 above 128K (seed 0 was 0.009 below), so the page now says about level, not better. Its seed spread (0.010) is about twice the tokenizers' (0.005, 0.006) | why-cvin-s1 (15378167) COMPLETED; vo-toklosses 15470524 |
| Q4c | Do the rare input rows memorise during the repeated passes? (owner, 2026-10-13) | 300m 128K, the input rows of ids 8000-127999 learn during pass 1 and then stop (VARIANT=fip, freeze_rows_muonh.py freeze_table=embed, from update 1430, in the full-data run too); the model is the baseline before the freeze, unlike the composed-input runs that remove those rows from step 0; fig-cvin placeholder popup-approved | why-fip8k 15523345 (commit dd88ea3a05) |
| Q10 | Can a large vocabulary repeat well? (owner, 2026-10-11) | 300m 128K: reshuffle every pass (PASS_DATA=rs, also 8K), new BPE-dropout 0.1 segmentation of passes 2-8 (bpd0.1, ignore_merges off), no loss on targets outside 8K from pass 5 (VARIANT=rm, loss sum(w m l)/sum(w)), composed input + 0.1 residual (VARIANT=cvr, full + rep8); pass caches from vocab_materialize_passes.py; fig-fixes (popup-approved) | why-cvr 15390567, vo-passes 15390501, then rm + 128K rs and bpd + 8K rs. PARTIAL (2026-10-13): cvr full 1.4013, rep8 1.4547, cost 0.053 (128K 1.3828 / 1.4651 / 0.082; 8K 1.451 / 1.4772 / 0.026; cvin s0/s1 full 1.4086/1.4098): the residual takes back ~30% of cvin's full-data loss and keeps the 8-pass gain; still twice 8K's cost; one seed. rm + rs8k (15394392) and rs + bpd (15399204) training; bpd lane slowed by a throttled GPU (93 C, 345 MHz on della-k17g3), resumes in 15482785 RESULTS (2026-10-13): 128K rs 8 passes 1.4574 (cost 0.075), 8K rs 1.473 (0.022), 128K bpd0.1 1.4525 (0.070; bpd ran 3454 steps on a throttled GPU then resumed in 15482785); plain rm8kp5 collapsed (1.53 at 5000 -> 2.17 at 6000 -> 2.84 at 7000; final 2.845): masking rare targets leaves the softmax pushing their logits down; not drawn (owner). Corrections (owner): why-rm-fix 15520302, rm8kp5f (mask + rare output rows frozen) and rm8kp5d (also out of the training softmax), both with the kept-block AdamH step; commit 13f8257057. |
| Q10b | Rescue the 128K tokenizer, building on the composed input + residual (owner, 2026-10-13: "keep going until it is rescued") | A: cvr seed 1; B: residual scale 0.03 and 0.3 (0.1 so far); C: VARIANT=cvd, rare input rows replaced by their 8K pieces' mean with p 0.5 in training only (dropout on the rare rows); D: cvr with reshuffled and with re-segmented repeated passes (8 passes only). New figure fig-fixes-in, placeholders popup-approved. Goal: a 128K setup whose 8-pass cost approaches 8K's (0.026) without losing the full-data fit, or whose 8-pass value beats both tokenizers beyond seed noise | why-cvr-s1 15523784, why-cvr-r003 15523785, why-cvr-r03 15523786, why-cvr-data 15523787, why-cvd 15523933 (commit 5f8c5b0dfe) |

| Bytes | Does the trend hold at the byte extreme? (owner, 2026-10-08) | K=256 (the 256 byte tokens, no merges; 4.74x the 128K tokens, so 4.74x the steps): 300m and 130m, window 4096, full + 8 passes; 520m too (owner, 2026-10-09; 47014 steps, ~82 h per run, 4 segments bytes-520m-seg1..4, jobs 15283034-37, ~620 H100h); window 16384 at batch 32 (same tokens per step) at 130m and 300m; Q4-Q6 grouped analyses on the 300m pair. Added to fig-vocab-final, fig-vocab, fig-sizes, fig-freq-in, fig-freq-out, fig-window (popup-approved captions) | bytes-300m-seg{1,2}, bytes-130m(-seg2), bytes-300m-sl16384-seg{1..4}; ~600-720 H100h |

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
    v8k-rep8 equal W&B eval/paloma loss within 1e-4 (audit 2026-10-11: the seed-1 8K pair within 3.1e-4 nats per subset,
    bf16 eval nondeterminism; the 128K pairs within 1e-6).
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
    Audit 2026-10-11: the page's bpb is Levanter's token-weighted mean of per-batch ratios times that factor, not
    sum(loss)/sum(bytes): absolute values run ~0.004 high (128K seed 1: 1.3923 against an exact 1.3884 from the per-token
    losses), costs agree within 5e-4 (0.0776 against 0.0771); the page compares costs and same-metric values only.

## Open risks

- Q8 keeps every TPP=20 hyperparameter (the speedrun recipe was tuned at TPP=20; 130m: linear schedule, decay 0.8, no warmup). Each pass of the 2.6-pass data repeats the same permutation (levanter RESTART, no reshuffle), as in every rep run. The smoke copies do not exercise the TPP step math (SMOKE_STEPS overrides it); the dry-run configs were checked instead.

- Bytes keep the 1000-step warmup of every run here: 1/54 of training at 300m against 1/11 at 128K. Consistent with the
  truncated tokenizers; Q7 (more steps) speaks to the step count, not to the warmup share.
- Bytes frequency deciles (Q4/Q5 analysis) over 256 types: frequent bytes straddle decile edges and are split by the
  seeded tie-break; read the byte panels as coarse.
- Bytes in the analyses: vocab_cost_groups.py has the K=256 pair "bytes" (tables checked: 512 ids, bytes 0-255 one byte,
  specials 0, EOS 257; train_counts_256.npy precomputed, 204 byte values seen in 201M tokens); the figure and record
  builders already list the bytes points. Still to do at submit time: SEQ_LEN=16384 in the token-loss specs of the
  sl16384 runs (vocab_token_losses.py defaults to 4096).
- vo-toklosses moved from pli-low (priority 122, never started) to pli-short (15255396), with the 8K seed-1 pair added.

- why-* jobs hold one run per lane: a run that fails twice (deterministic crash) leaves its 4 GPUs idle; Della cancels the
  job after 90 idle minutes, taking the healthy partner with it. The retry covers transient crashes only. Decision
  (2026-10-07, delegated by the owner): keep the 8-GPU jobs. Each job first runs its own smoke copy of the same setting,
  which catches a deterministic crash before either full run starts; an 8-GPU job starts sooner than two 4-GPU jobs, and
  resubmitting would reset the queue age.
- tools/packed_task.sh writes a ledger line only when timeout returns: a job ended by Slurm (wall, idle cancel, scancel)
  loses the running tasks' lines, and their hours count as other compute (tool owned by the controller).
- The vo130 job's last step (128K, 8 passes) runs on 4 GPUs for about an hour while the other 4 idle (under 90 min).

## Audit 2026-10-11 (12 dimensions, each finding checked by 3 independent refuters; tmp/audit/)

Fixed: seg3 of bytes-300m-sl16384 (15213238) re-planned to lanes (rep8 ran ~2.3 h ahead of full: idle-cancel risk) with
the v64k/v32k TPP=20 anchors moved in from 15299913; watcher log name and false STALL; page claims restated (saving vs size
is largest at 130m and level from 300m to 520m; 8-pass ordering; Q3/Q4/Q5/Q6 at seed-noise altitude) and disclosures
added (untied tables, per-tokenizer repeated parts, warmup share, eval windows, the -s1 rerun); grouped macro skips
(group, subset) cells under 1000 positions; negative group costs drawn; record: fig 11 tags, first-segment code commit,
running hours from W&B _runtime, toklosses jobs in other compute, the 14940537 pair split, no silent NO ROW.
Grouped results of the 300m byte pair (15293174): macro total +0.0041 (equals the W&B cost); input deciles 0.007-0.014
except decile 2 -0.033, target decile 1 +0.016, no rare-byte peak; context cost 0.015 falling to ~0 by 4096 bytes.
Q8 first results (2026-10-11, exact bpb; cost = 8 passes minus 2.6 passes): TPP=200 128K +0.0240 (1.4645 -> 1.4885), 64K +0.0252 (1.4821 -> 1.5073); 32K +0.0135 (1.5179 -> 1.5314); 16K -0.0018, 8K -0.0196; TPP=20 anchors done (2026-10-12): 128K +0.1003, 64K +0.0715, 32K +0.0637, 16K +0.0319, 8K +0.0204, bytes +0.0139. Caution (2026-10-12): at TPP=200 a few subsets swing between runs (8K: m2d2_wikipedia -0.158, programming languages -0.143, redpajama -0.061 while c4_en, falcon, mc4 are within 0.001; the wiki and code losses rise and fall mid-training in both runs), so the small-vocabulary TPP=200 costs (32K-8K) are within single-run noise; 128K and 64K are positive in 15/16 subsets. Before step 8376 both runs of a pair see the same data, yet differ by 0.008-0.016 at step 5000 (high-LR eval noise). Owner (2026-10-12): second seeds (init only) of the TPP=200 128K and 8K pairs, tpp200-128k-s1 15421594 and tpp200-8k-s1 15421595; fig-tpp-seeds (popup-approved). TPP=20 64K +0.0715, 32K +0.0637 (128K anchor pending): with a 10x larger repeated part the extra cost of 8 passes is about a third, and 128K and 64K are level at TPP=200. Owner decisions (popup, 2026-10-11): bytes-520m seg4 cancelled and resubmitted at 23:55 with 1380m limits (15375325);
pending jobs exclude della-k14g3 (GPU 0-3 half 7-13% slower); figs 5/7/8 retitled "Where the cost sits: ..."; TPP != 20
keeps no step-interval checkpoints (1965f50f0a; hashes of every TPP=20 config unchanged). Done: the page's bpb is now
exact, loss x T / (ln2 x B) per subset (tmp_plots/vocab/bpb_exact_tables.json); the old logged-bpb x whole-subset factor
ran ~0.004 high for BPE vocabularies and ~0.0004 for bytes; costs moved < 7e-4 and every claim still holds. Numbers in
this doc written before 2026-10-11 use the old metric (e.g. 300m 128K full 1.3867 is 1.3828 exact). To do: packed_job hardening (Slurm stop drops ledger lines, denylist env isolation, plan parser edge cases,
job_tmpdir EXIT trap under set -e) for new submissions only.
