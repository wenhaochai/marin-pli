# Project record: "How Vocabulary Size Shapes Overfitting" (inventory for the 2026-10-21 per-tokenizer goal)

Scope: every run in the vocab-overfitting project (300m MuonH Qwen3, Marin speedrun recipe; also 130m/520m), read from
the living doc, the job YAML, the public page and its figure data, the model/optimizer code, the row-norm outputs and
the earlier sampled-softmax / Over-Tokenized memory note. Read-only survey on 2026-10-21; no jobs run.

Conventions used below (all from the sources):
- "full" = unique tokens (each token seen once; 0.6 of fineweb-edu-10B at 300m); "rep8" = a random 1/8 subset repeated
  8 passes, same compute; "cost" = rep8 minus full, in Paloma macro bits per byte (bpb). Each tokenizer draws its own
  random repeated part with the same bytes. Smaller vocabularies take proportionally more steps (128K 11444 steps at
  300m; bytes 54247 = 4.74x). — [doc][doc], [page][page]
- "exact bpb" = loss x T / (ln2 x B) per subset (tmp_plots/vocab/bpb_exact_tables.json), adopted 2026-10-11. Numbers
  written earlier use Levanter's logged bpb x factor, which runs ~0.004 high for BPE vocabularies (~0.0004 for bytes);
  costs moved < 7e-4. Example: 300m 128K full 1.3867 (old) = 1.3828 (exact). Below, "(old)" marks old-metric values;
  everything else is exact. — [doc audit 2026-10-11][doc]
- Seeds: seed 0 is the default; "seed 1/2" changes the init only (data order and repeated part unchanged) unless noted
  (cvd seeds also change the replacement draws). — [page fig-seeds][figs], [yaml][yaml]
- Goal criteria used before 2026-10-21 (Q10b): criterion 1 = a 128K setup whose 8-pass cost approaches 8K's (0.026)
  without losing the full-data fit; criterion 2 = an 8-pass value that beats both 128K and 8K beyond seed noise
  (threshold 0.012). The 2026-10-21 goal (Q11) is stricter: every tokenizer's 8-pass bpb equal to its own full-data bpb.
  — [doc Q10b, Q11][doc]

## Q1. Baselines: unique-token bpb, 8-pass bpb and cost for each tokenizer and size

### Takeaway
Cost of 8 passes falls monotonically with every halving of the vocabulary at 130m and 300m (300m: 128K 0.0823 -> 64K
0.0747 -> 32K 0.0589 -> 16K 0.0436 -> 8K 0.0262 -> bytes 0.0041, seed 0, exact), and at 520m 128K 0.0946 > 8K 0.0390 >
bytes 0.0274. At 2.4 passes every tokenizer's cost is <= 0.016 with no vocabulary trend. Only 128K and 8K have second
seeds at 300m (bytes seed 1 is running).

### Cited Findings
- 300m, exact bpb, seed 0 (full / 2.4 passes / 8 passes): 128K 1.3828 / 1.3905 / 1.4651; 64K 1.3912 / 1.3965 / 1.4659;
  32K 1.4020 / 1.4184 / 1.4609; 16K 1.4229 / 1.4358 / 1.4665; 8K 1.4510 / 1.4498 / 1.4772; bytes 1.3695 / 1.3671 /
  1.3736. Derived 8-pass costs: 0.0823, 0.0747, 0.0589, 0.0436, 0.0262, 0.0041. Derived 2.4-pass costs: 0.0077, 0.0053,
  0.0164, 0.0129, -0.0012, -0.0024. — [figs fig-vocab-final / fig-sizes][figs]
- 300m second seeds (init only): 128K s1 full 1.3884 (the "-s1-rerun"), rep8 1.4655, cost 0.0771; 8K s1 full 1.4567,
  rep8 1.4766, cost 0.0199. Two-seed means: 128K full 1.3856, rep8 1.4653, cost 0.0797; 8K full 1.4539, rep8 1.4769,
  cost 0.0230. Older W&B-metric statements: 128K +0.078 (s1) / +0.083 (s0), 8K +0.020 / +0.026 (old). — [figs fig-seeds,
  fig-progress][figs], [doc status][doc], [yaml why-seed128k 15168739, why-seed8k 15168740][yaml]
- An earlier 128K seed-1 full run ("-s1") ended at 1.3822 against the rerun's 1.3884, i.e. a 0.006 spread between two
  runs of the same nominal setting; the page uses the rerun. — [doc Q10f][doc], [page fig-seeds caption][page]
- 130m (exact; full / rep8 / cost): 128K 1.5117 / 1.6214 / 0.1097; 64K 1.5252 / 1.6028 / 0.0776; 32K 1.5535 / 1.6084 /
  0.0549; 16K 1.5926 / 1.6231 / 0.0305; 8K 1.6161 / 1.6335 / 0.0174; bytes 1.5707 / 1.5773 / 0.0066 (job vo130 15168741
  recorded old-metric costs +0.1104, +0.0781, +0.0552, +0.0308, +0.0177). One seed. — [figs fig-sizes][figs], [yaml][yaml]
- 520m (exact; full / rep8 / cost): 128K 1.3001 / 1.3947 / 0.0946 (old 1.3038 / 1.3989 / +0.0951); 8K 1.3567 / 1.3957 /
  0.0390 (old 1.3607 / 1.3997); bytes 1.2581 / 1.2855 / 0.0274. 64K-16K not run at 520m. One seed. — [figs fig-sizes][figs],
  [yaml vocab-bytes, vo520][yaml], [doc Q10c E][doc]
- The 128K-minus-8K cost saving: 130m 0.093, 300m 0.057, 520m 0.056 (page corrected "shrinks as the model grows" in the
  2026-10-11 audit). At 520m 128K and 8K end level at 8 passes (1.3947 vs 1.3957). — [doc status][doc]
- Bytes: 300m window 16384 / batch 32: full 1.3308, rep8 1.3361, cost 0.0053; 130m window 16384: 1.5408 / 1.5552, cost
  0.0143 (yaml: +0.0143). The bytes' 300m gap stays within +-0.015 of zero at every evaluation. — [figs fig-window][figs],
  [doc status][doc], [yaml 15213238, 15213497][yaml]
- The 128K 8-pass gap grows from about pass 5 on; at 2.4 passes the gap stays small for every tokenizer.
  — [page fig-vocab caption][page]
- Overtrained regime Q8 (130m, TPP=200 = 10x steps on the same 10B tokens, so "full" = 2.6 passes; cost = 8 minus 2.6
  passes; one seed): 128K 1.4645 -> 1.4885 (+0.0240), 64K 1.4821 -> 1.5073 (+0.0252), 32K 1.5179 -> 1.5314 (+0.0135),
  16K 1.5307 -> 1.5289 (-0.0018), 8K 1.5884 -> 1.5688 (-0.0196), bytes 1.5687 -> 1.5398 (-0.029). TPP=20 anchors (8 minus
  2.6 passes): 128K +0.1003, 64K +0.0715, 32K +0.0637, 16K +0.0319, 8K +0.0204, bytes +0.0139. Second seeds: 128K
  1.4674 -> 1.5066 (+0.039), 8K 1.5892 -> 1.6332 (+0.044). The 8K seed spread (0.064) exceeds the whole range across K
  (0.053): answered "could not tell". — [doc Q8][doc], [figs fig-tpp, fig-tpp-seeds][figs], [yaml vocab-tpp200][yaml]

### Inferences
- At 300m the 64K, 32K and 16K truncations end level with or below 128K at 8 passes (1.4659 / 1.4609 / 1.4665 vs
  1.4651) although they are worse on unique tokens, so their goal gaps (0.075 / 0.059 / 0.044) are mostly the
  tokenizer's own repetition cost, not a weaker 8-pass endpoint.
- The bytes' 8-pass result (1.3736) is lower than every BPE tokenizer's unique-token result at 300m, but it uses 4.74x
  the optimizer steps; the goal compares each tokenizer only with itself, so bytes are already within 0.004 of their goal.

### Gaps
- 64K, 32K and 16K have one seed at 300m; their per-K costs are not seed-checked (the page notes the 128K->64K step is
  within seed spread).
- No 520m 64K/32K/16K runs; no 130m second seeds; bytes 300m seed 1 is still running (15832611).

## Q2. Interventions tried: effect on unique-token bpb, 8-pass bpb and cost, seeds, verdict

### Takeaway
At 300m 128K, the only families that lower the 8-pass result beyond seed noise are (a) random training-time replacement
of rare (id >= 8000) input rows by their 8K pieces' mean ("cvd", -0.018 over 3 seeds) and its variants, and (b) a new
order of the repeated part in every pass ("rs", about -0.008, adds on top of cvd). The best 128K 8-pass value is
detach + rs 1.4389 (2 seeds) / cvd + rs 1.4397 (3 seeds), still 0.053-0.054 above the 128K unique-token line (1.3856).
Every output-side intervention and every rare-token stop failed. Nothing has been tried yet on 64K/32K/16K; 8K has only
rs (one seed).

### Cited Findings
Reference values for this section (300m exact): 128K s0 1.3828 / 1.4651 (0.0823), s1 1.3884 / 1.4655 (0.0771); 8K s0
1.4510 / 1.4772 (0.0262), s1 1.4567 / 1.4766. "Lead" = 128K rep8 of the same seed minus the setup's rep8.
Code: `VARIANT=cv/fo/fop/fip/lr/rm/cvr/cvd` are 128K-only (the launcher raises if VOCAB_K is set); `drop` and `inoise`
work with any VOCAB_K. — [scaling script L378-379][scal]

A. Composed input rows (VARIANT=cv CV_SIDE=in, "cvin": every input row of ids >= 8000 is the mean of its 8K pieces' rows,
at train and eval; small rows start as the baseline's own rows; f32 compose).
- s0: full 1.4086, rep8 1.4557, cost 0.047 (old 1.4127 / 1.4601 / +0.0474); s1: 1.4098 / 1.4674, cost 0.058. Two-seed
  mean 1.4092 / 1.4616 / 0.0524. Removes 63% (s0) / 44% (s1) of the 128K-8K cost gap; at 8 passes s0 is 0.009 below
  128K, s1 0.002 above, so "about level". Its seed spread (0.010) is about twice the tokenizers' (0.005-0.006).
  Verdict: lower cost by giving up unique-token fit; 8-pass level. — [doc Q4, Q4b][doc], [figs fig-cvin][figs]

B. Composed input + residual (VARIANT=cvr: pieces' mean + CV_RES_SCALE x learned residual row per rare token, residual in
plain Adam, init zero).
- res 0.1: s0 1.4013 / 1.4547 / 0.0534; s1 1.4026 / 1.4578 / 0.0552; s2 1.3965 / 1.4587 / 0.062 (page: 1.3968 / 1.4588).
  3-seed rep8 mean 1.4571, lead 0.008 over 128K (every seed ahead by 0.006-0.011) < 0.012: fails criterion 2.
  — [doc Q10, Q10b A, Q10c F][doc], [yaml 15390567, 15523784, 15556775][yaml]
- res 0.03: 1.4014 / 1.4532 / 0.0518 (lead 0.0119, one seed); res 0.3: 1.3991 / 1.4567 / 0.0576. Across 0.03 / 0.1 / 0.3
  the cost is 0.052 / 0.053 / 0.058 and the fit 1.4014 / 1.4013 / 1.3991: the residual rate trades fit for cost only
  slightly. — [doc Q10b B1/B2][doc]
- cvr + rs (rep8 only): 1.4509 (page 1.4511) vs 128K rs 1.4572: lead 0.006. cvr + bpd0.1: 1.4531 vs 128K bpd 1.4523: no
  gain. "Data changes and composed rows stack only partly." — [doc Q10b D][doc], [yaml 15523787][yaml]
- J, residual rows frozen after pass 1 (CVR_RES_FREEZE, in the full run too): 1.4090 / 1.4571 / 0.048; residuals help
  the full fit, their repeated-pass learning does not change rep8. — [doc Q10d][doc]
- K, output head composed too (CV_SIDE=both, output residual 0.1, moves ~0.6x a head row's rate): 1.4243 / 1.4818 /
  0.058. L, cvr + rare output rows slowed to 0.1x after pass 1: 1.4249 / 1.4871 / 0.062. Both worse fit, higher rep8,
  no lower cost: "the cost left by cvr is not removed through the rare output rows". — [doc Q10e][doc]
- 520m cvr: full 1.3144, rep8 1.3917, cost 0.077 (128K 0.095): cost down about a fifth, lead 0.003 over 128K, one seed.
  — [doc Q10c E][doc], [yaml 15556774][yaml]

C. Random replacement "cvd" (VARIANT=cvd: the baseline's own 128K table; in training only, at each position whose input
id is in [8000, V-257), with probability CVD_P=0.5 the input row is replaced by the mean of its 8K pieces' rows, with
gradient into the pieces; eval uses own rows; the draw is a fixed function of (seed, step), identical in the full and
rep8 runs). — [table_variants L329-416][tv], [doc Q10b C][doc]
- p0.5 seeds: s0 1.3934 / 1.4504 / 0.057; s1 1.3915 / 1.4488 / 0.057; s2 1.3900 / 1.4429 / 0.053. 3-seed mean full
  1.3916 (+0.006 vs 128K 1.3856), rep8 1.4474, cost 0.056; lead 0.018 over 128K and 0.030 over 8K; every cvd seed beats
  both 128K seeds by >= 0.0147; cvd spread 0.0075: criterion 2 met at 300m, criterion 1 not (cost twice 8K's).
  — [doc Q10e][doc], [yaml 15523933, 15625219, 15657636][yaml]
- cvd + rs (rep8 only, full = cvd's): s0 1.4386, s1 1.4402, s2 1.4404, mean 1.4397 (spread 0.0018), cost 0.045-0.050
  (mean 0.048) vs cvd's full points; lead 0.026 over 128K, 0.037 over 8K, 0.008 over cvd alone: gains add.
  — [doc Q10e][doc], [yaml 15657637, 15714416][yaml]
- cvd + bpd0.1 (rep8 only): s0 1.4472, s1 1.4454: -0.003 per seed vs cvd, far less than rs (-0.012). — [doc Q10g][doc]
- cvd p0.8: 1.3981 / 1.4507 / 0.053: more replacement costs fresh fit, same rep8. — [doc Q10h][doc], [yaml 15704202][yaml]
- cvd from pass 2 (CVD_FROM_PASS=2; replacement from update round(steps/8); full point = 128K itself): rep8 s0 1.4449,
  s1 1.4493 (mean 1.4471); lead 0.020 / 0.016 (mean 0.018, same as cvd from the start) at no fresh-data cost; cost
  0.062 / 0.061 vs 128K full. — [doc Q10h][doc], [yaml 15713743][yaml]
- cvd from pass 2 + rs: rep8 s0 1.4423, s1 1.4402, mean 1.4412, cost 0.056 vs 128K full (the best result whose full
  point is the unmodified 128K). — [doc Q10l][doc], [yaml 15837660][yaml]
- cvd with 32K pieces on ids >= 32000 only (CV_K=32000; mean 2.52 pieces): 1.3839 / 1.4513 / 0.067, fresh fit within
  0.001 of 128K, 0.014 below 128K at 8 passes, one seed: "the rarest tokens carry most of the gain". — [doc Q10h][doc]
- Ablations Q10i (same drawn positions, one seed each unless noted): grad (own row kept, gradient blocked) 1.3856 /
  1.4525 / 0.067 (keeps 0.013 of cvd's 0.015 lead, better fit than cvd); unk (one shared learned row, id V-1) 1.4116 /
  1.4687 / 0.057 (cost = cvd's but fit 0.018 worse, ends above 128K); detach (pieces' mean without gradient into
  pieces) s0 1.3936 / 1.4421 / 0.049, s1 1.3946 / 1.4460 / 0.051, s2 1.3921 / 1.4441; 3-seed rep8 mean 1.4441 vs cvd
  1.4474, seed-matched -0.008 / -0.003 / +0.001: level with cvd. Ranking at 8 passes (seed 0): detach < cvd < grad <
  128K < unk. — [doc Q10i][doc], [yaml 15718460, 15718462, 15718470, 15743404, 15743423][yaml]
- detach + rs (rep8): s0 1.4402, s1 1.4377, mean 1.4389 (cost 0.045 vs detach full; page 0.0448): level with cvd + rs.
  detach from pass 2: s0 1.4547, s1 1.4501, mean 1.4524, cost 0.067: loses part of its gain when started after pass 1
  (unlike cvd fp2). — [doc Q10i, Q10j][doc], [yaml 15743441, 15795041][yaml]
- 520m cvd: s0 1.3038 / 1.3716, s1 1.3039 / 1.3740 (2-seed rep8 1.3728, spread 0.0024), lead 0.021-0.023 over both
  128K (1.3947) and 8K (1.3957) with both seeds, cost 0.068 vs 128K 0.095: criterion 2 met at 520m. 520m cvd fp2: rep8
  1.3707, cost 0.071 vs 128K full. 520m detach full 1.3047, detach + rs rep8 1.3627, cost 0.058 (one seed), the lowest
  520m 8-pass result. — [doc Q10h, Q10k][doc], [yaml 15713730, 15801440, 15801455, 15801564][yaml]

D. Data-order / segmentation changes of the repeated passes (PASS_DATA, pre-built caches; pass 1 = baseline subset in
its order).
- rs (new order each pass 2-8): 128K rep8 1.4574 (doc also says 1.4572), cost 0.075, one seed; 8K rep8 1.473, cost
  0.022, one seed (plain 1.4772). — [doc Q10][doc], [yaml 15394392, 15399204][yaml]
- bpd0.1 (new BPE-dropout 0.1 segmentation each pass 2-8; +4.06% tokens per pass, 11846 steps): 128K s0 1.4525 (doc also
  1.4523), s1 1.4546 (full point 128K s1 1.3884); 2-seed lead over 128K 0.0117, about the seed spread: misses criterion
  2 narrowly; mean cost 0.068 vs 1.3856. — [doc Q10f][doc], [yaml 15602274, 15482785][yaml]

E. Freezing / slowing rare rows after pass 1 (freeze_rows_muonh.py, from update 1430 = round(11444/8), in the full run
too).
- fip (rare input rows 8000+ stop): s0 1.3956 / 1.4555 / 0.0599; s1 1.3981 / 1.4563 / 0.058 (page s1 1.3984 / 1.4564).
  ~40% of the 128K-8K gap removed; full run loses 0.013; leads 128K by 0.009-0.010 < 0.012. — [doc Q4c, Q10d H][doc]
- fop (rare output rows stop): first run 1.4806 / 1.5665 / 0.086 CONFOUNDED (AdamH sizes the head step by the whole
  table, so the 8256 kept rows took ~4x their rate); rerun fop8kb with kept-block step (FOP_BLOCK_NORM=1) 1.4776 / 1.5292
  / 0.052: removes ~54% of the 128K-8K gap but fit worse than 8K (1.451): rare output rows keep learning from new text
  after pass 1. — [doc Q5b][doc], [yaml 15429947, 15521703][yaml]
- G, rare output rows slowed to 0.1x (FOP_SLOW=0.1): 1.4080 / 1.4870 / 0.079: keeps part of the fit, none of the cost
  drop. I, fip + output slow 0.1: 1.4251 / 1.4862 / 0.061: input and output effects do not add; worst fit.
  — [doc Q10c G, Q10d I][doc]

F. Stopping training on rare tokens (rare-target mask from pass 5, VARIANT=rm, K=8000).
- rm8kp5 (loss mask only): collapse, rep8 2.845 (1.53 at step 5000 -> 2.17 at 6000 -> 2.84 at 7000); rm8kp5f (mask +
  rare output rows frozen) 2.095 (collapsed slower; wikitext 3.65 -> 5.24); rm8kp5d (also out of the training softmax)
  1.633 (no collapse, eval rises after switch). All far worse than 128K 1.4651; not drawn. — [doc Q10][doc], [yaml 15520302][yaml]

G. Output-side composition and fixed rows (causal tests).
- cvout mean head: full 1.7670 (old metric), rep8 1.8613: the mean caps a token's logit at its best piece. cvout sum:
  gate failed (smoke eval 8.43; train loss 4.36 at step 890 vs 3.73), cancelled. fo8k (rare output rows fixed at init
  from step 0): gate failed (4.65 vs 3.72 at step 1000), cancelled. — [doc Q5, Q5b][doc], [yaml 15168736, 15228167, 15378165][yaml]

H. Probes (causes, not fixes).
- Shorter window (128K at SEQ_LEN 3072, batch 172, same tokens): exact 1.3921 / 1.4746 / 0.0825 (old +0.0829): no change.
  More steps (batch 96, 1.33x steps): exact 1.3850 / 1.4652 / 0.080 (old +0.0808): no change. — [figs fig-window,
  fig-steps][figs], [yaml 15168737, 15168738][yaml]
- Low-rank tables (rank-48 input and output, ~115M params like 8K): 1.5350 / 1.5711 / 0.036; gate failed (4.15 vs 3.72 at
  step 1000) but kept running; fit worse than 8K: inconclusive. — [doc Q9][doc]

I. Earlier related project (sampled softmax / Over-Tokenized Transformer): OT n-gram input tables beat the baseline on
<1 pass but repeated data flips it (300m x2.4: OT +0.153 vs baseline; repetition costs OT 0.21, baseline 0.02); tables
frozen from pass 2 end +0.071 vs OT-H +0.153, and frozen rows trained under the warmup backbone were never refreshed;
the vocabulary question split off from that project's Q5. — [memory ss/ov][mem]

### Inferences
- Two independent levers stack on 128K: input-side rare-row handling (cvd/detach, ~0.018 at 8 passes) and data order
  (rs, ~0.008; cvd + rs adds to 0.026). Segmentation change (bpd) overlaps with both and adds little on top of cvd.
- Removing gradient into the own rare row (grad) gives most of cvd's gain; the replacement input (pieces' mean) adds a
  further cost reduction while preserving fit; the gradient into the pieces is neutral (detach level with cvd).
- Every attempt to remove the rare-target (output) cost either destroyed the unique-token fit or did not lower the
  cost; that is where the remaining 128K gap sits (see Q4).

### Gaps
- 8K rs, 128K rs and cvr variants are single-seed; 520m detach + rs, cvd32k, grad and unk are single-seed.
- No intervention at 130m; at 520m no rs-only, no 8K or bytes interventions.
- No result yet for any dropout / noise variant (all queued).

## Q3. What is queued or running now (sacct 2026-10-21) and what each tests

### Takeaway
Two bytes jobs are running; seven jobs for the per-tokenizer goal are pending, all seed 0, all changing only passes 2-8
so each tokenizer's full point stays its own.

### Cited Findings
- Running: bytes-300m-s1 15832611 (started 2026-10-21 03:59; bytes seed 1, full + rep8; tests whether bytes rep8 below
  the old Figure-1 goal holds and moves the bytes full point); bytes-rs 15883244 (started 05:49; bytes rep8 with
  PASS_DATA=rs, seeds 0 and 1; tests whether bytes reach their own full result 1.3695 with a new order). — [sacct][yaml],
  [doc Q10m, Bytes row][doc]
- Pending (code bd5161f921 / 41b3e6a0b6; each 128K and 8K rep8, seed 0):
  drop-fp2 15954894 = residual dropout p 0.1 on attention and MLP outputs of every layer from pass 2 (VARIANT=drop,
  DROP_FROM_PASS=2; inverted dropout; train only; motivated by Xue et al. 2023); drop02-fp2 15960983 = dropout 0.2
  (dose); drop-fp2-rs 15961018 = dropout 0.1 + PASS_DATA=rs (does dropout stack on the order gain);
  inzero-fp2 15961203 = input-row dropout 0.1 per position, kept rows scaled 1/(1-p) (VARIANT=inoise INOISE_MODE=zero;
  AWD-LSTM embedding dropout); inrand-fp2 15961253 = input tokens replaced by uniform random ordinary tokens at 0.1
  (INOISE_MODE=rand; data noising, Xie et al. 2017). — [doc Q11, Q11b][doc], [dropout_qwen3.py][drop], [yaml][yaml]
- Pending, truncations: rs-mid 15961318 = 64K and 32K rep8 with PASS_DATA=rs (8 GPUs); rs-v16k 15961377 = 16K rep8 with
  rs (4 GPUs): gives the 64K/32K/16K panels their first setup. — [yaml][yaml], [logs/plans/rs-mid.txt][plans]
- The input-noise variants hit every position (not only rare tokens) and draw from the trainer's per-step,
  per-microbatch key; residual dropout acts only when a train step and key exist, so eval and all pre-pass-2 steps equal
  the baseline. — [dropout_qwen3.py][drop]

### Inferences
- The 8K arms of these jobs are the first interventions on 8K beyond rs; 64K/32K/16K get only rs. No queued job applies
  cvd-like rare-row handling to any truncated tokenizer (the cvd code is 128K-only).

### Gaps
- No queued job tests dropout from the start, other dropout rates for the truncations, any bytes intervention beyond rs,
  or any 520m/130m goal run.

## Q4. Mechanistic findings: where the cost sits, rare rows, row norms and directions, optimizer facts

### Takeaway
The 128K cost concentrates on the rarest input decile (0.219) and the rarest target decile (0.210) while 8K is flat and
low; cvd halves the rare-input peak (0.126) but leaves the rare-target peak (0.179), so most remaining cost sits on rare
targets. Repetition does not grow rare input-row norms; it rotates rare rows away from their pieces' mean (less
compositional), which cvd counteracts. In this recipe no table has effective weight decay: linears and lm_head are
norm-preserving and the embedding is plain Adam with no decay.

### Cited Findings
- Per-decile cost (seed-1 pairs, macro bpb, 10 equal groups by training frequency, rarest first): by input token, 128K
  0.219, 0.090, 0.073, 0.065, 0.071, 0.058, 0.048, 0.054, 0.029, 0.039; 8K 0.033, 0.020, 0.018, 0.018, 0.021, 0.018,
  0.021, 0.019, 0.036, 0.015; cvin-s1 0.105, 0.069, 0.055, 0.053, 0.055, 0.048, 0.039, 0.041, 0.078, 0.029; bytes 0.010,
  -0.033, 0.008 ... 0.014 (no rare-byte peak). By target token, 128K 0.210, 0.080, 0.065, 0.047, 0.053, 0.050, 0.035,
  0.034, 0.029, 0.028; 8K 0.002, 0.024, 0.029, 0.018, 0.021, 0.038, 0.025, 0.018, 0.033, 0.008; bytes 0.016, -0.021,
  0.013 ... -0.003. Macro totals equal the W&B costs. — [figs fig-freq-in, fig-freq-out][figs], [doc status, audit][doc]
- cvd seed-1 pair (B1): total cost 0.057 vs 128K s1 0.077 (8K s1 0.020); rarest input decile 0.126 vs 0.219, about half
  of the drop (0.093 of the 10-decile sum 0.19); rarest target decile 0.179 (128K 0.210): "most of the cost left sits
  on rare targets, which cvd does not touch". — [doc Q10i B1][doc]
- Context length (fig-pos): no cost grows with context bytes; 128K 0.090 at 32 bytes, ~0.070 at 1.4-2.9 KB, 0.076 at
  23 KB; 8K lower everywhere (0.013-0.035); bytes 0.015 falling to ~0 by 4096 bytes. — [figs fig-pos][figs], [doc Q6][doc]
- Repeated-text fit (B2, last-300-step train loss rep8 minus full, in bpb): 128K 0.043, 8K 0.031, cvd 0.037 (3 seeds
  0.037-0.038; lower bound since measured with replacement on), cvd + rs 0.032, cvr 0.040, fip 0.040. — [doc Q10i B2][doc]
- Row directions (B3, seed-1 final checkpoints): repetition lowers the cosine between a rare token's own input row and
  its pieces' mean in 128K (training-count deciles 4-7: 0.067->0.032, 0.106->0.063, 0.168->0.122, 0.198->0.166), much
  less in cvd (0.104->0.090, 0.168->0.150, 0.267->0.244, 0.338->0.318); cvd's rare rows lie ~2x closer to their pieces'
  mean (decile 10: 0.507 vs 0.255). — [doc Q10i B3][doc]
- Row norms (Q10n, median input-row norm by count in the first 200M tokens, seed 1): 128K full vs rep8: counts 1-9
  1.607 / 1.628; 10-99 2.942 / 2.963; 100-999 3.408 / 3.417; 1000-9999 3.509 / 3.511; 10^4-10^5 3.504 / 3.505; >10^5
  3.094 / 3.103. Count-0 rows 0.522 (full) vs 0.037 (rep8) only because the 8-pass subset never shows them. Conclusion:
  "the memorisation is in the direction of the rare rows, not their size"; weakens the case for embedding weight decay.
  — [doc Q10n][doc], [rownorm.json][rn]
- Same rownorm.json, other runs: cvd s1 full / rep8: 1.278 / 1.286 (counts 1-9), 2.731 / 2.746, 3.349 / 3.354, 3.481 /
  3.483; fip s1 (rare rows frozen at update 1430) full / rep8: 0.659 / 0.659 (1-9), 1.195 / 1.195 (10-99), 1.364 / 1.364,
  1.387 / 1.387 (1000-9999), then 3.458 / 3.461 for counts >= 10^4 (ids < 8000, not frozen); cvd + rs s1 rep8 1.286 (1-9).
  — [rownorm.json][rn]
- Optimizer (MuonH config, 300m): linear weights via MuonH (Newton-Schulz, "simple" coefficients, 5 steps), update
  p - lr*u*||p||/||u|| then renormalised to ||p|| (exactly norm-preserving); lm_head via AdamH (norm-preserving Adam);
  input Embedding via plain Adam (scale_by_adam, -adam_lr) with no decay term anywhere in muonh.py although the launcher
  passes weight_decay=0.1. Each group clips its own global norm (max_grad_norm 1.0). 300m: lr 0.01, adam_lr 0.002,
  beta1 0.9, beta2 0.98, eps 1e-15, momentum 0.98, cosine, warmup 1000, min_lr_ratio 0, batch 128 x 4096, 11444 steps;
  520m: 24 layers, hidden 1024, batch 256, 9918 steps; 130m: lr 0.02, adam_lr 0.008, linear decay 0.8, no warmup, 4959
  steps. — [muonh.py][mh], [scaling script L386-390, L662-680][scal], [doc Q10n][doc]
- AdamH sizes the head step by the whole table, so freezing part of the head re-distributes the step to the kept rows
  (~4x); the per-block fix leaves the rare block's baseline rate at lr*sqrt(f_U/f_P). — [doc Q5b, Q10c G][doc]

### Inferences
- fip's frozen rare rows are about 40% of the norm the baseline's rare rows reach at the end (0.66 vs 1.61), and the
  baseline's full and rep8 runs end at the same norms, so rare input rows keep growing after pass 1 regardless of
  repetition (consistent with fip losing 0.013 on unique tokens). cvd's rare rows end about 20% smaller (1.28 vs 1.61).
- Because no table has weight decay and linears/head are norm-fixed, the recipe has essentially no explicit
  regulariser; the queued dropout/noise runs are the first explicit regularisers tested.
- For the goal, the remaining 128K cost (cvd + rs ~0.048 against own full) is mostly on rare targets, the side where
  every intervention so far failed; bytes and 8K show low, flat per-decile costs.

### Gaps
- No per-decile analysis exists for 64K/32K/16K, for 8K + rs, or for cvd + rs / detach (only the cvd s1 pair).
- No row-norm or cosine analysis of the output head (lm_head is norm-preserving as a whole; per-row behaviour not
  recorded).
- The original list of eight candidate reasons (the doc refers to "reasons 4, 6, 7, 8") is not stored in the docs read.

## Q5. Best result per tokenizer and remaining gap to its unique-token line

### Takeaway
Against the 2026-10-21 goal (8-pass bpb = the tokenizer's own unique-token bpb), only the bytes are close (gap 0.004).
128K's best closes about a third of its gap (0.0797 -> 0.0533); 8K's rs closes ~0.004; 64K/32K/16K have no setups.

### Cited Findings
- Per-tokenizer table at 300m (goal line = unmodified unique-token result, mean over finished seeds; Q11 gap list):
  128K goal 1.3856, unchanged rep8 1.4653 (gap 0.0797), best detach + rs 1.4389 (2 seeds; gap 0.0533), cvd + rs 1.4397
  (3 seeds; gap 0.0541), best with the unmodified full fit cvd fp2 + rs 1.4412 (2 seeds; gap 0.0556). 8K goal 1.4539,
  unchanged 1.4769 (gap 0.0230), best rs 1.4730 (1 seed; gap 0.0191; 0.022 vs its seed-0 full 1.4510). 64K goal 1.3912,
  rep8 1.4659 (gap 0.0747); 32K 1.4020 / 1.4609 (0.0589); 16K 1.4229 / 1.4665 (0.0436); no setups. Bytes goal 1.3695,
  rep8 1.3736 (gap 0.0041), seed 1 and bytes + rs running. — [figs fig-progress][figs], [doc Q11][doc]
- 520m: 128K goal 1.3001, unchanged 1.3947 (0.0946), best detach + rs 1.3627 (one seed; gap 0.0626); cvd 2-seed 1.3728
  (gap 0.0727); 8K 1.3567 / 1.3957 (0.0390) and bytes 1.2581 / 1.2855 (0.0274) without setups. — [figs fixes-in 520m
  panel][figs], [doc Q10k][doc]
- Goal definition history: 2026-10-18 Figure 1 goal = 128K full + 8K cost (criterion 1), then 1.3856 (128K full, reached
  by unchanged bytes 1.3736), raised 2026-10-19 to the bytes' full 1.3695, replaced 2026-10-21 by one panel per tokenizer
  with each tokenizer's own full line ("不同 tokenizer 不可直接对比"). — [doc open risks, Q10j, Q11][doc], [page script][page]

### Inferences
- For 128K, the remaining gap (~0.053) is larger than 8K's unchanged cost (0.023), so input-row fixes plus data order
  reach neither the 8K cost level nor the goal; a different mechanism (rare targets / output side, or a general
  regulariser) is needed.

### Gaps
- Whether a setup's "own full" should be the modified run's full (e.g. cvd full 1.3916) or the unmodified tokenizer's is
  a definitional choice; from-pass-2 setups make the two identical by construction, and the Q11 gaps use the unmodified
  line.

## Q6. Open leads and ideas recorded but never run (or half-run)

### Takeaway
The docs record several options deliberately not chosen and some single-seed results awaiting replication; the largest
untested area is anything for 64K/32K/16K and any output-side or regularisation fix that keeps the unique-token fit.

### Cited Findings
- Options not chosen: dropout from the start (full runs too) and waiting for the bytes (Q11); BPE-dropout 0.3 and seed 1
  of the two reshuffled runs (128K rs, 8K rs) (Q10f); a 4th cvd seed (Q10h); waiting for the 300m detach seeds (Q10k).
  — [doc Q10f, Q10h, Q10k, Q11][doc]
- M, weight decay on the input residual: not run ("r003 already tests a slower input residual"). Embedding weight
  decay / norm limit: argued against by Q10n (no norm growth). — [doc Q10e, Q10n][doc]
- "Not done (owner): reasons 4 (Adam on rare rows), 6 (shorter token strings), 7 (weaker model), 8 (composing words)".
  — [doc L47][doc]
- Half-tested: 128K rs and 8K rs (one seed each); cvd32k, grad, unk, cvd p0.8, cvr r0.03/0.3, cvboth, cvr-oslow, fiop,
  fop8kb (one seed each); 520m detach + rs and cvd fp2 (one seed); Q8 TPP=200 costs (2 seeds only for 128K and 8K, noise
  exceeds range). — [doc][doc], [yaml][yaml]
- Code limits: cvd/cvr/fip/fop/rm/lr/cv are 128K-only (launcher raises with VOCAB_K), so cvd-style rare-row replacement
  on 64K/32K/16K would need new code. — [scaling script L378][scal]
- Disk: non-final checkpoints of finished runs were deleted on 2026-10-18 (272 GB); "a rerun from a mid-training step now
  needs a fresh run". — [doc open risks][doc]

### Inferences
- Untested combinations that follow directly from the record: detach (or cvd) from pass 2 + rs at 520m; cvd-style
  replacement for 64K/32K/16K (with their own smaller pieces); dropout/noise stacked on cvd + rs for 128K; any
  intervention for 8K beyond rs and the queued dropout/noise.

### Gaps
- No recorded idea addresses rare targets while keeping the unique-token fit (all tried output-side changes failed).
  Output-side regularisers such as label smoothing or z-loss are not mentioned in any source read.

## Q7. Contradictions and stale entries found

### Takeaway
Mostly small rounding mismatches (<= 0.0004) between the doc and the page data, old-metric numbers in older doc rows,
and stale status fields in the YAML.

### Cited Findings
- Doc header "Status (2026-10-08)" uses old-metric numbers (128K +0.083, full 1.3867, cvin 1.4127 / 1.4601, 520m 128K
  1.3038 / 1.3989, 8K 1.3607 / 1.3997); the exact values are 1.3828, 1.4086 / 1.4557, 1.3001 / 1.3947, 1.3567 / 1.3957.
  Note cvd 520m full is 1.3038 exact, numerically equal to the old-metric 128K 520m full. — [doc][doc], [figs][figs]
- Doc line 47 lists "the parameter-count control (low-rank tables)" as not done, and spec item 3 says "No parameter-count
  control", but Q9 lr48 ran on 2026-10-12 (15378166). — [doc][doc], [yaml][yaml]
- Doc vs page values: cvr s2 1.3965 / 1.4587 vs 1.3968 / 1.4588; fip s1 1.3981 / 1.4563 vs 1.3984 / 1.4564; fop8kb
  1.4776 / 1.5292 vs 1.4779 / 1.5293; fop slow 1.4080 / 1.4870 vs 1.4084 / 1.4872; cvr + rs 1.4509 vs 1.4511; cvr + bpd
  1.4531 vs 1.4533; cvr-resf 1.4090 / 1.4571 vs 1.4092 / 1.4573; 128K rs 1.4572 vs 1.4574; 128K bpd s0 1.4523 vs 1.4525.
  — [doc][doc], [figs][figs]
- YAML: bytes-300m-rep2.4 15715263 still says RUNNING although its result (1.3671) is in the doc (2026-10-19); experiment
  blocks vocab-sizes / vocab-why / vocab-tpp200 / vocab-why2 still "status: running, result: pending"; synced NEVER
  everywhere. — [yaml][yaml]
- Doc Q10n says "seed 0 at 8 passes and cvd are the same" (norms unchanged by 8 passes within each setup), but cvd's
  rare-row norms differ from 128K's (1.278 vs 1.607 for counts 1-9); the claim is about full vs rep8, not about cvd vs
  128K. — [rownorm.json][rn], [doc Q10n][doc]
- Page Q1 text and fig-vocab-final caption: "32K ends lowest at 8 passes among truncations by a small margin that we did
  not replicate" (1.4609); this is one seed. — [page][page]

### Inferences
- None of these mismatches changes a verdict; they matter only if a report quotes 4-decimal values, in which case the
  page figure data (generated by build_vocab_figs.py from exact tables) is the most recent single source.

### Gaps
- The page "Takeaways" section is empty in the published HTML (heading only). — [page][page]

[doc]: file:///scratch/gpfs/GROUP/USER/project/marin/docs/della/vocab-overfitting.md
[yaml]: file:///scratch/gpfs/GROUP/USER/project/marin/logs/vocab_overfitting_h100x4.yaml
[page]: file:///scratch/gpfs/GROUP/USER/tmp/vocab_wt/exps/vocab-overfitting.html
[figs]: file:///scratch/gpfs/GROUP/USER/tmp/vocab_wt/assets/data/vocab-overfitting-figs.js
[scal]: file:///scratch/gpfs/GROUP/USER/project/marin/experiments/references/della_muonh_qwen3_scaling.py
[tv]: file:///scratch/gpfs/GROUP/USER/project/marin/experiments/references/table_variants_qwen3.py
[drop]: file:///scratch/gpfs/GROUP/USER/project/marin/experiments/references/dropout_qwen3.py
[mh]: file:///scratch/gpfs/GROUP/USER/project/marin/lib/levanter/src/levanter/optim/muonh.py
[rn]: file:///scratch/gpfs/GROUP/USER/tmp_plots/vocab/rownorm/rownorm.json
[plans]: file:///scratch/gpfs/GROUP/USER/project/marin/logs/plans/rs-mid.txt
[mem]: file:///home/USER/.claude/projects/-home-USER/memory/project_ss_ov_kaiyue_baseline.md
