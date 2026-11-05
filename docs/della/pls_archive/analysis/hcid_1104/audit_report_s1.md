# Audit: one-stream decomposition (arch3-hcs1f / hcs1id / hcs1scalar / hcs1static / plleak01, all -probes)

Read-only review, 2026-11-04. Nothing was submitted, cancelled, committed or pushed.
I ran: the CPU test (della-vis1), the dry run (della-vis1), a CPU init/optimizer-label comparison
(`tmp/pls_hcid/audit_s1_init.py`), a read of the hcs1 final checkpoint (`audit_s1_read.py`) and W&B history reads
(`audit_s1_wb*.py`). No GPU code.

## Verdict

**No blocker.** Each job name reaches exactly its knobs, each arm computes what its name says, no run id collides, and
`wq cancel` + `wq requeue` resumes from the last periodic checkpoint. The problems are in what the result can say
(the reading rule, one missing arm, the meaning of plleak01) and in tests that claim more than they check. Fix the
reading rule (M1, M2) before the first job reaches step 600.

## Major (not blocking submission; they limit what the result can claim)

**M1. "Layer 12" is W&B key `train/pls/L11`. Write this into the rule.** Keys run L0..L47 (L47 = final;
per_layer_qwen3.py:684-685). Steps 500-599 from W&B:
Pre-LN fa4thd: L11 5.219, L12 5.179. HC 1 stream (old backend): L11 4.296, L12 4.257.
The reference values in the task are the L11 values. The docs (pls.md item 26) also use L11 as "layer 12".
If a reading script takes key L12, every number moves by about 0.04.

**M2. The 0.3-nat bands can overlap and are close to the noise.**
- The anchors are 0.92 apart now (5.219 − 4.296). arch3-hcs1f replaces 4.296. On the 4-stream pair the backend change
  moved L11 by 0.27 (4.249 jax_flash g4 → 3.975 fa4thd, same seed). If hcs1f moves up by a similar amount, the gap
  drops below 0.6, and an arm can be "within 0.3 of both anchors". The rule has no tie case. Define it relative to the
  measured hcs1f, and add a tie case (e.g. "inconclusive when the gap < 0.6").
- Noise at layer 12, steps 500-599 (W&B, key L12): HC seed 0 vs seed 1 3.935 / 3.938; hcid seed 0 vs seed 1
  5.312 / 5.200 (0.11); Pre-LN vs its same-seed rerun 5.179 / 5.167 (0.01); 4-stream HC backend change 0.27.
  So 0.3 is 1-3 times the run-to-run spread of designs with learned routing, and every arm has one seed.
  A "partial" reading, or a reading within about 0.15 of a band edge, needs a second seed before it means anything.
- Compare hcs1f with 4.296 first. If they differ by more than about 0.2, the 1-stream design is as sensitive to
  the trajectory as the 4-stream one, and a single-seed 0.3 band is not defensible.

**M3. Step 500-599 is early. The order holds in the references, but gaps shrink and one arm type moves the wrong way.**
- Layer 12 (key L12), steps 500-599 → 2100-2196 → 4800+: Pre-LN 5.179 → 5.053 → 4.661; HC 1 stream 4.257 → 4.322
  → 4.030; hcid 5.312 → 5.177 → 4.793; HC 4 streams 3.935 → 3.840(L12) → 3.586(s1).
  Pre-LN − HC 1 stream: 0.92 → 0.73 → 0.63. The order of these runs does not change after step 500.
- The 4-stream scalar run (hcscalar) rose from 4.792 (500-599) to 4.978 (2200-2299), against the trend of the
  others. The scalar arm is the arm most at risk of a different verdict at a later step.
- The 1-stream HC's routing is still moving at step 600 (W&B params/norm, layer 11 attention: read offset 0.23 at
  step 600, 1.03 at 2200; dynamic scale 0.47 → 0.55). The LR decay starts at step ~992 (decay 0.8 of 4,959 steps).
  So 500-599 measures how fast each knob is learned, not what it reaches. State the claim at that altitude.

**M4. The arms do not form a complete 2 x 2 over g and w. One cell is missing: g learned, w = 1.**
What each pair isolates (r is inert, see M5):
- learned vs scalar: token dependence of g, with w dynamic.
- scalar vs identity: a static g vs g = 1, with w dynamic.
- scalar vs static: token dependence of w, with g static.
- identity vs Pre-LN: a dynamic w alone (plus one extra Adam group, see m3).
No arm has a learned g with w = 1. So if hcs1id lands near Pre-LN, the runs can say "w alone is not enough", but
not "g alone is enough". No env knob gives w = 1 (HC_WRITE=shared with m = 1 is the same as per-stream write).

**M5. With one stream, r does nothing except set a sign, and it never flips sign in the reference run.**
- Code: x = r H (depth_arch_qwen3.py:435), then the branch applies RN first (line 361). RN divides by
  sqrt(mean(x^2) + eps), so r cancels when r^2 mean(h^2) >> eps = 1e-5.
- Evidence: in the old hcs1 run the gradient norm of `read` is 1e-7 to 3e-7, against 1e-4 to 2e-2 for `write`,
  `res` and `scale` (W&B grad/norm, layers 0, 11, 30, steps 100/600/2200).
- Final checkpoint (step 4,958): static r = 1 + read is 0.865 at layer 0 attention and 2.0-3.1 elsewhere. The
  dynamic term is at most |s| = 0.834, so r stays >= 0.03 > 0. No sign flip.
- Consequence: "static: r, g, w constants" is in effect "g, w constants". "identity" is Pre-LN plus a dynamic w.
  No arm means something other than its name, but r should not appear in any claim.
- Small caveat: at layer 0 attention r can reach 0.03. There RN may not fully cancel it if the embedding RMS is
  small; I did not measure the embedding RMS.

**M6. plleak01 is not the g of this decomposition.** It applies a fixed decay derived from the 4-stream HC's row sums
(HC_SUM_GAIN ** 0.1, depth_arch_qwen3.py:227-236, 252-254). The 1-stream HC learns a different g:
- static g at layer 0 attention = 1 + 2.21 = 3.21 (it amplifies the embedding by about 3 times);
- static g at most other sublayers = 1 ± 0.12;
- pls.md item 26: cumulative static gain to layer 12 is 10^-0.03, measured stream ratio 10^+0.21.
plleak01 answers "does a small fixed decay of the sum help Pre-LN's layer 12?" It does not answer "is the 1-stream
HC's g what helps?". Keep its verdict separate from the g/w table.

**M7. The tests claim more than they check (item 7).**
- Test 1 (hc arms) compares the model with the hand computation only at a fresh init. At init read, res, write, all
  w_* are 0 and every dynamic tanh term is 0, so r = g = w = 1 in all four hc arms. The four arms are then the same
  function as Pre-LN. The test cannot detect an arm that uses the wrong g or w (for example scalar applying the
  dynamic g, or static keeping the dynamic terms). Fix: perturb read/res/write/w_*/scale to non-zero values before
  the comparison.
- Test 2 checks that the fixed knobs do nothing. It does not check that the free knobs do something. A scalar arm
  that ignored `res` (that is, identity) passes tests 1 and 2.
- Test 2, static arm, `scale`: with w_* = 0 the dynamic terms are tanh(0) = 0, so `scale` has no effect and zero
  gradient in any mode. That check cannot fail. The `w_read` value check almost cannot fail either, since RN removes
  r (M5); its zero-gradient check does work (dynamic mode gives ~1e-7, not 0).
- Test 3 multiplies numbers from the HC_SUM_GAIN table in numpy. It never calls the model. Its docstring claim
  "with g = 1 the state grows only by writes" is not tested anywhere.
- Test 4: the docstring says "the probes' gradient does not reach the backbone". The code checks only that the loss
  and gradients are finite and that the loss falls in 12 steps. Nothing checks the backbone gradient of the probe
  losses.
- Test 4 uses DepthArchMuonHConfig for the leak arm. Production uses MuonHConfig for ARCH=preln
  (della_muonh_qwen3_scaling.py:538-549). The groups are the same (preln has no `hyper` leaves), so this is cosmetic.
  The `try/except ImportError` fallback around the import (test lines 157-160) is dead code.
- The leak test's bfloat16 tolerance (rtol 4e-2, atol 0.15) is looser than the leak's effect over 8 sublayers
  (about 5%). The float32 branch (1e-5) does catch a missing leak, so the test as a whole is valid.
- real_step_s1.py: its recorded output (real_step_s1.txt) prints `detach False sep False`, 28 / 14 gradient leaves
  (47 separate heads would add far more) and one CE value. The current script prints three CE values (k = 0, 11, 47).
  So the recorded output is from another version, and it was a model without probes. Its docstring claims (probe
  gradient does not reach the backbone, SGD steps lower the loss) are not in its code either. The GPU smoke with the
  train-path gate (pls_h100x8_single.sbatch:203-223) is the only real-size check of the probes model.

What I confirmed independently of the tests: I read each arm's code path (below), and the CPU test passed
(`ALL OK`, log `tmp/pls_hcid/s1_test_audit.log`).

## Item-by-item evidence

### 1. Names to knobs, leftover env, run ids
- pls_arch3_fetchable.sbatch:157-161 maps each design to exactly the listed knobs, on
  `base` (line 112: SIZE=130m PLS_W=1 PLS_SEP=1 PLS_DETACH_BB=1 PARAM_MAP=vocab NUM_GPUS=4), LAYERS=48 PDP=4.
- The `-probes` suffix gives BB=1 (line 109). The name comes from FETCHED_JOB_NAME under wq (line 106).
- pls_h100x8_single.sbatch:31-32 unsets every knob these arms use (ARCH, HC_STREAMS, HC_RES, ROUTE, PRELN_LEAK,
  SHRINK_POWER) and every knob of the other designs (HC_RW, HC_WRITE, HC_RWDYN, HC_STATE_F32, BRANCH_BOUND, MHC_*,
  AR_*, PLS_*, SEED, RUN_TAG, ATTN_BACKEND, ...). ~/.bashrc exports none of them.
- Dry run (della-vis1), run ids:
  - `...-pls1-sep-bbfrozen-ahc-hcs1-fa4thd-vs-d48`
  - `...-ahc-hcid-hcs1-fa4thd-vs-d48`
  - `...-ahc-hcscalar-hcs1-fa4thd-vs-d48`
  - `...-ahc-rstatic-hcs1-fa4thd-vs-d48`
  - `...-apreln-plleak-sp0.1-fa4thd-vs-d48`
  None exists in `marin_store_big/speedrun/`, in W&B `reself/marin-della`, in `logs/pair_failed/` or in
  `logs/smoke_pass/`. No queued wq job has these names.
- Note: `arch3-hcs1f` (line 157) has the same knobs as the old `arch3-hc1` (line 129). Today both would give the same
  run id. Submit only one of them.

### 2. What each arm computes (m = 1)
- `_hc_constants(1, k)` gives read = mixing = write = 1 for every sublayer (depth_arch_qwen3.py:257-261; CPU test ok 1).
- learned: g = 1 + res + s tanh(RMS(H) w_res), w = 1 + write + s_w tanh(...), token-dependent (lines 416-418).
  g is cast to bfloat16 before it is applied (line 468), so g is rounded in steps of about 0.4% near 1.
- identity: H <- H + w F; g = 1 exactly (lines 444-446).
- scalar: g = sum(c_res (c_res + res)) / m = 1 + res, one static learned number per sublayer, applied in float32
  (lines 447-450). w stays dynamic. Unlike learned, this g is not rounded to bfloat16.
- static: r, g, w = constants + learned offsets, no dynamic term (lines 413-414). g and w then use the default path
  (line 468).
- 1/sqrt(m) at init: m = 1 gives a factor of 1 (lines 330-333). CPU check (`audit_s1_init.py`, same key):
  all 20 shared leaves of the hcs1id model and the Pre-LN model are bitwise equal, and the layer outputs at init
  are equal (max |diff| 0.0). So the scaling does not confound the comparison.
- Every other difference between hcs1id and Pre-LN:
  1. 14 extra leaves per layer (`hyper[0..1].{read,res,write,w_read,w_res,w_write,scale}`). In hcs1id, res and w_res
     get zero gradient (CPU test ok 2). read and w_read get about 1e-7 (M5).
  2. Those leaves are in their own Adam group `adam_new`: eps 1e-8 (Pre-LN's Adam has 1e-20), its own global-norm
     clip at 1.0, LR = adam_lr 0.008 on the same schedule (depth_arch_qwen3.py:946-987). All shared leaves have the
     same optimizer label in both models (CPU check). The clip of the other groups sees the same parameters as in Pre-LN.
  3. No weight decay anywhere: MuonH ignores weight_decay, and both Adam groups have no decay term.
  4. CE path: ARCH=hc gets HCMB (pls_h100x8_single.sbatch:90-92: PARAM_MAP_PDP=4, LEVANTER_CE_BTILED_MIN_BATCH=1048576),
     so the hc arms compute the loss with the XLA streaming CE. Pre-LN and plleak01 use the b-tiled CE with the LSE
     kernel. The microbatch is 4 in both. The comments put this at summation-order level (gradient rel 1.5e-6 to 1.6e-5,
     w306). Small, but it is a difference between hcs1id and Pre-LN that is not in the knob list.
  5. Speed only: HC 1 stream ~16.3 s/step (w195), Pre-LN probes ~6.1 s/step.

### 3. PRELN_LEAK=1 SHRINK_POWER=0.1
- Gain of sublayer 2l + j = HC_SUM_GAIN[2l + j] ** 0.1 in float32 (lines 252-254). Index: l is the 0-based layer
  from the scan, j = 0 attention, 1 MLP (line 390). Applied as (g * h_f32).astype(bf16) + F (line 391). The stream
  stays bfloat16, as in Pre-LN, with one rounding per sublayer, as in Pre-LN.
- CPU test: gains equal HC_SUM_GAIN ** 0.1; cumulative 0.383 at layer 12, 0.231 at layer 32, 0.260 at layer 48;
  smallest gain 0.893, largest 1.048.
- The launcher checks hold: SHRINK_POWER != 1 needs PRELN_LEAK (line 322); 48 layers <= 96 / 2 gains (line 712).
- Clean against Pre-LN: same seed (SEED unset, 0), same data order (data_seed 42, linear permutation), no new
  parameters, same optimizer class (MuonHConfig), same CE path (no HCMB for preln). The only change is g.
- Meaning: see M6.

### 4. Are the references comparable?
- The Pre-LN fa4thd and old hcs1 configs differ only in attn_backend (FA4_THD / JAX_FLASH), depth_arch and
  hc_streams. The hcs1 config file holds the model section only. Its model fields (pls_*, hidden 512, 48 layers,
  eps 1e-5, fold_norm on) match. Both use the same launcher recipe (batch 128, 4,959 steps, seed 0, data_seed 42,
  47 probes, microbatch 4: w195 log).
- The old hcs1 reference also predates later commits, but every later hc option is gated off by default.
- arch3-hcs1f does not correct the old run. It replaces it. Use hcs1f and Pre-LN fa4thd as the two anchors (both
  fa4thd). Use the old 4.296 only to measure how far a backend change moves the 1-stream design (M2).

### 5. Failure modes and stopping
- What a 30-step smoke and a step-100 check would miss:
  1. the slow drift of the learned knobs and the scalar-type upward drift (M3);
  2. W&B offline mode: if the proxy fails, the metrics must be synced (`logs/wandb_offline.txt`) before they can be read;
  3. a resumed segment overwrites `logs/slurm/<name>_<wN>_train.log` (pls_h100x8_single.sbatch:232-234,
     `> "$log"`), so the step 0-700 log is lost after a later requeue. W&B keeps the metrics.
  Known late failures start after step 700: w973's rise from step ~1,200; w890's float32 overflow at step 4,410.
- Stopping with `wq cancel` on a running --partial job:
  - fetch_pending.sh:475-476 sends TERM to the guest's bash.
  - Its EXIT trap (pls_h100x8_single.sbatch:136, `stop`) TERMs the task's `timeout`, which TERMs Python. KILL follows
    after 90 s.
  - Levanter has no TERM handler that saves a checkpoint, so the steps after the last periodic save are lost.
    Saves come every 30 min (CKPT_MIN=30): up to about 110 steps for HC 1 stream, about 300 for Pre-LN.
  - A save killed mid-write has no metadata.json (written last, checkpoint.py:843-846), and discovery skips it
    (checkpoint.py:1189). The previous temporary checkpoint is deleted only after a new one commits
    (keep_last_temporary_checkpoints 1, checkpoint.py:690-718). So the last complete checkpoint survives.
  - `end_run` marks the job "cancelled" (wq.py:445-446). `wq requeue` clears the cancel marker and queues it with
    runs = 0 (wq.py:346-360). A resumed segment skips its smoke when code and knobs are unchanged
    (pls_h100x8_single.sbatch:175-182).
- Evidence from today: w1002 (arch3-hcshrink05-probes, --partial) was cancelled while running at 12:41
  (wq events.log: "143 cancelled by the user while running"). It ran again from 12:45, and its train log shows
  "Discovered latest checkpoint ... step-1775", then "Resuming training from step 1776". It is training now
  (step 3,769 saved at 21:09).
- Recommendation: run `wq cancel` just after a "Saved checkpoint" line in the train log, to lose almost no steps.

### 6. Reading rule
See M1-M3. Summary:
- read key L11;
- define the bands on the measured hcs1f, with a tie case;
- treat 0.3 as about 1-3 run-to-run spreads, so a second seed is needed for a middle or edge reading;
- 500-599 keeps the reference order, but gaps shrink about 30% by the end and a scalar-type arm can drift up 0.2.

### 7. Tests
See M7.

## Minor
- m1. The untracked repo file `same` (0 bytes) is the stderr path of earlier wq jobs (w1002 spec: `"err": ".../marin-pls/same"`),
  probably from `-e same`. Do not pass `-e same` with these five jobs. Otherwise all their stderr goes to one file in
  the repo root.
- m2. In the learned and static arms g and w are rounded to bfloat16 before they are applied. In scalar, g is float32.
  This is a second difference between learned and scalar besides token dependence. It is probably small (steps of
  about 0.4% per sublayer); record it.
- m3. In hcs1id the dead read and res leaves keep their Adam state. They cost nothing. Their gradient is 0 or ~1e-7,
  so they do not affect the adam_new clip.
- m4. The uncommitted sbatch diff adds only comments and the five case lines. depth_arch_s1_test.py is untracked.
  The smoke key hashes lib/ and experiments/ only, so committing these files does not force new smokes.
