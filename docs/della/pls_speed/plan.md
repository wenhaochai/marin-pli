# pls speed plan: ranked harmless candidates, A/B protocols, rollout rules

**Repo state.** The speed copy is `/scratch/gpfs/GROUP/USER/project/marin-pls-speed` at `07b1bc1d9d` (branch pls-speed). Line numbers below are from this copy. Its launcher is offset by about +2 lines from `pls`, because of the PROFILE knob.

**Determinism floor.** `docs/della/pls.md` records no GPU run-to-run floor for training, so every A/B must measure one first (step 0 of the protocol). The doc does record these noise references:
- pls.md:32: GPU bf16 `eval/L{L-1}` against the main eval differ by at most 2e-7 relative (6-layer runs).
- Smoke-gate gap between the main eval and the last readout: 1e-4 to 5e-4. The gate threshold is 0.005.
- pls.md:582: AttnRes Full d48, final layer-48 train loss, seed 0 / seed 1. Separate heads 3.023 / 3.016. Probes 3.045 / 2.955. So the probe seed gap is 0.09.
- pls.md:57: 130m 6-layer baseline pool, c4_en bpb sd 0.00035 (n=8).
- Full GPU steps are probably not bitwise, because the embedding-gradient scatter uses atomics. A bitwise claim needs `--xla_gpu_exclude_nondeterministic_ops=true` on both arms.

**Expected total after Tier A.**
- PDP=4 designs (preln, sandwich, lns, keel, deepnorm, vis): about 13–14% wall-clock per 4960-step run.
- hc / mhc / attnres at d48: about 7–9%.
- Probe runs (`-bbfrozen`): another 5–6%.
- Tier B adds about 25–30% on hc/mhc, but only with the owner's OK (see Tier B).

---

## Global rules for every candidate

- **G1.** Edit only `marin-pls-speed`. Every change is an env knob whose default is today's behaviour. A sbatch change is a new file (`scripts/della/pls_h100x8_single_v2.sbatch`), never an edit of `pls_h100x8_single.sbatch`.
- **G2. Merge gate.** Queued chains import `pls` live at their next segment. So before any merge into `pls`, prove that "knobs off" changes nothing:
  - On della-vis1, compile `jit__train_step` with 4 fake CPU devices for preln d48 sep PDP=4 and hc d48 sep PDP=1, from `pls` and from `pls-speed` with knobs off.
  - The HLO fingerprints must be identical.
  - The eval program must also be identical. `scripts/della/pls_env.sh` already differs between the two branches (PROFILE), so include it.
- **G3. Metrics in every A/B:**
  - Loss: `train/loss`, and `train/pls/L0, L11, L23, L35, L46, L47`.
  - Gradients: `grad/norm/total`, and the clip rate.
  - Speed: `throughput/duration` (s/step) and tokens/s.
  - Memory: peak memory (`memory_stats()["peak_bytes_in_use"]`) and the XLA buffer-assignment temp.
  - Long A/Bs also compare `eval/L47/loss`, `eval/loss` and the Paloma macro at step 1000.
- **G4. Pass rules by numerics class:**
  - *Bitwise class:* the knob-on arm is bitwise equal at every step under exclude_nondeterministic_ops. Otherwise, at every step, the on/off |Δ| must stay inside the envelope of two knob-off twins (two RUN_TAGs, same seed).
  - *Changes-math class:* step-0 loss is bitwise equal. From step 200 to 1100, the 50-step-smoothed |Δ| of `train/loss`, of every `train/pls/L{k}` and of the grad norm stays inside the seed band of an existing seed pair at the same steps. The step-1000 eval is also inside that band.
  - *Seed pairs to use:* preln d48 sep s1 (w89) against apreln; preln probes s1 (w90) against preln probes; AttnRes Full g4 s0/s1 (finished).
- **G5. Where the tests run:**
  - 30–40-step smokes go to `pli-cp` (standing OK). One 8-GPU job holds two 4-GPU arms side by side, so their timings are matched.
  - The 1100-step A/Bs go through `wq` to `pli-short`. At d48 they are longer than 3 h, so they need `--partial`, or the owner's OK.
  - No submission happens in this phase.

---

## Tier A: harmless under rules (1)–(3). Ship after the checks.

Ranked by corrected saving per unit of risk.

### A1. One-hot label term in the H100 b-tiled CE backward (R1 = K1)

- **Saving.** About 10% (range 8–15%) of each PDP≥2 run: preln, sandwich, lns, keel, deepnorm, the vis runs, and the larger sizes. That is 2–3 h of a 21–22 h run.
- **Does not apply to** PDP=1 designs. Their compiled programs must stay unchanged.
- **Risk / effort.** Low risk. Small effort.
- **Mechanism, confirmed in the deviceless H100 HLO** `tmp/probe_oom/gpu_pdp8.hlo`. Today each 8192-row block runs five kernels:
  1. a GEMM that writes bf16 logits;
  2. a multiply fusion that writes f32 dlogits (4.2 GB);
  3. an in-place `input_scatter_fusion`;
  4. `wrapped_convert`, which reads the f32 and writes bf16;
  5. two GEMMs.

  A select fuses steps 2–4 into one kernel, so each block moves 8.4 GB less (12.6 → 4.2 GB). That is 6.45 TB less per step, about 2.2 s of 15.1 s.
- **Change.** In `lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/batched_xla.py:600-603` (`_backward_b_tiled_from_lse`, which starts at :552), behind `LEVANTER_CE_BTILED_ONEHOT=1`. Read the env var at trace time, as `xla.py:35/:714` does. Replace

  ```python
  dlogits = dlogits.at[row_indices, safe_labels].add(label_grad)
  ```

  with

  ```python
  hit = jnp.arange(v_dim, dtype=jnp.int32)[None, :] == safe_labels[:, None]
  dlogits = jnp.where(hit, dlogits + label_grad[:, None], dlogits)
  ```

  Masked rows add +0.0 at column 0, the same as the scatter. The only possible difference is the sign of a zero, which has no effect after the GEMM. Launcher: pass the knob through `SINGLE`. Add a run-id tag (`-ce1h`, launcher :329) only if check 4 below is not bitwise.
- **Numerics.** Bitwise on CPU (`tmp/pls_speed/onehot_test.py`, `readout/ce_numerics.py`). On GPU, an FMA contraction could move the label entry by 1 ulp, which is summation-order class at worst.
- **A/B protocol:**
  0. CPU unit test, bitwise against the scatter: B=16384, H=512, V=128256, 2 b-blocks, labels −1, nonzero `g_lse`, soft-cap set and unset.
  1. 1×H100 kernel test at the same shape. `grad_x` and `grad_w` must be bitwise equal; otherwise accept only differences of ≤1 bf16 ulp at label entries. Time each block: expect 2.5–3 ms less.
  2. GPU XLA dump of preln d48 sep PDP=4. Knob off must show the scatter; if it does not, drop the candidate. Knob on must show no scatter and no f32 [8192,128256] input to the convert, and a temp ≤ 30.61 GiB.
  3. The hc/mhc/attnres PDP=1 HLO fingerprint is unchanged.
  4. `pli-cp`, preln d48 sep PDP=4, 4 GPUs, seed 0, data_seed 42, 40 steps. Three arms: off, off-twin, on. Run them with exclude_nondeterministic_ops, plus one off/on pair without it for timing.
  5. Resume one arm from an existing preln checkpoint.

  AttnRes is not the A/B config here, because AttnRes is not on this path.
- **Pass.** Bitwise or inside the off/off envelope (G4). Median s/it over steps 15–40, excluding the checkpoint step, falls by ≥0.75 s (≥5%). If the step-level result is only inside the floor, also run a 1100-step A/B against the w89/apreln seed band before the first full run.

### A2. Probe readouts: drop the unused dX GEMM (R3b; replaces R3)

- **Saving.** About 6% of probe runs (`PLS_DETACH_BB=1`): 1.1 s of 15.1 s at PDP=4, about 2 s at PDP=1.
- **Risk / effort.** Medium-low risk. Medium effort.
- **Mechanism.** The aux norm scale g is trainable, so `x = g·rmsnorm(sg(h))` still needs dX to get dg. The CPU HLO has 4 GEMMs; it has 3 when dX is dead.
- **Change.** In `experiments/references/per_layer_qwen3.py:454-476` (`_layer` / `_readout_ce`, `detach_input` branch), behind knob `PLS_PROBE_NODX=1` with tag `-nodx`:
  - Keep the forward exactly as it is: x = bf16(g·n), W unchanged.
  - Wrap it in a `custom_vjp` whose backward takes only `gw` from the CE vjp (XLA then drops the `grad_x` GEMM).
  - Return `dg_j = Σ_v W_jv·gw_jv / g_j`, accumulated in f32, and no gradient for n.
  - Guard |g_j| < 1e-4: count such entries and fail loudly.
  - Keep the `pls_visible_frac` mask before the norm.
  - Check that the Qwen3 RmsNorm has no bias.
- **Numerics.** The forward and `gw` are bitwise. Only dg (512 values per head) is rounded differently. Without `-os`, the clip coupling carries that difference into the backbone at bf16-noise level, so this is changes-math in its last bits and needs the long A/B. R3 as proposed is rejected, because moving the forward rounding changes the measured probe losses.
- **A/B protocol:**
  - CPU at B=16384: GEMM count 3 on the b-tiled and the xla paths; loss and `gw` bitwise; dg within 1e-3 relative.
  - GPU dump: one dot fewer per aux readout; temp not larger.
  - 1100 steps of preln d48 probes PDP=4, with the exact effective args of w90 (seed 1), against w90's logged curve. Seed band: w90 against the seed-0 preln probes.
  - Also an attnres d48 probes HLO check (the gain at PDP=1).
- **Pass.** G4 changes-math rule. Step-0 losses bitwise. Median s/it ≥4% faster.

### A3. Save just before the time limit (C3), then a 30-min checkpoint interval (C2)

C2 ships only together with C3.

- **Saving.** About 1.5–2% in total: C2 1.1–1.6% net of the extra lost work at unplanned ends, plus C3 0.3–0.8%.
- **Risk / effort.** Low risk. C3 is medium effort; C2 is trivial.
- **Change, C2.** `experiments/references/della_muonh_qwen3_scaling.py:501`: `save_interval=timedelta(minutes=int(os.environ.get("CKPT_MIN", "10")))`. In the new sbatch v2, add `CKPT_MIN` and `TRAIN_DEADLINE` to the unset list (as at line 31). No tag.
- **Change, C3.**
  - Launcher monkeypatch next to :61-64. Wrap `levanter.checkpoint.Checkpointer.on_step` (checkpoint.py:578). When `TRAIN_DEADLINE` (epoch seconds) is set and `now + 2·step_ema + save_s (+ eval_s if the next steps_per_eval multiple falls before the deadline) ≥ deadline`, call `request_checkpoint()` (:653).
  - After `wait_until_finished()`, send `os.kill(os.getpid(), SIGTERM)`, so the exit stays rc 124/143.
  - sbatch v2, after line 155: `TRAIN_DEADLINE=$(( $(date +%s) + ${TRAIN_LIMIT%s} ))`, passed only to the train launch (:159), never to the smoke.
  - Optional addition, not counted in the saving: a SIGTERM handler that requests a save at the next step, for early rc143 ends (15 of them in the ledgers).
- **Numerics.** Bitwise: the change only reads the state.
- **A/B protocol (attnres d48 sep, 4 GPUs, seed 0):**
  - (a) `CKPT_MIN=1` against `=3`, 40 steps: bitwise or inside the off/off envelope; same hook_time per save.
  - (b) A 20-min `TRAIN_LIMIT` with C3. Expect: last checkpoint step = last train step; rc 124/143; `.executor_status` not SUCCESS; the next segment logs `Resuming training from step <last+1>` and no `Cowardly refusing`. The resumed curve is bitwise or within the floor against an unstopped run, over ≥50 steps.
  - (c) Put an eval step inside the deadline window: the save must come first.
  - (d) Kill during a save: the previous checkpoint still loads.
- **Pass.** All of (a)–(d). The smoke still writes its forced final checkpoint and passes all gates.

### A4. Turn off jaxpr/HLO debug logging (J1 = K6)

- **Saving.** 0.4–0.5% of every run: 3.5–4 min per segment of GPU idle (w78: 12 + 93 + 7 s in mid-run).
- **Risk / effort.** About zero risk. Trivial effort.
- **Change.** `della_muonh_qwen3_scaling.py:483-516`, in TrainerConfig: `log_jaxprs=LOG_JAXPR, log_xla_hlo=LOG_JAXPR` with `LOG_JAXPR = os.environ.get("LOG_JAXPR", "1") == "1"`. The defaults are at trainer.py:885/887, and the logging is in `_maybe_save_jaxpr` at :830-852. No tag.
- **Numerics.** Bitwise: same executable, same compile-cache key.
- **A/B protocol.** `pli-cp` smokes of attnres d48 sep, 30 steps, knob on and off.
- **Pass.**
  - The log has no `Tracing ... for jaxpr` lines.
  - Losses are bitwise or inside the envelope.
  - The `jit__train_step` cache entry is hit, not rewritten.
  - The first step plus the first hooks step are ≥60 s shorter in total.
  - Nothing reads `train_step.jaxpr.txt.gz`; grep confirms this.

### A5. One eval pass: log readout L{L-1} as the main eval (E5)

- **Saving.** About 0.9% (0.8–1.1%): 123–255 s per eval × 5 evals.
- **Risk / effort.** Low risk. Medium effort.
- **Change.** Knob `EVAL_ONE=1`, which sets model config `pls_eval_alias_main=True`:
  - `lib/levanter/src/levanter/main/train_lm.py:362` skips `cb_tagged_lm_evaluate` when the flag is set.
  - `per_layer_qwen3.py:775` (`cb_readout_evaluate`) also logs readout L-1 under `eval/` with `construct_log_dict`, and writes `eval_metrics.jsonl` with the same keys and step numbering.
  - Refuse the knob when `PLS_EVAL=0` or `EMA_BETA>0`.
  - sbatch v2 forces `EVAL_ONE=0` for the smoke, so the gap gate still compares the two programs.
- **Numerics.** Training is bitwise. The `eval/*` values become summation-order class (≤1e-4).
- **A/B protocol.** Eval-only job on the step-4958 checkpoint of hc d48 rstatic. Compare every `eval/*` key against the aliased L-1, check `eval_metrics.jsonl`, and run a 30-step smoke with the knob on.
- **Pass.** |Δ| ≤ 1e-4 on loss, macro, per-tag loss and bpb; identical key set. The smoke passes the gap gate. Train loss is bitwise. Eval wall time falls by about the main-pass time.

### A6. Static causal loops in JAX_FLASH attention (K3 Fix A)

Conditional: do it only if the profile shows the problem.

- **Saving.** About 3%: about 4% on PDP=1 designs, 1–2% on PDP=4. Speculative.
- **Risk / effort.** Low-medium risk. Medium effort.
- **Gate.** The pending profiler probes 16144708 / 16144709 must show ≥0.5 s/step of GPU idle at the predicate D2H syncs. If they do not, drop the candidate.
- **Change.** In `lib/levanter/src/levanter/models/flash_attention.py:278-282` (forward, `j_end = min(i+1, Tc)` while loop) and :416-419 (backward, `i_start = j` while loop), behind `LEVANTER_FLASH_STATIC_CAUSAL=1`:
  - Replace the data-dependent `while_loop` with a Python loop over the 10 static causal (i, j) blocks (Tr = Tc = 4 at seq 4096, block 1024).
  - Keep the same online-softmax update order, shapes and dtypes.
  - Fix B (WHILE command buffers) is rejected.
- **Numerics.** Bitwise in intent; it must be shown on GPU.
- **A/B protocol:**
  - 1-GPU kernel test (B = 1 and 4, 8 heads, 4096, 64): output, dQ, dK and dV bitwise.
  - GPU dump of hc d48 PDP=1: no while loop without `known_trip_count` is left; temp grows by less than 1 GiB.
  - 40-step smokes of attnres d48 sep and hc d48 sep at PDP=1: on, off and off-twin.
- **Pass.** Bitwise or inside the envelope. s/it falls by at least the measured idle time. If the result is not bitwise, it becomes changes-math class: run a 1100-step A/B against the AttnRes s0/s1 band.

### A7. Eval at 4 sequences per device for PDP=1 designs (R6 = E2)

- **Saving.** About 1.3% alone (about 6 min per eval × 5). About 0.6% after A5.
- **Risk / effort.** Low-medium risk (eval memory). Trivial effort.
- **Change.** `della_muonh_qwen3_scaling.py:497`, add `per_device_eval_parallelism=int(os.environ.get("EVAL_PDP", "-1"))`. With -1, trainer.py:1168-1175 still copies pdp. Set `EVAL_PDP=4` only for new hc/mhc/attnres/attnres_block/moda submissions. No tag.
- **Numerics.** Training is bitwise. Eval values are summation-order class; the zero-weight padding (loader.py:646-648) keeps the token set identical.
- **A/B protocol:**
  - Eval-only on an hc d48 checkpoint and an attnres d48 checkpoint, at eval batch 4 and 16.
  - Compiled eval `memory_analysis` on GPU.
  - 300 steps of hc d48 sep with `steps_per_eval≈25`, so that train steps and evals alternate more than 10 times past step 257.
- **Pass.** Every `eval/L{k}` and Paloma key within 5e-4; identical `loss_per_tag.total` counts. No OOM. Step times unchanged. Eval wall time ≥20% lower.

### A8. Skip the 30-step smoke on resume when the code is identical (S1)

- **Saving.** 0.3–0.5%. Commits that change code land about 2 per day, so only 20–40% of resume smokes would be skipped.
- **Risk / effort.** Low risk. Small effort. **This needs the owner's OK, because it removes a gate the owner set up.**
- **Change.** sbatch v2, lines 111-152. Write the marker `logs/pair_passed/<rid>.<hash>` only after the smoke passes rc=0, the gap gate and the train-path gate. The hash covers:
  - git HEAD and `git diff HEAD`;
  - untracked `.py` files under `experiments/` and `lib/`;
  - every source file in `sys.modules` of the launcher's dry run;
  - `pls_env.sh`, the sbatch copy and `tools/packed_task.sh`;
  - `SINGLE`, the venv path, the GPU model, and the JAX cache env.
- **Numerics.** None: wall-clock only.
- **A/B protocol.** Dry runs on della-vis1 for cases (a)–(f) from the skeptic's list. Then two real resume segments.
- **Pass.** The smoke is skipped only on an identical key. Any changed input runs the smoke. A `pair_failed` marker still blocks the run.

---

## Tier B: changes-math. Needs the owner's decision; not auto-shipped.

**Why Tier B is separate.** Every one of these moves hc/mhc (or attnres) off the PDP=1 `xla` CE fallback. That fallback (xla.py:280-395) feeds f32 delta to TF32 GEMMs and accumulates `grad_x` and `grad_w` in bf16. The b-tiled path or the fast backward uses bf16 dlogits and f32 accumulation instead.

**The skeptics disagreed.** The memory skeptic kept M1/M2/M3 as harmless if the A/B passes. The XLA skeptic rejected K2, the same mechanism, as not harmless. The plan resolves it this way:
- The project's d48 comparisons resolve 0.035–0.11 nats (separate-head cost per design).
- The seed noise of those comparisons is up to 0.09.
- A 1100-step A/B can exclude a gross change. It cannot certify that no final-loss shift of 0.01 or less exists.

**Rule.** Use Tier B only when every arm of a comparison runs at the new setting (rerun the sep run and the probe run together), or when the owner explicitly accepts mixing.

**Precedent and side argument.** Control 15341053 already ran steps 0–239 at mb4. Today PDP=1 and PDP=4 designs already train with different CE-backward numerics. That is an existing confound in the E1 table, and Tier B removes it.

### B1. hc/mhc d48 at microbatch 4 with a non-fragmenting allocator (M3; merges the allocator part of K2)

- **Saving.** About 30% of each hc/mhc/mhcid d48 run: 16.3 s/it against 24.3–25.3 s/it, measured. About 11 h of 35 h.
- **Risk / effort.** Medium risk. Trivial effort.
- **Change.** No code change. A new submission sets `XLA_PYTHON_CLIENT_ALLOCATOR=vmm` (fall back to `cuda_async`) and `PARAM_MAP_PDP=4` in `SINGLE`; the override already exists at launcher :303-304. Never set it in the launcher's PARAM_MAP block (:180-189). If pdp does not already show in the run id, add a tag.
- **A/B protocol:**
  1. 30 steps of hc d48 sep at mb4, BFC against vmm: must be bitwise.
  2. ≥1100 steps at mb4+vmm, packed as 4+4 GPUs on one node. It must pass step 257, the step-1000 eval, ≥5 checkpoints and one kill-and-resume. No OOM; no `Can't reduce memory` warning; s/it within 3% of 16.3.
  3. Numerics against w27's or w76's logged curve, inside the seed band (G4).
  4. Confirm that there is no PRNG-dependent op, because the microbatch count changes the per-microbatch keys.
- **Fallback, B1′ (M1).** Microbatch 2, using a 2-trip CE tile:
  - `batched_xla.py:369-389`: when `LEVANTER_CE_BTILED_SPLIT8K=1` and `x.shape[0] <= 8192`, return 4096.
  - Set `PARAM_MAP_PDP=2`.
  - About 25% saving. The CPU probe gives 64.45 → 8.08 GiB.
  - The `pls_d48_tempsize.sbatch` smoke must show temp ≤ 26 GiB for hc:2 and mhc:2.
- **Shallow runs, B1″ (M2).** `PARAM_MAP_PDP=4` alone, at d12 and d24, for new hc/mhc runs only: about 30% saving. It needs a tempsize smoke with `LAYERS` made overridable in the speed copy. Launch only if new shallow runs are planned.

### B2. Fast CE backward for designs that stay at microbatch 1 (R2; absorbs M4)

- **Saving.** For attnres, attnres_block and moda: ≤6% (the attnres mb4 evidence gives at most about 2 s/step). For hc/mhc it is moot once B1 ships.
- **Risk / effort.** Medium risk. Trivial effort.
- **Change.** Launcher knob `CE_FAST_BWD=1` with tag `-cefb`, which sets `LEVANTER_CE_XLA_FAST_BWD=1` (xla.py:35/:714) in-process. Never set it as an sbatch export.
- **Gate.** A 30-step timing smoke of attnres d48 sep must show ≥5% lower s/it, with a bitwise step-0 loss.
- **A/B protocol.** Then: a GPU dump with temp ≤ about 20 GiB and no `grad_w` copies that scale with the 48 heads; 300 steps at d48; 1100 steps against the AttnRes s0/s1 band.

---

## Rollout rules

1. Running and queued chains are never touched. No in-place edit, no change of defaults, no resubmission. Resubmitting a pending job loses its accrued age, which needs the owner's OK.
2. Every knob goes only into **new submissions**, through `SINGLE` in a new sbatch copy, and only after G2 passes and the merge into `pls`.
3. **Could a chain switch at a resume without breaking comparability?**
   - **Yes, in principle,** for A3, A4 and A8. They do not touch the math or any logged value.
   - **Yes, for training,** for A7. The eval values move by ≤5e-4, so mark the switch step.
   - **No mid-run** for A5, so that one run's `eval/loss` curve does not mix two programs.
   - **No mid-run** for A1 if it is only within the floor; allowed only if it is step-level bitwise.
   - **Never** for A2, A6-if-not-bitwise, B1 and B2.

   Even for the "yes" cases, switching needs either a resubmission or a default change. The first loses age; the second breaks rule (2). So in practice everything starts with new submissions. The owner may later flip the defaults of A3 and A4 after their checks pass.
4. Record the knobs, the CE path and the microbatch in each run's `logs/*.yaml`. Record Tier-B runs in `docs/della/pls.md` as a different CE backward.
5. Execution order:
   1. CPU work on della-vis1: unit tests and G2 fingerprints.
   2. One 8-GPU `pli-cp` job with 4+4-GPU arms: floor twins, A1, A4, and A3a.
   3. Eval-only jobs: A5 and A7.
   4. Read the profile result for A6.
   5. 1100-step `pli-short` A/Bs: A2, then Tier B if the owner approves.

## Rejected

| id | one-line reason |
|---|---|
| R4 vocab-parallel CE | Feasibility rests on an unconfirmed temp estimate. It changes the math (CE path and cross-device LSE). It brings back the sharded-program readout discrepancy of 15255032. 2–3 days of high-risk work. |
| R5 / K7 fused Pallas CE kernel | Roofline guess. Repo notes measured Pallas GPU forwards 8–9× slower than XLA. About a week of work. After A1, the remaining gain is about 10–20%. |
| K2 cuda_async + PDP=4 as a "free" speedup | Not harmless: the gain comes with a CE-precision change against the finished controls. Its allocator part is kept as B1 (Tier B). |
| K4 latency-hiding scheduler | Exposed NCCL time is unknown (some overlap already exists). About 1% gain. Raises peak memory in programs with a history of fragmentation deaths. |
| K5 8-GPU rendezvous timeout | No throughput gain: GPU-hours per run are the same or higher. Root cause unverified. A real hang would become a 10-min idle stall. |
| K8 AttnRes buffer slicing | Unprofiled, about 5% at most. Summation-order change against finished AttnRes controls. Restructures the tightest program (53 GiB temp). |
| C1 pinned checkpoint staging | About 0.4% after A3. Risk of a host OOM with two guests per node. Changes levanter internals. |
| E3 faster readout-eval CE | Kernel work that belongs to R5/K7. Not harmless when shared with the train step. |
| M4 fast backward for attnres | Merged into B2. Unmeasured, and the attnres mb4 evidence caps the gain at about 6%. |
| R3 (as proposed) | Moving the forward rounding changes the probe losses, which are the measured quantity. Replaced by A2 (R3b). |
| K3 Fix B (WHILE command buffers) | CUDA-graph conditionals inside a scanned, remat'd loop are high-risk, for the same gain as Fix A. |

## Side findings for the owner (no speed effect)

- **Launcher comment is wrong.** `della_muonh_qwen3_scaling.py:186-187` (speed copy) says a PDP change alters only the summation order. That is false: PDP=1 takes the `xla` fallback CE backward, with f32 delta and bf16 `grad_x`/`grad_w` accumulation. PDP≥2 takes the b-tiled path with bf16 dlogits.
  - So PDP=1 designs (hc, mhc, attnres, moda) and PDP=4 designs (preln, sandwich, lns, keel, deepnorm) train with different CE-gradient numerics. On CPU against f32, the `grad_x` error is 5.3e-3 against 7.6e-4.
  - This is a confound in the cross-design comparisons of the E1 table.
- **Stale smoke gate on resume.** On wq resume segments, the smoke's train-path readout gate reads the segment-1 `wandb-summary.json`. The smoke run id is reused, and W&B drops steps 0–29. Verified on w27: identical values (7.175948619842529) and the same `_timestamp` in both segments.