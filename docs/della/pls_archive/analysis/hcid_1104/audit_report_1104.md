# Preflight audit: arch3-hcidfb100-sep (commit bf33d6a7f4), 2026-11-04

I found no code blocker. The bound reaches every forward pass that training and eval use, and the run matches w810 except for the bound. Three problems can make the result hard to read. Fix them, or accept them, before you submit (R1-R3).

## Must settle before submission (readability, not code)
R1. **The criterion is the same size as the seed spread.** The two seeds of hcid separate heads end at 3.100 (w810) and 3.169 (w893), 0.069 apart, at L48 train (docs pls.md item 21). The criterion "about 3.02 explains the gap, about 3.10 does not" separates its two outcomes by 0.08. One seed can decide only the two ends: below about 3.04, or above about 3.10. A result from 3.05 to 3.08 is ambiguous. Write the thresholds down now, or queue seed 1 too (`arch3-hcidfb100-s1` needs a new case line).
R2. **The bound acts on F, not on the write.** `upd = write * _bound_norm(f)` (depth_arch_qwen3.py:439-442). The static write offset `p.write` has no bound and trains with Adam. In w810 it grew from 11.73 to 21.76 (logs/pls_depth_controls.yaml:518). A run can move the gain from the MLP into the write and keep the state frozen. A null result is then uninformative. Read the per-sublayer write norms, which W&B logs every 10 steps (`params/norm/transformer.layers.{l}.hyper.{j}.write`), next to w810's.
R3. **The control has no bound.** w811 is the ordinary model without the bound. Its 3.8e-5 share above 100 was measured only at step 4,958. Early-training norms were not measured. So "the bound leaves the ordinary model unchanged" is an assumption. `arch3-hcidfb100-probes` already parses (the sbatch case handles `-probes`) and gives a clean cost. Without it, report the result as an absolute L48, not as a cost.

## Checks (item, result, how verified)
1. **Path.** hyper_step has one call site, `DepthArchTransformer.outputs` → `hstep` → `scan_via` (depth_arch_qwen3.py:737-743). A grep for `hyper_step` and `.branch(` over experiments/ and lib/ found no other hc step. Training (`compute_next_token_loss`), the per-layer eval (`readout_losses`, per_layer_qwen3.py:695-699), and the main eval (`__call__`, depth_arch_qwen3.py:939) all go through `layer_outputs` → `outputs` (per_layer_qwen3.py:418-419). Remat recomputes the whole hyper_step with the bound inside. fa4thd sits inside `branch`, upstream of the bound. `pls_fold_norm` and `pls_serial_readouts` change only the readouts. A checkpoint resume rebuilds the model from the current config, with no second compute path. `config` is a static field, so `float(getattr(...))` reads a Python float at trace time.
   **Second paths outside the model:** tmp/pls_hcid/branch_norm.py:35-50, replay_hc*.py, blown_*.py and grad_tail*.py rewrite the hc step by hand, without the bound. If you run them on this run's checkpoints, they compute states the model never had. Add the bound to them before you analyze the run. mlp_spike_all.py reads weights only and is safe.
2. **Same as w810.** The launcher dry run (tmp/pls_hcid/dryrun_bound.sh, della-vis1, output audit_1104/dryrun.out) printed `...-sep-ahc-hcid-fb100-fa4thd-vs-d48`. Without the bound it printed w810's exact id. `ARCH=preln BRANCH_BOUND=100` fails with ValueError. A field-by-field diff of the built PerLayerQwen3Config against w810's W&B model config (audit_1104/cfg_and_overflow.py cfg) found only `depth_branch_bound=100` plus three new fields at their defaults (`depth_hc_state_f32`, `depth_preln_leak`, `depth_shrink_power`). The other differences are only printing (enum and rope repr). SEED 0, data_seed 42 (hard-coded), fa4thd, PLS_FOLD_NORM=1 and PLS_SERIAL=0 match. Since w810's commit 33a3063f97, nothing changed under lib/, nor in pls_env.sh. The venv has not changed since 10-26. In `hyper_step`, the identity branch is unchanged: `prec=None` without hc_state_f32, and `HyperParams.init(joint=False)` makes the same leaves. HCMB (`PARAM_MAP_PDP=4 LEVANTER_CE_BTILED_MIN_BATCH=1048576`) applies to both runs: w810's outer log shows it, and the new SINGLE contains ` ARCH=hc `. Two runs of identical code still differ by up to about 0.02 (GEMM autotune, w205 vs w229).
3. **`_bound_norm` (depth_arch_qwen3.py:239-246).** It computes the norm in float32 over Embed only, per token, before the broadcast to the streams. Below the bound, the scale is exactly 1.0, so the bf16 value comes back bit for bit and no gradient flows through the scale. Above the bound, the gradient loses its radial part and is scaled by c/|F|. The 1e-12 matters only at F = 0. Overflow test (cfg_and_overflow.py overflow, f32 and bf16, D=512):
   - When the sum of squares overflows (|F| > 1.8e19), n = inf, the scale is 0, and **F becomes 0, not c·direction**. The value stays finite.
   - At 1e18 per coordinate, the gradient is finite. At 3e37 per coordinate, the gradient is **NaN**.
   - An inf or NaN in F gives NaN.
   The right behavior is n = m·‖f/m‖ with m = max|f|, which keeps c·direction. In practice no run reaches this: w810's largest value is 1.55e6. This is a note, not a blocker.
   **Test weaknesses** (scripts/della/depth_arch_branchbound_test.py, which passes on vis1: audit_1104/unittest.out):
   - Test 5's "real optimizer with bound 20" never clips. I ran it with bound 20 and with bound 1e9 (audit_1104/t5_probe.py): the 12 losses are bit-identical. So that test proves nothing about the bound.
   - Test 3's reference calls `_bound_norm` itself, and its `maxn <= bound` check reads the reference's own F. So it tests where the bound sits, not what the model's F is.
   - The tests use VANILLA attention on CPU without a mesh. Only the job's smoke covers fa4thd and the 4-GPU sharding.
   - The GPU production test (untracked scripts/della/depth_arch_branchbound_real_test.py) logged only allocator OOM lines (tmp/pls_hcid/bound_real_test.log, 09:32) and no result.
4. **The default 0 and other jobs.** With 0, `if bound > 0` skips the code in Python, so the jaxpr of other runs does not change. Run ids add `-fb` only when the bound is nonzero (the dry run reproduced w810's id). Checkpoints hold arrays only, and the config comes from the launcher, so a resume loads. Resumed W&B configs gain `depth_branch_bound: 0.0`, which is harmless. Cost: `smoke_key` hashes `HEAD:experiments` (pls_h100x8_single.sbatch:176-178). So the next segment of every running pls job runs its smoke again, about 11-19 min on 4 GPUs each.
5. **Bound 100.** The bound is reasonable for blocking the freeze: HC's maximum is 107, and w811 has 3.8e-5 of pairs above 100. The bound does not block cliff-edge tokens below 100 (docs item 22). To decide (a) gain blocked versus (b) gain still grows, read all of these:
   (i) the MLP gain and the ln_2 maximum at the final checkpoint (mlp_spike_all.py);
   (ii) the ln_2 norm per layer over time from W&B (w810 layer 5: 25.7 → 32.1 from step 2,500 to 3,500);
   (iii) the write norms (R2);
   (iv) the grad median and the spikes in layers 1-2 (w810: 16.9 → 39.2, spikes up to 28,954);
   (v) the share of clipped pairs, with a bound-aware branch_norm.py.
   If the gain still grows while L48 stays near 3.10, the run neither confirms nor refutes the explanation. **Only one checkpoint survives:** `keep every 10000` keeps one permanent checkpoint, and the temp checkpoint keeps only the last one. w810 has only step-4958. So a gain-over-time read needs live_1104b.py on the temp checkpoint during each segment. Otherwise only W&B norms remain.
6. **wq command.**
   - The name parses: `*-sep` → BB=0, DESIGN `arch3-hcidfb100` matches line 150.
   - The memory estimate is 4 × 32G = 131,072 MB, `--time` 20 h, `--partial`, the same as w810's spec (wq/jobs/810/spec.json).
   - **`%w` does not expand.** wq.py:257 and fetch_pending.sh:211 expand only `%j`, `%x`, `%u` and `%N`. w810's log is literally `logs/slurm/arch3-hcid-sep_%w.out`, and later segments append to it. Use `%j` (→ w<n>). This is cosmetic.
   - w810 ran with `--prio 10 -x della-k16g1`. The new command has neither. Pilots exclude bad_nodes.txt anyway, so only the start order differs.
   - Python is read from the working tree at each segment start. A commit to `pls` between segments changes the code mid-run. w810 had the same exposure.

## Not checked
- The fa4thd and 4-GPU compile with the bound. The smoke covers it.
- The production-scale GPU test. It hit OOM and has no result.
- Early-training branch norms of w811 and HC.
- Whether the write offsets of w810 carry the gain on the bad tokens.
- I ran no GPU code and submitted nothing.
