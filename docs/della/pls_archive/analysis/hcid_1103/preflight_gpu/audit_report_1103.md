# Audit of the four new designs (2026-11-03): sumshrink, half dose, move, Pre-LN leak, float32 streams

Repo `marin-pls`, branch `pls`, HEAD eaf75de78c (commits 5538c2842b, ef2e9ab9e4, 1ebbd6edbe, 52942aeb4e). Jobs w972, w973, w1000-w1007 are queued in wq. None of them has started.
Scratch scripts and outputs: `tmp/pls_hcid/audit_1103/` (`checks.py`, `scan_vs_loop.py`, `sens.py`, `tf32.py`, `dryrun.sh`, `singles.txt`, `tests_out.txt`). Everything I ran, I ran on della-vis1 on CPU.

## Verdict per design

| design (jobs) | verdict |
|---|---|
| 1. sumshrink (arch3-hcshrink, w972/w973) | OK |
| 2. half dose (arch3-hcshrink05, w1002/w1003) | OK |
| 3. move (arch3-hcmove, w1000/w1001) | **BLOCKER** (design confound, F1). Fix: shift the soft non-negativity so that the zeros of I map to about 1e-4, for example `sp(A - 0.3)`. |
| 4. Pre-LN leak (arch3-plleak, w1004/w1005) | OK |
| 5. float32 streams (arch3-hcsumf32, w1006/w1007) | **BLOCKER, probable** (F2; not confirmed on GPU). Fix: give the mixing dot `precision=jax.lax.Precision.HIGHEST`, or write it as an elementwise multiply and sum. |

## Findings, most severe first

### F1. move: at init the mixing averages the streams at 3.1% per entry, so stream differences shrink 0.874x per sublayer. (High; design confound)
`experiments/references/depth_arch_qwen3.py:436`, `HC_MOVE_BETA` at :231.
- sp(a) = softplus(20 a) / 20 gives sp(1) = 1.000 and sp(0) = ln 2 / 20 = 0.0347. The zeros of I therefore become 0.0347. After row normalisation, P = 0.906 on the diagonal and 0.0314 off it.
- The eigenvalues of P are 1 (the stream sum) and 0.874 three times (every difference between streams). Over 96 sublayers the differences get a gain of 2.5e-6.
- Ran: with distinct streams, writes off and init parameters, one layer of move scales the difference norm by 0.765. HC (learned) at the same init scales it by 1.000 (`checks.py` part B).
- The docstring says that "sp is close to a for a > 0.1". That is true, but the off-diagonal entries start at a = 0, where sp is not close to 0.
- "At init this is HC's function" holds only at step 0, because all streams are equal there (the same embedding in each stream, and w = 1 on every stream). From step 1 on, the writes differ per stream, and move then applies a fixed averaging that HC and hcid do not have. The arm therefore does not test "moving content between streams" against hcid. It adds a strong built-in collapse toward the stream mean, and the parameters must learn to undo it. To make the off-diagonal entry small, res must move to about -0.2 to -0.3: offset -0.2 gives an off-diagonal entry of 9.0e-4 and a gain of 0.9964 per sublayer; offset -0.3 gives 1.2e-4 and 0.9995. With Adam at lr 0.008 that takes tens of steps per entry.
- In the replay (`replay2/3.jsonl`), move did learn to keep differences (dif/sum 0.46-0.73 at step 120, against 0.63 for HC). So the run will train. The problem is what the run measures, not whether it trains.
- Fix (one line): `pos = softplus(BETA * (res - 0.3)) / BETA + 1e-6 * c_res`. Then diagonal = sp(0.7) = 0.70 and off-diagonal = sp(-0.3) = 1.2e-4. The difference gain is 0.9995 per sublayer (0.95 over 96). The gradient still flows below 0. A second way: let P = row-softmax(tau (I + res)) with tau ≈ 9. The same parameters are kept in either case, so nothing else changes. Rerun `depth_arch_hcmove_test.py` after the fix and add the check in F5.

### F2. hcsumf32: the float32 mixing is a float32 dot_general. On H100 the JAX default precision for float32 dots is TF32. (High; probable; not confirmed on GPU)
`depth_arch_qwen3.py:443`: `hax.dot(res, f32(H), axis="stream")`. Also :415 (read), though that output is cast to bfloat16 anyway.
- Nothing in the launcher, levanter or marin sets `jax_default_matmul_precision` (grep; only `optim/kron.py` uses it). The JAX docs (`jax/_src/lax/lax.py:2146-2150`, jax 0.11.1) say that Precision.DEFAULT on A100/H100 uses TF32.
- All three tests and the replay set `jax_default_matmul_precision="highest"` and run on CPU, where TF32 does not exist. The claim "float32 did not diverge in 800 steps" was therefore measured with true float32 arithmetic, which the H100 run may not use.
- Ran (`tf32.py`, numpy emulation of TF32 rounding of both operands of one row-stochastic 4x4 mixing). The table gives the relative error of the stream sum after one mixing:

  | dif/sum | f32 | TF32 | bf16 |
  |---|---|---|---|
  | 300 | 4.3e-6 | 7.2e-2 | 5.7e-1 |
  | 1e4 | 1.4e-4 | 2.4 | 19 |
  | 1e5 | 1.4e-3 | 24 | 190 |

  In the float32 replays, dif/sum reached 1e3 at step 120, 4e4 at step 280 (`replay4.jsonl`) and 6.5e5 at step 240 (`replay_f32.jsonl`). TF32 is about 8x better than bfloat16 storage. At those ratios it still loses the sum, so w959's failure would come back a few tens of steps later.
- I could not check whether XLA:GPU lowers this K = 4 contraction to cuBLAS (TF32) or to a fused multiply-reduce (exact float32). Fix: pass `precision=jax.lax.Precision.HIGHEST` to this `hax.dot`, or compute `sum_i res[i, o] * H_i` as a broadcast multiply plus `hax.sum` (no dot_general, so no TF32). Either costs almost nothing at K = 4. The readouts use `hax.sum(H, axis="stream")`, a reduce, so they are exact float32 and need no change.
- Without the fix, the smoke cannot detect this: w959's own smoke passed and w959 diverged at step 27.

### F3. hcsumf32: even true float32 is marginal at the ratios the replay reached. (Medium; read and computed)
At dif/sum = 1e5 (the hcsumf32 test's own case), one mixing loses 1.4e-3 of the sum in float32. The error compounds over 96 sublayers. In the CPU replay, dif/sum still rose to 6.5e5 by step 240, and maxeig of the mixing kept rising (1.26 at step 800). The design keeps the instability. Float32 only moves the point where rounding wins. This is not a code bug, but expect a late divergence. Watch the state ratio (dif/sum) in the run, not only the loss.

### F4. Launcher: SHRINK_POWER, PRELN_LEAK and HC_STATE_F32 are missing from the unset list. (Low)
`scripts/della/pls_h100x8_single.sbatch:32` unsets HC_RES but not the three new variables.
- Under wq/fetch_pending this is harmless: guests start from `env -i` plus a fresh `~/.bashrc` (`tools/fetch_pending.sh:147`), and `~/.bashrc` sets none of them.
- A direct `sbatch` from a shell that exported, for example, SHRINK_POWER=0.5 would carry it into every SINGLE that does not set it. The run id would then carry `-sp0.5`, so it would not resume a wrong checkpoint, but it would run another design.
- Fix: append `SHRINK_POWER PRELN_LEAK HC_STATE_F32` to line 32.

### F5. Tests that cannot fail, or that run a friendlier regime than production. (Medium for move and hcsumf32)
- `depth_arch_hcmove_test.py:93` (test 1, "at init it computes hc's layer outputs"): the streams are equal at init, so any row-stochastic P passes. This test cannot see F1. Add: distinct streams, writes off, init parameters; assert that the difference norm is kept (my check gives 0.765 now).
- `depth_arch_hcsumf32_test.py:104-113` ("differences 1e5 times the sum"): it rounds numpy arrays and sums them. It never calls model code, so it cannot fail on the model.
- All three tests (`:32`, `:36`, `:37`) set `jax_default_matmul_precision="highest"` and run on CPU. No test or replay exercises the H100 precision path (F2).
- `depth_arch_hcmove_test.py` part 2 (bounded over 24 layers) uses static mixing only (`scale` = 0, `w_res` = 0), which is friendlier than production's dynamic term. Its `hc_res="sum"` growth of 1.5e5 uses res scale 0.5, which is not production's.
- The replay (`replay_hc4.py`) uses hidden 96, 64 tokens, VANILLA attention and `gradient_checkpointing=False`. That is acceptable for its purpose, but it is not the FA4_THD / remat path.
- The sumshrink tests are sound: test 2 asserts no dependence on res/w_res with exact 0, and test 4 asserts exactly zero gradient.

### F6. Analysis scripts build the streams in bfloat16. (Low)
`scripts/della/pls_hc_hidden.py:62`, `pls_hidden_ablate.py:64`, `pls_hc_hidden_cpu.py:77` and `pls_hc_hidden_ablate_cpu.py:73` stack the embedding into streams themselves and do not apply the float32 cast that `outputs` (:716-717) applies. On an hcsumf32 checkpoint they would run bfloat16 streams and give wrong stream statistics. Training and evaluation are not affected.

### F7. move: entries far below 0 lose their gradient. (Low)
`checks.py` part C: a row at -3 gives dP/dres = 8.8e-21. A diagonal at -0.5 with the others at -5 gives 1.4e-6. "An entry that turns negative can come back" therefore holds only within about 0.3-0.5 below 0. All rows stay finite, and the 1e-6 guard keeps the own stream. This is unlikely to matter at Adam lr 0.008.

## Checked and found correct

- Axis convention: `res` is (stream, stream_out). `hax.dot(res, H, axis="stream")` contracts the input stream, so H_o = sum_i A[i, o] H_i. A row is the outflow of input i, and row sums of 1 keep the stream sum. `"sum"` subtracts the row mean over stream_out, which is correct. move normalises over stream_out, which is correct.
- HC_SUM_GAIN equals `sumfail_data.json` mix.HC.rows[*].rowsum (step 4958) to 4 decimals (max difference 0.0), in the order 2l + j. rowsum = mean over i of sum over o of (I + res), i.e. the outflow; static part only, as the task states. Cumulative gain: 6.7e-5 at layer 12, 1.0e-6 at layer 24, 1.4e-6 at layer 48. The largest product over consecutive sublayers is 5.56, so there is no overflow. Half dose: 8.2e-3 at layer 12.
- Sublayer index: preln and hc are in SCANNED (`:171`). `outputs` passes `hax.arange(Layers)` (traced) to both `rstep` and `hstep`. The Python loop path is used only for attnres/moda, so `layer` is never None for these designs. `pls_local_heads` (the only other path that calls layers) is rejected for depth_arch != baseline (`per_layer_qwen3.py:297`). Ran: scan vs Python-int loop gives f32 rel 4e-7 (preln, leak, leak p = 0.5), 1e-5 to 6e-5 (sumshrink, move, sum f32). A shifted index gives rel 1.2, so the index is honoured. The larger f32 and bf16 scan-vs-loop gaps of sumshrink and move come from conditioning, not from the path: with the random test parameters, a 1e-3 input change moves layer 6 by 5e-2 (sumshrink) and 0.13 (move), and scan-bf16 vs scan-f32 differs as much as scan vs loop (`scan_vs_loop.py`, `sens.py`).
- Leak with a traced index under jit equals h <- g h + F per sublayer (rel 2e-7; layers 0, 3, 5).
- Eval path: the main eval (`compute_next_token_loss(key=None)` → `activations` → `DepthArchTransformer.__call__` → `outputs`) and the readout eval (`readout_losses` → `layer_outputs` → `outputs`) use the same forward pass as training. Both apply `mp.cast_to_compute` (`trainer.py:679`, `per_layer_qwen3.py:711`).
- hcsumf32 dtypes under p=f32, c=bf16 (ran): the embedding is bf16, the carry is f32 after a layer and stays f32 (the write promotes, `.astype(H.dtype)` keeps it), the branch input is cast to bf16 (:417), the outputs are bf16, and the gradients are float32 after `cast_to_param`. No f32 value reaches the attention kernel: x is cast before `branch`.
- Optimizer: res, w_res, read and scale are labelled `adam_new` (eps 1e-8). sumshrink: res/w_res get exactly zero gradient; after 3 real optimizer steps (eps 1e-20 recipe, clip 1) all parameters are finite and res did not move (Adam 0/(0+1e-8) = 0). Zero gradients do not change the per-group clip norm. HC parameters are not MuonH, so there is no norm projection. move: res moves 2.4e-2 in 3 steps, all finite.
- Launcher: all 10 job names resolve to SINGLE strings with the right knobs (`singles.txt`). `env $SINGLE` passes HC_RES, SHRINK_POWER, PRELN_LEAK and HC_STATE_F32 to both the smoke and the training (same `env $SINGLE $HCMB`). The launcher validates the combinations. HCMB (PARAM_MAP_PDP=4) applies to the hc jobs only.
- Run ids (ran dry runs): 10 distinct ids with the right tags (`-hcshrink`, `-hcshrink-sp0.5`, `-hcmove`, `-apreln-plleak`, `-hcsum-hcf32`). None exists in the store or in the temp checkpoints, and none has a fail marker or a smoke key. Nothing can resume from a wrong checkpoint. `tmp/pls_hcid/pls_h100x8_single.sbatch.new` is identical to the committed script.
- Config: the new fields are plain str/bool/float on `PerLayerQwen3Config`, so they serialise to W&B and draccus. Their validation is in `__post_init__` and in `DepthArchTransformer.init`. There are no new parameters, so checkpoints keep the HC tree.
- sumshrink: no learned part acts through the mixing. Read, write and the dynamic weights get gradient. `d_res` is computed and then unused (wasted, harmless).
- Memory for hcsumf32: the policy is `gradient_checkpointing=True` → `ScanCheckpointPolicy(simple=True)`, so each layer is checkpointed and the scan saves each layer's input carry. Per device: 128 / 4 GPUs = 32 sequences, microbatch 4 (HCMB). One carry is 4 × 4096 × 4 × 512 × 2 B = 64 MiB in bf16, 128 MiB in f32. 48 saved carries go from 3 GiB to 6 GiB, so about +3 GiB, plus one f32 cotangent carry (+64 MiB). This is the same carry size as 8 bf16 streams, and HC with 8 streams measured 24.86 GiB temp against 21.67 GiB for 4 streams (w473 vs w461): +3.2 GiB, matching the estimate. Expected about 25 GiB against a pool of about 72 GB (0.9 × 80 GB). It fits. If microbatch 4 fails, the smoke falls back to microbatch 1, which is smaller.
- The existing tests pass (`tests_out.txt`): hcshrink, hcmove and hcsumf32 print ALL OK.

## What I could not check

- Whether XLA:GPU lowers the 4-wide float32 mixing dot with TF32 on H100 (F2). It needs a GPU: dump the optimised HLO of one hc layer, or compare `Precision.DEFAULT` with `HIGHEST` on a 4x4 mixing with dif/sum = 1e4.
- FA4_THD attention and the real 48-layer, 4-GPU program: CPU tests use VANILLA attention.
- Real GPU memory: only estimated, above.
- Long-run behaviour of the designs at production scale.
