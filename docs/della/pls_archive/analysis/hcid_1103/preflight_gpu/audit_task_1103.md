# Task: independent audit of four new model designs before their jobs start

You are an independent reviewer. Assume the author's tests and reasoning are wrong until you have checked them against the code. Do not edit the repo, commit, push, submit, or cancel anything. Write your report to `/scratch/gpfs/GROUP/USER/tmp/pls_hcid/audit_report_1103.md`.

Rules for running anything: you are probably on a Della login node. Do not run Python, JAX or tests there. Reading and grep are fine. For any computation use `ssh della-vis1 'cd /scratch/gpfs/GROUP/USER/project/marin-pls && PYTHONPATH=. CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu nice -n 19 timeout 500 .venv/bin/python -u /path/to/script.py'`. Put scratch scripts in `/scratch/gpfs/GROUP/USER/tmp/pls_hcid/audit_1103/` (the only place you may write besides the report). Never source `scripts/della/pls_env.sh` on a vis node.

## Repo and commits

Repo: `/scratch/gpfs/GROUP/USER/project/marin-pls`, branch `pls`. Read with `git show`:
- 5538c2842b: depth_hc_res="sumshrink"
- ef2e9ab9e4: depth_hc_res="move", depth_shrink_power, depth_preln_leak
- 1ebbd6edbe: "move" with softplus
- 52942aeb4e: depth_hc_state_f32 with depth_hc_res="sum"

Main files: `experiments/references/depth_arch_qwen3.py` (read the module docstring, `HyperParams`, `hyper_step`, `residual_step`, `outputs`, `_hc_constants`, `DepthArchMuonHConfig`), `experiments/references/per_layer_qwen3.py`, `experiments/references/della_muonh_qwen3_scaling.py` (the launcher), `scripts/della/pls_arch3_fetchable.sbatch`, `scripts/della/pls_h100x8_single.sbatch`. Tests: `scripts/della/depth_arch_hcshrink_test.py`, `depth_arch_hcmove_test.py`, `depth_arch_hcsumf32_test.py`. Small-model replay: `/scratch/gpfs/GROUP/USER/tmp/pls_hcid/replay_hc4.py`.

## Background

Hyper-Connections (HC) keeps m = 4 streams. Per sublayer: x = sum_s r_s H_s, H_o <- sum_i A[i, o] H_i + w_o F(RN(x)). Every readout reads the stream sum. Production regime: compute policy p=f32, c=bfloat16; 48 layers; width 512; DepthArchMuonHConfig with lr 0.02, Adam lr 0.008, no warmup; FA4_THD attention; 4 GPUs. The designs under audit, and the claim each must meet:

1. sumshrink (job arch3-hcshrink): H_o <- H_o - (1 - g) mean_s H_s + w_o F, g = HC_SUM_GAIN[2 l + j]. The stream sum has gain g; every difference between streams has gain 1. HC_SUM_GAIN must be the mean row sum of I + res of HC's ordinary model at step 4,958, per sublayer, in the order 2 l + j. Source data: `/scratch/gpfs/GROUP/USER/tmp/pls_hcid/sumfail_data.json`, key mix.HC.rows, written by `sumfail_data.py`.
2. half dose (arch3-hcshrink05): g ** 0.5.
3. move (arch3-hcmove): every row of the mixing is non-negative and sums to 1 over stream_out, on the whole mixing (static plus dynamic). The stream sum stays. At init this is HC's function.
4. Pre-LN leak (arch3-plleak): h <- g h + F with the same gains; the layer index must be the real one in every code path, in training and in evaluation.
5. float32 streams (arch3-hcsumf32): with depth_hc_res="sum", only the streams are float32; the branch and the readouts run in the compute dtype.

## What to check

1. Axis conventions: is the row of `res` really the outflow of an input stream, so that a row sum of 1 keeps the stream sum? Check against `hax.dot(res, H, axis="stream")`. Compare hc_res="sum" too.
2. The sublayer index: traced in the scan, Python int elsewhere. Are there code paths where `layer` is None or wrong (evaluation, layer_outputs, per-layer probes, the non-scanned path, the Pre-LN path in `outputs`)? `SCANNED` and what it holds for preln.
3. Dtypes under the real policy, including what the optimizer sees, the scan carry staying a stable dtype, and `mp.cast_to_param` on gradients. Any place a float32 carry reaches a kernel that requires bfloat16, or a bfloat16 value gets written back into the carry.
4. How new code meets existing infrastructure: optimizer labels for `res` and `w_res` when they get zero gradient (Adam with eps 1e-8, and 130m's eps 1e-20 for other labels; NaN risk with zero gradient and zero second moment), MuonH's norm projection, gradient clipping, checkpoint save and resume, the W&B config, `PerLayerQwen3Config` serialization and draccus parsing of the new fields, run name and checkpoint path tags (no collision with an existing run, so that nothing resumes from the wrong checkpoint), `PARAM_MAP=vocab` and PDP settings.
5. Launcher: does each job case in `pls_arch3_fetchable.sbatch` actually reach the model config? `pls_h100x8_single.sbatch` line 32 unsets a list of variables and the job passes settings through `SINGLE`: confirm HC_RES, SHRINK_POWER, PRELN_LEAK and HC_STATE_F32 get through, and that the smoke copy uses the same settings. This is a likely failure point: a variable unset or dropped means a job silently runs another design.
6. Eval path: evaluation under `pls_eval`; does it use the same forward path and dtype as training? Does anything other than `outputs` build the streams (for example `hax.stack` of the embedding elsewhere, a layer_outputs copy, or scripts/della/pls_*.py analyses)?
7. Numerics: softplus at large negative input, the 1e-6 guard, gain > 1 at late layers, float32 overflow, `_rms` on large float32 states.
8. Memory: estimate the extra GPU memory of arch3-hcsumf32 from how gradient checkpointing and the scan store the carry. Does a float32 carry likely fit a 4-GPU run with XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 on 80 GB H100s? Batch 128, sequence 4096, 4 streams, width 512, 48 layers. State your assumptions.
9. Do the tests test what they claim? Point out any assertion that cannot fail or any test that runs a friendlier regime than production.

## Report

Sections: Verdict per design (OK, or BLOCKER with the fix). Findings ranked by severity, each with file:line, the concrete failure and how you confirmed it (say whether you ran it or only read it). Things you checked and found correct, one line each. Do not pad. Say plainly what you could not check.
