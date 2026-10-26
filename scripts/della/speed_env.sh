# Speed settings of the 300m runs (sourced by muonh_qwen3_h100x4.sbatch and the smokes, 2026-10-26), so a smoke runs the
# main run's code paths: until now the smokes ran python without these, so a smoke could pass while the main run crashed
# (the GIF fallback bug). Each knob keeps an environment override; 0 gives the old path.

# XLA_CMD_BUFFER_WHILE=1 (speed, 2026-10-24; default 1 since 2026-10-26 after the H100 A/B w203/w204, 0 = the old launch):
# XLA also captures while/conditional loops in CUDA graphs. The JAX_FLASH inner causal loops have data-dependent trip
# counts; without this each iteration copies its predicate to the host and relaunches its kernels (~55 us per iteration in
# the charwin logs: 20K-34K iterations per step at flash block 128). Same kernels in the same order, so the same math. The
# NCCL watchdog flag is repeated here because marin's training env adds it only when XLA_FLAGS is unset
# (os.environ.setdefault), so the job keeps it.
if [ "${XLA_CMD_BUFFER_WHILE:-1}" = 1 ]; then
  case " ${XLA_FLAGS:-} " in *xla_gpu_nccl_termination_timeout_seconds*) ;; *) XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_nccl_termination_timeout_seconds=600" ;; esac
  case " ${XLA_FLAGS} " in *xla_gpu_enable_command_buffer*) ;; *) XLA_FLAGS="${XLA_FLAGS} --xla_gpu_enable_command_buffer=+WHILE,+CONDITIONAL" ;; esac
  export XLA_FLAGS
  echo "XLA_CMD_BUFFER_WHILE=1: XLA_FLAGS=$XLA_FLAGS"
fi
# LEVANTER_CKPT_DEADLINE (speed, 2026-10-25; levanter commit from the pls session): one temporary checkpoint 150 s before
# this job's end, SLURM_JOB_END_TIME (Slurm's own, or the host's end for a wq guest), so a resumable run that the end stops
# redoes almost no work on resume. No end time: no effect.
if [ -n "${SLURM_JOB_END_TIME:-}" ]; then
  export LEVANTER_CKPT_DEADLINE=$((SLURM_JOB_END_TIME - 150))
  echo "LEVANTER_CKPT_DEADLINE=$LEVANTER_CKPT_DEADLINE ($(date -d @$LEVANTER_CKPT_DEADLINE +%F' '%T))"
fi
# LEVANTER_CE_BTILED_ONEHOT=1 (speed, 2026-10-25; the pls session's CE commit, bitwise equal to the scatter on H100 in its
# deterministic A/B): the cross-entropy label term as a select instead of a scatter, so XLA fuses the dlogits pass. The
# same additions on the same values: every run's numbers are unchanged.
export LEVANTER_CE_BTILED_ONEHOT=${LEVANTER_CE_BTILED_ONEHOT:-1}
# LEVANTER_CE_GRAD_IN_FWD=1 (speed, 2026-10-26; the pls session's CE commit): the fused cross-entropy computes its
# gradients in the forward from the same logits GEMM per block (3 logits GEMMs instead of 4). Not bitwise: H100 det round
# w245 at 300m 128K gives per-step loss within 1.9e-6 of the base over 58 steps, far inside the run-to-run spread of
# GEMM autotuning (pls w205 vs w229: 0.076). -4.7% step time on top of A1 under FA4 (w227). Off from 02:51 to 04:20 on
# 2026-10-26: its fallback crashed under shard_map for shards off the b-tiled path (every vocabulary below 65536). The pls
# fix (6935700283) runs those shards through the ordinary fused CE; H100 det check w256 at 8K: bitwise equal to GIF off.
export LEVANTER_CE_GRAD_IN_FWD=${LEVANTER_CE_GRAD_IN_FWD:-1}
# LEVANTER_CE_LSE_KERNEL=1 (speed, 2026-10-26; Triton kernel by this session, integrated by the pls session b3f40f216f):
# in the gradient-in-forward CE block (vocabularies >= 65536), one Triton GEMM writes the bf16 logits, each row's f32 label
# logit and per-tile logsumexp partials, so no f32 logits reach HBM and the two lse passes go away. Not bitwise (GEMM and
# lse merge summation order; lse within 5.8e-8 of float64). H100 at 300m 128K: -5.9% (w280: 1213 -> 1142 ms), loss within
# bf16 noise. No effect below 65536.
export LEVANTER_CE_LSE_KERNEL=${LEVANTER_CE_LSE_KERNEL:-1}

# LEVANTER_ASYNC_STEP=1 (speed, owner popup 2026-10-26): Trainer.training_steps dispatches step N+1 before it reads step
# N's loss and runs its hooks, so the host's work between steps (hooks, logging, the next dispatch) overlaps the devices'
# work instead of leaving them idle; the train step keeps its input state instead of donating it (one more copy of the
# trainer state, ~1 GB per GPU at 300m) and a NaN loss stops the run one step later. Same computation: losses within
# noise. 4 H100s, 300m, wall per step: 128K 558.5 -> 523.7 ms (-6.2%, w331), 8K 437.6 -> 408.2 ms (-6.7%, w332); full
# evaluations and checkpoints every 20 steps pass (w325). Not with REMAT=fdots: at 128K the next step's 37 GiB buffer
# did not fit beside the running step's (w324, w325).
export LEVANTER_ASYNC_STEP=${LEVANTER_ASYNC_STEP:-1}
# LEVANTER_EMBED_GRAD_F32=1 (speed and numerics, owner popup 2026-10-26, no baseline reruns): the input embedding's
# gradient is summed in f32 and cast once, in place of a bf16 scatter-add that rounds once per occurrence of a token
# (17-20% relative error on the most frequent rows of 131072 Zipf ids, w327). Changes training: the train loss over the
# first 200 steps is 0.002 nats lower at 128K and 0.0013 at 8K, so runs that start after 2026-10-26 12:17 are not bitwise
# comparable to earlier ones. -1.7% (128K) and -1.3% (8K) per step on top of the above.
export LEVANTER_EMBED_GRAD_F32=${LEVANTER_EMBED_GRAD_F32:-1}
echo "speed_env: CE onehot=$LEVANTER_CE_BTILED_ONEHOT gif=$LEVANTER_CE_GRAD_IN_FWD lse_kernel=$LEVANTER_CE_LSE_KERNEL async_step=$LEVANTER_ASYNC_STEP embed_grad_f32=$LEVANTER_EMBED_GRAD_F32 ATTN_BACKEND=${ATTN_BACKEND:-code default: fa4thd at 130m/300m, fa4 above} REMAT=${REMAT:-flash (code default)}"
