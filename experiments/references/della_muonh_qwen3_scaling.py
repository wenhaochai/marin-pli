# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Kaiyue Wen's ``muonh_qwen3_scaling`` speedrun baselines on Della 4xH100.

Reproduces the released Qwen3 (LLaMA-geometry-matched) MuonH runs at 130m, 300m, 520m and 1_2b (W&B
``marin-community/marin`` ``qwen3_<size>_muonh_4096_*``, 2025-10-12, 64 TPU devices). Shapes, batch, steps and
optimizer values are transcribed from those runs' logged configs. Current main matches their training math except
for two defaults set explicitly here: the Newton-Schulz coefficients (main switched to per-iteration quintic
coefficients in 2026-01) and the data order (the originals used a full linear permutation with data seed 42; main
defaults to a Feistel block shuffle). Paloma is tokenized with the training tokenizer, as in the originals.
SMOKE_STEPS turns a run into a short smoke test with its own run id and output, evaluating one batch at the end.
VARIANT=fbt swaps the model for the full-bandwidth transformer (experiments.references.full_bandwidth_qwen3, two
fixed passes) with everything else unchanged; its run ids end in ``-fbt2``. VARIANT=twin | sr | twinsr add the non-NTP
auxiliary objectives of experiments.references.objective_qwen3 (Twin-Networks state matching, successor-representation
TD head); run ids end in ``-twin<w>`` / ``-sr<gamma>w<w>``. VARIANT=ss trains with the sampled softmax of
experiments.references.sampled_softmax_qwen3 (per-device candidate sets, full softmax at the end, full-softmax eval);
SS_SCHEDULE sets its stages and the run id ends in ``-ss<P/1024>k<fraction>...``. VARIANT=ov is the Over-Tokenized
Transformer's over-encoding + over-decoding (experiments.references.over_vocab_qwen3: hashed 2-/3-gram input embeddings,
OV_M rows per table; a 2-gram output vocabulary in the paper's product decomposition, weight OV_OD_W); VARIANT=ovss adds
the sampled softmax to both output heads. VARIANT=pls adds every layer's LM loss through the shared final norm and lm_head
(experiments.references.per_layer_qwen3), weight PLS_W, and evaluates every layer's readout (eval/L{k}/...); PLS_W=0 is
the baseline's training with per-layer monitoring. Run ids end in ``-pls<w>``.

    SIZE=130m python -m experiments.references.della_muonh_qwen3_scaling        # DRY_RUN=1 prints the plan
"""

import dataclasses
import os
from datetime import timedelta

import jmp
from fray.cluster import ResourceConfig
from haliax.partitioning import ResourceAxis
from haliax.quantization import QuantizationConfig
from levanter.optim.model_averaging import EmaModelAveragingConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.layers.attention import AttentionBackend
from levanter.main import train_lm
from levanter.models.qwen import Qwen3Config
from levanter.optim.muonh import MuonHConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.execution.lazy import ArtifactStep, StepContext, lower, run
from marin.execution.remote import remote
from marin.experiment.data import mixture
from marin.training import training as _training
from marin.training.training import LevanterCheckpoint, TrainLmOnPodConfig, resolve_training_env, run_levanter_train_lm

from experiments.datasets.paloma import paloma_datasets
from experiments.datasets.prebuilt_caches import fineweb_edu_10B_dataset
from experiments.marin_tokenizer import marin_tokenizer
from experiments.references.full_bandwidth_qwen3 import FullBandwidthQwen3Config
from experiments.references.objective_qwen3 import ObjectiveQwen3Config
from experiments.references.over_vocab_qwen3 import ROWS as OV_ROWS
from experiments.references.over_vocab_qwen3 import OverVocabQwen3Config
from experiments.references.per_layer_qwen3 import PerLayerQwen3Config
from experiments.references.sampled_softmax_qwen3 import SampledSoftmaxQwen3Config

# With a local MARIN_PREFIX the temporary checkpoint base comes back as a file:// URL, which the tensorstore
# writer treats as a relative path (arrays land in <cwd>/file:/...), so resume finds only metadata.json.
_temp_ckpt_base = _training.temporary_checkpoint_base_path
_training.temporary_checkpoint_base_path = lambda output_path: _temp_ckpt_base(output_path).removeprefix("file://")

if "della-proxy" in os.environ.get("https_proxy", ""):
    # Della's compute-node proxy passes api.wandb.ai but not storage.googleapis.com: file/artifact uploads
    # (levanter logs requirements.txt as an artifact) hang run.finish(). Metrics still stream.
    os.environ.update(WANDB_IGNORE_GLOBS="*", WANDB_DISABLE_CODE="true", WANDB_CONSOLE="off", WANDB_X_DISABLE_META="true")
    import wandb.sdk.wandb_run

    wandb.sdk.wandb_run.Run.log_artifact = lambda self, *args, **kwargs: None

# On GPU the fused cross-entropy sweeps 7 block-size candidates on a cache miss, and here the result cannot be cached
# (its kernel jaxpr is unavailable), so every run and resume segment re-sweeps: ~10 min at 1_2b before the first step.
# Block sizes only change tiling, not the loss, so use the inferred sizes.
os.environ.setdefault("LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS", "0")

VERSION = "2026.09.13"
NUM_GPUS = 4
SEQ_LEN = 4096
SMOKE_STEPS = int(os.environ.get("SMOKE_STEPS", "0"))
VARIANT = os.environ.get("VARIANT", "baseline")  # baseline | fbt | ss | ov | ovss | pls | twin | sr | twinsr | ...
DEVICE_TAG = os.environ.get("DEVICE_TAG", "h100")  # run ids carry the GPU type so an A100 copy is a separate run
# PRECISION=fp8 quantizes every Linear to FP8 (delayed scaling, E4M3 forward / E5M2 gradients); H100 only.
PRECISION = os.environ.get("PRECISION", "bf16")  # bf16 | fp8
FEEDBACK_PASSES = 2
# FBT ablations (each is one change from the fixed-two-pass FBT run): FBT_NOISE=0 drops the jitter noise, TIE=1 ties
# the embedding and LM head (paper recipe; also applies to a baseline run), FBT_RESIDUAL=1 uses the residual form with
# a zero-initialised scalar gate. INIT_FROM=<levanter checkpoint dir> continues pretraining from another run's
# checkpoint (full-state restore: step, optimizer state, data position; leaves the checkpoint lacks, such as the FBT
# parameters, keep their fresh init) with TOTAL_STEPS as the new schedule length; CPT_TAG names the run.
FBT_NOISE = float(os.environ.get("FBT_NOISE", "0.02"))
TIE = os.environ.get("TIE", "0") == "1"
FBT_RESIDUAL = os.environ.get("FBT_RESIDUAL", "0") == "1"
# Readout variants (all need FBT_RESIDUAL=1): FBT_LAYERWISE=1 adds per-layer aligned injection, FBT_INPUT=0 drops the
# input-level term (layerwise only), FBT_INPUT_LAYERS=k makes the input-level term read a softmax mix of the last k layers.
FBT_LAYERWISE = os.environ.get("FBT_LAYERWISE", "0") == "1"
FBT_INPUT = os.environ.get("FBT_INPUT", "1") == "1"
FBT_INPUT_LAYERS = int(os.environ.get("FBT_INPUT_LAYERS", "1"))
FBT_ALPHA_MULT = float(os.environ.get("FBT_ALPHA_MULT", "1"))  # Adam-invariant LR multiplier for the scalar gates
# Objective hill-climb (experiments.references.objective_qwen3): VARIANT=twin adds the Twin-Networks backward model and
# state-matching penalty (TWIN_W weight, TWIN_OFF offset), VARIANT=sr the successor-representation TD head (SR_W weight,
# SR_GAMMA discount), VARIANT=twinsr both. The forward model, data, batch, schedule and optimizer stay the baseline's.
OBJECTIVE_VARIANTS = ("twin", "sr", "twinsr", "pi", "eos", "ebm", "dn", "mtp", "swap")
TWIN_W = float(os.environ.get("TWIN_W", "0.1"))
TWIN_OFF = int(os.environ.get("TWIN_OFF", "2"))
SR_W = float(os.environ.get("SR_W", "0.1"))
SR_GAMMA = float(os.environ.get("SR_GAMMA", "0.9"))
# VARIANT=pi: predictive-information InfoNCE between h_t and h_{t+PI_K} (PI_W weight, PI_TAU temperature, PI_NEG negatives).
PI_W = float(os.environ.get("PI_W", "0.1"))
PI_K = int(os.environ.get("PI_K", "4"))
PI_TAU = float(os.environ.get("PI_TAU", "0.1"))
PI_NEG = int(os.environ.get("PI_NEG", "512"))
# VARIANT=eos: 13-way log2-binned cross-entropy on the distance to the current document's EOS (EOS_W weight).
EOS_W = float(os.environ.get("EOS_W", "0.1"))
# VARIANT=ebm: binary NCE between the clean prefix and a copy corrupted by the model's own next-token samples (EBM_W weight,
# EBM_RHO max per-sequence corruption rate); second trunk pass, scalar head on unit-RMS states.
EBM_W = float(os.environ.get("EBM_W", "0.1"))
EBM_RHO = float(os.environ.get("EBM_RHO", "0.5"))
# EBM_TEMP: sampling temperature of the corruption samples (tag -T{t} when != 1). DENOISE_W > 0 adds denoising NTP on the
# same corrupted copy (tag -dn{w}); VARIANT=dn runs denoising NTP alone (default weight 0.1), no energy head.
EBM_TEMP = float(os.environ.get("EBM_TEMP", "1"))
# EBM_STEPS=k > 1: the ebm/dn corruption is a k-step autoregressive rollout of the model's own continuation in aligned
# blocks of k instead of independent one-step draws (tag k{k} after the ebm/dn tag). 1 = the original one-step path.
EBM_STEPS = int(os.environ.get("EBM_STEPS", "1"))
DENOISE_W = float(os.environ.get("DENOISE_W", "0"))
# ADV_W > 0 (ebm only): REINFORCE on the sampler with the ebm head's score as reward, weight ADV_W (tag -adv{w}).
ADV_W = float(os.environ.get("ADV_W", "0"))
# VARIANT=mtp: multi-token prediction auxiliary (D x D projection + shared lm_head) predicting x_{t+MTP_K}, weight MTP_W.
MTP_W = float(os.environ.get("MTP_W", "0.1"))
MTP_K = int(os.environ.get("MTP_K", "2"))
# VARIANT=swap: real-text-wrong-context negatives -- SWAP_SPANS spans of length U(SWAP_MIN, SWAP_MAX) per sequence spliced in
# from the previous sequence in the batch; scalar head, binary NCE, second trunk pass (weight SWAP_W). Tag -sww{w}[n{spans}][l{min}-{max}].
SWAP_W = float(os.environ.get("SWAP_W", "0.1"))
SWAP_SPANS = int(os.environ.get("SWAP_SPANS", "1"))
SWAP_MIN = int(os.environ.get("SWAP_MIN", "16"))
SWAP_MAX = int(os.environ.get("SWAP_MAX", "128"))
# FREE_HEADS=1 (default): auxiliary heads are plain arrays in MuonH's adam group (norm free). FREE_HEADS=0 reproduces the
# 2026-09-18 pinned-head runs (hnn.Linear heads that MuonH keeps at their init norm). Run ids carry -fh when free.
FREE_HEADS = os.environ.get("FREE_HEADS", "1") == "1"
# AUX_LAYER=-1 (default): auxiliary heads read the final normed h. AUX_LAYER=k: they read the RMS-normalised residual
# stream after layer k, so the top layers / final norm / lm_head operating point stay NTP's alone (run tag -L{k}).
AUX_LAYER = int(os.environ.get("AUX_LAYER", "-1"))
# AUX_GATE="hi:lo" anneals the auxiliary weight on the NTP loss level: gate = clip((L_ntp - lo)/(hi - lo), 0, 1); empty = off.
# 130m baseline train loss: 4.0 ~ step 500, 3.6 ~ step 2400, 3.25 at the end. Run tag -g{hi}-{lo}.
AUX_GATE = os.environ.get("AUX_GATE", "")
AUX_GATE_HI, AUX_GATE_LO = (tuple(float(v) for v in AUX_GATE.split(":")) if AUX_GATE else (0.0, 0.0))
# SEED (default 0 = the baseline's): trainer init seed for seed replicates; data order stays data_seed=42. Run tag -s{SEED}.
SEED = int(os.environ.get("SEED", "0"))
# LAYERS (default 0 = the size's own depth): a shallower or deeper model at the same width, batch, steps and schedule,
# e.g. the depth-matched controls for the per-layer readouts of VARIANT=pls. Run tag -d{LAYERS}.
LAYERS = int(os.environ.get("LAYERS", "0"))
# LR_MUL (default 1) scales both peak learning rates (MuonH and its Adam group) of the size's recipe; WARMUP (default: the
# size's own) sets the warmup steps. For depth probes (2026-10-05: at LAYERS=48 the 130m recipe stalls near 6.6 nats).
# Run tags -lrm{LR_MUL}, -wu{WARMUP}.
LR_MUL = float(os.environ.get("LR_MUL", "1"))
WARMUP = int(os.environ["WARMUP"]) if os.environ.get("WARMUP") else None
if LR_MUL <= 0 or (WARMUP is not None and WARMUP < 0):
    raise ValueError(f"LR_MUL must be > 0 and WARMUP >= 0, got {LR_MUL}, {WARMUP}")
# EMA_BETA > 0 keeps an exponential moving average of the weights (levanter ModelAveraging) and evaluates it alongside the
# raw weights (eval/ema/...). Training is untouched; this is an evaluation-noise reduction (Polyak 1992). Tag -ema{beta}.
EMA_BETA = float(os.environ.get("EMA_BETA", "0"))
# RUN_TAG appends a literal suffix to the run id ONLY (output path and W&B name); it changes nothing about training.
# Its purpose is determinism probes: two runs with the same SEED and different RUN_TAGs are the identical computation
# under two names, so their difference measures the hardware/compiler nondeterminism floor.
RUN_TAG = os.environ.get("RUN_TAG", "")
# VARIANT=ss: sampled softmax for the training loss (nanoGPT speedrun record #92). SS_SCHEDULE = comma-separated
# "fraction:P" stages: P candidate classes per device until that fraction of the steps, then the next stage; the full
# softmax runs from the last fraction on. The default is record #92's schedule as fractions of the vocabulary (P / V =
# 0.19, 0.29, 0.51 against its 0.20, 0.28, 0.49) and of training (0.57, 0.81, 0.93). MFU keeps the full-vocabulary FLOP
# count, so it reads as baseline-equivalent throughput.
SS_SCHEDULE = tuple((float(f), int(p)) for f, p in (st.split(":") for st in os.environ.get("SS_SCHEDULE", "0.57:24576,0.81:36864,0.93:65536").split(",")))
# VARIANT=ov | ovss: OE-12.8M + OD (n = 2) of arXiv 2501.16975 (n = 3, k from d_model / (n k) ~ 256). OV_M = rows per
# hashed n-gram table, OV_OD_W = lambda_2 of over-decoding (0 turns it off). ovss also applies SS_SCHEDULE to both heads.
OV_M = int(float(os.environ.get("OV_M", "12.8e6")))
OV_OD_W = float(os.environ.get("OV_OD_W", "0.1"))
# VARIANT=pls: every layer's residual stream decoded by the shared final norm + lm_head and trained on NTP, all the time.
# PLS_W = weight of each intermediate layer's LM loss (the final layer's is 1); 0 = baseline training with stop-gradient
# per-layer readouts on every PLS_MON-th position for monitoring. PLS_EVAL=0 turns off the per-layer eval (eval/L{k}/...).
PLS_W = float(os.environ.get("PLS_W", "1"))
PLS_MON = int(os.environ.get("PLS_MON", "16"))
PLS_EVAL = os.environ.get("PLS_EVAL", "1") == "1"
# PLS_DETACH=1: the intermediate layers' losses do not update the shared final norm / lm_head (trunk only). Tag -dh.
PLS_DETACH = os.environ.get("PLS_DETACH", "0") == "1"
# PLS_SEP=1: every intermediate layer gets its own RMSNorm + lm_head, trained by its own loss only. Tag -sep.
PLS_SEP = os.environ.get("PLS_SEP", "0") == "1"
# PLS_DETACH_BB=1 (with PLS_SEP=1): the per-layer losses train their own heads only, the backbone trains as the baseline. Tag -bbfrozen.
PLS_DETACH_BB = os.environ.get("PLS_DETACH_BB", "0") == "1"
# PLS_LOCAL=1 (with PLS_SEP=1): each layer loss trains its own head and its own layer only. Tag -local.
PLS_LOCAL = os.environ.get("PLS_LOCAL", "0") == "1"
# PLS_SHAPE=depth: intermediate weights rise with depth, PLS_W * 2(k+1)/L, same sum as PLS_W uniform. Tag -shdepth.
# PLS_PCGRAD=1: per-token gradient surgery at every layer output (per_layer_qwen3.pls_pcgrad). Tag -pcg.
PLS_SHAPE = os.environ.get("PLS_SHAPE", "uniform")
PLS_PCGRAD = os.environ.get("PLS_PCGRAD", "0") == "1"
# PLS_OFF=E or S:E: the intermediate weight holds until step S, falls linearly to 0 at step E (E alone: a switch at E) and
# is 0 after, with the readouts skipped. Tag -off{E} or -off{S}-{E}.
PLS_OFF = os.environ.get("PLS_OFF", "")
PLS_OFF_START, PLS_OFF_END = (int(x) for x in (PLS_OFF.split(":", 1) if ":" in PLS_OFF else (PLS_OFF, PLS_OFF))) if PLS_OFF else (0, 0)
# PLS_PROBE=1: log per-layer gradient alignment and in-context scores every step (diagnostic runs; one extra forward and
# L backward passes per step). Tag -probe.
PLS_PROBE = os.environ.get("PLS_PROBE", "0") == "1"
# ARCH=<name> (VARIANT=pls): the residual design of every layer, one of depth_arch_qwen3.ARCHS (DepthBench, arXiv
# 2609.32534); empty = the baseline (hybrid_norm, i.e. Sandwich-LN, on the Stacked transformer). Tag -a{ARCH}.
ARCH = os.environ.get("ARCH", "")
if VARIANT not in ("baseline", "fbt", "ss", "ov", "ovss", "pls", *OBJECTIVE_VARIANTS):
    raise ValueError(f"unknown VARIANT={VARIANT!r}")
INIT_FROM = os.environ.get("INIT_FROM") or None
TOTAL_STEPS = int(os.environ["TOTAL_STEPS"]) if os.environ.get("TOTAL_STEPS") else None
CPT_TAG = os.environ.get("CPT_TAG", "-cpt") if INIT_FROM else ""
# HEADS_FROM=<levanter checkpoint dir> (VARIANT=pls with PLS_SEP=1; set the finished run's own PLS_* knobs): refit
# the readout heads of that run. The model loads its weights only (initialize_model_from_checkpoint_path: fresh
# optimizer, step 0, data from the start), every readout reads a stop-gradient layer output (pls_heads_only), and
# HeadsOnlyMuonHConfig freezes every other parameter. The heads keep their pretraining optimizer and peak learning
# rates on a new schedule of TOTAL_STEPS steps with HEADS_WARMUP warmup steps. Tag -headft{TOTAL_STEPS}.
HEADS_FROM = os.environ.get("HEADS_FROM") or None
HEADS_WARMUP = int(os.environ.get("HEADS_WARMUP", "100"))
if HEADS_FROM and (VARIANT != "pls" or not PLS_SEP or not TOTAL_STEPS or INIT_FROM or ARCH):
    raise ValueError("HEADS_FROM needs VARIANT=pls, PLS_SEP=1 and TOTAL_STEPS, without INIT_FROM or ARCH")
# Transcribed from the original runs' W&B configs; ref_c4_en_bpb is their final eval/paloma/c4_en/bpb.
SIZES = {
    "130m": dict(hidden=512, inter=1792, layers=6, heads=8, kv=8, batch=128, steps=4959, lr=0.02, adam_lr=0.008, eps=1e-20, momentum=0.95, schedule="linear", decay=0.8, warmup=0, max_grad_norm=1.0, ref_c4_en_bpb=1.16354),
    "300m": dict(hidden=768, inter=2688, layers=12, heads=12, kv=12, batch=128, steps=11444, lr=0.01, adam_lr=0.002, eps=1e-15, momentum=0.98, schedule="cosine", decay=None, warmup=1000, max_grad_norm=1.0, ref_c4_en_bpb=1.05602),
    "520m": dict(hidden=1024, inter=3584, layers=24, heads=16, kv=8, batch=256, steps=9918, lr=0.01, adam_lr=0.002, eps=1e-15, momentum=0.98, schedule="cosine", decay=None, warmup=1000, max_grad_norm=1.0, ref_c4_en_bpb=0.98824),
    "1_2b": dict(hidden=2048, inter=7168, layers=16, heads=16, kv=8, batch=256, steps=22888, lr=0.01, adam_lr=0.0015, eps=1e-15, momentum=0.98, schedule="cosine", decay=None, warmup=1000, max_grad_norm=2.0, ref_c4_en_bpb=0.92731),
}
# At 64 sequences per 80 GB GPU, 1_2b fails a 44.6 GiB allocation after a few steps (A100 smoke 13851002) and 520m a
# 35.5 GiB one at step ~115 (H100 13852139, after a 40-step smoke had passed); two microbatches per step keep the batch
# math and fit. 130m/300m run one microbatch as the originals did.
PER_DEVICE_PARALLELISM = {"520m": 32, "1_2b": 32}


def _run_size(config: TrainLmOnPodConfig) -> None:
    env = resolve_training_env(config.env_vars, config.resources)
    remote(run_levanter_train_lm, resources=config.resources, env_vars=env)(dataclasses.replace(config, env_vars=env))


def muonh_qwen3_run(size: str) -> ArtifactStep[LevanterCheckpoint]:
    s = SIZES[size]
    variant_tags = (f"-fbt{FEEDBACK_PASSES}" if VARIANT == "fbt" else "") + ("-fp8" if PRECISION == "fp8" else "") + ("-tied" if TIE else "")
    if VARIANT == "fbt":
        variant_tags += ("-nonoise" if FBT_NOISE == 0 else "") + ("-res" if FBT_RESIDUAL else "")
        variant_tags += ("-lw" if FBT_LAYERWISE else "") + ("-noin" if not FBT_INPUT else "") + (f"-ml{FBT_INPUT_LAYERS}" if FBT_INPUT_LAYERS > 1 else "")
        variant_tags += f"-am{FBT_ALPHA_MULT:g}" if FBT_ALPHA_MULT != 1 else ""
    # Mirrored by the SUFFIX case in scripts/della/muonh_qwen3_smoke.sbatch; pass weights in their shortest form (0.1, not 0.10).
    if VARIANT in ("twin", "twinsr"):
        variant_tags += f"-twin{TWIN_W:g}" + (f"o{TWIN_OFF}" if TWIN_OFF != 2 else "")
    if VARIANT in ("sr", "twinsr"):
        variant_tags += f"-sr{SR_GAMMA:g}w{SR_W:g}"
    if VARIANT == "pi":
        variant_tags += f"-pik{PI_K}w{PI_W:g}" + (f"t{PI_TAU:g}" if PI_TAU != 0.1 else "") + (f"n{PI_NEG}" if PI_NEG != 512 else "")
    if VARIANT == "eos":
        variant_tags += f"-eosw{EOS_W:g}"
    if VARIANT == "ebm":
        variant_tags += f"-ebmr{EBM_RHO:g}w{EBM_W:g}" + (f"T{EBM_TEMP:g}" if EBM_TEMP != 1 else "") + (f"k{EBM_STEPS}" if EBM_STEPS != 1 else "") + (f"-adv{ADV_W:g}" if ADV_W > 0 else "")
    if VARIANT == "mtp":
        variant_tags += f"-mtpk{MTP_K}w{MTP_W:g}"
    if VARIANT == "swap":
        variant_tags += f"-sww{SWAP_W:g}" + (f"n{SWAP_SPANS}" if SWAP_SPANS != 1 else "") + (f"l{SWAP_MIN}-{SWAP_MAX}" if (SWAP_MIN, SWAP_MAX) != (16, 128) else "")
    if VARIANT in ("ov", "ovss"):
        variant_tags += f"-ov{OV_M / 1e6:g}m" + (f"od{OV_OD_W:g}" if OV_OD_W != 0.1 else "")
    if VARIANT == "pls":
        variant_tags += f"-pls{PLS_W:g}" + (f"m{PLS_MON}" if PLS_W == 0 and PLS_MON != 16 else "") + ("-dh" if PLS_DETACH else "") + ("-sep" if PLS_SEP else "") + ("-bbfrozen" if PLS_DETACH_BB else "") + ("-local" if PLS_LOCAL else "") + ("-shdepth" if PLS_SHAPE == "depth" else "") + ("-pcg" if PLS_PCGRAD else "") + ((f"-off{PLS_OFF_END}" if PLS_OFF_START == PLS_OFF_END else f"-off{PLS_OFF_START}-{PLS_OFF_END}") if PLS_OFF_END else "") + ("-probe" if PLS_PROBE else "") + ("" if PLS_EVAL else "-noev") + (f"-a{ARCH}" if ARCH else "")
    if VARIANT in ("ss", "ovss"):
        variant_tags += "-ss" + "-".join(f"{p // 1024}k{f:g}".replace("k0.", "k.") if p % 1024 == 0 else f"{p}p{f:g}".replace("p0.", "p.") for f, p in SS_SCHEDULE)
    if VARIANT == "dn":
        variant_tags += f"-dnr{EBM_RHO:g}w{DENOISE_W or 0.1:g}" + (f"T{EBM_TEMP:g}" if EBM_TEMP != 1 else "") + (f"k{EBM_STEPS}" if EBM_STEPS != 1 else "")
    elif VARIANT in OBJECTIVE_VARIANTS and DENOISE_W > 0:
        variant_tags += f"-dn{DENOISE_W:g}"
    if VARIANT in OBJECTIVE_VARIANTS and AUX_GATE:
        variant_tags += f"-g{AUX_GATE_HI:g}-{AUX_GATE_LO:g}"
    if VARIANT in OBJECTIVE_VARIANTS and AUX_LAYER >= 1:
        variant_tags += f"-L{AUX_LAYER}"
    if VARIANT in OBJECTIVE_VARIANTS and FREE_HEADS:
        variant_tags += "-fh"
    if EMA_BETA > 0:
        variant_tags += f"-ema{EMA_BETA:g}"
    if LAYERS:
        variant_tags += f"-d{LAYERS}"
    if SEED != 0:
        variant_tags += f"-s{SEED}"
    if LR_MUL != 1:
        variant_tags += f"-lrm{LR_MUL:g}"
    if WARMUP is not None:
        variant_tags += f"-wu{WARMUP}"
    if HEADS_FROM:
        variant_tags += f"-headft{TOTAL_STEPS}"
    run_id = f"muonh-qwen3-{size}-della4x{DEVICE_TAG}" + variant_tags + CPT_TAG + RUN_TAG + (f"-smoke{SMOKE_STEPS}" if SMOKE_STEPS else "")
    train = {fineweb_edu_10B_dataset(): 1.0}
    validation = list(paloma_datasets(tokenizer=marin_tokenizer).values())
    if VARIANT == "fbt":
        model_cls, model_extra = FullBandwidthQwen3Config, dict(
            feedback_passes=FEEDBACK_PASSES,
            feedback_noise=FBT_NOISE,
            feedback_residual=FBT_RESIDUAL,
            feedback_layerwise=FBT_LAYERWISE,
            feedback_input=FBT_INPUT,
            feedback_input_layers=FBT_INPUT_LAYERS,
            feedback_alpha_lr_mult=FBT_ALPHA_MULT,
        )
    elif VARIANT in OBJECTIVE_VARIANTS:
        model_cls, model_extra = ObjectiveQwen3Config, dict(
            twin=VARIANT in ("twin", "twinsr"),
            twin_weight=TWIN_W,
            twin_offset=TWIN_OFF,
            sr=VARIANT in ("sr", "twinsr"),
            sr_weight=SR_W,
            sr_gamma=SR_GAMMA,
            pi=VARIANT == "pi",
            pi_weight=PI_W,
            pi_k=PI_K,
            pi_tau=PI_TAU,
            pi_negatives=PI_NEG,
            eos=VARIANT == "eos",
            eos_weight=EOS_W,
            ebm=VARIANT == "ebm",
            ebm_weight=EBM_W,
            ebm_rho_max=EBM_RHO,
            ebm_temp=EBM_TEMP,
            ebm_steps=EBM_STEPS,
            denoise=VARIANT == "dn" or DENOISE_W > 0,
            denoise_weight=DENOISE_W or 0.1,
            adv=VARIANT == "ebm" and ADV_W > 0,
            adv_weight=ADV_W or 0.03,
            mtp=VARIANT == "mtp",
            mtp_weight=MTP_W,
            mtp_k=MTP_K,
            swap=VARIANT == "swap",
            swap_weight=SWAP_W,
            swap_spans=SWAP_SPANS,
            swap_min=SWAP_MIN,
            swap_max=SWAP_MAX,
            free_heads=FREE_HEADS,
            aux_layer=AUX_LAYER,
            aux_gate_hi=AUX_GATE_HI,
            aux_gate_lo=AUX_GATE_LO,
        )
    elif VARIANT == "ss":
        num_steps = SMOKE_STEPS or TOTAL_STEPS or s["steps"]
        model_cls, model_extra = SampledSoftmaxQwen3Config, dict(
            ss_candidates=tuple(p for _, p in SS_SCHEDULE),
            ss_stage_ends=tuple(round(f * num_steps) for f, _ in SS_SCHEDULE),
        )
    elif VARIANT in ("ov", "ovss"):
        num_steps = SMOKE_STEPS or TOTAL_STEPS or s["steps"]
        model_cls, model_extra = OverVocabQwen3Config, dict(
            oe_m=OV_M,
            od_weight=OV_OD_W,
            ss_candidates=tuple(p for _, p in SS_SCHEDULE) if VARIANT == "ovss" else (),
            ss_stage_ends=tuple(round(f * num_steps) for f, _ in SS_SCHEDULE) if VARIANT == "ovss" else (),
        )
    elif VARIANT == "pls":
        model_cls, model_extra = PerLayerQwen3Config, dict(pls_weight=PLS_W, pls_monitor_stride=PLS_MON, pls_eval=PLS_EVAL, pls_detach_head=PLS_DETACH, pls_separate_heads=PLS_SEP, pls_detach_backbone=PLS_DETACH_BB, pls_local_heads=PLS_LOCAL, pls_weight_shape=PLS_SHAPE, pls_pcgrad=PLS_PCGRAD, pls_off_start=PLS_OFF_START, pls_off_end=PLS_OFF_END, pls_probe=PLS_PROBE, pls_heads_only=HEADS_FROM is not None, depth_arch=ARCH or "baseline", **({"scan_layers": False} if ARCH else {}))
    else:
        model_cls, model_extra = Qwen3Config, {}
    if TIE:
        model_extra = dict(model_extra, tie_word_embeddings=True)
    model = model_cls(
        max_seq_len=SEQ_LEN,
        hidden_dim=s["hidden"],
        intermediate_dim=s["inter"],
        num_layers=LAYERS or s["layers"],
        num_heads=s["heads"],
        num_kv_heads=s["kv"],
        hybrid_norm=True,
        # No Transformer Engine on Della, so the GPU default (NVTE) would fall back to unfused O(S^2) attention.
        attn_backend=AttentionBackend.JAX_FLASH,
        **model_extra,
    )
    # Designs that add parameters (hc, mhc, attnres*) give those their own Adam (depth_arch_qwen3.DepthArchMuonHConfig);
    # every other run, the baseline included, keeps MuonHConfig exactly.
    if ARCH in ("hc", "mhc", "attnres", "attnres_block"):
        from experiments.references.depth_arch_qwen3 import DepthArchMuonHConfig as _Opt
    elif HEADS_FROM:
        from experiments.references.per_layer_qwen3 import HeadsOnlyMuonHConfig as _Opt
    else:
        _Opt = MuonHConfig
    optimizer = _Opt(
        learning_rate=s["lr"] * LR_MUL,
        adam_lr=s["adam_lr"] * LR_MUL,
        beta1=0.9,
        beta2=0.98,
        epsilon=s["eps"],
        momentum=s["momentum"],
        nesterov=True,
        backend_steps=5,
        muon_epsilon=1e-5,
        max_grad_norm=s["max_grad_norm"],
        weight_decay=0.1,
        lr_schedule=s["schedule"],
        decay=s["decay"],
        warmup=HEADS_WARMUP if HEADS_FROM else (s["warmup"] if WARMUP is None else WARMUP),
        min_lr_ratio=0.0,
        # The originals ran the fixed (3.4445, -4.7750, 2.0315) iteration; main defaults to quintic coefficients.
        coefficient_type="simple",
    )

    def build_config(ctx: StepContext) -> TrainLmOnPodConfig:
        data = dataclasses.replace(mixture(ctx, train, validation=validation, shuffle=True), permutation_type="linear")
        inner = train_lm.TrainLmConfig(
            data=data,
            trainer=TrainerConfig(
                tracker=WandbConfig(
                    entity=os.environ.get("WANDB_ENTITY"),
                    project=os.environ.get("WANDB_PROJECT", "marin-della"),
                    group="muonh-qwen3-smoke" if SMOKE_STEPS else ("muonh-qwen3-fbt-della" if VARIANT == "fbt" else "muonh-qwen3-ss-della" if VARIANT == "ss" else "muonh-qwen3-ov-della" if VARIANT in ("ov", "ovss") else "muonh-qwen3-pls-della" if VARIANT == "pls" else "muonh-qwen3-objective-della" if VARIANT in OBJECTIVE_VARIANTS else "muonh-qwen3-fp8-della" if PRECISION == "fp8" else "muonh-qwen3-della"),
                    tags=["speedrun", "muonh", "qwen3", size, f"della4x{DEVICE_TAG}", "jax_flash", *([f"fbt{FEEDBACK_PASSES}"] if VARIANT == "fbt" else []), PRECISION, *[t for t in variant_tags.split("-") if t], *(["cpt"] if INIT_FROM else [])],
                ),
                initialize_from=INIT_FROM,
                allow_partial_checkpoint=INIT_FROM is not None,
                mp=jmp.get_policy("p=f32,c=bfloat16"),
                # FP8 replaces every Linear's dot_general with the delayed-scaling FP8 op (fwd e4m3, grads e5m2, f32 accumulate);
                # embeddings, norms, attention softmax and the loss stay in the bf16/f32 policy above.
                quantization=QuantizationConfig(fp8=True) if PRECISION == "fp8" else None,
                train_batch_size=s["batch"],
                per_device_parallelism=int(os.environ.get("PDP", PER_DEVICE_PARALLELISM.get(size, -1))),  # PDP=<n>: microbatch per device
                num_train_steps=SMOKE_STEPS or TOTAL_STEPS or s["steps"],
                steps_per_eval=SMOKE_STEPS or 1000,
                max_eval_batches=1 if SMOKE_STEPS else None,
                checkpointer=CheckpointerConfig(save_interval=timedelta(minutes=10), keep=[dict(every=10000)]),
                mesh=MeshConfig(
                    axes={"data": -1, "replica": 1, "model": 1},
                    compute_mapping={
                        "token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
                        "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
                        # ov: the hashed n-gram tables stay row-sharded in compute too (never all-gathered)
                        **({OV_ROWS: ResourceAxis.DATA} if VARIANT in ("ov", "ovss") else {}),
                    },
                    **({"param_mapping": {"embed": "data", OV_ROWS: "data"}} if VARIANT in ("ov", "ovss") else {}),
                ),
                seed=SEED,
                model_averaging=EmaModelAveragingConfig(beta=EMA_BETA) if EMA_BETA > 0 else None,
                allow_nondivisible_batch_size=True,
            ),
            train_seq_len=SEQ_LEN,
            initialize_model_from_checkpoint_path=HEADS_FROM,
            model=model,
            optimizer=optimizer,
            z_loss_weight=0.0,
            data_seed=42,
            # No HF export: compute nodes cannot reach the Hub, and loading the tokenizer-only marin-tokenizer repo through
            # AutoTokenizer needs a config.json lookup that fails offline. The levanter checkpoint is the artifact.
            hf_save_steps=None,
        )
        return TrainLmOnPodConfig(
            train_config=inner,
            resources=ResourceConfig.with_gpu(DEVICE_TAG.upper(), count=NUM_GPUS),
            output_path=ctx.output_path,
            env_vars={"RUN_ID": run_id},
        )

    return ArtifactStep(
        name=f"speedrun/{run_id}",
        version=VERSION,
        artifact_type=LevanterCheckpoint,
        run=remote(_run_size, resources=ResourceConfig.with_cpu()),
        build_config=build_config,
        deps=(*train, *validation),
    )


if __name__ == "__main__":
    step = muonh_qwen3_run(os.environ["SIZE"])
    if os.environ.get("DRY_RUN") == "1":
        spec = lower(step)
        print("output:", spec.override_output_path or spec.name)
        print("deps:", len(spec.deps))
    else:
        run(step)
