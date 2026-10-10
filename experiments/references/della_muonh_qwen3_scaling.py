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
the sampled softmax to both output heads; VARIANT=ovgramss is ovgram with the sampled softmax on its main head.

    SIZE=130m python -m experiments.references.della_muonh_qwen3_scaling        # DRY_RUN=1 prints the plan
"""

import dataclasses
import os
from datetime import timedelta

# 1_2b OV on 8 GPUs dies under XLA's default BFC allocator: ~27 GiB of persistent state plus one contiguous 25-30 GiB
# temp buffer per step, and a long-lived small block lands beside the temp until no contiguous block is left (ovss at
# step ~120 every time, ov at ~1,290). scripts/della/ov_alloc_diag.sbatch (job 14824921, 1.2B ovss, 300 steps): BFC
# OOMs at step 119; cuda_async with a preallocated pool runs through at +1.4% step time in the first stage (+4% over
# the run); cuda_async without preallocation +30%; vmm +17%. So unless the job set an allocator itself, 1_2b OV uses
# cuda_async with a pool sized to its temp (ov 30.2 GiB -> 0.8, ovss 25.1 GiB -> 0.75). It must run before JAX
# initialises its backend, which reads these variables once.
if os.environ.get("VARIANT") in ("ov", "ovss", "ovfocal", "ovgram", "ovgram3", "ovgramss") and os.environ.get("SIZE") == "1_2b" and "XLA_PYTHON_CLIENT_ALLOCATOR" not in os.environ:
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "cuda_async"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = {"ov": "0.8", "ovss": "0.75", "ovfocal": "0.8", "ovgram": "0.75", "ovgram3": "0.7", "ovgramss": "0.7"}[os.environ["VARIANT"]]
    print(f"1_2b OV allocator: cuda_async, memory fraction {os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']}", flush=True)

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
from experiments.references.composed_vocab_qwen3 import ComposedVocabQwen3Config
from experiments.references.focal_qwen3 import FocalQwen3Config
from experiments.references.over_vocab_qwen3 import ROWS as OV_ROWS
from experiments.references.over_vocab_qwen3 import OverVocabQwen3Config
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
# NUM_GPUS (default 4, the baselines' count) and PDP (sequences per device per microbatch) change only the data-parallel
# layout, not the training math: the global batch is fixed, so fewer sequences per microbatch means more gradient
# accumulation. Used by OV at 520m/1_2b, whose hashed n-gram tables need the memory. Run ids carry della{NUM_GPUS}x.
NUM_GPUS = int(os.environ.get("NUM_GPUS", "4"))
PDP = int(os.environ["PDP"]) if os.environ.get("PDP") else None
# SEQ_LEN / BATCH override the sequence length (default 4096) and sequences per step (default the size's); the step count
# scales so a run reads the same tokens (vocabulary Q6: a 3072-token window holds about the bytes of 4096 8K tokens;
# Q7: batch 96 gives 128K about the 8K run's step count). Run tags -sl<SEQ_LEN>, -b<BATCH>.
SEQ_LEN = int(os.environ.get("SEQ_LEN", "4096"))
BATCH = int(os.environ.get("BATCH", "0"))
SMOKE_STEPS = int(os.environ.get("SMOKE_STEPS", "0"))
VARIANT = os.environ.get("VARIANT", "baseline")  # baseline | fbt | ss | twin | sr | twinsr
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
# OV_FREEZE_AT > 0: the n-gram tables stop getting gradients from that step on (over_vocab_qwen3.oe_freeze_step). Run ids
# get -oefreeze<step>. With DATA_EPOCHS=2.4 at 300m, 4768 freezes them as the second pass over the subset starts.
OV_FREEZE_AT = int(os.environ.get("OV_FREEZE_AT", "0"))
# VARIANT=focal: the baseline with its cross-entropy replaced by focal loss (experiments.references.focal_qwen3);
# VARIANT=ovfocal: OV with focal loss on both of its heads. FOCAL_GAMMA is gamma; run ids end in -focal<gamma>.
FOCAL_GAMMA = float(os.environ.get("FOCAL_GAMMA", "1"))
# VARIANT=ovgram: OV with a real 2-gram output vocabulary in place of the product-decomposed over-decoding: (x_{t+1}, x_{t+2})
# hashed into OD_M classes, each with its own output embedding, scored jointly by a per-device sampled softmax
# (over_vocab_qwen3.hashed_od_loss). Run ids end in -odhash<OD_M/1e6>m.
OD_M = int(float(os.environ.get("OD_M", "12.8e6")))
# VARIANT=ovgram3: OV with real 2-gram AND 3-gram output vocabularies (od_orders (2, 3)), each its own hashed head at
# weight OV_OD_W; run ids end in -odhash<OD_M/1e6>m-o23.
# VARIANT=ovgramss: ovgram with SS_SCHEDULE on the main head too (the 2-gram head keeps its own per-device sampled
# softmax), the counterpart of ovss; run ids end in -odhash<OD_M/1e6>m-ss<...>.
OV_VARIANTS = ("ov", "ovss", "ovfocal", "ovgram", "ovgram3", "ovgramss")
# DATA_EPOCHS > 0 trains on a random subset of fineweb-edu-10B sized so the run makes that many passes over it: levanter
# shuffles (linear permutation, data seed 42), keeps the first steps / DATA_EPOCHS batches (max_train_batches) and
# restarts at its end. 1.2B makes 2.4 passes over the full 10.0B tokens; DATA_EPOCHS=2.4 reproduces that repetition at
# a smaller size, every other setting unchanged. Run tag -rep{epochs}.
DATA_EPOCHS = float(os.environ.get("DATA_EPOCHS", "0"))
# VOCAB_K > 0: the Marin tokenizer truncated to its K lowest-ranked tokens (experiments/references/small_vocab_tokenizer.py;
# specials at K..K+255), on fineweb-edu-10B and the Paloma sets converted to it (convert_small_vocab_cache.py). Fixed
# text: the step count grows by the token ratio of the converted training data, so a run reads the same text the same
# number of passes (with DATA_EPOCHS too). Compare runs by bits per byte. Run ids get -v<K/1000>k, or -bytes for
# K=256 (the 256 byte tokens, no merges).
VOCAB_K = int(os.environ.get("VOCAB_K", "0"))
# TPP: training tokens per parameter, as a multiple of the speedrun recipe's ~20 (owner, 2026-10-09: the overtrained
# regime). TPP=200 trains 10x the recipe's steps on the same data (levanter restarts a finished dataset, so all of
# fineweb-edu-10B is 2.6 passes at 130m), every other setting unchanged: the learning-rate schedule stretches with the
# steps. Evaluations every TPP/40 x as many steps (5000 at TPP=200). Run tag -tpp<TPP>.
TPP = float(os.environ.get("TPP", "20"))
if TPP < 20:
    raise ValueError(f"TPP={TPP}: 20 (the recipe) or more")
if VOCAB_K and VOCAB_K < 1000 and VOCAB_K != 256:
    raise ValueError(f"VOCAB_K={VOCAB_K}: 256 (bytes, run tag -bytes) or a multiple of 1000 (tag -v<K/1000>k)")
# VARIANT=cv: the Marin tokenizer with its input embedding (CV_SIDE=in), output head (out) or both composed from the rows of
# its CV_K-token truncation (composed_vocab_qwen3.py; vocabulary Q4 and Q5). Run tag -cv<side><CV_K/1000>k.
# CV_OUT_REDUCE=sum: an output row is the sum of its pieces' rows, not their mean (the mean caps a token's logit at its
# largest piece's); run tag -sum.
CV_SIDE = os.environ.get("CV_SIDE", "in")
CV_OUT_REDUCE = os.environ.get("CV_OUT_REDUCE", "mean")
if CV_OUT_REDUCE not in ("mean", "sum"):
    raise ValueError(f"CV_OUT_REDUCE={CV_OUT_REDUCE!r}: mean or sum")
if CV_OUT_REDUCE != "mean" and (os.environ.get("VARIANT") != "cv" or CV_SIDE == "in"):
    raise ValueError("CV_OUT_REDUCE=sum needs VARIANT=cv with a composed output head (CV_SIDE=out or both)")
CV_K = int(os.environ.get("CV_K", "8000"))
if CV_SIDE not in ("in", "out", "both"):
    raise ValueError(f"CV_SIDE={CV_SIDE!r}: in, out or both")
_PREFIX = os.environ.get("MARIN_PREFIX", "/scratch/gpfs/GROUP/USER/marin_store_big")


def _vocab_ratio(k: int) -> float:
    """Converted / original training tokens of fineweb-edu-10B for the K-token tokenizer (written by the conversion)."""
    import json
    path = f"{_PREFIX}/tokenized/convert_v{k}.json"
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} is missing: run scripts/della/convert_small_vocab.sbatch for K={k} first")
    return json.load(open(path))["fineweb-edu-10B"]["ratio"]


def _small_vocab_data(k: int, train_names, validation_names):
    """The run's data config with every component pointed at its K-token conversion, under the same component names (so
    the eval metric keys match the full-vocabulary runs)."""
    from levanter.data.text.datasets import DatasetComponent, LmDataConfig, UrlDatasetSourceConfig
    from levanter.data.text.formats import TextLmDatasetFormat

    def comp(path):
        if not os.path.isdir(path):
            raise FileNotFoundError(f"{path} is missing: run the K={k} conversion first")
        src = UrlDatasetSourceConfig(tags=[], train_urls=[], validation_urls=[], cache_dir=os.path.dirname(path), format=TextLmDatasetFormat())
        return DatasetComponent(source=src, cache_dir=src.cache_dir, format=src.format, tags=[])

    components, weights = {}, {}
    for name in train_names:
        assert name == "fineweb-edu-10B", name
        components[name], weights[name] = comp(f"{_PREFIX}/fineweb-edu-10B-v{k}/train"), 1.0
    for name in validation_names:   # paloma/<subset>-marin-tokenizer
        sub = name.split("/", 1)[1].rsplit("-marin-tokenizer", 1)[0]
        components[name], weights[name] = comp(f"{_PREFIX}/tokenized/paloma-v{k}/{sub}/validation"), 0.0
    return LmDataConfig(components=components, train_weights=weights, tokenizer=f"{_PREFIX}/tokenizers/marin-small/v{k}",
                        cache_dir=None, shuffle=True, permutation_type="linear")
if VARIANT == "cv" and VOCAB_K:
    raise ValueError("VARIANT=cv keeps the Marin tokenizer: unset VOCAB_K")
if VARIANT not in ("baseline", "fbt", "ss", "ov", "ovss", "ovfocal", "ovgram", "ovgram3", "ovgramss", "focal", "cv", *OBJECTIVE_VARIANTS):
    raise ValueError(f"unknown VARIANT={VARIANT!r}")
INIT_FROM = os.environ.get("INIT_FROM") or None
TOTAL_STEPS = int(os.environ["TOTAL_STEPS"]) if os.environ.get("TOTAL_STEPS") else None
CPT_TAG = os.environ.get("CPT_TAG", "-cpt") if INIT_FROM else ""
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
    eval_every = 1000
    if TPP != 20:   # before the vocabulary ratio: the same text, TPP/20 times as long
        s = dict(s, steps=round(s["steps"] * TPP / 20))
        eval_every = 1000 * max(1, round(TPP / 40))
    if VOCAB_K:
        s = dict(s, steps=round(s["steps"] * _vocab_ratio(VOCAB_K)))   # fixed text: more steps for the same text
        # bytes (4.7x the steps): evaluate every round(ratio) x 1000 steps, about as often per text as the Marin
        # tokenizer's runs; ratios below 1.5 (64K-8K) keep 1000
        eval_every = max(eval_every, 1000 * max(1, round(_vocab_ratio(VOCAB_K))))
    if BATCH or SEQ_LEN != 4096:   # the same tokens in steps of BATCH x SEQ_LEN
        b = BATCH or s["batch"]
        s = dict(s, batch=b, steps=round(s["steps"] * s["batch"] * 4096 / (b * SEQ_LEN)))
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
    if VARIANT in OV_VARIANTS:
        variant_tags += f"-ov{OV_M / 1e6:g}m" + (f"od{OV_OD_W:g}" if OV_OD_W != 0.1 else "") + (f"-oefreeze{OV_FREEZE_AT}" if OV_FREEZE_AT else "")
    if VARIANT in ("focal", "ovfocal"):
        variant_tags += f"-focal{FOCAL_GAMMA:g}"
    if VARIANT == "cv":
        variant_tags += f"-cv{CV_SIDE}{CV_K // 1000}k" + ("-sum" if CV_OUT_REDUCE == "sum" else "")
    if VARIANT in ("ovgram", "ovgram3", "ovgramss"):
        variant_tags += f"-odhash{OD_M / 1e6:g}m" + ("-o23" if VARIANT == "ovgram3" else "")
    if VARIANT in ("ss", "ovss", "ovgramss"):
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
    if VOCAB_K:
        variant_tags += "-bytes" if VOCAB_K == 256 else f"-v{VOCAB_K // 1000}k"
    if TPP != 20:
        variant_tags += f"-tpp{TPP:g}"
    if DATA_EPOCHS > 0:
        variant_tags += f"-rep{DATA_EPOCHS:g}"
    if EMA_BETA > 0:
        variant_tags += f"-ema{EMA_BETA:g}"
    if SEQ_LEN != 4096:
        variant_tags += f"-sl{SEQ_LEN}"
    if BATCH:
        variant_tags += f"-b{BATCH}"
    if SEED != 0:
        variant_tags += f"-s{SEED}"
    run_id = f"muonh-qwen3-{size}-della{NUM_GPUS}x{DEVICE_TAG}" + variant_tags + CPT_TAG + RUN_TAG + (f"-smoke{SMOKE_STEPS}" if SMOKE_STEPS else "")
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
    elif VARIANT == "focal":
        model_cls, model_extra = FocalQwen3Config, dict(focal_gamma=FOCAL_GAMMA)
    elif VARIANT == "cv":
        cv_map = f"{_PREFIX}/tokenizers/marin-small/v{CV_K}/expand_full.npz"
        if not os.path.exists(cv_map):
            raise FileNotFoundError(f"{cv_map} is missing: run small_vocab_tokenizer expansion-npz {CV_K} first")
        model_cls, model_extra = ComposedVocabQwen3Config, dict(cv_map=cv_map, cv_input=CV_SIDE in ("in", "both"), cv_output=CV_SIDE in ("out", "both"),
                                                               cv_output_reduce=CV_OUT_REDUCE)
    elif VARIANT in OV_VARIANTS:
        num_steps = SMOKE_STEPS or TOTAL_STEPS or s["steps"]
        model_cls, model_extra = OverVocabQwen3Config, dict(
            oe_m=OV_M,
            od_weight=OV_OD_W,
            focal_gamma=FOCAL_GAMMA if VARIANT == "ovfocal" else 0.0,
            ss_candidates=tuple(p for _, p in SS_SCHEDULE) if VARIANT in ("ovss", "ovgramss") else (),
            ss_stage_ends=tuple(round(f * num_steps) for f, _ in SS_SCHEDULE) if VARIANT in ("ovss", "ovgramss") else (),
            od_mode="hashed" if VARIANT in ("ovgram", "ovgram3", "ovgramss") else "product",
            od_orders=(2, 3) if VARIANT == "ovgram3" else (2,),
            od_m=OD_M,
            oe_freeze_step=OV_FREEZE_AT,
        )
    else:
        model_cls, model_extra = Qwen3Config, {}
    if TIE:
        model_extra = dict(model_extra, tie_word_embeddings=True)
    model = model_cls(
        max_seq_len=SEQ_LEN,
        hidden_dim=s["hidden"],
        intermediate_dim=s["inter"],
        num_layers=s["layers"],
        num_heads=s["heads"],
        num_kv_heads=s["kv"],
        hybrid_norm=True,
        # No Transformer Engine on Della, so the GPU default (NVTE) would fall back to unfused O(S^2) attention.
        attn_backend=AttentionBackend.JAX_FLASH,
        **model_extra,
    )
    optimizer = MuonHConfig(
        learning_rate=s["lr"],
        adam_lr=s["adam_lr"],
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
        warmup=s["warmup"],
        min_lr_ratio=0.0,
        # The originals ran the fixed (3.4445, -4.7750, 2.0315) iteration; main defaults to quintic coefficients.
        coefficient_type="simple",
    )

    def build_config(ctx: StepContext) -> TrainLmOnPodConfig:
        if VOCAB_K:
            data = _small_vocab_data(VOCAB_K, [d.name for d in train], [d.name for d in validation])
        else:
            data = dataclasses.replace(mixture(ctx, train, validation=validation, shuffle=True), permutation_type="linear")
        if DATA_EPOCHS > 0:
            steps = SMOKE_STEPS or TOTAL_STEPS or s["steps"]
            data = dataclasses.replace(data, max_train_batches={d.name: round(steps / DATA_EPOCHS) for d in train})
        inner = train_lm.TrainLmConfig(
            data=data,
            trainer=TrainerConfig(
                tracker=WandbConfig(
                    entity=os.environ.get("WANDB_ENTITY"),
                    project=os.environ.get("WANDB_PROJECT", "marin-della"),
                    group="muonh-qwen3-smoke" if SMOKE_STEPS else ("muonh-qwen3-fbt-della" if VARIANT == "fbt" else "muonh-qwen3-ss-della" if VARIANT == "ss" else "muonh-qwen3-ov-della" if VARIANT in OV_VARIANTS else "muonh-qwen3-focal-della" if VARIANT == "focal" else "muonh-qwen3-objective-della" if VARIANT in OBJECTIVE_VARIANTS else "muonh-qwen3-fp8-della" if PRECISION == "fp8" else "muonh-qwen3-della"),
                    tags=["speedrun", "muonh", "qwen3", size, f"della{NUM_GPUS}x{DEVICE_TAG}", "jax_flash", *([f"fbt{FEEDBACK_PASSES}"] if VARIANT == "fbt" else []), PRECISION, *[t for t in variant_tags.split("-") if t], *(["cpt"] if INIT_FROM else [])],
                ),
                initialize_from=INIT_FROM,
                allow_partial_checkpoint=INIT_FROM is not None,
                mp=jmp.get_policy("p=f32,c=bfloat16"),
                # FP8 replaces every Linear's dot_general with the delayed-scaling FP8 op (fwd e4m3, grads e5m2, f32 accumulate);
                # embeddings, norms, attention softmax and the loss stay in the bf16/f32 policy above.
                quantization=QuantizationConfig(fp8=True) if PRECISION == "fp8" else None,
                train_batch_size=s["batch"],
                per_device_parallelism=PDP or PER_DEVICE_PARALLELISM.get(size, -1),
                num_train_steps=SMOKE_STEPS or TOTAL_STEPS or s["steps"],
                steps_per_eval=SMOKE_STEPS or eval_every,
                max_eval_batches=1 if SMOKE_STEPS else None,
                # OV checkpoints carry the n-gram tables with their Adam state (79 GB at 300m, 112 GB at 520m, ~227 GB at
                # 1_2b), so OV variants (a) keep no step-interval checkpoints (the GROUP fileset was 96% full on
                # 2026-09-30) and (b) save the temporary (resume) checkpoint hourly: a save stalls training while it stages
                # to host (300m ov: ~70 s per save, ~11% of wall-clock at the 10-minute interval). The final checkpoint is
                # unchanged and training is unaffected; a killed job loses at most an hour.
                checkpointer=CheckpointerConfig(
                    save_interval=timedelta(minutes=60 if VARIANT in OV_VARIANTS else 10),
                    keep=[] if VARIANT in OV_VARIANTS else [dict(every=10000)],
                ),
                mesh=MeshConfig(
                    axes={"data": -1, "replica": 1, "model": 1},
                    compute_mapping={
                        "token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
                        "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
                        # ov: the hashed n-gram tables stay row-sharded in compute too (never all-gathered)
                        **({OV_ROWS: ResourceAxis.DATA} if VARIANT in OV_VARIANTS else {}),
                    },
                    **({"param_mapping": {"embed": "data", OV_ROWS: "data"}} if VARIANT in OV_VARIANTS else {}),
                ),
                seed=SEED,
                model_averaging=EmaModelAveragingConfig(beta=EMA_BETA) if EMA_BETA > 0 else None,
                allow_nondivisible_batch_size=True,
            ),
            train_seq_len=SEQ_LEN,
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


def _start_memstats_thread(every: int) -> None:
    """Peak device memory, a P0 signal the trainer does not log: every `every` seconds (MEMSTATS_EVERY, default 600, 0
    turns it off) print the allocator stats reduced over the local devices, max for usage and peak, min for the limit.
    The first read waits one period so the trainer brings the backend up first."""
    import threading
    import time

    import jax

    def loop():
        while True:
            time.sleep(every)
            try:
                stats = [d.memory_stats() or {} for d in jax.local_devices()]
                red = {k: max(st.get(k, 0) for st in stats) for k in ("bytes_in_use", "peak_bytes_in_use", "largest_alloc_size")}
                red["largest_free_block_bytes"] = min(st.get("largest_free_block_bytes", 0) for st in stats)
                red["bytes_limit"] = min(st.get("bytes_limit", 0) for st in stats)
                print(f"MEMSTATS max-over-{len(stats)}-devices " + " ".join(f"{k}={v / 2**30:.2f}" for k, v in red.items()) + " GiB", flush=True)
            except Exception as e:  # diagnostics must never take the run down
                print(f"MEMSTATS error {e!r}", flush=True)

    threading.Thread(target=loop, daemon=True).start()


if __name__ == "__main__":
    if int(os.environ.get("MEMSTATS_EVERY", "600")) > 0 and os.environ.get("DRY_RUN") != "1":
        _start_memstats_thread(int(os.environ.get("MEMSTATS_EVERY", "600")))
    step = muonh_qwen3_run(os.environ["SIZE"])
    if os.environ.get("DRY_RUN") == "1":
        spec = lower(step)
        print("output:", spec.override_output_path or spec.name)
        print("deps:", len(spec.deps))
    else:
        run(step)
