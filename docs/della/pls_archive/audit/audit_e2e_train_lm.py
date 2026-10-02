"""Audit C: levanter.main.train_lm.main end to end on CPU (4 fake devices) with PerLayerQwen3Config.

Exercises the real hook in train_lm (extra_eval_callbacks), the real Trainer (bf16 compute policy p=f32,c=bfloat16,
microbatched grad accumulation: batch 8, per_device_parallelism 1 -> 2 microbatches of 4), the real eval dataset
plumbing (NamedLmDataset / DomainTaggedDataset / DataLoader with a partial last batch), a real tokenizer (marin
tokenizer, bytes-per-token over the 128256 vocab), steps_per_eval, the forced end-of-training hooks, and checkpoint
resume. Every tracker.log call is recorded.

Checks per eval step: eval/L{L-1}/<key> == eval/<key> for every main-eval key (loss, macro, per-tag loss, bpb, ...);
eval/L{k}/<key> exists for every k and key; train/pls/L{k} logged every step; train/loss == L_{L-1} + w * sum_k L_k.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4"
os.environ.setdefault("LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS", "0")
os.environ.setdefault("HF_HOME", "/scratch/gpfs/GROUP/USER/cache/huggingface")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("WANDB_MODE", "disabled")

import collections  # noqa: E402
import math  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jmp  # noqa: E402
import numpy as np  # noqa: E402

from haliax.partitioning import ResourceAxis  # noqa: E402
import levanter.main.train_lm as train_lm  # noqa: E402
from levanter.checkpoint import CheckpointerConfig  # noqa: E402
from levanter.data.dataset import ListAsyncDataset  # noqa: E402
from levanter.data.text.datasets import DirectDatasetComponent, LmDataConfig  # noqa: E402
from levanter.data.text.examples import GrugLmExample  # noqa: E402
from levanter.distributed import DistributedConfig  # noqa: E402
from levanter.layers.attention import AttentionBackend  # noqa: E402
from levanter.optim.config import AdamConfig  # noqa: E402
from levanter.tracker.json_file import JsonFileTracker, JsonFileTrackerConfig  # noqa: E402
from levanter.trainer import TrainerConfig  # noqa: E402
from levanter.utils.mesh import MeshConfig  # noqa: E402

from experiments.references.per_layer_qwen3 import PerLayerQwen3Config  # noqa: E402

assert jax.device_count() == 4
T, L = 64, 3
RECORDS: list[tuple[int | None, dict]] = []
_orig_log = JsonFileTracker.log


def _rec_log(self, metrics, *, step, commit=None):
    flat = {}
    for k, v in metrics.items():
        try:
            flat[k] = float(np.asarray(v))
        except Exception:  # noqa: BLE001
            flat[k] = v
    RECORDS.append((step, flat))
    return _orig_log(self, metrics, step=step, commit=commit)


JsonFileTracker.log = _rec_log

rng = np.random.default_rng(0)
EOS = 128001


def seqs(n, lo=0, hi=5000):
    out = []
    for _ in range(n):
        s = rng.integers(lo, hi, size=T).astype(np.int32)
        s[rng.integers(5, T - 5)] = EOS
        out.append(GrugLmExample.causal(jnp.asarray(s), eos_id=EOS))
    return ListAsyncDataset(out)


ds_train = seqs(64)
ds_a = seqs(6, 0, 3000)
ds_b = seqs(7, 1000, 9000)


def make_config(tmpdir, *, w, steps, run_id):
    data = LmDataConfig(
        components={
            "tiny": DirectDatasetComponent(datasets={"train": ds_train}),
            "paloma/a": DirectDatasetComponent(datasets={"validation": ds_a}),
            "paloma/b": DirectDatasetComponent(datasets={"validation": ds_b}),
        },
        train_weights={"tiny": 1.0, "paloma/a": 0.0, "paloma/b": 0.0},
        tokenizer="marin-community/marin-tokenizer",
    )
    model = PerLayerQwen3Config(max_seq_len=T, hidden_dim=32, intermediate_dim=64, num_layers=L, num_heads=2, num_kv_heads=2,
                                hybrid_norm=True, attn_backend=AttentionBackend.VANILLA, pls_weight=w, pls_monitor_stride=16)
    trainer = TrainerConfig(
        id=run_id,
        num_train_steps=steps,
        train_batch_size=8,
        per_device_parallelism=1,  # microbatch = 1 * 4 devices = 4 -> 2 accumulation steps
        steps_per_eval=2,
        max_eval_batches=None,
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        tracker=JsonFileTrackerConfig(output_path=tmpdir),
        checkpointer=CheckpointerConfig(base_path=os.path.join(tmpdir, "checkpoints")),
        require_accelerator=False,
        distributed=DistributedConfig(initialize_jax_distributed=False),
        mesh=MeshConfig(axes={"data": -1, "replica": 1, "model": 1}, compute_mapping={
            "token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
            "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA)}),
        allow_nondivisible_batch_size=True,
    )
    return train_lm.TrainLmConfig(data=data, model=model, trainer=trainer, optimizer=AdamConfig(learning_rate=3e-3), z_loss_weight=0.0,
                                  data_seed=42, hf_save_steps=None)


def check_run(tag, w, first_step, last_step):
    fails = []
    merged = collections.defaultdict(dict)
    for st, d in RECORDS:
        merged[st].update(d)
    train_steps = sorted(st for st, d in merged.items() if st is not None and "train/pls/L0" in d)
    print(f"[{tag}] train steps logged: {train_steps}")
    if train_steps != list(range(first_step, last_step + 1)):
        fails.append(f"train steps {train_steps}")
    for st in train_steps:
        d = merged[st]
        per = [d.get(f"train/pls/L{k}") for k in range(L)]
        if any(p is None or not math.isfinite(p) for p in per):
            fails.append(f"step {st} missing per-layer {per}")
            continue
        if "train/loss" not in d:  # log_step_info (train/loss) is a hook: it skips info.step <= 1 unless forced (baseline behaviour)
            print(f"[{tag}] step {st}: no train/loss logged (hook cadence), per-layer {[round(p, 4) for p in per]}")
            continue
        want = per[L - 1] + w * sum(per[: L - 1])
        if abs(d["train/loss"] - want) > 1e-4 * abs(want):
            fails.append(f"step {st}: train/loss {d['train/loss']} != {want}")
        print(f"[{tag}] step {st}: train/loss {d['train/loss']:.4f} per-layer {[round(p, 4) for p in per]}")
    main = [(st, d) for st, d in RECORDS if "eval/loss" in d]
    ro = [(st, d) for st, d in RECORDS if f"eval/L{L - 1}/loss" in d]
    print(f"[{tag}] main eval log steps {[st for st, _ in main]}  readout eval log steps {[st for st, _ in ro]}")
    if [st for st, _ in main] != [st for st, _ in ro] or not main:
        fails.append("eval cadence mismatch")
    for (st, dm), (st2, dr) in zip(main, ro):
        worst = 0.0
        keys = [k for k in dm if k.startswith("eval/") and "time" not in k]
        for k in keys:
            rk = f"eval/L{L - 1}/" + k[len("eval/"):]
            if rk not in dr:
                fails.append(f"step {st}: {rk} missing")
                continue
            dd = abs(dr[rk] - dm[k]) / max(abs(dm[k]), 1e-12)
            worst = max(worst, dd)
            if dd > 1e-5:
                fails.append(f"step {st}: {rk}={dr[rk]} != {k}={dm[k]}")
            for kk in range(L - 1):
                if f"eval/L{kk}/" + k[len("eval/"):] not in dr:
                    fails.append(f"step {st}: eval/L{kk}/{k[5:]} missing")
        extra = sorted(k for k in dr if not any(k == f"eval/L{kk}/" + m[5:] for kk in range(L) for m in dm))
        if extra:
            fails.append(f"step {st}: readout keys without a main counterpart: {extra[:6]}")
        print(f"[{tag}] eval@{st}: {len(keys)} main keys; L-1 vs main max rel diff {worst:.2e}; loss by layer "
              f"{[round(dr[f'eval/L{kk}/loss'], 4) for kk in range(L)]} main {dm['eval/loss']:.4f}; "
              f"paloma/a/bpb by layer {[round(dr.get(f'eval/L{kk}/paloma/a/bpb', float('nan')), 4) for kk in range(L)]}")
    if main:
        print(f"[{tag}] main-eval keys: {sorted(k for k in main[0][1] if 'time' not in k)}")
    return fails


all_fails = {}
tmp = tempfile.mkdtemp(prefix="pls_e2e_", dir="/scratch/gpfs/GROUP/USER/tmp/pls/audit")
try:
    # C1: w=1, 2 steps, then resume to 4 steps (evals at 2 and 4, forced at the end of each segment)
    d1 = os.path.join(tmp, "w1")
    RECORDS.clear()
    train_lm.main(make_config(d1, w=1.0, steps=2, run_id="pls-audit-w1"))
    all_fails["C1 w=1 steps 0-1"] = check_run("w1 seg1", 1.0, 0, 1)
    RECORDS.clear()
    train_lm.main(make_config(d1, w=1.0, steps=4, run_id="pls-audit-w1"))
    all_fails["C1 w=1 resumed steps 2-3"] = check_run("w1 seg2 (resumed)", 1.0, 2, 3)
    # C2: w=0 (monitor), 2 steps
    d0 = os.path.join(tmp, "w0")
    RECORDS.clear()
    train_lm.main(make_config(d0, w=0.0, steps=2, run_id="pls-audit-w0"))
    f = check_run("w0", 0.0, 0, 1)
    merged0 = collections.defaultdict(dict)
    for st, d in RECORDS:
        merged0[st].update(d)
    for st, d in merged0.items():
        if "train/pls/L0" in d and "train/loss" in d and abs(d["train/loss"] - d[f"train/pls/L{L - 1}"]) > 1e-6:
            f.append(f"w0 step {st}: train/loss != train/pls/L{L - 1}")
    all_fails["C2 w=0"] = f
finally:
    shutil.rmtree(tmp, ignore_errors=True)

for k, v in all_fails.items():
    print(k, "OK" if not v else f"FAIL: {v[:8]}")
print("AUDIT E2E", "PASSED" if not any(all_fails.values()) else "FAILED")
sys.exit(0 if not any(all_fails.values()) else 1)
