# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Paloma losses of one checkpoint at several softmax temperatures (logits / T), for the vocabulary page's question why
one setup flips sign across tokenizers: is the 8-pass model overconfident everywhere, and more so with more classes?

A copy of vocab_token_losses.py (same caches, windows, loss path) that divides the output head by each T in TEMPS and
saves per subset only the weighted loss sums per T and the weight sum (nats); bits per byte come afterwards from the
window tokens of the same tokenizer's vocab_token_losses npz. T = 1 reproduces the run's own loss.

    CKPT=<.../checkpoints/step-N> SIZE=300m [VOCAB_K=8000] [TEMPS="0.95 1 1.05 1.1 1.2"] OUT=<file.json>
    python -m experiments.references.vocab_temp_losses

Original docstring of vocab_token_losses.py follows.

Evaluates CKPT on every Paloma validation subset of its tokenizer (the Marin tokenizer, or with VOCAB_K its truncation)
exactly as the run's evals do (same caches, SEQ_LEN windows, the model's own loss path through activations and
get_lm_head), and saves for each subset the window token ids [n, SEQ_LEN], the loss in nats of predicting token t+1 at
position t [n, SEQ_LEN] (float32) and its loss weight. A repeated run and its full-data partner use the same tokenizer and
windows, so their arrays line up position by position. vocab_cost_groups.py groups the differences.

    CKPT=<.../checkpoints/step-N> SIZE=300m [VOCAB_K=8000] [VARIANT=cv CV_SIDE=in|out [CV_OUT_REDUCE=sum]] [SEQ_LEN=4096] OUT=<file.npz>
    python -m experiments.references.vocab_token_losses
"""

import json
import os
import time

import jax
import jax.numpy as jnp
import jmp
import numpy as np

import haliax as hax
from haliax import Axis
from haliax.partitioning import ResourceAxis, round_axis_for_partitioning
from levanter.data.loader import DataLoader
from levanter.data.text.datasets import DatasetComponent, LmDataConfig, UrlDatasetSourceConfig
from levanter.data.text.formats import TextLmDatasetFormat
from levanter.layers.attention import AttentionBackend
from levanter.model_loading import load_levanter_checkpoint
from levanter.models.loss import maybe_fused_next_token_loss
from levanter.models.qwen import Qwen3Config
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from levanter.utils.tree_utils import inference_mode

from experiments.references.composed_vocab_qwen3 import ComposedVocabQwen3Config

CKPT = os.environ["CKPT"]
OUT = os.environ["OUT"]
SIZE = os.environ.get("SIZE", "300m")
VOCAB_K = int(os.environ.get("VOCAB_K", "0"))
VARIANT = os.environ.get("VARIANT", "baseline")
CV_SIDE = os.environ.get("CV_SIDE", "in")
# a sum head and a mean head have the same tree, so a sum checkpoint would load and be scored as a mean one: the run id
# in the checkpoint path (-cvout8k-sum) decides, and an explicit CV_OUT_REDUCE must agree with it
CV_OUT_REDUCE = "sum" if "-sum" in CKPT.split("/checkpoints/")[0].rsplit("/", 2)[-2] else "mean"
if os.environ.get("CV_OUT_REDUCE", CV_OUT_REDUCE) != CV_OUT_REDUCE:
    raise ValueError(f"CV_OUT_REDUCE={os.environ['CV_OUT_REDUCE']} but the run id in {CKPT} says {CV_OUT_REDUCE}")
SEQ_LEN = int(os.environ.get("SEQ_LEN", "4096"))
PREFIX = os.environ.get("MARIN_PREFIX", "/scratch/gpfs/GROUP/USER/marin_store_big")
EVAL_BATCH = int(os.environ.get("EVAL_BATCH", "8"))
ONLY = os.environ.get("ONLY", "").split()          # test: these subsets only
MAX_WINDOWS = int(os.environ.get("MAX_WINDOWS", "0"))
TEMPS = tuple(float(t) for t in os.environ.get("TEMPS", "0.95 1 1.05 1.1 1.2").split())   # test: the first windows of each subset only
HIDDEN = {"130m": (512, 1792, 6, 8, 8), "300m": (768, 2688, 12, 12, 12), "520m": (1024, 3584, 24, 16, 8)}


def component(path: str) -> DatasetComponent:
    src = UrlDatasetSourceConfig(tags=[], train_urls=[], validation_urls=[], cache_dir=path, format=TextLmDatasetFormat())
    return DatasetComponent(source=src, cache_dir=path, format=src.format, tags=[])


def data_config() -> LmDataConfig:
    """Every Paloma validation subset of the tokenizer, at weight 0 (validation only)."""
    if VOCAB_K:
        root, tok = f"{PREFIX}/tokenized/paloma-v{VOCAB_K}", f"{PREFIX}/tokenizers/marin-small/v{VOCAB_K}"
        subs = {d: os.path.join(root, d) for d in sorted(os.listdir(root))}
    else:
        import glob
        root = f"{PREFIX}/tokenized/paloma"
        tok = os.path.dirname(glob.glob("/scratch/gpfs/GROUP/USER/cache/huggingface/hub/models--marin-community--marin-tokenizer/snapshots/*/tokenizer.json")[0])
        subs = {d.rsplit("-", 1)[0]: os.path.join(root, d) for d in sorted(os.listdir(root))}
    comps = {n: component(p) for n, p in subs.items() if os.path.isdir(os.path.join(p, "validation"))}
    assert len(comps) == 16, f"expected the 16 Paloma subsets, found {sorted(comps)}"
    return LmDataConfig(components=comps, train_weights={n: 0.0 for n in comps}, tokenizer=tok, cache_dir=None, shuffle=False)


def model_config():
    h, inter, layers, heads, kv = HIDDEN[SIZE]
    common = dict(max_seq_len=SEQ_LEN, hidden_dim=h, intermediate_dim=inter, num_layers=layers, num_heads=heads, num_kv_heads=kv,
                  hybrid_norm=True, attn_backend=AttentionBackend.JAX_FLASH)
    if VARIANT == "cv":
        return ComposedVocabQwen3Config(**common, cv_map=f"{PREFIX}/tokenizers/marin-small/v{int(os.environ.get('CV_K', '8000'))}/expand_full.npz",
                                        cv_input=CV_SIDE in ("in", "both"), cv_output=CV_SIDE in ("out", "both"),
                                        cv_output_reduce=CV_OUT_REDUCE)
    assert VARIANT == "baseline", VARIANT
    return Qwen3Config(**common)


def main():
    cfg = model_config()
    Pos = Axis("position", SEQ_LEN)
    trainer = TrainerConfig(mp=jmp.get_policy("p=f32,c=bfloat16"), per_device_eval_parallelism=EVAL_BATCH,
                            mesh=MeshConfig(axes={"data": -1, "replica": 1, "model": 1},
                                            compute_mapping={"token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
                                                             "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA)}))
    data = data_config()
    out = {}
    with trainer.use_device_mesh(), hax.axis_mapping(trainer.compute_axis_mapping):
        Vocab = round_axis_for_partitioning(Axis("vocab", len(data.the_tokenizer)), trainer.parameter_axis_mapping)
        model = inference_mode(load_levanter_checkpoint(cfg, CKPT, Vocab=Vocab, axis_mapping=trainer.parameter_axis_mapping, key=jax.random.PRNGKey(0)), True)
        mp = trainer.mp

        @hax.named_jit(axis_resources=trainer.compute_axis_mapping)
        def loss_sums(m, ex):
            m = mp.cast_to_compute(m)
            h = m.activations(ex.tokens, attn_mask=ex.attn_mask, key=None)
            head = m.get_lm_head()
            sums = [jnp.sum(maybe_fused_next_token_loss(m.Pos, m.Embed, m.Vocab, h, head * (1.0 / t), ex.tokens, loss_weight=ex.loss_weight,
                                                        reduction=None, reduction_axis=(), dtype=jnp.float32).array) for t in TEMPS]
            return jnp.stack(sums), jnp.sum(ex.loss_weight.array)

        for name, ds in data.validation_sets(Pos).items():
            if ONLY and name not in ONLY:
                continue
            t0 = time.time()
            n = len(ds.as_sync_dataset())
            if MAX_WINDOWS:
                n = min(n, MAX_WINDOWS)
                ds = ds.slice_dataset(end_index=n)
            ds = ds.slice_dataset(end_index=n - n % trainer.EvalBatch.size)   # whole batches only: no padded windows in the sums
            tot, wsum, nw = np.zeros(len(TEMPS)), 0.0, 0
            for batch in DataLoader(ds, trainer.EvalBatch, mesh=trainer.device_mesh, axis_resources=trainer.compute_axis_mapping):
                l, w = loss_sums(model, batch)
                tot += np.asarray(jax.device_get(l), np.float64); wsum += float(jax.device_get(w)); nw += trainer.EvalBatch.size
            out[name] = {"loss_sum": tot.tolist(), "weight_sum": wsum, "windows": nw}
            print(f"{name}: {nw} windows, mean loss by T " + " ".join(f"{t:g}:{x / wsum:.4f}" for t, x in zip(TEMPS, tot)) + f" nats ({time.time() - t0:.0f}s)", flush=True)
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    json.dump({"ckpt": CKPT, "temps": TEMPS, "subsets": out}, open(OUT, "w"), indent=1)
    print("written", OUT, flush=True)


if __name__ == "__main__":
    main()
