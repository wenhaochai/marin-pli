# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Scores a checkpoint of a run with a shrinking vocabulary (vocab_shrink_qwen3.py) on Paloma tokenized at any K (owner 2026-11-04).

A path 1 run trains on several views, so it has one score per view. This script loads CKPT (read-only; any 128,256-row
checkpoint, the plain Marin run's too), evaluates all 16 Paloma subsets at EVAL_K with a softmax over the ids below K and the
specials, at K's equal-text window by default, and writes per subset the mean loss and the scored targets, plus the exact
macro bits per byte (the page's formula, loss x T / (ln 2 x B) per subset), to OUT as JSON. The tables must count the same
targets (checked).

    CKPT=<.../step-N> EVAL_K=32000 OUT=<file.json> [SEQ_LEN=4480] [EVAL_BATCH=8] [ONLY=ptb] [MAX_WINDOWS=n] [TIE=1] [EMA=1] python -m experiments.references.vocab_shrink_eval

TIE=1: a run with tied tables. EMA=1: the checkpoint's moving average (a run with EMA_BETA) in place of its weights.
"""

import glob
import json
import math
import os
import time

import equinox as eqx
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
from levanter.checkpoint import load_checkpoint
from levanter.model_loading import load_levanter_checkpoint
from levanter.trainer import TrainerConfig
from levanter.utils.jax_utils import use_cpu_device
from levanter.utils.mesh import MeshConfig
from levanter.utils.tree_utils import inference_mode

from experiments.references.vocab_shrink_qwen3 import VocabShrinkQwen3Config

CKPT, OUT, K = os.environ["CKPT"], os.environ["OUT"], int(os.environ["EVAL_K"])
EQW = {128000: 4096, 64000: 4224, 32000: 4480, 16000: 4992, 8000: 5504}
SEQ_LEN = int(os.environ.get("SEQ_LEN", EQW[K]))
EVAL_BATCH = int(os.environ.get("EVAL_BATCH", "8"))
ONLY = os.environ.get("ONLY", "").split()
MAX_WINDOWS = int(os.environ.get("MAX_WINDOWS", "0"))
TIE = os.environ.get("TIE") == "1"
EMA = os.environ.get("EMA") == "1"
PREFIX = os.environ.get("MARIN_PREFIX", "/scratch/gpfs/GROUP/USER/marin_store_big")
TABLES = os.environ.get("BPB_TABLES", "/scratch/gpfs/GROUP/USER/tmp_plots/vocab/bpb_exact_tables.json")
TOK = "/scratch/gpfs/GROUP/USER/cache/huggingface/hub/models--marin-community--marin-tokenizer/snapshots/a5ca45f2feb6c959bd87b81689aa7279b5bdcaa2"
SUBSETS = sorted(d for d in os.listdir(f"{PREFIX}/tokenized/paloma-v8000") if os.path.isdir(f"{PREFIX}/tokenized/paloma-v8000/{d}/validation"))


def component(val_dir: str) -> DatasetComponent:
    root = os.path.dirname(val_dir)
    src = UrlDatasetSourceConfig(tags=[], train_urls=[], validation_urls=[], cache_dir=root, format=TextLmDatasetFormat())
    return DatasetComponent(source=src, cache_dir=root, format=src.format, tags=[])


def data_config() -> LmDataConfig:
    comps = {}
    for sub in SUBSETS:
        if K == 128000:
            (path,) = glob.glob(f"{PREFIX}/paloma/{sub}-marin-tokenizer/*/validation")
        else:
            path = f"{PREFIX}/tokenized/paloma-v{K}m/{sub}/validation"
        comps[sub] = component(path)
    assert len(comps) == 16, sorted(comps)
    return LmDataConfig(components=comps, train_weights={n: 0.0 for n in comps}, tokenizer=TOK, cache_dir=None, shuffle=False)


def main():
    cfg = VocabShrinkQwen3Config(max_seq_len=SEQ_LEN, hidden_dim=768, intermediate_dim=2688, num_layers=12, num_heads=12, num_kv_heads=12, hybrid_norm=True,
                                 attn_backend=AttentionBackend.JAX_FLASH, flash_attention_block_size=128, vs_eval_k=K, tie_word_embeddings=TIE)   # every equal-text window is a multiple of 128
    Pos = Axis("position", SEQ_LEN)
    trainer = TrainerConfig(mp=jmp.get_policy("p=f32,c=bfloat16"), per_device_eval_parallelism=EVAL_BATCH,
                            mesh=MeshConfig(axes={"data": -1, "replica": 1, "model": 1},
                                            compute_mapping={"token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
                                                             "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA)}))
    data = data_config()
    tables = json.load(open(TABLES)).get(str(K), {}).get(str(SEQ_LEN))
    out = {}
    with trainer.use_device_mesh(), hax.axis_mapping(trainer.compute_axis_mapping):
        Vocab = round_axis_for_partitioning(Axis("vocab", len(data.the_tokenizer)), trainer.parameter_axis_mapping)
        assert Vocab.size == 128256, Vocab
        model = inference_mode(load_levanter_checkpoint(cfg, CKPT, Vocab=Vocab, axis_mapping=trainer.parameter_axis_mapping, key=jax.random.PRNGKey(0)), True)
        if EMA:
            with use_cpu_device():
                ema = load_checkpoint(eqx.filter_eval_shape(cfg.build, Vocab, key=jax.random.PRNGKey(0)), CKPT, subpath="model_averaging/model")
            model = inference_mode(hax.shard_with_axis_mapping(ema, trainer.parameter_axis_mapping), True)
        mp = trainer.mp

        @hax.named_jit(axis_resources=trainer.compute_axis_mapping)
        def per_position(m, ex):
            m = mp.cast_to_compute(m)
            return m.compute_next_token_loss(ex, reduction=None, reduction_axis=(), loss_dtype=jnp.float32).array, ex.loss_weight.array

        for sub, ds in data.validation_sets(Pos).items():
            if ONLY and sub not in ONLY:
                continue
            t0 = time.time()
            n = len(ds.as_sync_dataset())
            if MAX_WINDOWS:
                n = min(n, MAX_WINDOWS)
                ds = ds.slice_dataset(end_index=n)
            s, tw = 0.0, 0.0
            left = n
            for batch in DataLoader(ds, trainer.EvalBatch, mesh=trainer.device_mesh, axis_resources=trainer.compute_axis_mapping):
                lo, wt = (np.asarray(jax.device_get(a)) for a in per_position(model, batch))
                lo, wt = lo[:left], wt[:left]
                left -= len(lo)
                s += float((lo.astype(np.float64) * wt).sum())
                tw += float(wt.astype(np.float64).sum())
            out[sub] = dict(windows=n, scored=tw, loss=s / tw)
            print(f"{sub}: {n} windows, loss {s / tw:.4f} nats per token ({time.time() - t0:.0f}s)", flush=True)
    res = dict(ckpt=CKPT, eval_k=K, seq_len=SEQ_LEN, subsets=out)
    if tables and len(out) == 16 and not MAX_WINDOWS:
        for a, v in out.items():
            assert abs(v["scored"] - tables[a][0]) < 1e-6 * tables[a][0], (a, v["scored"], tables[a][0])
        res["exact_bpb"] = sum(v["loss"] * tables[a][0] / (math.log(2) * tables[a][1]) for a, v in out.items()) / 16
        print(f"K={K} window {SEQ_LEN}: exact macro bits per byte {res['exact_bpb']:.4f}", flush=True)
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)
    print("written", OUT, flush=True)


if __name__ == "__main__":
    main()
