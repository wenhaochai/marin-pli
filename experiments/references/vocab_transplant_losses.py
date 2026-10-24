# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Weight transplant between a tokenizer's unique-token run (U) and its 8-pass run (R), for the vocabulary page's question
why the cost of repetition grows with the vocabulary (owner popup 2026-10-24, test A). If the extra cost of a large
vocabulary sits in token-specific parameters, it should move with the token rows: R with U's rows should lose most of
it, and U with R's rows should gain it.

Hybrids: the token rows are the input embedding (embeddings.token_embeddings) and the output head (lm_head), all rows or
with RARE_FROM set only the rows of ids >= RARE_FROM (for the Marin tokenizer, the tokens outside the 8K truncation):
    U, R                        the two runs as trained (checks against their exact finals)
    R+U.in, R+U.out, R+U.io     R with U's input rows, output rows, or both
    U+R.in, U+R.out, U+R.io     U with R's rows
    R+U.rare, U+R.rare          both sides of the rows of ids >= RARE_FROM only (when RARE_FROM is set)
    avg                         every weight the mean of U and R (the two runs share the init and diverge at step 0,
                                so this checks whether hybrids of the two are meaningful at all)
Each variant's Paloma loss sums per subset go to OUT_DIR/<variant>.json (skipped when present), in the format of
vocab_temp_losses.py with T = 1 only, so temp_analyze's byte tables give bits per byte.

    CKPT_U=<...step-N> CKPT_R=<...step-N> SIZE=300m [VOCAB_K=8000] [RARE_FROM=8000] OUT_DIR=<dir>
    python -m experiments.references.vocab_transplant_losses
"""

import json
import os
import time

os.environ.setdefault("CKPT", os.environ["CKPT_U"])   # vocab_temp_losses reads these at import
os.environ.setdefault("OUT", "/dev/null")
os.environ["TEMPS"] = "1"

import equinox as eqx
import jax
import numpy as np

import haliax as hax
from haliax import Axis
from haliax.partitioning import round_axis_for_partitioning
from levanter.data.loader import DataLoader
from levanter.model_loading import load_levanter_checkpoint
from levanter.models.loss import maybe_fused_next_token_loss
from levanter.utils.tree_utils import inference_mode

import experiments.references.vocab_temp_losses as vtl

CKPT_U, CKPT_R, OUT_DIR = os.environ["CKPT_U"], os.environ["CKPT_R"], os.environ["OUT_DIR"]
RARE_FROM = int(os.environ.get("RARE_FROM", "0"))
ONLY_VARIANTS = os.environ.get("VARIANTS", "").split()


def in_rows(m):
    return m.embeddings.token_embeddings.weight


def out_rows(m):
    return m.lm_head.weight


def swap(dst, src, getters, rare_mask=None):
    """dst with the arrays at `getters` taken from src (only the rows where rare_mask holds, when given)."""
    for g in getters:
        new = g(src) if rare_mask is None else hax.where(rare_mask, g(src), g(dst))
        dst = eqx.tree_at(g, dst, new)
    return dst


def main():
    cfg = vtl.model_config()
    Pos = Axis("position", vtl.SEQ_LEN)
    trainer = vtl.TrainerConfig(mp=vtl.jmp.get_policy("p=f32,c=bfloat16"), per_device_eval_parallelism=vtl.EVAL_BATCH,
                                mesh=vtl.MeshConfig(axes={"data": -1, "replica": 1, "model": 1},
                                                    compute_mapping={"token": (vtl.ResourceAxis.REPLICA_DCN, vtl.ResourceAxis.REPLICA, vtl.ResourceAxis.DATA),
                                                                     "token_repeat": (vtl.ResourceAxis.REPLICA_DCN, vtl.ResourceAxis.REPLICA, vtl.ResourceAxis.DATA)}))
    data = vtl.data_config()
    os.makedirs(OUT_DIR, exist_ok=True)
    with trainer.use_device_mesh(), hax.axis_mapping(trainer.compute_axis_mapping):
        Vocab = round_axis_for_partitioning(Axis("vocab", len(data.the_tokenizer)), trainer.parameter_axis_mapping)
        load = lambda c: inference_mode(load_levanter_checkpoint(cfg, c, Vocab=Vocab, axis_mapping=trainer.parameter_axis_mapping, key=jax.random.PRNGKey(0)), True)
        U, R = load(CKPT_U), load(CKPT_R)
        assert U.lm_head is not None, "untied head expected"
        rare = hax.arange(Vocab) >= RARE_FROM if RARE_FROM else None
        variants = {
            "U": lambda: U, "R": lambda: R,
            "R+U.in": lambda: swap(R, U, [in_rows]), "R+U.out": lambda: swap(R, U, [out_rows]), "R+U.io": lambda: swap(R, U, [in_rows, out_rows]),
            "U+R.in": lambda: swap(U, R, [in_rows]), "U+R.out": lambda: swap(U, R, [out_rows]), "U+R.io": lambda: swap(U, R, [in_rows, out_rows]),
            "avg": lambda: jax.tree_util.tree_map(lambda a, b: (a + b) / 2 if eqx.is_inexact_array(a) else a, U, R),
        }
        if RARE_FROM:
            variants["R+U.rare"] = lambda: swap(R, U, [in_rows, out_rows], rare)
            variants["U+R.rare"] = lambda: swap(U, R, [in_rows, out_rows], rare)
        mp = trainer.mp

        @hax.named_jit(axis_resources=trainer.compute_axis_mapping)
        def loss_sum(m, ex):
            m = mp.cast_to_compute(m)
            h = m.activations(ex.tokens, attn_mask=ex.attn_mask, key=None)
            per = maybe_fused_next_token_loss(m.Pos, m.Embed, m.Vocab, h, m.get_lm_head(), ex.tokens, loss_weight=ex.loss_weight,
                                              reduction=None, reduction_axis=(), dtype=vtl.jnp.float32).array
            return vtl.jnp.sum(per), vtl.jnp.sum(ex.loss_weight.array)

        sets = data.validation_sets(Pos)
        for name, make in variants.items():
            if ONLY_VARIANTS and name not in ONLY_VARIANTS:
                continue
            path = os.path.join(OUT_DIR, f"{name}.json")
            if os.path.exists(path):
                print("SKIP", name, flush=True)
                continue
            model = make()
            t0, out = time.time(), {}
            for sub, ds in sets.items():
                n = len(ds.as_sync_dataset())
                if vtl.MAX_WINDOWS:
                    n = min(n, vtl.MAX_WINDOWS)
                    ds = ds.slice_dataset(end_index=n)
                ds = ds.slice_dataset(end_index=n - n % trainer.EvalBatch.size)   # whole batches only, as vocab_temp_losses
                tot, wsum, nw = 0.0, 0.0, 0
                for batch in DataLoader(ds, trainer.EvalBatch, mesh=trainer.device_mesh, axis_resources=trainer.compute_axis_mapping):
                    l, w = loss_sum(model, batch)
                    tot += float(jax.device_get(l)); wsum += float(jax.device_get(w)); nw += trainer.EvalBatch.size
                out[sub] = {"loss_sum": [tot], "weight_sum": wsum, "windows": nw}
            json.dump({"ckpt_u": CKPT_U, "ckpt_r": CKPT_R, "variant": name, "temps": [1.0], "subsets": out}, open(path, "w"), indent=1)
            mean = np.mean([v["loss_sum"][0] / v["weight_sum"] for v in out.values()])
            print(f"{name}: mean subset loss {mean:.4f} nats ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
