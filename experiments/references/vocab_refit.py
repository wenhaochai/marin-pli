# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Re-fit one part of a final checkpoint on fresh text, then score Paloma: where does the cost of 8 passes live?
(owner popup 2026-10-24, test A redesigned after the weight transplant broke the models.)

From a tokenizer's unique-token run (U) or 8-pass run (R), train only PART on fineweb-edu sequences that neither run saw
(the run order's sequences past the unique-token run's last step, with a margin of a few mixture blocks), everything
else frozen, for FT_STEPS steps of FT_BATCH sequences; then write the Paloma loss sums per subset (T = 1, the format of
vocab_temp_losses.py). PART: rows (input embedding and output head), body (every other weight), or rare (the input and
output rows of ids >= RARE_FROM only). The rows' share of R's cost is R's gain from re-fitting the rows minus U's gain;
the body run is the control. Adam directions scaled by each weight block's RMS times FT_LR (a relative step),
linear warmup over 10% of the steps, cosine to 0, no weight decay.

    CKPT=<...step-N> SIZE=300m [VOCAB_K=8000] STEPS_RUN=<the unique-token run's steps> PART=rows|body|rare [RARE_FROM=8000]
    FT_STEPS=400 FT_BATCH=32 FT_LR=1e-3 OUT=<file.json> python -m experiments.references.vocab_refit
"""

import json
import os
import time

os.environ.setdefault("OUT", "/dev/null")
os.environ["TEMPS"] = "1"

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

import haliax as hax
from haliax import Axis
from haliax.partitioning import round_axis_for_partitioning
from levanter.data.loader import DataLoader
from levanter.data.text.datasets import LmDataConfig, NamedLmDataset
from levanter.model_loading import load_levanter_checkpoint
from levanter.models.loss import maybe_fused_next_token_loss

import experiments.references.vocab_temp_losses as vtl

PART = os.environ["PART"]
assert PART in ("rows", "body", "rare"), PART
RARE_FROM = int(os.environ.get("RARE_FROM", "0"))
assert PART != "rare" or RARE_FROM > 0, "PART=rare needs RARE_FROM"
STEPS_RUN = int(os.environ["STEPS_RUN"])
FT_STEPS, FT_BATCH, FT_LR = int(os.environ.get("FT_STEPS", "400")), int(os.environ.get("FT_BATCH", "32")), float(os.environ.get("FT_LR", "1e-3"))
RUN_BATCH = 128          # 300m training batch (della_muonh_qwen3_scaling SIZES)
DATA_SEED = 42
MARGIN = 4 * 2048        # mixture blocks: the run reads order index r within its block of j, so stay a few blocks past
OUT = os.environ["OUT"]


def train_order(Pos):
    """The run's shuffled fineweb-edu-10B order (data seed 42, linear permutation), as ov_memorization_eval rebuilds it."""
    if vtl.VOCAB_K:
        path, tok = f"{vtl.PREFIX}/fineweb-edu-10B-v{vtl.VOCAB_K}/train", f"{vtl.PREFIX}/tokenizers/marin-small/v{vtl.VOCAB_K}"
    else:
        import glob
        path = f"{vtl.PREFIX}/fineweb-edu-10B/2026.06.28/train"
        tok = os.path.dirname(glob.glob("/scratch/gpfs/GROUP/USER/cache/huggingface/hub/models--marin-community--marin-tokenizer/snapshots/*/tokenizer.json")[0])
    cfg = LmDataConfig(components={"t": vtl.component(os.path.dirname(path))}, train_weights={"t": 1.0}, tokenizer=tok, cache_dir=None,
                       shuffle=True, permutation_type="linear")
    _, shuffle_key = jax.random.split(jax.random.PRNGKey(DATA_SEED))
    return cfg.train_sets(Pos, key=shuffle_key, initial_batch_size=RUN_BATCH)["t"]


def trainable_mask(model, rare):
    """A pytree of bools (or, for PART=rare, row masks) over the model's arrays: what the re-fit may change."""
    is_row = lambda path: any(getattr(p, "name", None) in ("token_embeddings", "lm_head") for p in path)
    flat, treedef = jax.tree_util.tree_flatten_with_path(model)
    out = []
    for path, leaf in flat:
        row = is_row(path)
        out.append({"rows": row, "body": not row, "rare": row}[PART] and eqx.is_inexact_array(leaf))
    return jax.tree_util.tree_unflatten(treedef, out)


def main():
    cfg = vtl.model_config()
    Pos = Axis("position", vtl.SEQ_LEN)
    trainer = vtl.TrainerConfig(mp=vtl.jmp.get_policy("p=f32,c=bfloat16"), per_device_eval_parallelism=FT_BATCH,
                                mesh=vtl.MeshConfig(axes={"data": -1, "replica": 1, "model": 1},
                                                    compute_mapping={"token": (vtl.ResourceAxis.REPLICA_DCN, vtl.ResourceAxis.REPLICA, vtl.ResourceAxis.DATA),
                                                                     "token_repeat": (vtl.ResourceAxis.REPLICA_DCN, vtl.ResourceAxis.REPLICA, vtl.ResourceAxis.DATA)}))
    data = vtl.data_config()
    order = train_order(Pos)
    n_order = len(order.as_sync_dataset())
    start = STEPS_RUN * RUN_BATCH + MARGIN
    end = start + FT_STEPS * trainer.EvalBatch.size
    assert end <= n_order, f"fresh text {start}..{end} past the order's {n_order} sequences"
    fresh = NamedLmDataset(order.slice_dataset(start_index=start, end_index=end), Pos)
    print(f"order {n_order} sequences; unique-token run used {STEPS_RUN * RUN_BATCH}; re-fit on {start}..{end}", flush=True)
    with trainer.use_device_mesh(), hax.axis_mapping(trainer.compute_axis_mapping):
        Vocab = round_axis_for_partitioning(Axis("vocab", len(data.the_tokenizer)), trainer.parameter_axis_mapping)
        model = load_levanter_checkpoint(cfg, vtl.CKPT, Vocab=Vocab, axis_mapping=trainer.parameter_axis_mapping, key=jax.random.PRNGKey(0))
        model = vtl.inference_mode(model, True)   # no dropout in this model; inference mode only fixes attention-path flags
        mask = trainable_mask(model, None)
        rare = (hax.arange(Vocab) >= RARE_FROM) if PART == "rare" else None
        train_p, frozen_p = eqx.partition(model, mask)
        sched = optax.warmup_cosine_decay_schedule(0.0, FT_LR, max(1, FT_STEPS // 10), FT_STEPS, 0.0)
        # Adam direction, step sized relative to each weight block's RMS (FT_LR = relative change per step at the peak), so
        # one FT_LR suits the rows and the MuonH-trained matrices alike
        opt = optax.chain(optax.scale_by_adam(), optax.scale_by_param_block_rms(), optax.scale_by_schedule(lambda c: -sched(c)))
        opt_state = opt.init(train_p)
        mp = trainer.mp

        def loss_of(tp, ex):
            m = mp.cast_to_compute(eqx.combine(tp, frozen_p))
            h = m.activations(ex.tokens, attn_mask=ex.attn_mask, key=None)
            per = maybe_fused_next_token_loss(m.Pos, m.Embed, m.Vocab, h, m.get_lm_head(), ex.tokens, loss_weight=ex.loss_weight,
                                              reduction=None, reduction_axis=(), dtype=jnp.float32).array
            return jnp.sum(per) / jnp.maximum(jnp.sum(ex.loss_weight.array), 1.0)

        @hax.named_jit(axis_resources=trainer.compute_axis_mapping)
        def step(tp, os_, ex):
            loss, g = eqx.filter_value_and_grad(loss_of)(tp, ex)
            if rare is not None:   # only the rows of ids >= RARE_FROM move
                g = jax.tree_util.tree_map(lambda x: hax.where(rare, x, hax.zeros_like(x)) if isinstance(x, hax.NamedArray) and x.array is not None and m_has_vocab(x) else x, g,
                                           is_leaf=lambda x: isinstance(x, hax.NamedArray))
            upd, os_ = opt.update(g, os_, tp)
            return eqx.apply_updates(tp, upd), os_, loss

        def m_has_vocab(x):
            return any(a.name == Vocab.name for a in x.axes)

        t0, losses = time.time(), []
        for i, batch in enumerate(DataLoader(fresh, trainer.EvalBatch, mesh=trainer.device_mesh, axis_resources=trainer.compute_axis_mapping)):
            train_p, opt_state, loss = step(train_p, opt_state, batch)
            losses.append(float(loss))
            if i % 50 == 0 or i == FT_STEPS - 1:
                print(f"step {i} loss {losses[-1]:.4f} ({time.time() - t0:.0f}s)", flush=True)
        model = eqx.combine(train_p, frozen_p)

        @hax.named_jit(axis_resources=trainer.compute_axis_mapping)
        def loss_sum(m, ex):
            m = mp.cast_to_compute(m)
            h = m.activations(ex.tokens, attn_mask=ex.attn_mask, key=None)
            per = maybe_fused_next_token_loss(m.Pos, m.Embed, m.Vocab, h, m.get_lm_head(), ex.tokens, loss_weight=ex.loss_weight,
                                              reduction=None, reduction_axis=(), dtype=jnp.float32).array
            return jnp.sum(per), jnp.sum(ex.loss_weight.array)

        evb = Axis(trainer.EvalBatch.name, 8)
        out = {}
        for sub, ds in data.validation_sets(Pos).items():
            n = len(ds.as_sync_dataset())
            if vtl.MAX_WINDOWS:
                n = min(n, vtl.MAX_WINDOWS)
                ds = ds.slice_dataset(end_index=n)
            ds = ds.slice_dataset(end_index=n - n % evb.size)   # whole batches, as vocab_temp_losses (EVAL_BATCH 8)
            tot, wsum, nw = 0.0, 0.0, 0
            for batch in DataLoader(ds, evb, mesh=trainer.device_mesh, axis_resources=trainer.compute_axis_mapping):
                l, w = loss_sum(model, batch)
                tot += float(jax.device_get(l)); wsum += float(jax.device_get(w)); nw += evb.size
            out[sub] = {"loss_sum": [tot], "weight_sum": wsum, "windows": nw}
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    json.dump({"ckpt": vtl.CKPT, "part": PART, "rare_from": RARE_FROM, "ft_steps": FT_STEPS, "ft_batch": FT_BATCH, "ft_lr": FT_LR,
               "fresh": [start, end], "train_loss": losses, "temps": [1.0], "subsets": out}, open(OUT, "w"), indent=1)
    print("written", OUT, "mean subset loss", np.mean([v["loss_sum"][0] / v["weight_sum"] for v in out.values()]), flush=True)


if __name__ == "__main__":
    main()
