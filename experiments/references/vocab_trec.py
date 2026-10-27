# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Training re-evaluation curve (TREC; Bergsma, Dey and Hestness 2025, arXiv 2509.25380) of one checkpoint on the
unique-token run's training stream (owner 2026-10-26: read the training data again with the final weights, to see which
part of it the repeated run does worse on).

The unique-token run's fineweb-edu component is linearly permuted (data seed 42), and training reads it through a
MixtureDataset that permutes the sequences again inside every block of 2048 (16 batches). main() scores the component in
position order (training_stream): its "step" t is component positions t*B .. t*B+B-1, the right 16-batch block of the run
(read at steps 16*floor(t/16) .. +15) but not the exact batch; training_batches() gives the exact batches. Positions are
what the analysis needs: the 8-pass partner reads exactly positions 0 .. round(T/8)*B - 1 (about 8 times each, in a new
order inside every block), so a sequence is subset, read once by the unique run only, or unread by both, by its position
alone (the block that holds position T*B is partly read: leave it out). This scores CKPT on N_STEPS position blocks spread
over the unique run's T steps and on N_TAIL blocks after them, which neither run of a pair has read (held-out text from the
same shuffled set). For every sequence it keeps the summed loss in nats and the loss weight, and for the first KEEP
sequences of each block the per-position loss and the tokens, for token-frequency splits.

Output npz: steps [n] (component position blocks t, positions t*B .. t*B+B-1; tail blocks all >= T), total_steps (T),
seq_loss [n, B], seq_weight [n, B], pos_loss [n, KEEP, SEQ_LEN], tokens [n, KEEP, SEQ_LEN], ckpt.

    CKPT=<.../checkpoints/step-N> SIZE=300m [VOCAB_K=8000] [N_STEPS=192] [N_TAIL=16] [KEEP=4] OUT=<file.npz>
    python -m experiments.references.vocab_trec
"""

import dataclasses
import glob
import os
import time
from unittest import mock

SIZE = os.environ.get("SIZE", "300m")
K = int(os.environ.get("VOCAB_K", "0"))
N_STEPS = int(os.environ.get("N_STEPS", "192"))
N_TAIL = int(os.environ.get("N_TAIL", "16"))
KEEP = int(os.environ.get("KEEP", "4"))
SEQ = int(os.environ.get("SEQ_LEN", "4096"))
# the unique-token run's own data config: no repetition, no variant
for k in ("PASS_DATA", "VARIANT", "SMOKE_STEPS", "TOTAL_STEPS", "BATCH", "TPP", "SEED", "DATA_EPOCHS"):
    os.environ.pop(k, None)
os.environ.update(SIZE=SIZE, DRY_RUN="1", MEMSTATS_EVERY="0")
if K:
    os.environ["VOCAB_K"] = str(K)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import haliax as hax  # noqa: E402
from haliax import Axis  # noqa: E402
from haliax.partitioning import round_axis_for_partitioning  # noqa: E402
from levanter.data.loader import DataLoader  # noqa: E402
from levanter.data.text.datasets import NamedLmDataset  # noqa: E402
from levanter.model_loading import load_levanter_checkpoint  # noqa: E402
from levanter.models.loss import maybe_fused_next_token_loss  # noqa: E402
from levanter.utils.tree_utils import inference_mode  # noqa: E402

import experiments.references.vocab_token_losses as vtk  # noqa: E402
from experiments.references import della_muonh_qwen3_scaling as scaling  # noqa: E402


def training_batches():
    """The unique-token run's training examples in the order it read them (a NamedLmDataset: batch t is examples
    t*B .. t*B+B-1), its batch size and its step count. Built as train_lm builds it, data.train_set(Pos, batch_schedule,
    key=PRNGKey(data_seed)): the MixtureDataset around the one component permutes the sequences inside every block of
    mixture_block_size (2048 sequences, 16 batches) with its own key, so training_stream() below (the component before the
    mixture) has the right 16-batch blocks but not the right batches. Found by the TREC session (2026-10-26): its replay of
    the last 12 batches of -s1-rerun with training_stream() missed W&B train/loss by up to 0.05 nats, at a learning rate
    near 0 where the final weights must reproduce it."""
    tc = scaling.muonh_qwen3_run(SIZE).build_config(mock.MagicMock()).train_config
    batch, data = tc.trainer.train_batch_size, tc.data
    assert not data.max_train_batches, data.max_train_batches
    if not K:
        (name,) = [n for n, w in data.train_weights.items() if w > 0]
        tok = os.path.dirname(glob.glob("/scratch/gpfs/GROUP/USER/cache/huggingface/hub/models--marin-community--marin-tokenizer/snapshots/*/tokenizer.json")[0])
        data = dataclasses.replace(data, tokenizer=tok, components={**data.components, name: scaling._cache_component(f"{scaling._PREFIX}/fineweb-edu-10B/2026.06.28/train")})
    ds = data.train_set(Axis("position", SEQ), tc.trainer.batch_schedule, key=jax.random.PRNGKey(tc.data_seed))
    return ds, batch, tc.trainer.num_train_steps


def training_stream():
    """The unique-token run's shuffled component before the mixture (an AsyncDataset), its batch size and its step count.
    Its index p is a component position: p // 2048 is the mixture block, read at steps 16 * (p // 2048) .. +15. The 8-pass
    run reads exactly positions 0 .. round(T/8)*B - 1 (about 8 times each), so classify sequences by position here."""
    tc = scaling.muonh_qwen3_run(SIZE).build_config(mock.MagicMock()).train_config
    batch, data = tc.trainer.train_batch_size, tc.data
    assert not data.max_train_batches, data.max_train_batches
    (name,) = [n for n, w in data.train_weights.items() if w > 0]
    if not K:   # the 128K component's path is resolved by the executor at run time; point it at the same prebuilt cache
        tok = os.path.dirname(glob.glob("/scratch/gpfs/GROUP/USER/cache/huggingface/hub/models--marin-community--marin-tokenizer/snapshots/*/tokenizer.json")[0])
        data = dataclasses.replace(data, tokenizer=tok, components={**data.components, name: scaling._cache_component(f"{scaling._PREFIX}/fineweb-edu-10B/2026.06.28/train")})
    _, shuffle_key = jax.random.split(jax.random.PRNGKey(tc.data_seed))   # train_lm: data_key = PRNGKey(data_seed); train_set splits it
    ds = data.train_sets(Axis("position", SEQ), key=shuffle_key, initial_batch_size=batch)[name]
    return ds, batch, tc.trainer.num_train_steps


def main():
    ds, batch, total = training_stream()
    n_seq = len(ds.as_sync_dataset())
    steps = np.unique(np.linspace(0, total - 1, N_STEPS).round().astype(np.int64))
    spare = n_seq // batch - total
    assert spare >= N_TAIL, f"only {spare} unread batches after the {total} steps"
    tail = total + (np.arange(N_TAIL) * (spare // max(N_TAIL, 1))).astype(np.int64)
    every = np.concatenate([steps, tail])
    print(f"{n_seq} sequences, batch {batch}, {total} steps; scoring {len(steps)} steps and {len(tail)} unread batches", flush=True)

    cfg = vtk.model_config()
    Pos = Axis("position", SEQ)
    trainer = vtk.TrainerConfig(mp=vtk.jmp.get_policy("p=f32,c=bfloat16"), per_device_eval_parallelism=vtk.EVAL_BATCH,
                                mesh=vtk.MeshConfig(axes={"data": -1, "replica": 1, "model": 1},
                                                    compute_mapping={"token": (vtk.ResourceAxis.REPLICA_DCN, vtk.ResourceAxis.REPLICA, vtk.ResourceAxis.DATA),
                                                                     "token_repeat": (vtk.ResourceAxis.REPLICA_DCN, vtk.ResourceAxis.REPLICA, vtk.ResourceAxis.DATA)}))
    named = NamedLmDataset(ds, Pos)
    seq_loss = np.zeros((len(every), batch), np.float32)
    seq_weight = np.zeros((len(every), batch), np.float32)
    pos_loss = np.zeros((len(every), KEEP, SEQ), np.float32)
    tokens = np.zeros((len(every), KEEP, SEQ), np.int32)
    with trainer.use_device_mesh(), hax.axis_mapping(trainer.compute_axis_mapping):
        # the tokenizer of the checkpoint's evaluations (vocab_token_losses: the Marin tokenizer or its K-token truncation)
        Vocab = round_axis_for_partitioning(Axis("vocab", len(vtk.data_config().the_tokenizer)), trainer.parameter_axis_mapping)
        model = inference_mode(load_levanter_checkpoint(cfg, vtk.CKPT, Vocab=Vocab, axis_mapping=trainer.parameter_axis_mapping, key=jax.random.PRNGKey(0)), True)
        mp = trainer.mp

        @hax.named_jit(axis_resources=trainer.compute_axis_mapping)
        def per_position(m, ex):
            m = mp.cast_to_compute(m)
            h = m.activations(ex.tokens, attn_mask=ex.attn_mask, key=None)
            loss = maybe_fused_next_token_loss(m.Pos, m.Embed, m.Vocab, h, m.get_lm_head(), ex.tokens, loss_weight=ex.loss_weight,
                                               reduction=None, reduction_axis=(), dtype=jnp.float32)
            return loss.array, ex.loss_weight.array, ex.tokens.array

        t0 = time.time()
        for i, t in enumerate(every):
            sub = named.slice_dataset(start_index=int(t) * batch, end_index=int(t + 1) * batch)
            ls, ws, ts = [], [], []
            for b in DataLoader(sub, trainer.EvalBatch, mesh=trainer.device_mesh, axis_resources=trainer.compute_axis_mapping):
                l, w, tk = per_position(model, b)
                ls.append(np.asarray(jax.device_get(l))); ws.append(np.asarray(jax.device_get(w))); ts.append(np.asarray(jax.device_get(tk)))
            L, W, T = np.concatenate(ls)[:batch], np.concatenate(ws)[:batch], np.concatenate(ts)[:batch]
            seq_loss[i], seq_weight[i] = (L * W).sum(1), W.sum(1)
            pos_loss[i], tokens[i] = L[:KEEP], T[:KEEP]
            if i % 16 == 0 or i == len(every) - 1:
                print(f"step {t}: mean loss {seq_loss[i].sum() / seq_weight[i].sum():.4f} nats ({time.time() - t0:.0f}s)", flush=True)
    os.makedirs(os.path.dirname(vtk.OUT) or ".", exist_ok=True)
    np.savez(vtk.OUT, ckpt=vtk.CKPT, steps=every, total_steps=total, seq_loss=seq_loss, seq_weight=seq_weight, pos_loss=pos_loss, tokens=tokens)
    print("written", vtk.OUT, flush=True)


if __name__ == "__main__":
    main()
