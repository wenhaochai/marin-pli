# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Training re-evaluation curve (TREC; Bergsma, Dey and Hestness 2025, arXiv 2509.25380) of one checkpoint on the
unique-token run's training stream (owner 2026-10-26: read the training data again with the final weights, to see which
part of it the repeated run does worse on).

The unique-token run reads its shuffled training set in order: batch t is sequences t*B .. t*B+B-1 of the fineweb-edu
component after its linear permutation (data seed 42; MixtureDataset of one component keeps the order). Its 8-pass
partner reads the first T/8 of those batches 8 times in the same order (vocab_materialize_passes.py rebuilds that subset
the same way and checks it). This scores CKPT on the batches of N_STEPS steps spread evenly over the unique run's T steps
and on N_TAIL batches spread over the sequences after them, which neither run of a pair has read (held-out text from
the same shuffled set). For every sequence it keeps the summed loss in nats and the loss weight, and for the first KEEP
sequences of each batch the per-position loss and the tokens, for token-frequency splits.

Output npz: steps [n] (training steps; tail batches carry their stream index, all >= T), total_steps (T), seq_loss
[n, B], seq_weight [n, B], pos_loss [n, KEEP, SEQ_LEN], tokens [n, KEEP, SEQ_LEN], ckpt.

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
if os.environ.get("TREC_BATCH"):   # trec_score: a run trained at another batch size (-b96) reads the same stream in other batches
    os.environ["BATCH"] = os.environ["TREC_BATCH"]
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


def training_stream():
    """The unique-token run's shuffled training sequences (an AsyncDataset), its batch size and its step count."""
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
