# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Training re-evaluation curve (TREC; Bergsma, Dey and Hestness 2025, arXiv 2509.25380) of one final checkpoint: its
loss on every batch it trained on, in training order, and on batches after them that it never read.

Every run here reads the same stream: the fineweb-edu-10B cache under the Marin tokenizer (or its K-token truncation),
linearly permuted with data seed 42, batch t = sequences t*128 .. t*128+127 (vocab_trec.training_stream(), checked
against the materialized 8-pass subset). The 130m depth control of the pls runs (12 layers, 4959 steps) reads the first
4959 batches of the 300m stream (same cache, seed, permutation and batch), so its tail batches are taken after step
11444, where neither size read, and the two models are scored on the same unread text.

STEPS=all scores the run's T training batches and N_TAIL batches spread evenly over the unread part of the stream;
STEPS=smoke the last 12 and first 4 training batches and 4 tail batches (the last ones must match the W&B train loss,
since the schedules end at lr 0). Results go to OUT/<first index>.npz per chunk of CHUNK stream indices (index [n],
seq_loss [n, B] summed nats, seq_weight [n, B] loss weights, seconds); a rerun skips the chunks present, so the job
resumes where it stopped.

    CKPT=<.../checkpoints/step-N> MODEL=300m|130m-d12 [VOCAB_K=8000] OUT=<dir> [STEPS=all|smoke] [N_TAIL=512]
    python -m experiments.references.trec_score
"""

import os
import time

MODEL = os.environ.get("MODEL", "300m")
STEPS = os.environ.get("STEPS", "all")
N_TAIL = int(os.environ.get("N_TAIL", "512"))
CHUNK = int(os.environ.get("CHUNK", "256"))
OUT_DIR = os.environ["OUT"]
os.environ.setdefault("EVAL_BATCH", "32")   # per device; training ran 32 sequences per device with the backward pass
os.environ["N_STEPS"], os.environ["N_TAIL"] = "0", "0"   # vocab_trec's own sampling is not used
os.environ["SIZE"] = "300m"   # the stream: every model here reads the 300m stream (see the docstring)
if MODEL == "130m-d12" and os.environ.get("VOCAB_K", "0") != "0":
    raise ValueError("the 130m depth control was trained on the Marin tokenizer")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import haliax as hax  # noqa: E402
from haliax import Axis  # noqa: E402
from haliax.partitioning import round_axis_for_partitioning  # noqa: E402
from levanter.data.loader import DataLoader  # noqa: E402
from levanter.data.text.datasets import NamedLmDataset  # noqa: E402
from levanter.layers.attention import AttentionBackend  # noqa: E402
from levanter.model_loading import load_levanter_checkpoint  # noqa: E402
from levanter.models.loss import maybe_fused_next_token_loss  # noqa: E402
from levanter.models.qwen import Qwen3Config  # noqa: E402
from levanter.utils.tree_utils import inference_mode  # noqa: E402

from experiments.references import vocab_trec as vt  # noqa: E402

vtk = vt.vtk
TOTAL = {"130m-d12": 4959}   # the depth control's steps; the 300m runs take theirs from the run config
TAIL_FROM = {"130m-d12": 11444}   # after the 300m run's steps, so both sizes are scored on the same unread batches


def model_config():
    if MODEL == "130m-d12":   # marin-pls della_muonh_qwen3_scaling.py, SIZE=130m LAYERS=12 (run tag -d12)
        return Qwen3Config(max_seq_len=vt.SEQ, hidden_dim=512, intermediate_dim=1792, num_layers=12, num_heads=8, num_kv_heads=8,
                           hybrid_norm=True, attn_backend=AttentionBackend.JAX_FLASH)
    assert MODEL == "300m", MODEL
    return vtk.model_config()


def plan(n_seq, batch, total):
    """The stream indices to score, in order: training batches, then unread ones."""
    tail_from = TAIL_FROM.get(MODEL, total)
    spare = n_seq // batch - tail_from
    if STEPS == "smoke":
        train, n_tail = np.r_[0:4, total - 12:total], 4
    else:
        assert STEPS == "all", STEPS
        train, n_tail = np.arange(total), N_TAIL
    assert spare >= n_tail, f"only {spare} unread batches after step {tail_from}"
    tail = tail_from + (np.arange(n_tail) * (spare // max(n_tail, 1))).astype(np.int64)
    return np.concatenate([train, tail]).astype(np.int64)


def main():
    ds, batch, total = vt.training_stream()
    total = TOTAL.get(MODEL, total)
    n_seq = len(ds.as_sync_dataset())
    every = plan(n_seq, batch, total)
    chunks = [every[i:i + CHUNK] for i in range(0, len(every), CHUNK)]
    todo = [c for c in chunks if not os.path.exists(os.path.join(OUT_DIR, f"{c[0]:06d}.npz"))]
    print(f"{MODEL} {vtk.CKPT}: {n_seq} sequences, batch {batch}, {total} steps; {len(every)} batches in {len(chunks)} chunks, "
          f"{len(todo)} to do", flush=True)
    if not todo:
        return
    os.makedirs(OUT_DIR, exist_ok=True)

    Pos = Axis("position", vt.SEQ)
    trainer = vtk.TrainerConfig(mp=vtk.jmp.get_policy("p=f32,c=bfloat16"), per_device_eval_parallelism=vtk.EVAL_BATCH,
                                mesh=vtk.MeshConfig(axes={"data": -1, "replica": 1, "model": 1},
                                                    compute_mapping={"token": (vtk.ResourceAxis.REPLICA_DCN, vtk.ResourceAxis.REPLICA, vtk.ResourceAxis.DATA),
                                                                     "token_repeat": (vtk.ResourceAxis.REPLICA_DCN, vtk.ResourceAxis.REPLICA, vtk.ResourceAxis.DATA)}))
    named = NamedLmDataset(ds, Pos)
    with trainer.use_device_mesh(), hax.axis_mapping(trainer.compute_axis_mapping):
        Vocab = round_axis_for_partitioning(Axis("vocab", len(vtk.data_config().the_tokenizer)), trainer.parameter_axis_mapping)
        t0 = time.time()
        model = inference_mode(load_levanter_checkpoint(model_config(), vtk.CKPT, Vocab=Vocab, axis_mapping=trainer.parameter_axis_mapping,
                                                        key=jax.random.PRNGKey(0)), True)
        print(f"loaded in {time.time() - t0:.0f}s", flush=True)
        mp = trainer.mp

        @hax.named_jit(axis_resources=trainer.compute_axis_mapping)
        def per_sequence(m, ex):
            m = mp.cast_to_compute(m)
            h = m.activations(ex.tokens, attn_mask=ex.attn_mask, key=None)
            loss = maybe_fused_next_token_loss(m.Pos, m.Embed, m.Vocab, h, m.get_lm_head(), ex.tokens, loss_weight=ex.loss_weight,
                                               reduction=None, reduction_axis=(), dtype=jnp.float32)
            return (loss.array * ex.loss_weight.array).sum(-1), ex.loss_weight.array.sum(-1)

        for c in todo:
            t1 = time.time()
            seq_loss = np.zeros((len(c), batch), np.float32)
            seq_weight = np.zeros((len(c), batch), np.float32)
            for i, t in enumerate(c):
                sub = named.slice_dataset(start_index=int(t) * batch, end_index=int(t + 1) * batch)
                ls, ws = [], []
                for b in DataLoader(sub, trainer.EvalBatch, mesh=trainer.device_mesh, axis_resources=trainer.compute_axis_mapping):
                    l, w = per_sequence(model, b)
                    ls.append(np.asarray(jax.device_get(l)))
                    ws.append(np.asarray(jax.device_get(w)))
                seq_loss[i], seq_weight[i] = np.concatenate(ls)[:batch], np.concatenate(ws)[:batch]
            dt = time.time() - t1
            path = os.path.join(OUT_DIR, f"{c[0]:06d}.npz")
            np.savez(path + ".tmp.npz", ckpt=vtk.CKPT, model=MODEL, index=c, total_steps=total, seq_loss=seq_loss,
                     seq_weight=seq_weight, seconds=dt)
            os.replace(path + ".tmp.npz", path)
            mean = seq_loss.sum(1) / seq_weight.sum(1)
            print(f"chunk {c[0]}-{c[-1]}: {dt / len(c):.3f}s per batch, mean loss {mean.mean():.4f} nats, last {mean[-1]:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    print("done", OUT_DIR, flush=True)


if __name__ == "__main__":
    main()
