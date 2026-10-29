# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Training re-evaluation curve (TREC; Bergsma, Dey and Hestness 2025, arXiv 2509.25380) of one final checkpoint: its
loss on every batch it trained on, in training order, and on batches after them that it never read.

Every run here reads the same stream: the fineweb-edu-10B cache under the Marin tokenizer (or its K-token truncation),
linearly permuted with data seed 42 and then permuted inside every block of 2048 sequences by MixtureDataset, exactly as
train_lm builds it (training_batches()); batch t = indices t*128 .. t*128+127 of that dataset. The 130m depth control of the pls runs (12 layers, 4959 steps) reads the first
4959 batches of the 300m stream (same cache, seed, permutation and batch), so its tail batches are taken after step
11444, where neither size read, and the two models are scored on the same unread text.

RUN=<W&B run id> (owner popups 2026-10-27: the other MuonH checkpoints, one setting changed at a time) takes the model
shape, batch size, step count and tokenizer from the run's cached W&B config (experiments/trec/fetch_wandb.py) instead of
MODEL. A batch of 128 reads the stream above, 96 the same stream in batches of 96 (-b96), 256 the 520m stream (520m and
1.2b); the 130m runs of 4959 steps or fewer take their tail after step 11444 as above. A run that read its stream more
than once (520m 1.04 passes, 1.2b 2.40, 130m -tpp200 2.60; levanter restarts a finished dataset) has no unread batches
and gets no tail. SUBPATH=model_averaging/model scores the EMA weights of an -ema run instead of the raw ones.

STEPS=all scores the run's T training batches and N_TAIL batches spread evenly over the unread part of the stream;
SAMPLE=s scores 1 in s of the training batches, as blocks of BLOCK consecutive steps every s * BLOCK steps, plus the
last BLOCK steps; STEPS=smoke the last 12 and first 4 training batches and 4 tail batches (the last ones must match the
W&B train loss, since the schedules end at lr 0). Results go to OUT/<first index>.npz per chunk of CHUNK stream indices
(index [n], seq_loss [n, B] summed nats, seq_weight [n, B] loss weights, seconds); a rerun skips the chunks present, so
the job resumes where it stopped.

    CKPT=<.../checkpoints/step-N> MODEL=300m|130m-d12 [VOCAB_K=8000] OUT=<dir> [STEPS=all|smoke] [N_TAIL=512]
    CKPT=<.../checkpoints/step-N> RUN=<run id> [SUBPATH=model_averaging/model] [SAMPLE=8 BLOCK=8] OUT=<dir> [STEPS=all|smoke]
    python -m experiments.references.trec_score
"""

import json
import os
import time

MODEL = os.environ.get("MODEL", "300m")
RUN = os.environ.get("RUN", "")
STEPS = os.environ.get("STEPS", "all")
N_TAIL = int(os.environ.get("N_TAIL", "512"))
SAMPLE = int(os.environ.get("SAMPLE", "1"))
BLOCK = int(os.environ.get("BLOCK", "8"))
TRAIN_RANGE = os.environ.get("TRAIN_RANGE", "")   # a:b scores training batches a..b-1 only (the branch runs of Figure 10)
SUBPATH = os.environ.get("SUBPATH", "model")
CHUNK = int(os.environ.get("CHUNK", "256"))
OUT_DIR = os.environ["OUT"]
os.environ.setdefault("EVAL_BATCH", "32")   # per device; training ran 32 sequences per device with the backward pass
os.environ["N_STEPS"], os.environ["N_TAIL"] = "0", "0"   # vocab_trec's own sampling is not used
os.environ["SIZE"] = "300m"   # the stream: every model here reads the 300m stream (see the docstring)
if RUN:
    CFG = json.load(open(f"/scratch/gpfs/GROUP/USER/project/trec/cache/wandb/{RUN}.json"))["config"]
    BATCH = CFG["trainer"]["train_batch_size"]
    tok = CFG["data"]["tokenizer"]
    os.environ["VOCAB_K"] = "0" if tok == "marin-community/marin-tokenizer" else tok.rsplit("/v", 1)[1]
    os.environ["SIZE"] = {128: "300m", 96: "300m", 256: "520m"}[BATCH]
    if BATCH == 96:
        os.environ["TREC_BATCH"] = "96"
    MODEL = RUN
if MODEL == "130m-d12" and os.environ.get("VOCAB_K", "0") != "0":
    raise ValueError("the 130m depth control was trained on the Marin tokenizer")

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import haliax as hax  # noqa: E402
from haliax import Axis  # noqa: E402
from haliax.partitioning import round_axis_for_partitioning  # noqa: E402
from levanter.checkpoint import load_checkpoint  # noqa: E402
from levanter.data.loader import DataLoader  # noqa: E402
from levanter.data.text.datasets import NamedLmDataset  # noqa: E402
from levanter.layers.attention import AttentionBackend  # noqa: E402
from levanter.models.loss import maybe_fused_next_token_loss  # noqa: E402
from levanter.models.qwen import Qwen3Config  # noqa: E402
from levanter.utils.jax_utils import use_cpu_device  # noqa: E402
from levanter.utils.tree_utils import inference_mode  # noqa: E402

from experiments.references import vocab_trec as vt  # noqa: E402

vtk = vt.vtk
TOTAL = {"130m-d12": 4959}   # the depth control's steps; the 300m runs take theirs from the run config
TAIL_FROM = {"130m-d12": 11444}   # after the 300m run's steps, so both sizes are scored on the same unread batches


def training_batches(Pos):
    """The training examples in the order train_lm read them: LmDataConfig.train_set(Pos, batch_schedule,
    key=PRNGKey(data_seed)), whose index i is sequence i of the run (batch t = indices 128t .. 128t+127). On top of the
    shuffled component that vocab_trec.training_stream() rebuilds, MixtureDataset permutes the sequences inside every
    block of mixture_block_size (2048) sequences with mix_key, even for a single component (randomize_blocks=True); a
    replay of the component alone scores the right 16-step blocks but not the right batches (smoke w439: the last
    batches missed the W&B train loss by up to 0.07 nats)."""
    tc = vt.scaling.muonh_qwen3_run(vt.SIZE).build_config(vt.mock.MagicMock()).train_config
    data = tc.data
    (name,) = [n for n, w in data.train_weights.items() if w > 0]
    if not vt.K:   # as in vocab_trec.training_stream(): the executor resolves this path at run time; the same prebuilt cache
        tok = os.path.dirname(vt.glob.glob("/scratch/gpfs/GROUP/USER/cache/huggingface/hub/models--marin-community--marin-tokenizer/snapshots/*/tokenizer.json")[0])
        data = vt.dataclasses.replace(data, tokenizer=tok, components={**data.components, name: vt.scaling._cache_component(f"{vt.scaling._PREFIX}/fineweb-edu-10B/2026.06.28/train")})
    return data.train_set(Pos, tc.trainer.batch_schedule, key=jax.random.PRNGKey(tc.data_seed))


def model_config():
    if RUN:   # every run here differs from the 300m one only in these fields (W&B configs diffed 2026-10-27)
        m = CFG["model"]
        return Qwen3Config(max_seq_len=vt.SEQ, hidden_dim=m["hidden_dim"], intermediate_dim=m["intermediate_dim"], num_layers=m["num_layers"],
                           num_heads=m["num_heads"], num_kv_heads=m["num_kv_heads"], hybrid_norm=m["hybrid_norm"],
                           attn_backend=AttentionBackend.JAX_FLASH)
    if MODEL == "130m-d12":   # marin-pls della_muonh_qwen3_scaling.py, SIZE=130m LAYERS=12 (run tag -d12)
        return Qwen3Config(max_seq_len=vt.SEQ, hidden_dim=512, intermediate_dim=1792, num_layers=12, num_heads=8, num_kv_heads=8,
                           hybrid_norm=True, attn_backend=AttentionBackend.JAX_FLASH)
    assert MODEL == "300m", MODEL
    return vtk.model_config()


def tail_from(batch, total):
    if RUN and batch == 128 and vt.K == 0 and total <= 11444:   # the 130m runs share the 300m run's unread batches
        return 11444
    return TAIL_FROM.get(MODEL, total)


def plan(n_seq, batch, total):
    """The stream indices to score, in order: training batches, then unread ones."""
    start = tail_from(batch, total)
    spare = n_seq // batch - start
    if STEPS == "smoke":
        train, n_tail = np.r_[0:4, total - 12:total], 4
    else:
        assert STEPS == "all", STEPS
        train, n_tail = np.arange(total), N_TAIL
        if SAMPLE > 1:
            train = np.union1d(train[train % (SAMPLE * BLOCK) < BLOCK], train[-BLOCK:])
        if TRAIN_RANGE:
            a, b = (int(v) for v in TRAIN_RANGE.split(":"))
            train = np.arange(a, b)
    if spare <= 0:   # read more than once: nothing in the stream is unread
        print(f"{total} steps of {batch} read {total * batch / n_seq:.2f} passes: no unread batches", flush=True)
        n_tail, spare = 0, 0
    assert spare >= n_tail, f"only {spare} unread batches after step {start}"
    tail = start + (np.arange(n_tail) * (spare // max(n_tail, 1))).astype(np.int64)
    return np.concatenate([train, tail]).astype(np.int64)


def main():
    ds, batch, total = vt.training_stream()
    total = TOTAL.get(MODEL, total)
    if RUN:
        assert batch == BATCH, (batch, BATCH)
        total = CFG["trainer"]["num_train_steps"]
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
    named = training_batches(Pos)
    assert isinstance(named, NamedLmDataset), type(named)
    with trainer.use_device_mesh(), hax.axis_mapping(trainer.compute_axis_mapping):
        Vocab = round_axis_for_partitioning(Axis("vocab", len(vtk.data_config().the_tokenizer)), trainer.parameter_axis_mapping)
        t0 = time.time()
        with use_cpu_device():   # as levanter.model_loading.load_levanter_checkpoint, with the subpath (raw or EMA weights)
            model = load_checkpoint(eqx.filter_eval_shape(model_config().build, Vocab, key=jax.random.PRNGKey(0)), vtk.CKPT, subpath=SUBPATH)
        model = inference_mode(hax.shard_with_axis_mapping(model, trainer.parameter_axis_mapping), True)
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
            ls, ws = [], []
            # each run of consecutive batches goes through one DataLoader, which prefetches while the GPUs work
            for run in np.split(c, np.where(np.diff(c) != 1)[0] + 1):
                sub = named.slice_dataset(start_index=int(run[0]) * batch, end_index=int(run[-1] + 1) * batch)
                rl, rw = [], []
                for b in DataLoader(sub, trainer.EvalBatch, mesh=trainer.device_mesh, axis_resources=trainer.compute_axis_mapping):
                    l, w = per_sequence(model, b)
                    rl.append(np.asarray(jax.device_get(l)))
                    rw.append(np.asarray(jax.device_get(w)))
                n = len(run) * batch
                rl, rw = np.concatenate(rl), np.concatenate(rw)
                assert len(rl) >= n, (len(rl), n)
                ls.append(rl[:n].reshape(len(run), batch))
                ws.append(rw[:n].reshape(len(run), batch))
            seq_loss, seq_weight = np.concatenate(ls), np.concatenate(ws)
            dt = time.time() - t1
            path = os.path.join(OUT_DIR, f"{c[0]:06d}.npz")
            np.savez(path + ".tmp.npz", ckpt=vtk.CKPT, model=MODEL, subpath=SUBPATH, index=c, total_steps=total, seq_loss=seq_loss,
                     seq_weight=seq_weight, seconds=dt)
            os.replace(path + ".tmp.npz", path)
            mean = seq_loss.sum(1) / seq_weight.sum(1)
            print(f"chunk {c[0]}-{c[-1]}: {dt / len(c):.3f}s per batch, mean loss {mean.mean():.4f} nats, last {mean[-1]:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    print("done", OUT_DIR, flush=True)


if __name__ == "__main__":
    main()
