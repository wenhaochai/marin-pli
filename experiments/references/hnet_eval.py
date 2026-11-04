# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Paloma scores of a unique-text H-Net and its 8-epoch partner under swapped and uncapped boundaries (owner popup
2026-11-03): why is the H-Net's multi-epoch regret (0.031 bits per byte at 300m) above the plain bytes' (0.0175)?

Loads two checkpoints of one H-Net configuration (CKPT_U: the unique-text run, CKPT_R: the 8-epoch run; read-only), builds
the run's model config and Paloma data through the scaling script itself (della_muonh_qwen3_scaling.py with the run's
environment, parsed from RUN_U), and scores every window of the 16 Paloma subsets, as the trainer's evaluator cuts them
(the v256 caches, SEQ_LEN-byte windows, the eval loss path: the compute policy p=f32,c=bfloat16, no key), under these conditions:
  own_u, own_r      each model with its own boundaries, as trained (must reproduce the logged final results);
  swap_u, swap_r    each model with the other model's routing output p (so its boundaries and EMA weights); the chunk
                    vectors still come from the model's own encoder (HNetQwen3LMHeadModel.activations route_p);
  big_u, big_r      each model with its own boundaries and enough slots for every boundary of every window (slots = the
                    window, eval only), to measure what overflow costs.
Saves, per subset, the log-probability (nats) of every scored byte under every condition (float32, [windows, SEQ_LEN];
position t holds the log-probability of byte t + 1, weight 0 where nothing is scored), the loss weights, the window bytes
(int16) and both models' boundaries (bool, before the slot cap) and slot overflow, to OUT/<subset>.npz, and per condition the
exact bits per byte (the page's formula: loss x T / (ln 2 x B) per subset, then the macro mean over 16 subsets) to
OUT/summary.json. Subsets can be split over processes with SUBSETS; ``merge`` combines their summaries.

Then (REP_STRIDE > 0, default 1) it routes the repeated eighth of the training text, the windows the 8-epoch run reads 8
times: the first round(steps / 8) x batch windows of the unique run's shuffled training data (linear permutation, data seed
42, as hnet_route.py finds them), every REP_STRIDE-th of them, with both models, and saves their boundaries (before the slot
cap; np.packbits per window) and the windows' raw cache indices to OUT/rep.npz, from which analyze_eval.py counts how often
each chunk string occurs there. Both models' routing (embedding, encoder, q, k) on the 3.6B bytes is the cheap part of the
job (a 2-layer encoder pass per model).

    RUN_U=<run id> CKPT_U=<.../step-N> CKPT_R=<.../step-N> OUT=<dir> [SUBSETS=a,b] [MAX_WINDOWS=n] [EVAL_BATCH=1] [REP_STRIDE=1]
    REP_ONLY=1 ... python -m experiments.references.hnet_eval    (the repeated-eighth routing alone, on every visible GPU)
    python -m experiments.references.hnet_eval merge             (the shards' summaries -> summary.json, exact bits per byte)
    python -m experiments.references.hnet_eval
MP (default p=f32,c=bfloat16, the trainer's policy) sets the precision. A CPU test: TEST_TINY=1 builds two tiny random models that differ only in q_proj (so only in their routing), with few
slots, instead of loading checkpoints, and checks: swap_u equals own_r and swap_r equals own_u (with the other model's p
nothing of the model's own routing is left); big equals own on windows that do not overflow and differs on those that do.
"""

import asyncio
import json
import math
import os
import re
import sys
import time
from unittest import mock

import numpy as np

RUN_U = os.environ["RUN_U"]
OUT = os.environ["OUT"]
SUBSETS = [x for x in os.environ.get("SUBSETS", "").split(",") if x]
MAX_WINDOWS = int(os.environ.get("MAX_WINDOWS", "0"))
EVAL_BATCH = int(os.environ.get("EVAL_BATCH", "1"))   # windows per step (per device; the batch is sharded over the GPUs)
TINY = os.environ.get("TEST_TINY") == "1"
REP_STRIDE = int(os.environ.get("REP_STRIDE", "1"))
REP_MAX = int(os.environ.get("REP_MAX", "0"))   # test: at most this many repeated-eighth windows
REP_ONLY = os.environ.get("REP_ONLY") == "1"    # only the repeated-eighth routing (a job runs it after the Paloma shards)
SUMMARY_ONLY = os.environ.get("SUMMARY_ONLY") == "1"   # no per-byte npz files, only the summaries (a rescore of a pair)
STORE = os.environ.get("MARIN_PREFIX", "/scratch/gpfs/GROUP/USER/marin_store_big")
TABLES = os.environ.get("BPB_TABLES", "/scratch/gpfs/GROUP/USER/tmp_plots/vocab/bpb_exact_tables.json")

# the run's environment, from its id (the H-Net run ids of della_muonh_qwen3_scaling.py)
m = re.fullmatch(r"muonh-qwen3-(\w+)-della(\d)xh100-hnet(\d+)x(\d+)n([\d.]+)((?:-[a-z][\w.]*?)*)-bytes-sl(\d+)(?:-s(\d+))?", RUN_U)
if not m:
    raise ValueError(f"RUN_U={RUN_U}: not a unique-text H-Net run id")
SIZE, SEQ_LEN = m.group(1), int(m.group(7))
TAGS = [t for t in m.group(6).split("-") if t]
# EVAL_ATTN: the attention backend; fa4 (the default) is what the H-Net runs trained and evaluated with (logs/plans/hnet-*.txt);
# jax_flash (block 128: the slots, e.g. 5248, are multiples of 128, not of the default 1024) as a fallback; vanilla for CPU
EVAL_ATTN = os.environ.get("EVAL_ATTN", "fa4")
if EVAL_ATTN not in ("fa4", "jax_flash", "vanilla"):
    raise ValueError(f"EVAL_ATTN={EVAL_ATTN}: fa4, jax_flash or vanilla")
ENV = dict(SIZE=SIZE, VARIANT="hnet", VOCAB_K="256", SEQ_LEN=str(SEQ_LEN), ATTN_BACKEND="fa4", HNET_ENC=m.group(3),
           HNET_DEC=m.group(4), HNET_N=m.group(5), NUM_GPUS=m.group(2), DRY_RUN="1", MEMSTATS_EVERY="0")
if m.group(8):
    ENV["SEED"] = m.group(8)
for t in TAGS:
    if t.startswith("r"):
        ENV["HNET_ROUTE"] = t[1:]
    elif t.startswith("sm"):
        ENV["HNET_SLOT_MARGIN"] = t[2:]
    elif t.startswith("chfz"):
        ENV["HNET_FREEZE_FROM_PASS"] = t[4:]
    elif t in ("nrg", "nores", "noema"):
        ENV[{"nrg": "HNET_NO_ROUTE_GRAD", "nores": "HNET_NO_RESIDUAL", "noema": "HNET_NO_EMA"}[t]] = "1"
    else:
        raise ValueError(f"RUN_U={RUN_U}: unknown tag -{t}")
for k in ("DATA_EPOCHS", "PASS_DATA", "SMOKE_STEPS", "TOTAL_STEPS", "BATCH", "TPP", "SEED", "VOCAB_SET", "EMA_BETA", "RUN_TAG", "PDP"):
    os.environ.pop(k, None)
os.environ.update(ENV)
if TINY:
    SEQ_LEN = int(os.environ.get("TEST_SEQ_LEN", "1024"))
    os.environ["SEQ_LEN"] = str(SEQ_LEN)

import dataclasses  # noqa: E402

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jmp  # noqa: E402

import haliax as hax  # noqa: E402
from haliax import Axis  # noqa: E402
from haliax.partitioning import ResourceAxis  # noqa: E402
from levanter.data._prp import Permutation  # noqa: E402
from levanter.data.loader import DataLoader  # noqa: E402
from levanter.store.cache import TreeCache  # noqa: E402
from levanter.utils.jax_utils import key_iterator  # noqa: E402
from levanter.layers.attention import AttentionBackend, AttentionMask  # noqa: E402
from levanter.model_loading import load_levanter_checkpoint  # noqa: E402
from levanter.models.loss import maybe_fused_next_token_loss  # noqa: E402
from levanter.trainer import TrainerConfig  # noqa: E402
from levanter.utils.mesh import MeshConfig  # noqa: E402
from levanter.utils.tree_utils import inference_mode  # noqa: E402

from experiments.references import della_muonh_qwen3_scaling as scaling  # noqa: E402
from experiments.references.hnet_qwen3 import HNetQwen3LMHeadModel  # noqa: E402

CONDITIONS = ("own_u", "own_r", "swap_u", "swap_r", "big_u", "big_r")


def run_config():
    """(model config, Paloma data config, repeated-eighth window count, training cache dir, data seed) of the unique-text
    run, from the scaling script."""
    step = scaling.muonh_qwen3_run(SIZE)
    if not TINY and step.name != f"speedrun/{RUN_U}":
        raise ValueError(f"the scaling script builds {step.name}, not speedrun/{RUN_U}")
    tc = step.build_config(mock.MagicMock()).train_config
    cfg = {"fa4": tc.model, "vanilla": dataclasses.replace(tc.model, attn_backend=AttentionBackend.VANILLA),
           "jax_flash": dataclasses.replace(tc.model, attn_backend=AttentionBackend.JAX_FLASH, flash_attention_block_size=128)}[EVAL_ATTN]
    if TINY:
        cfg = dataclasses.replace(cfg, hidden_dim=64, intermediate_dim=128, num_layers=2, num_heads=4, num_kv_heads=4,
                                  enc_layers=1, dec_layers=1, attn_backend=AttentionBackend.VANILLA, chunk_ratio=4.74,
                                  chunk_slots=int(os.environ.get("TEST_SLOTS", "256")))
    data = dataclasses.replace(tc.data, shuffle=False)
    (comp,) = [c for n, c in tc.data.components.items() if tc.data.train_weights[n] > 0]
    rep = round(tc.trainer.num_train_steps / 8) * tc.trainer.train_batch_size   # the 8-epoch run's max_train_batches x batch
    return cfg, data, rep, os.path.join(comp.cache_dir, "train"), tc.data_seed


def exact_tables():
    tb = json.load(open(TABLES))["256"]
    return tb.get(str(SEQ_LEN))


def main():
    cfg, data, rep, train_dir, data_seed = run_config()
    big = dataclasses.replace(cfg, chunk_slots=SEQ_LEN)
    Pos = Axis("position", SEQ_LEN)
    trainer = TrainerConfig(mp=jmp.get_policy(os.environ.get("MP", "p=f32,c=bfloat16")), per_device_eval_parallelism=EVAL_BATCH,
                            mesh=MeshConfig(axes={"data": -1, "replica": 1, "model": 1},
                                            compute_mapping={"token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
                                                             "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA)}))
    os.makedirs(OUT, exist_ok=True)
    print(f"{RUN_U}: env {ENV}; slots {cfg.slots}, uncapped {big.slots}; window {SEQ_LEN}; policy {trainer.mp}", flush=True)
    with trainer.use_device_mesh(), hax.axis_mapping(trainer.compute_axis_mapping):
        Vocab = Axis("vocab", 512)
        if TINY:
            mu = HNetQwen3LMHeadModel.init(Vocab, cfg, key=jax.random.PRNGKey(0))
            mr = HNetQwen3LMHeadModel.init(Vocab, cfg, key=jax.random.PRNGKey(0))
            noise = lambda w, k: hax.named(w.array + 0.3 * jax.random.normal(jax.random.PRNGKey(k), w.array.shape), w.axes)  # noqa: E731
            mu = eqx.tree_at(lambda x: x.q_proj.weight, mu, noise(mu.q_proj.weight, 1))
            mr = eqx.tree_at(lambda x: x.q_proj.weight, mr, noise(mr.q_proj.weight, 2))
        else:
            load = lambda c: inference_mode(load_levanter_checkpoint(cfg, c, Vocab=Vocab, axis_mapping=trainer.parameter_axis_mapping,  # noqa: E731
                                                                     key=jax.random.PRNGKey(0)), True)
            mu, mr = load(os.environ["CKPT_U"]), load(os.environ["CKPT_R"])
        mp = trainer.mp
        # the same arrays under the uncapped config (the model reads its slots from the main network's static config)
        as_big = lambda mdl: eqx.tree_at(lambda x: x.transformer, mdl, dataclasses.replace(mdl.transformer, config=big))  # noqa: E731
        bu, br = as_big(mu), as_big(mr)
        for x in (bu, br):
            assert x.config.slots == SEQ_LEN, x.config.slots

        @hax.named_jit(axis_resources=trainer.compute_axis_mapping)
        def step(mu, mr, bu, br, ex):
            mu, mr, bu, br = (mp.cast_to_compute(x) for x in (mu, mr, bu, br))
            pu = mu.routing(ex.tokens, ex.attn_mask)
            pr = mr.routing(ex.tokens, ex.attn_mask)

            def logp(mdl, route_p=None):
                h = mdl.activations(ex.tokens, ex.attn_mask, key=None, route_p=route_p)
                loss = maybe_fused_next_token_loss(mdl.Pos, mdl.Embed, mdl.Vocab, h, mdl.get_lm_head(), mdl.split_ids(ex.tokens)[0],
                                                   loss_weight=ex.loss_weight, reduction=None, reduction_axis=(), dtype=jnp.float32)
                return -loss.array

            lp = jnp.stack([logp(mu), logp(mr), logp(mu, pr), logp(mr, pu), logp(bu), logp(br)])
            return lp, ex.loss_weight.array, ex.tokens.array, pu > 0.5, pr > 0.5

        if REP_ONLY:
            route_rep(mu, mr, mp, trainer, rep, train_dir, data_seed)
            return
        tables = exact_tables()
        summary = {}
        for name, ds in data.validation_sets(Pos).items():
            sub = name.split("/", 1)[1].rsplit("-marin-tokenizer", 1)[0] if "/" in name else name
            if SUBSETS and sub not in SUBSETS:
                continue
            t0 = time.time()
            n = len(ds.as_sync_dataset())
            if MAX_WINDOWS:
                n = min(n, MAX_WINDOWS)
                ds = ds.slice_dataset(end_index=n)
            parts = []
            for batch in DataLoader(ds, trainer.EvalBatch, mesh=trainer.device_mesh, axis_resources=trainer.compute_axis_mapping):
                parts.append([np.asarray(jax.device_get(a)) for a in step(mu, mr, bu, br, batch)])
            lp, w, tok, bu_, br_ = (np.concatenate([p[i] for p in parts], axis=1 if i == 0 else 0) for i in range(5))
            lp, w, tok, bu_, br_ = lp[:, :n], w[:n], tok[:n], bu_[:n], br_[:n]
            tok = np.where(tok >= 512, tok - 512, tok)   # route="bpe": the byte ids without the token-start flag
            if TINY:
                tiny_checks(lp, bu_, br_, cfg.slots)
            T = float(w.sum())
            loss = {c: float(-(lp[i] * w).sum() / T) for i, c in enumerate(CONDITIONS)}
            over_u, over_r = bu_.sum(1) - cfg.slots, br_.sum(1) - cfg.slots
            if not SUMMARY_ONLY:
                np.savez(f"{OUT}/{sub}.npz", logp=lp.astype(np.float32), conditions=np.array(CONDITIONS), weight=w.astype(np.float16),
                         tokens=tok.astype(np.int16), b_u=bu_, b_r=br_, slots=cfg.slots)
            summary[sub] = dict(windows=n, scored=T, scored_bytes=float((w * (np.roll(tok, -1, axis=1) < 256)).sum()), loss=loss, overflow_windows_u=int((over_u > 0).sum()), overflow_windows_r=int((over_r > 0).sum()),
                                bytes_per_chunk_u=float(bu_.size / bu_.sum()), bytes_per_chunk_r=float(br_.size / br_.sum()))
            print(f"{sub}: {n} windows, {T:.0f} scored bytes, loss (nats/byte) " + " ".join(f"{c} {v:.4f}" for c, v in loss.items())
                  + f"; overflow windows u {summary[sub]['overflow_windows_u']} r {summary[sub]['overflow_windows_r']} ({time.time() - t0:.0f}s)", flush=True)
        if REP_STRIDE and not SUBSETS:
            route_rep(mu, mr, mp, trainer, rep, train_dir, data_seed)
        shard = "_".join(SUBSETS) if SUBSETS else "all"
        json.dump(dict(run_u=RUN_U, ckpt_u=os.environ.get("CKPT_U"), ckpt_r=os.environ.get("CKPT_R"), slots=cfg.slots,
                       subsets=summary), open(f"{OUT}/summary_{shard}.json", "w"), indent=1)
        compare_logged(summary)
        if not SUBSETS and tables and not MAX_WINDOWS:
            print_bpb(summary, tables)


def route_rep(mu, mr, mp, trainer, rep, train_dir, data_seed):
    """Both models' boundaries on the repeated eighth (see the module docstring), to OUT/rep.npz."""
    t0 = time.time()
    c = TreeCache.load(train_dir, {"input_ids": np.zeros((0,), dtype=np.int32)})
    nw = c.store.tree["input_ids"].data_size // SEQ_LEN
    # train_lm: data_key = PRNGKey(data_seed); LmDataConfig.train_set: mix_key, shuffle_key = split(data_key); train_sets: the
    # first key of key_iterator(shuffle_key) shuffles the single component with the linear permutation (hnet_route.py)
    _, shuffle_key = jax.random.split(jax.random.PRNGKey(data_seed))
    perm = Permutation.make("linear", nw, next(key_iterator(shuffle_key)))
    sh = np.arange(0, min(rep, REP_MAX or rep), REP_STRIDE)
    raw = np.asarray(perm(sh)).astype(np.int64)
    nd = len(jax.devices())
    B = Axis("batch", nd * EVAL_BATCH)
    Pos = Axis("position", SEQ_LEN)

    @hax.named_jit(axis_resources=trainer.compute_axis_mapping)
    def route(mu, mr, ids, seg):
        mu, mr = mp.cast_to_compute(mu), mp.cast_to_compute(mr)
        tok = hax.named(ids, (B, Pos))
        mask = AttentionMask.causal().with_segment_ids(hax.named(seg, (B, Pos)))
        return mu.routing(tok, mask) > 0.5, mr.routing(tok, mask) > 0.5

    eos = 257
    io = 64 * B.size   # windows per read
    bu, br = [], []
    for a0 in range(0, len(raw), io):
        block = np.stack([np.asarray(r, np.int32) for r in asyncio.run(c.get_flat_field_batch("input_ids", (raw[a0:a0 + io] * SEQ_LEN).tolist(), SEQ_LEN))])
        for a in range(0, len(block), B.size):
            k = min(B.size, len(block) - a)
            ids = np.zeros((B.size, SEQ_LEN), np.int32)
            ids[:k] = block[a:a + k]
            prev = np.roll(ids, 1, axis=1) == eos
            prev[:, 0] = False
            seg = np.cumsum(prev, axis=1).astype(np.int32)   # CausalLmDataset's segments: a new document after each EOS
            u, r = route(mu, mr, jnp.asarray(ids), jnp.asarray(seg))
            bu.append(np.packbits(np.asarray(jax.device_get(u))[:k], axis=1))
            br.append(np.packbits(np.asarray(jax.device_get(r))[:k], axis=1))
        print(f"  repeated eighth: {a0 + len(block)}/{len(raw)} windows ({time.time() - t0:.0f}s)", flush=True)
    np.savez(f"{OUT}/rep.npz", raw=raw, shuffled=sh, stride=REP_STRIDE, rep=rep, seq_len=SEQ_LEN, train_dir=train_dir,
             b_u=np.concatenate(bu), b_r=np.concatenate(br))
    print(f"repeated eighth: {len(raw)} windows (every {REP_STRIDE}th of {rep}) routed by both models, {time.time() - t0:.0f}s", flush=True)


def tiny_checks(lp, b_u, b_r, slots):
    c = {k: i for i, k in enumerate(CONDITIONS)}
    swap = np.array_equal(lp[c["swap_u"]], lp[c["own_r"]]) and np.array_equal(lp[c["swap_r"]], lp[c["own_u"]])
    over = b_u.sum(1) > slots
    d = np.abs(lp[c["big_u"]] - lp[c["own_u"]]).max(axis=1)
    print(f"  TEST swap identity {'ok' if swap else 'BAD'}; big vs own max |d logp| on windows without overflow "
          f"{d[~over].max() if (~over).any() else float('nan'):.1e}, with overflow {d[over].max() if over.any() else float('nan'):.1e} "
          f"({int(over.sum())} of {len(over)} overflow)", flush=True)
    assert swap
    assert not (~over).any() or d[~over].max() < (2e-3 if os.environ.get("MP") == "p=f32,c=f32" else 1e-1)
    assert not over.any() or d[over].max() > 1e-2


def logged_losses(ckpt):
    """The run's own final Paloma losses per subset (eval_metrics.jsonl beside the checkpoints, the line of the
    checkpoint's step), or None."""
    if not ckpt:
        return None
    path = os.path.join(os.path.dirname(ckpt.rstrip("/")), "eval_metrics.jsonl")
    step = int(re.search(r"step-(\d+)", ckpt).group(1))
    if not os.path.exists(path):
        return None
    rows = [json.loads(x) for x in open(path) if x.strip()]
    rows = [r for r in rows if r.get("step") == step]
    if not rows:
        return None
    return {k.split("/")[2].rsplit("-marin-tokenizer", 1)[0]: v for k, v in rows[-1].items()
            if k.startswith("eval/paloma/") and k.endswith("-marin-tokenizer/loss")}


def compare_logged(summary):
    """own_u and own_r against the runs' logged losses: per subset |d| and the macro exact bits per byte of both."""
    out = {}
    for c, ck in (("own_u", os.environ.get("CKPT_U")), ("own_r", os.environ.get("CKPT_R"))):
        lg = logged_losses(ck)
        if not lg:
            print(f"{c}: no logged losses found beside {ck}", flush=True)
            continue
        d = {s: summary[s]["loss"][c] - lg[s] for s in summary if s in lg}
        out[c] = d
        worst = max(d.items(), key=lambda kv: abs(kv[1]))
        print(f"{c} against the run's logged final losses: {len(d)} subsets, max |d| {abs(worst[1]):.5f} nats ({worst[0]}), "
              f"mean d {np.mean(list(d.values())):+.5f}", flush=True)
    return out


def print_bpb(summary, tables):
    """Exact macro bits per byte per condition: per subset loss x T / (ln 2 x B), T and B from the page's tables (which must
    count the same scored targets); without a table for this window (a test), B = the scored targets that are bytes."""
    out = {}
    if tables is None:
        print(f"no exact table for a {SEQ_LEN}-byte window: bits per byte over the scored targets (a test only)", flush=True)
        tables = {s: (v["scored"], v["scored_bytes"]) for s, v in summary.items()}
    for s in summary:
        assert abs(summary[s]["scored"] - tables[s][0]) < 1e-6 * tables[s][0], (s, summary[s]["scored"], tables[s][0])   # "scored" is a float32 sum, which rounds near 4e7
    for c in CONDITIONS:
        out[c] = sum(summary[s]["loss"][c] * tables[s][0] / (math.log(2) * tables[s][1]) for s in summary) / len(summary)
    print("exact macro bits per byte: " + " ".join(f"{c} {v:.4f}" for c, v in out.items()), flush=True)
    return out


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "merge":   # python -m experiments.references.hnet_eval merge: the shards' summaries
        shards = [json.load(open(os.path.join(OUT, f))) for f in sorted(os.listdir(OUT)) if f.startswith("summary_") and f.endswith(".json")]
        subs = {k: v for sh in shards for k, v in sh["subsets"].items()}
        assert len(subs) == 16, sorted(subs)
        compare_logged(subs)
        bpb = print_bpb(subs, exact_tables())
        json.dump(dict(shards[0], subsets=subs, exact_bpb=bpb), open(os.path.join(OUT, "summary.json"), "w"), indent=1)
        print("written", os.path.join(OUT, "summary.json"))
    else:
        main()
