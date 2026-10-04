# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Seen vs unseen training data: does OV's loss gap under data repetition come from memorization?

Rebuilds the training run's shuffled fineweb-edu-10B order (data seed 42, linear permutation, as in
della_muonh_qwen3_scaling) and evaluates one checkpoint on SLICES evenly spaced slices of that order, SEQS sequences
each. Sequence i of the order is trained on at mixture steps k * S + i (k = 0, 1, ...) below the run's total, where S is
the subset the run cycles through (all of it, or steps / DATA_EPOCHS batches); so each slice has a known number of
passes (0 = never seen) and a known last-seen step. Paloma subsets (HELDOUT) give an out-of-distribution reference.

Modes: ``full`` (the model as trained) and, for OV checkpoints, ``oe_scramble``: every n-gram index shifted by a fixed
offset mod m, so each position reads the row of an unrelated n-gram. How much scrambling costs on seen versus unseen
slices measures how much of the tables' use is specific to n-grams that were trained on.

    CKPT=<.../checkpoints/step-N> SIZE=300m MODEL=ov [OV_M=12.8e6 OV_OD_W=0.1 OD_MODE=product OD_M=12.8e6]
    DATA_EPOCHS=0 SLICES=16 SEQS=256 HELDOUT="c4_en" MODES="full oe_scramble" python -m experiments.references.ov_memorization_eval
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
from levanter.data.text.datasets import DatasetComponent, LmDataConfig, NamedLmDataset, UrlDatasetSourceConfig
from levanter.data.text.formats import TextLmDatasetFormat
from levanter.eval import TaggedEvaluator, eval_model
from levanter.layers.attention import AttentionBackend
from levanter.model_loading import load_levanter_checkpoint
from levanter.models.loss import maybe_fused_next_token_loss
from levanter.models.qwen import Qwen3Config
from levanter.schedule import BatchSchedule
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from levanter.utils.tree_utils import inference_mode

from experiments.marin_tokenizer import marin_tokenizer
from experiments.references.della_muonh_qwen3_scaling import SEQ_LEN, SIZES
import experiments.references.over_vocab_qwen3 as ovq
from experiments.references.over_vocab_qwen3 import ROWS, OverVocabQwen3Config, _segment_ids, ngram_index, shifted_tokens

CKPT = os.environ["CKPT"]
SIZE = os.environ.get("SIZE", "300m")
MODEL = os.environ.get("MODEL", "base")  # base (baseline or ss: same parameters) or ov
PREFIX = os.environ.get("MARIN_PREFIX", "/scratch/gpfs/GROUP/USER/marin_store_big")
DATA_EPOCHS = float(os.environ.get("DATA_EPOCHS", "0"))
SLICES = int(os.environ.get("SLICES", "16"))
SEQS = int(os.environ.get("SEQS", "256"))
HELDOUT = os.environ.get("HELDOUT", "c4_en").split()
MODES = os.environ.get("MODES", "full oe_scramble" if MODEL == "ov" else "full").split()
DATA_SEED = 42
TRAIN_NAME = "fineweb-edu-10B"
RUN = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(CKPT.rstrip("/")))))
OUT = os.environ.get("OUT", f"logs/ov_memorization_{RUN}_{os.path.basename(CKPT.rstrip('/'))}.json")


def component(path: str) -> DatasetComponent:
    source = UrlDatasetSourceConfig(tags=[], train_urls=[], validation_urls=[], cache_dir=path, format=TextLmDatasetFormat())
    return DatasetComponent(source=source, cache_dir=path, format=source.format, tags=[])


def data_config(max_train_batches=None) -> LmDataConfig:
    """The run's training data: fineweb-edu-10B alone at weight 1, plus the HELDOUT Paloma subsets at weight 0."""
    components = {TRAIN_NAME: component(os.path.join(PREFIX, "fineweb-edu-10B", "2026.06.28"))}
    paloma = os.path.join(PREFIX, "tokenized", "paloma")
    for d in sorted(os.listdir(paloma)):
        if d.rsplit("-", 1)[0] in HELDOUT and os.path.isdir(os.path.join(paloma, d, "validation")):
            components[d.rsplit("-", 1)[0]] = component(os.path.join(paloma, d))
    return LmDataConfig(
        components=components,
        train_weights={n: (1.0 if n == TRAIN_NAME else 0.0) for n in components},
        tokenizer=marin_tokenizer,
        cache_dir=None,
        shuffle=True,
        permutation_type="linear",
        max_train_batches=max_train_batches,
    )


def model_config(s) -> Qwen3Config:
    common = dict(max_seq_len=SEQ_LEN, hidden_dim=s["hidden"], intermediate_dim=s["inter"], num_layers=s["layers"], num_heads=s["heads"],
                  num_kv_heads=s["kv"], hybrid_norm=True, attn_backend=AttentionBackend.JAX_FLASH)
    if MODEL == "base":
        return Qwen3Config(**common)
    return OverVocabQwen3Config(**common, oe_m=int(float(os.environ.get("OV_M", "12.8e6"))), od_weight=float(os.environ.get("OV_OD_W", "0.1")),
                                od_mode=os.environ.get("OD_MODE", "product"), od_m=int(float(os.environ.get("OD_M", "12.8e6"))))


def scrambled_embed(m, input_ids, attn_mask):
    """OverVocabQwen3LMHeadModel.embed with every n-gram index shifted by about m / 2 (mod m)."""
    cfg = m.config
    Pos = input_ids.resolve_axis(m.Pos.name)
    x = m.embeddings.token_embeddings(input_ids)
    prev = shifted_tokens(input_ids, _segment_ids(attn_mask), Pos, cfg.oe_n)
    for table, proj, (order, mod) in zip(m.oe_tables, m.oe_proj, cfg.moduli()):
        idx = (ngram_index([z.array for z in prev[:order]], m.Vocab.size, mod) + (mod // 2 + 7919)) % mod
        x = x + proj(table.take(ROWS, hax.named(idx, input_ids.axes)))
    x = x / (1 + cfg.k * (cfg.oe_n - 1))
    if m.embeddings.norm is not None:
        x = m.embeddings.norm(x)
    return x


def main():
    s = SIZES[SIZE]
    B, steps = s["batch"], s["steps"]
    cfg = model_config(s)
    Pos = cfg.max_Pos
    ov = MODEL == "ov"
    trainer = TrainerConfig(
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        train_batch_size=B,
        per_device_eval_parallelism=int(os.environ.get("EVAL_PER_DEVICE", "8")),
        mesh=MeshConfig(
            axes={"data": -1, "replica": 1, "model": 1},
            compute_mapping={
                "token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
                "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
                **({ROWS: ResourceAxis.DATA} if ov else {}),
            },
            **({"param_mapping": {"embed": "data", ROWS: "data"}} if ov else {}),
        ),
    )

    # The training order: train_lm's data_key = PRNGKey(data_seed); train_set splits it into (mix, shuffle) keys and
    # train_sets shuffles the one nonzero-weight component with the first key of the shuffle key's iterator.
    mtb = {TRAIN_NAME: round(steps / DATA_EPOCHS)} if DATA_EPOCHS > 0 else None
    data = data_config()
    _, shuffle_key = jax.random.split(jax.random.PRNGKey(DATA_SEED))
    order = data.train_sets(Pos, key=shuffle_key, initial_batch_size=B)[TRAIN_NAME]
    N = len(order.as_sync_dataset())
    S = mtb[TRAIN_NAME] * B if mtb else N
    total = steps * B
    # Check the reconstruction against the run's own mixture (with its max_train_batches) at a few mixture indices,
    # including ones past the first pass. The mixture permutes ids within blocks of mixture_block_size, so mixture
    # index j reads raw index r (same block as j) of the order, wrapped mod S: a sequence's visits are known to +-1 block.
    mix = data_config(mtb).train_set(Pos, BatchSchedule(B), key=jax.random.PRNGKey(DATA_SEED))
    md, bs = mix.dataset, data.mixture_block_size
    sync_order, sync_mix = order.as_sync_dataset(), mix.as_sync_dataset()
    for j in (0, 12345, S - 1, S + 7, min(total - 1, 2 * S + 3)):
        _, r = md._index_into_dataset_for_id(md._get_block(j // bs)[j % bs], j // bs)
        assert r // bs == j // bs, (j, r)
        assert np.array_equal(np.asarray(sync_mix[j].tokens.array), np.asarray(sync_order[r % S].tokens)), f"mixture index {j} != order index {r % S}"
    print(f"order rebuilt and checked: N={N} sequences, cycled subset S={S}, run consumes {total} ({total / S:.2f} passes of S, "
          f"{total / N:.2f} of N)", flush=True)

    span = min(N, 2 * S) if S < N else N
    starts = [int(round(x)) for x in np.linspace(0, span - SEQS, SLICES)]
    meta = {}
    datasets = []
    for a in starts:
        tag = f"slice{a:09d}"
        # passes over sequence i: the k >= 0 with k * S + i < total (none past the cycled subset). A slice whose first and
        # last sequences differ in pass count straddles a boundary; it is kept but labelled "mixed" and left out of groups.
        npass = lambda i: 0 if i >= S else max(0, -(-(total - i) // S))
        passes = npass(a) if npass(a) == npass(a + SEQS - 1) else "mixed"
        meta[tag] = dict(start=a, passes=passes, last_seen_step=((passes - 1) * S + a) // B if isinstance(passes, int) and passes else None)
        datasets.append((NamedLmDataset(order.slice_dataset(start_index=a, end_index=a + SEQS), Pos), [tag]))
    for name, ds in data.validation_sets(Pos).items():
        if name in HELDOUT:
            datasets.append((ds, [f"heldout_{name}"]))
            meta[f"heldout_{name}"] = dict(passes=None)
    print(json.dumps(meta), flush=True)
    if os.environ.get("DRY"):
        return

    with trainer.use_device_mesh(), hax.axis_mapping(trainer.compute_axis_mapping):
        tokenizer = data.the_tokenizer
        Vocab = round_axis_for_partitioning(Axis("vocab", len(tokenizer)), trainer.parameter_axis_mapping)
        try:
            model = load_levanter_checkpoint(cfg, CKPT, Vocab=Vocab, axis_mapping=trainer.parameter_axis_mapping, key=jax.random.PRNGKey(0))
        except ValueError as e:
            # OV checkpoints from before the tables' rows were padded to ROW_ALIGN hold exactly m rows per table
            if not (ov and ROWS in str(e)):
                raise
            print(f"unpadded OV tables in this checkpoint ({e}); loading with ROW_ALIGN = 1", flush=True)
            ovq.ROW_ALIGN = 1
            model = load_levanter_checkpoint(cfg, CKPT, Vocab=Vocab, axis_mapping=trainer.parameter_axis_mapping, key=jax.random.PRNGKey(0))
        model = inference_mode(model, True)
        mp = trainer.mp
        results = {"ckpt": CKPT, "size": SIZE, "model": MODEL, "N": N, "S": S, "total": total, "slices": meta, "modes": {}}
        for mode in MODES:

            def loss_fn(m, ex, mode=mode):
                m = mp.cast_to_compute(m)
                if mode == "oe_scramble":
                    x = scrambled_embed(m, ex.tokens, ex.attn_mask)
                    h = m.transformer(x, attn_mask=ex.attn_mask, key=None)
                else:
                    h = m.activations(ex.tokens, attn_mask=ex.attn_mask, key=None)
                per_pos = maybe_fused_next_token_loss(m.Pos, m.Embed, m.Vocab, h, m.get_lm_head(), ex.tokens, loss_weight=ex.loss_weight,
                                                      reduction=None, reduction_axis=(), dtype=jnp.float32).array
                return per_pos, ex.loss_weight.array, jnp.roll(ex.tokens.array, -1, axis=-1)

            evaluator = TaggedEvaluator(EvalBatch=trainer.EvalBatch, tagged_eval_sets=datasets, loss_fn=loss_fn, tokenizer=tokenizer,
                                        axis_mapping=trainer.compute_axis_mapping)
            t0 = time.time()
            log = eval_model(evaluator, model, prefix="eval")
            results["modes"][mode] = {t: float(log[f"eval/{t}/loss"]) for t in meta if f"eval/{t}/loss" in log}
            print(f"MODE {mode} ({time.time() - t0:.0f}s)", flush=True)
            for t, v in results["modes"][mode].items():
                print(f"  {t:24s} passes={meta[t].get('passes')} last_seen={meta[t].get('last_seen_step')} loss={v:.4f}", flush=True)
            os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
            json.dump(results, open(OUT, "w"), indent=1)
        print("written", OUT)


if __name__ == "__main__":
    main()
