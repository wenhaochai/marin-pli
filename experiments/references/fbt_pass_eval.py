# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Eval-only diagnostics for a full-bandwidth (FBT) checkpoint: how much each pass contributes.

Runs Paloma under several readout regimes with the same weights and prints one row per regime:

* ``pass1``      standard transformer (no feedback) -- the paper's "standard" regime
* ``pass2``      fused pass (what training logged as eval/paloma/*): position t reads pass-1's state at t-1
* ``pass3``      one more prefill pass (pass 2's states fed back again) -- extrapolation
* ``pass2_shuf`` pass 2 fed with the states of a *different* sequence (batch rolled by one): if this matches
                 ``pass2`` the model only reacts to the input-norm change, not to the content of h
* ``pass2_pre25/50/75``  pass 2 with a fixed plain prefix covering the first 25/50/75% of positions:
                 how much of the fused-pass gain needs feedback everywhere

    CKPT=<levanter checkpoint dir> SIZE=300m python -m experiments.references.fbt_pass_eval
    (FBT_* env flags as for the training launcher; MARIN_PREFIX gives the Paloma caches under tokenized/paloma)
"""

import dataclasses
import json
import os
import time

import jax
import jmp

import haliax as hax
import haliax.nn as hnn
from haliax import Axis
from haliax.partitioning import ResourceAxis, round_axis_for_partitioning
from levanter.data.text.datasets import DatasetComponent, LmDataConfig, UrlDatasetSourceConfig
from levanter.data.text.formats import TextLmDatasetFormat
from levanter.eval import TaggedEvaluator, eval_model
from levanter.layers.attention import AttentionBackend
from levanter.model_loading import load_levanter_checkpoint
from levanter.models.loss import maybe_fused_next_token_loss
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from levanter.utils.tree_utils import inference_mode

from experiments.marin_tokenizer import marin_tokenizer
from experiments.references.della_muonh_qwen3_scaling import FBT_ALPHA_MULT, FBT_INPUT, FBT_INPUT_LAYERS, FBT_LAYERWISE, FBT_NOISE, FBT_RESIDUAL, SEQ_LEN, SIZES, TIE
from experiments.references.full_bandwidth_qwen3 import FullBandwidthQwen3Config

CKPT = os.environ["CKPT"]
SIZE = os.environ.get("SIZE", "300m")
PREFIX = os.environ.get("MARIN_PREFIX", "/scratch/gpfs/KARTHIKN/wc9403/marin_store_big")
OUT = os.environ.get("OUT", f"logs/fbt_pass_eval_{os.path.basename(os.path.dirname(os.path.dirname(CKPT.rstrip('/'))))}.json")
MODES = os.environ.get("MODES", "pass1 pass2 pass3 pass2_shuf pass2_pre25 pass2_pre50 pass2_pre75").split()


def paloma_data_config() -> LmDataConfig:
    root = os.path.join(PREFIX, "tokenized", "paloma")
    components = {}
    for d in sorted(os.listdir(root)):
        path = os.path.join(root, d)
        if not os.path.isdir(os.path.join(path, "validation")):
            continue
        name = d.rsplit("-", 1)[0]
        source = UrlDatasetSourceConfig(tags=[], train_urls=[], validation_urls=[], cache_dir=path, format=TextLmDatasetFormat())
        components[name] = DatasetComponent(source=source, cache_dir=path, format=source.format, tags=[])
    return LmDataConfig(components=components, train_weights={n: 0.0 for n in components}, tokenizer=marin_tokenizer, cache_dir=None)


def main():
    s = SIZES[SIZE]
    cfg = FullBandwidthQwen3Config(
        max_seq_len=SEQ_LEN,
        hidden_dim=s["hidden"],
        intermediate_dim=s["inter"],
        num_layers=s["layers"],
        num_heads=s["heads"],
        num_kv_heads=s["kv"],
        hybrid_norm=True,
        attn_backend=AttentionBackend.JAX_FLASH,
        feedback_passes=3,  # params do not depend on the pass count; 3 gives pass1/pass2/pass3 in one forward
        feedback_noise=FBT_NOISE,
        feedback_residual=FBT_RESIDUAL,
        feedback_layerwise=FBT_LAYERWISE,
        feedback_input=FBT_INPUT,
        feedback_input_layers=FBT_INPUT_LAYERS,
        feedback_alpha_lr_mult=FBT_ALPHA_MULT,
        tie_word_embeddings=TIE,
    )
    trainer = TrainerConfig(
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        train_batch_size=s["batch"],
        per_device_eval_parallelism=int(os.environ.get("EVAL_PER_DEVICE", "8")),
        mesh=MeshConfig(
            axes={"data": -1, "replica": 1, "model": 1},
            compute_mapping={
                "token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
                "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
            },
        ),
    )
    data = paloma_data_config()
    tokenizer = data.the_tokenizer
    Pos = cfg.max_Pos
    datasets = data.tagged_eval_sets(Pos)
    print(f"{len(datasets)} Paloma subsets: {[t[-1] for _, t in datasets]}", flush=True)
    compute_axis_mapping = trainer.compute_axis_mapping
    parameter_axis_mapping = trainer.parameter_axis_mapping
    mp = trainer.mp

    with trainer.use_device_mesh():
        Vocab = round_axis_for_partitioning(Axis("vocab", len(tokenizer)), compute_axis_mapping)
        model = load_levanter_checkpoint(cfg, CKPT, Vocab=Vocab, axis_mapping=parameter_axis_mapping, key=jax.random.PRNGKey(0))
        model = inference_mode(model, True)

        def ntp(m, h, ex):
            return maybe_fused_next_token_loss(
                m.Pos, m.Embed, m.Vocab, h, m.get_lm_head(), ex.tokens, loss_weight=ex.loss_weight, reduction=None, reduction_axis=(), dtype=jax.numpy.float32
            )

        def fused_input(m, e, h_prev, plain):
            """Pass-2 input from the previous-position states ``h_prev`` (paper form or residual form)."""
            c = m.config
            gate = hnn.sigmoid(m.w_g(m.input_norm(e)).rename({"fused_embed": "embed"}))
            fused = m.w_u(hax.roll(h_prev, 1, Pos)).rename({"fused_embed": "embed"}) * gate
            if c.feedback_residual:
                return e + (m.alpha * c.feedback_alpha_lr_mult).astype(e.dtype) * hax.where(plain, 0.0, fused)
            return m.input_norm(hax.where(plain, e, fused))

        def states_for(mode, m, ex):
            m = mp.cast_to_compute(m)
            tokens, mask = ex.tokens, ex.attn_mask
            Batch = tokens.resolve_axis("batch")
            position = hax.arange(Pos).broadcast_axis((Batch,))
            if mode in ("pass1", "pass2", "pass3"):
                st = m._passes(tokens, mask, key=None, pos_ids=None, train=False)
                return st[{"pass1": 0, "pass2": 1, "pass3": 2}[mode]]
            e = m.embeddings.embed(tokens)
            h1 = m.transformer(e, attn_mask=mask, key=None, pos_ids=None)
            if mode == "pass2_shuf":
                h_prev = hax.roll(h1, 1, Batch)  # states of another sequence in the batch
                plain = position <= 0
            else:
                frac = {"pass2_pre25": 0.25, "pass2_pre50": 0.5, "pass2_pre75": 0.75}[mode]
                h_prev = h1
                plain = position <= int(frac * Pos.size)
            x = fused_input(m, e, h_prev, plain)
            if m.config.feedback_layerwise:
                raise NotImplementedError("shuffle/prefix modes for the layerwise variant")
            return m.transformer(x, attn_mask=mask, key=None, pos_ids=None)

        results = {}
        for mode in MODES:

            def loss_fn(m, ex, mode=mode):
                h = states_for(mode, m, ex)
                per_pos = ntp(m, h, ex).array
                return per_pos, ex.loss_weight.array, jax.numpy.roll(ex.tokens.array, -1, axis=-1)

            evaluator = TaggedEvaluator(EvalBatch=trainer.EvalBatch, tagged_eval_sets=datasets, loss_fn=loss_fn, tokenizer=tokenizer, axis_mapping=compute_axis_mapping)
            t0 = time.time()
            log = eval_model(evaluator, model, prefix="eval")
            row = {k.replace("eval/", ""): v for k, v in log.items() if k.endswith("/bpb") or k.endswith("macro_bpb") or k.endswith("macro_loss") or k in ("eval/loss", "eval/bpb")}
            results[mode] = row
            print(f"MODE {mode}: c4_en/bpb={row.get('c4_en/bpb')} macro_bpb={row.get('macro_bpb')} macro_loss={row.get('macro_loss')} loss={row.get('loss')} ({time.time() - t0:.0f}s)", flush=True)
            json.dump(results, open(OUT, "w"), indent=1)

        base = results.get("pass1", {})
        print("\nSUMMARY (delta vs pass1):")
        for mode, row in results.items():
            print(f"  {mode:12s} c4_en {row.get('c4_en/bpb'):.5f} ({row.get('c4_en/bpb') - base.get('c4_en/bpb', 0):+.5f})  macro_bpb {row.get('macro_bpb'):.5f} ({row.get('macro_bpb') - base.get('macro_bpb', 0):+.5f})")
        print("written", OUT)


if __name__ == "__main__":
    main()
