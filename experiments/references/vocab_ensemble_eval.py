# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Does ensembling close the gap between 8-pass and unique-token training? Kim et al. 2025 (arXiv 2509.14786) find that
an ensemble of independently trained models lowers the loss asymptote in data-constrained pretraining.

Loads two checkpoints CKPT_A and CKPT_B of one tokenizer and size, runs both on every Paloma validation window (the
run's own eval windows and loss path, as vocab_token_losses.py) and reports, per subset, the mean loss in nats per
scored target (weights = the eval's loss_weight) of model A, of model B and of their ensemble, the average of the two
next-token distributions: -log((p_A + p_B) / 2) at the target = -logsumexp(-l_A, -l_B) + log 2. Each loss also becomes
the exact macro bits per byte, loss x T / (ln 2 x B) per subset with T, B from tmp_plots/vocab/bpb_exact_tables.json,
averaged over the 16 subsets. On a full run each model's per-subset loss must match its logged eval at the same step
(the checkpoint's eval_metrics.jsonl) within 1e-3 and each subset's scored targets must equal the table's T; the json is
written first, then a mismatch raises.

    CKPT_A=<...> CKPT_B=<...> SIZE=300m [VOCAB_K=8000] OUT=<file.json> [MAX_WINDOWS=N] python -m experiments.references.vocab_ensemble_eval
"""

import json
import math
import os
import time

# vocab_token_losses reads CKPT and OUT at import; this script reads CKPT_A, CKPT_B and OUT in main (the test imports it bare)
os.environ.setdefault("CKPT", os.environ.get("CKPT_A", "/unset/run/d/checkpoints/step-0"))
os.environ.setdefault("OUT", "/unset")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jmp  # noqa: E402
import numpy as np  # noqa: E402

import haliax as hax  # noqa: E402
from haliax import Axis  # noqa: E402
from haliax.partitioning import ResourceAxis, round_axis_for_partitioning  # noqa: E402
from levanter.data.loader import DataLoader  # noqa: E402
from levanter.model_loading import load_levanter_checkpoint  # noqa: E402
from levanter.models.loss import maybe_fused_next_token_loss  # noqa: E402
from levanter.trainer import TrainerConfig  # noqa: E402
from levanter.utils.mesh import MeshConfig  # noqa: E402
from levanter.utils.tree_utils import inference_mode  # noqa: E402

import experiments.references.vocab_token_losses as vtk  # noqa: E402

BPB_TABLES = "/scratch/gpfs/GROUP/USER/tmp_plots/vocab/bpb_exact_tables.json"
LOGGED_TOL = 1e-3
MODELS = ("a", "b", "ens")


def ensemble_loss(loss_a, loss_b):
    """-log of the mean of the two target probabilities, from the two per-target losses (nats), computed stably."""
    return -jnp.logaddexp(-loss_a, -loss_b) + math.log(2.0)


def position_losses(m, ex, mp):
    """Per-position loss (nats) of predicting token t+1 at position t, times the eval's loss weight (vocab_token_losses' path)."""
    m = mp.cast_to_compute(m)
    h = m.activations(ex.tokens, attn_mask=ex.attn_mask, key=None)
    return maybe_fused_next_token_loss(m.Pos, m.Embed, m.Vocab, h, m.get_lm_head(), ex.tokens, loss_weight=ex.loss_weight,
                                       reduction=None, reduction_axis=(), dtype=jnp.float32).array


def window_sums(ma, mb, ex, mp):
    """Per window: weighted loss sums of A, B and the ensemble, the weight sum, and the count of weights not in {0, 1}."""
    w = ex.loss_weight.array.astype(jnp.float32)
    la, lb = position_losses(ma, ex, mp), position_losses(mb, ex, mp)   # already x w; w in {0, 1} so unweighted where w = 1
    le = ensemble_loss(la, lb) * w
    axes = tuple(range(1, w.ndim))
    return (la.sum(axes), lb.sum(axes), le.sum(axes), w.sum(axes), ((w != 0) & (w != 1)).sum(axes))


def macro_bpb(losses: dict[str, float], table: dict[str, list[int]]) -> float:
    """Mean over subsets of loss x T / (ln 2 x B)."""
    assert sorted(losses) == sorted(table), (sorted(losses), sorted(table))
    return float(np.mean([losses[s] * table[s][0] / (math.log(2) * table[s][1]) for s in losses]))


def logged_losses(ckpt: str) -> dict[str, float]:
    """The per-subset Paloma loss the run logged at the checkpoint's step (eval_metrics.jsonl beside the checkpoint)."""
    step = int(ckpt.rstrip("/").rsplit("step-", 1)[1])
    path = os.path.join(os.path.dirname(ckpt.rstrip("/")), "eval_metrics.jsonl")
    rows = [json.loads(line) for line in open(path) if line.strip()]
    row = [r for r in rows if r.get("step") == step]
    assert row, f"no eval at step {step} in {path}"
    out = {}
    for key, v in row[-1].items():
        if key.startswith("eval/paloma/") and key.endswith("/loss"):
            out[key[len("eval/paloma/"):-len("/loss")].removesuffix("-marin-tokenizer")] = float(v)
    return out


def main():
    ckpt_a, ckpt_b, out_path = os.environ["CKPT_A"], os.environ["CKPT_B"], os.environ["OUT"]
    assert out_path != "/unset", "set OUT"
    cfg =vtk.model_config()
    Pos = Axis("position", vtk.SEQ_LEN)
    trainer = TrainerConfig(mp=jmp.get_policy("p=f32,c=bfloat16"), per_device_eval_parallelism=vtk.EVAL_BATCH,
                            mesh=MeshConfig(axes={"data": -1, "replica": 1, "model": 1},
                                            compute_mapping={"token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
                                                             "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA)}))
    data = vtk.data_config()
    k = vtk.VOCAB_K or 128000
    table = json.load(open(BPB_TABLES))[str(k)][str(vtk.SEQ_LEN)]
    sums: dict[str, np.ndarray] = {}
    with trainer.use_device_mesh(), hax.axis_mapping(trainer.compute_axis_mapping):
        Vocab = round_axis_for_partitioning(Axis("vocab", len(data.the_tokenizer)), trainer.parameter_axis_mapping)
        load = lambda c: inference_mode(load_levanter_checkpoint(cfg, c, Vocab=Vocab, axis_mapping=trainer.parameter_axis_mapping,  # noqa: E731
                                                                 key=jax.random.PRNGKey(0)), True)
        ma, mb = load(ckpt_a), load(ckpt_b)
        step = hax.named_jit(lambda ma, mb, ex: window_sums(ma, mb, ex, trainer.mp), axis_resources=trainer.compute_axis_mapping)
        for name, ds in data.validation_sets(Pos).items():
            if vtk.ONLY and name not in vtk.ONLY:
                continue
            t0 = time.time()
            n = len(ds.as_sync_dataset())
            if vtk.MAX_WINDOWS:
                n = min(n, vtk.MAX_WINDOWS)
                ds = ds.slice_dataset(end_index=n)
            parts = [np.stack([np.asarray(jax.device_get(x), np.float64) for x in step(ma, mb, batch)], axis=1)
                     for batch in DataLoader(ds, trainer.EvalBatch, mesh=trainer.device_mesh, axis_resources=trainer.compute_axis_mapping)]
            sums[name] = np.concatenate(parts)[:n].sum(0)   # [a, b, ens, weight, non-binary weights]
            assert sums[name][4] == 0, f"{name}: loss weights outside {{0, 1}}"
            la, lb, le = sums[name][:3] / sums[name][3]
            print(f"{name}: {n} windows, loss A {la:.4f} B {lb:.4f} ens {le:.4f} ({time.time() - t0:.0f}s)", flush=True)

    full = not vtk.MAX_WINDOWS and not vtk.ONLY
    loss = {m: {s: float(v[i] / v[3]) for s, v in sums.items()} for i, m in enumerate(MODELS)}
    res = {"ckpt_a": ckpt_a, "ckpt_b": ckpt_b, "vocab_k": k, "seq_len": vtk.SEQ_LEN, "full": full,
           "targets": {s: int(v[3]) for s, v in sums.items()}, "loss": loss,
           "mean_loss": {m: float(np.mean(list(loss[m].values()))) for m in MODELS}}
    problems = []
    if full:
        res["macro_bpb"] = {m: macro_bpb(loss[m], table) for m in MODELS}
        for s in sums:
            if res["targets"][s] != table[s][0]:
                problems.append(f"{s}: {res['targets'][s]} scored targets, table T {table[s][0]}")
        for m, ckpt in (("a", ckpt_a), ("b", ckpt_b)):
            logged = logged_losses(ckpt)
            diff = {s: loss[m][s] - logged[s] for s in loss[m]}
            res[f"logged_diff_{m}"] = diff
            problems += [f"{m} {s}: ours {loss[m][s]:.5f} logged {logged[s]:.5f}" for s, d in diff.items() if abs(d) > LOGGED_TOL]
    res["problems"] = problems
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    json.dump(res, open(out_path, "w"), indent=1)
    if full:
        bpb = res["macro_bpb"]
        print(f"macro bpb: A {bpb['a']:.4f} B {bpb['b']:.4f} ensemble {bpb['ens']:.4f} "
              f"(ensemble - mean of A, B {bpb['ens'] - (bpb['a'] + bpb['b']) / 2:+.4f}); "
              f"max |loss - logged| {max(abs(d) for m in 'ab' for d in res[f'logged_diff_{m}'].values()):.2e}", flush=True)
    else:
        ml = res["mean_loss"]
        print(f"partial: mean subset loss A {ml['a']:.4f} B {ml['b']:.4f} ensemble {ml['ens']:.4f}", flush=True)
    print("written", out_path, flush=True)
    if problems:
        raise ValueError("checks failed:\n" + "\n".join(problems))


if __name__ == "__main__":
    main()
