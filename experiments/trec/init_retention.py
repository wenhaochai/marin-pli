"""How much of its random initialization each weight matrix of a final checkpoint still holds (not yet run; owner
approval first).

MuonH (matrices) and AdamH (lm_head) rescale every matrix to its starting norm after each step. If update t, of
relative size eta_t, is orthogonal to the weights, the old weights shrink by 1/sqrt(1 + eta_t^2), so the final matrix is
W_T = a W_0 + (the sum of later updates) with a = prod_t (1 + eta_t^2)^(-1/2) = exp(-sum eta_t^2 / 2), about 0.81 for the
300m cosine run and 0.63 for the 130m WSD run. This measures, per layer and per matrix, cos(W_T, W_0) = <W_T, W_0> /
(|W_T| |W_0|), against that a: the same decay product that sets the TREC prediction (fig-coef), read off the weights
without the training-fraction term. W_0 is rebuilt from the run's seed as train_lm builds it (model_key = the third key
of split(PRNGKey(seed), 4); config.build(Vocab, key=model_key)); |W_T| = |W_0| for every fixed-norm matrix checks the
rebuild. The embedding (plain Adam, no decay) has no prediction and is reported as measured.

    JAX_PLATFORMS=cpu CKPT=<.../step-N> MODEL=300m|130m-d12 SEED=<trainer seed> OUT=<file.json> \
        python -m experiments.trec.init_retention        # from the worktree, on a vis node
"""

import json
import os

import jax
import numpy as np

import haliax as hax
from haliax import Axis
from haliax.partitioning import round_axis_for_partitioning
from levanter.model_loading import load_levanter_checkpoint

SEED = int(os.environ["SEED"])   # read before vocab_trec, whose import clears SEED from the environment
os.environ.setdefault("OUT", "/dev/null")
from experiments.references import trec_score as ts  # noqa: E402
from experiments.trec import predict  # noqa: E402


def leaves(model):
    flat, _ = jax.tree_util.tree_flatten_with_path(model, is_leaf=lambda x: isinstance(x, hax.NamedArray))
    for path, leaf in flat:
        arr = leaf.array if isinstance(leaf, hax.NamedArray) else leaf
        if hasattr(arr, "ndim") and arr.ndim >= 2:
            yield jax.tree_util.keystr(path), np.asarray(arr, dtype=np.float64)


def main():
    cfg = ts.model_config()
    vtk = ts.vtk
    trainer = vtk.TrainerConfig(mp=vtk.jmp.get_policy("p=f32,c=bfloat16"), mesh=vtk.MeshConfig(axes={"data": -1, "replica": 1, "model": 1}))
    with trainer.use_device_mesh(), hax.axis_mapping(trainer.parameter_axis_mapping):
        Vocab = round_axis_for_partitioning(Axis("vocab", len(vtk.data_config().the_tokenizer)), trainer.parameter_axis_mapping)
        model_key = jax.random.split(jax.random.PRNGKey(SEED), 4)[2]   # train_lm: data, loader, model, training
        init = cfg.build(Vocab, key=model_key)
        final = load_levanter_checkpoint(cfg, vtk.CKPT, Vocab=Vocab, axis_mapping=trainer.parameter_axis_mapping, key=jax.random.PRNGKey(0))
    eta = predict.lr_schedule(ts.MODEL)
    a = float(np.exp(-np.log1p(eta ** 2).sum() / 2))
    rows = []
    for (p0, w0), (p1, w1) in zip(leaves(init), leaves(final)):
        assert p0 == p1 and w0.shape == w1.shape, (p0, p1, w0.shape, w1.shape)
        stacked = w0.ndim > 2   # as norm_preserving_update: a higher-rank array is a stack of matrices along axis 0
        for k in range(w0.shape[0] if stacked else 1):
            x0, x1 = (w0[k], w1[k]) if stacked else (w0, w1)
            n0, n1 = np.linalg.norm(x0), np.linalg.norm(x1)
            rows.append({"param": p0, "slice": k, "norm_init": n0, "norm_final": n1, "cos": float((x0 * x1).sum() / (n0 * n1))})
    json.dump({"ckpt": ts.vtk.CKPT, "model": ts.MODEL, "predicted_fixed_norm": a, "rows": rows}, open(os.environ["OUT"], "w"), indent=1)
    print("predicted", round(a, 3))
    for r in rows:
        print(f"{r['param'][:60]:60s} {r['slice']:3d} norm {r['norm_init']:.4f} -> {r['norm_final']:.4f} cos {r['cos']:.3f}")


if __name__ == "__main__":
    main()
