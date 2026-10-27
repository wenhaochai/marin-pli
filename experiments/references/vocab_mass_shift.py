# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Where the 8-pass model moves probability mass on Paloma (owner popup 2026-10-27, "A 质量去向诊断"). The softmax
temperature test (w115/w116) left most of the cost: it is in which tokens get the mass, not in how sharp the distribution
is. Hypothesis: in the 7 repeated passes a token that the repeated eighth holds rarely only ever gets the softmax's push
down, so its log-probability drifts lower everywhere; a large vocabulary has many such tokens, the byte vocabulary none.

Loads the unique-token checkpoint CKPT_U and the 8-pass checkpoint CKPT_R of one tokenizer, runs both on every Paloma
validation window (the run's own eval windows, as vocab_token_losses.py) and, from the full log-softmax of both models at
every scored position, sums per Paloma subset:
  shift  [V]  log p_R(v) - log p_U(v) over all scored positions (every token v, target or not)
  mass   [V]  p_R(v) - p_U(v) over all scored positions
  tshift [V]  log p_R(t) - log p_U(t) at the positions whose target is t (minus the cost of those positions)
  tcount [V]  the number of such positions
and the number of scored positions. vocab_mass_shift_groups.py groups them by each token's count in the repeated eighth.

    CKPT_U=<...> CKPT_R=<...> SIZE=300m [VOCAB_K=8000] OUT=<file.npz> [MAX_WINDOWS=N] python -m experiments.references.vocab_mass_shift
"""

import os
import time

os.environ.setdefault("EVAL_BATCH", "1")   # one window per step: full logits of both models fit beside each other
os.environ.setdefault("CKPT", os.environ["CKPT_U"])   # vocab_token_losses reads CKPT at import; CKPT_U/CKPT_R are loaded below

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jmp  # noqa: E402
import numpy as np  # noqa: E402

import haliax as hax  # noqa: E402
from haliax import Axis  # noqa: E402
from haliax.partitioning import ResourceAxis, round_axis_for_partitioning  # noqa: E402
from levanter.data.loader import DataLoader  # noqa: E402
from levanter.model_loading import load_levanter_checkpoint  # noqa: E402
from levanter.trainer import TrainerConfig  # noqa: E402
from levanter.utils.mesh import MeshConfig  # noqa: E402
from levanter.utils.tree_utils import inference_mode  # noqa: E402

import experiments.references.vocab_token_losses as vtk  # noqa: E402

CKPT_U, CKPT_R, OUT = os.environ["CKPT_U"], os.environ["CKPT_R"], os.environ["OUT"]


def main():
    cfg = vtk.model_config()
    Pos = Axis("position", vtk.SEQ_LEN)
    trainer = TrainerConfig(mp=jmp.get_policy("p=f32,c=bfloat16"), per_device_eval_parallelism=vtk.EVAL_BATCH,
                            mesh=MeshConfig(axes={"data": -1, "replica": 1, "model": 1},
                                            compute_mapping={"token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
                                                             "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA)}))
    data = vtk.data_config()
    n_real = len(data.the_tokenizer)
    out = {}
    with trainer.use_device_mesh(), hax.axis_mapping(trainer.compute_axis_mapping):
        Vocab = round_axis_for_partitioning(Axis("vocab", n_real), trainer.parameter_axis_mapping)
        load = lambda c: inference_mode(load_levanter_checkpoint(cfg, c, Vocab=Vocab, axis_mapping=trainer.parameter_axis_mapping,  # noqa: E731
                                                                 key=jax.random.PRNGKey(0)), True)
        mu, mr = load(CKPT_U), load(CKPT_R)
        mp = trainer.mp

        @hax.named_jit(axis_resources=trainer.compute_axis_mapping)
        def step(mu, mr, ex):
            def logp(m):
                m = mp.cast_to_compute(m)
                h = m.activations(ex.tokens, attn_mask=ex.attn_mask, key=None)
                z = hax.dot(h, m.get_lm_head(), axis=m.Embed).astype(jnp.float32)
                z = z.rearrange((*[ax for ax in z.axes if ax.name != m.Vocab.name], m.Vocab)).array
                return jax.nn.log_softmax(z.reshape(-1, z.shape[-1]), axis=-1)[:, :n_real]
            lu, lr = logp(mu), logp(mr)
            w = ex.loss_weight.array.reshape(-1).astype(jnp.float32)
            tok = ex.tokens.array
            tgt = jnp.concatenate([tok[..., 1:], tok[..., :1]], axis=-1).reshape(-1)   # position t predicts token t + 1
            d = lr - lu
            dt = jnp.take_along_axis(d, tgt[:, None], axis=1)[:, 0] * w
            return (jnp.sum(d * w[:, None], axis=0), jnp.sum((jnp.exp(lr) - jnp.exp(lu)) * w[:, None], axis=0),
                    jnp.zeros(n_real, jnp.float32).at[tgt].add(dt), jnp.zeros(n_real, jnp.float32).at[tgt].add(w), jnp.sum(w))

        for name, ds in data.validation_sets(Pos).items():
            t0 = time.time()
            n = len(ds.as_sync_dataset())
            if vtk.MAX_WINDOWS:
                n = min(n, vtk.MAX_WINDOWS)
                ds = ds.slice_dataset(end_index=n)
            acc = None
            for batch in DataLoader(ds, trainer.EvalBatch, mesh=trainer.device_mesh, axis_resources=trainer.compute_axis_mapping):
                r = step(mu, mr, batch)
                acc = r if acc is None else tuple(a + b for a, b in zip(acc, r))
            shift, mass, tshift, tcount, npos = (np.asarray(jax.device_get(a), np.float64) for a in acc)
            out.update({f"{name}/shift": shift, f"{name}/mass": mass, f"{name}/tshift": tshift, f"{name}/tcount": tcount,
                        f"{name}/npos": npos})
            print(f"{name}: {n} windows, {npos:.0f} positions, cost {-tshift.sum() / npos:.4f} nats/token, "
                  f"mean log-prob shift of all tokens {shift.mean() / npos:.4f} ({time.time() - t0:.0f}s)", flush=True)
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    np.savez(OUT, ckpt_u=CKPT_U, ckpt_r=CKPT_R, **out)
    print("written", OUT, flush=True)


if __name__ == "__main__":
    main()
