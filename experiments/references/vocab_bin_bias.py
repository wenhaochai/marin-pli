# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Is the 8-pass cost on rare targets a global prior error? (owner popup 2026-10-27, "A 分频段偏置校准"). The mass-shift
diagnostic (vocab_mass_shift.py, wq w506) found that 8 passes lower a rare token's log-probability everywhere (128K: -0.49
nats for tokens seen 100-1K times in the repeated eighth). If that push-down is one offset per frequency band, one log-prob
offset per band of the repeated-eighth count should give most of the cost back.

Loads the unique-token checkpoint CKPT_U and the 8-pass checkpoint CKPT_R of one tokenizer, runs both on every Paloma
validation window (the run's own eval windows, as vocab_token_losses.py) and keeps, per scored position and model, the
target's logit and one logsumexp of the logits per band (the 9 count bands of vocab_mass_shift_groups.py, plus one band
for the padded vocabulary rows). Calibrated loss with offsets b: logsumexp_k(lse_k + b_k) - z_t - b_band(t), so the
offsets are fitted exactly, by Newton's method, on the positions alone. Each model gets its own offsets, fitted on the
even windows of every subset; the odd windows are scored. Cost = loss of the 8-pass model - loss of the unique model, per
byte of target text (nats/byte; byte-level token lengths), before and after calibration. The padded band keeps offset 0.

    CKPT_U=<...> CKPT_R=<...> SIZE=300m [VOCAB_K=8000] OUT=<file.json> [MAX_WINDOWS=N] python -m experiments.references.vocab_bin_bias
"""

import json
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
COUNTS = "/scratch/gpfs/GROUP/USER/project/marin/logs/vocab_token_losses/subset_counts_{}.npy"
EDGES = [0, 10, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e12]
NB = len(EDGES)   # 9 count bands + the padded rows


def bands(n_real: int, n_pad: int) -> np.ndarray:
    """Band of every vocabulary row: its count in the repeated eighth (rows beyond the count file count 0); padding NB-1."""
    c = np.load(COUNTS.format(vtk.VOCAB_K or 128000)).astype(np.float64)
    c = np.pad(c, (0, max(0, n_real - len(c))))[:n_real]
    b = np.full(n_pad, NB - 1, np.int32)
    b[:n_real] = np.searchsorted(EDGES, c, side="right") - 1
    return b


def byte_lengths(tok, n: int) -> np.ndarray:
    """Bytes per token id: the byte-level BPE writes one character per byte; special tokens count 0."""
    special = set(tok.all_special_ids)
    out = np.zeros(n, np.float32)
    for i in range(n):
        p = tok.convert_ids_to_tokens(i)
        out[i] = 0 if i in special or p is None else len(p)
    return out


def fit(lse, zt, tb, w, iters=25):
    """Offsets b (NB, the padded band fixed at 0) minimising sum_i w_i [logsumexp(lse_i + b) - zt_i - b[tb_i]]."""
    onehot_t = jax.nn.one_hot(tb, NB, dtype=jnp.float32) * w[:, None]
    nt = onehot_t.sum(0)
    free = jnp.arange(NB) < NB - 1
    present = jnp.isfinite(lse).any(0) & free

    def loss(b):
        return jnp.sum(w * (jax.nn.logsumexp(lse + b, axis=1) - zt)) - jnp.dot(nt, b)

    b = jnp.zeros(NB, jnp.float32)
    for _ in range(iters):
        p = jax.nn.softmax(lse + b, axis=1)
        g = (w[:, None] * p).sum(0) - nt
        h = jnp.einsum("i,ij,ik->jk", w, p, p)
        h = jnp.diag((w[:, None] * p).sum(0)) - h
        m = present[:, None] & present[None, :]
        h = jnp.where(m, h, jnp.eye(NB)) + 1e-6 * w.sum() * jnp.eye(NB)
        step = jnp.where(present, jnp.linalg.solve(h, jnp.where(present, g, 0.0)), 0.0)
        cur, t = loss(b), 1.0
        while t > 1e-4 and loss(b - t * step) > cur:
            t /= 2
        b = b - t * step
    return b


def calibrated(lse, zt, tb, w, b):
    """Per-position weighted loss under offsets b."""
    return w * (jax.nn.logsumexp(lse + b, axis=1) - zt - b[tb])


def main():
    cfg = vtk.model_config()
    Pos = Axis("position", vtk.SEQ_LEN)
    trainer = TrainerConfig(mp=jmp.get_policy("p=f32,c=bfloat16"), per_device_eval_parallelism=vtk.EVAL_BATCH,
                            mesh=MeshConfig(axes={"data": -1, "replica": 1, "model": 1},
                                            compute_mapping={"token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
                                                             "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA)}))
    data = vtk.data_config()
    tok = data.the_tokenizer
    n_real = len(tok)
    with trainer.use_device_mesh(), hax.axis_mapping(trainer.compute_axis_mapping):
        Vocab = round_axis_for_partitioning(Axis("vocab", n_real), trainer.parameter_axis_mapping)
        band = bands(n_real, Vocab.size)
        onehot = jnp.asarray(np.eye(NB, dtype=np.float32)[band])   # (V, NB)
        load = lambda c: inference_mode(load_levanter_checkpoint(cfg, c, Vocab=Vocab, axis_mapping=trainer.parameter_axis_mapping,  # noqa: E731
                                                                 key=jax.random.PRNGKey(0)), True)
        mu, mr = load(CKPT_U), load(CKPT_R)
        mp = trainer.mp

        @hax.named_jit(axis_resources=trainer.compute_axis_mapping)
        def step(mu, mr, ex):
            tok_ = ex.tokens.array
            tgt = jnp.concatenate([tok_[..., 1:], tok_[..., :1]], axis=-1).reshape(-1)   # position t predicts token t + 1

            def stats(m):
                m = mp.cast_to_compute(m)
                h = m.activations(ex.tokens, attn_mask=ex.attn_mask, key=None)
                z = hax.dot(h, m.get_lm_head(), axis=m.Embed).astype(jnp.float32)
                z = z.rearrange((*[ax for ax in z.axes if ax.name != m.Vocab.name], m.Vocab)).array
                z = z.reshape(-1, z.shape[-1])
                top = z.max(axis=1, keepdims=True)
                s = jnp.exp(z - top) @ onehot
                lse = jnp.where(s > 0, jnp.log(jnp.maximum(s, 1e-38)) + top, -jnp.inf)
                return lse, jnp.take_along_axis(z, tgt[:, None], axis=1)[:, 0]
            lu, zu = stats(mu)
            lr, zr = stats(mr)
            return lu, zu, lr, zr, ex.loss_weight.array.reshape(-1).astype(jnp.float32), tgt

        parts, names, sub_of, odd_of = [], [], [], []
        for name, ds in data.validation_sets(Pos).items():
            t0 = time.time()
            n = len(ds.as_sync_dataset())
            if vtk.MAX_WINDOWS:
                n = min(n, vtk.MAX_WINDOWS)
                ds = ds.slice_dataset(end_index=n)
            names.append(name)
            for i, batch in enumerate(DataLoader(ds, trainer.EvalBatch, mesh=trainer.device_mesh, axis_resources=trainer.compute_axis_mapping)):
                parts.append(step(mu, mr, batch))
                sub_of.append(len(names) - 1)
                odd_of.append(i % 2)
            print(f"{name}: {n} windows ({time.time() - t0:.0f}s)", flush=True)
        lu, zu, lr, zr, w, tgt = (jnp.concatenate([p[j] for p in parts]) for j in range(6))
        del parts
        per = lu.shape[0] // len(sub_of)
        sub = jnp.asarray(np.repeat(np.asarray(sub_of, np.int32), per))
        odd = jnp.asarray(np.repeat(np.asarray(odd_of, np.float32), per))
        tb = jnp.asarray(band)[tgt]
        nbytes = w * jnp.asarray(np.pad(byte_lengths(tok, n_real), (0, Vocab.size - n_real)))[tgt]

        res = {"ckpt_u": CKPT_U, "ckpt_r": CKPT_R, "vocab_k": vtk.VOCAB_K or 128000, "edges": EDGES, "positions": int((w > 0).sum())}
        zero = jnp.zeros(NB, jnp.float32)
        rows = {}
        for tag, lse, zt in (("u", lu, zu), ("r", lr, zr)):
            b = fit(lse, zt, tb, w * (1 - odd))
            res[f"b_{tag}"] = np.asarray(b).round(4).tolist()
            rows[tag] = (calibrated(lse, zt, tb, w, zero), calibrated(lse, zt, tb, w, b))
            print(f"offsets {tag}: {res[f'b_{tag}']}", flush=True)
        ws = odd   # score on the odd windows
        tot = float((nbytes * ws).sum())

        def cost(a, b, mask=1.0):
            m = ws * mask
            return float(((rows["r"][a] - rows["u"][b]) * m).sum() / (nbytes * m).sum())
        res["bytes_scored"] = tot
        res["cost_raw"] = cost(0, 0)          # neither calibrated
        res["cost_cal"] = cost(1, 1)          # each model with its own offsets
        res["cost_r_only"] = cost(1, 0)       # only the 8-pass model calibrated
        res["loss_u_raw"], res["loss_u_cal"] = (float((rows["u"][j] * ws).sum() / tot) for j in (0, 1))
        res["by_band"] = [dict(lo=EDGES[k], share=float((nbytes * ws * (tb == k)).sum() / tot),
                               raw=cost(0, 0, tb == k) if float((nbytes * ws * (tb == k)).sum()) else None,
                               cal=cost(1, 1, tb == k) if float((nbytes * ws * (tb == k)).sum()) else None) for k in range(NB - 1)]
        res["by_subset"] = {nm: dict(raw=cost(0, 0, sub == s), cal=cost(1, 1, sub == s)) for s, nm in enumerate(names)}
    print(f"cost (nats/byte, odd windows): raw {res['cost_raw']:.4f}, both calibrated {res['cost_cal']:.4f}, "
          f"8-pass only {res['cost_r_only']:.4f}; unique loss {res['loss_u_raw']:.4f} -> {res['loss_u_cal']:.4f}", flush=True)
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)
    print("written", OUT, flush=True)


if __name__ == "__main__":
    main()
