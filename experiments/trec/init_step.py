"""Does the last MuonH step still point along the initialization? (owner's approval 2026-10-27; idea from the vocab
session.) The final matrices keep cos(W_T, W_0) of a few percent, more in the first layer's attention, while the
momentum reading of the rescaling predicts almost none (fig-init). One explanation is that later updates keep adding
the initial matrix back. This reads, for every MuonH matrix of a final checkpoint, the momentum buffer M that sets the
next step, -eta R NS(M) / |NS(M)| (Nesterov adds the new gradient, which the buffer of 0.95 or 0.98 outweighs), and the
initial matrix rebuilt from the run's seed (init_retention.models), and reports

  cos_init  <W_T, W_0> / (|W_T| |W_0|)
  r_O       -<NS(M), W_T> / (|NS(M)| |W_T|), the outward radial part of the step
  q_O       -<NS(M), W_0> / (|NS(M)| |W_0|), the step's overlap with the initial matrix
  q_perp    q_O - r_O cos_init, the part of q_O that the radial part along W_T does not account for
  floor     1 / sqrt(number of entries), the size of the overlap of two random directions

The checkpoint layout, the (Out, In) flattening and the Newton-Schulz iteration (levanter "simple" coefficients, 5 steps,
eps 1e-5) follow the vocab session's tmp/trec/scripts/measure_radial.py.

    JAX_PLATFORMS=cpu CKPT=<.../step-N> MODEL=300m|130m-d12 SEED=<trainer seed> OUT=<file.json> \
        python -m experiments.trec.init_step          # from the worktree, on a vis node
"""

import json
import os

import numpy as np
import tensorstore

from experiments.trec import init_retention as ir

NS_COEFF = (3.4445, -4.7750, 2.0315)
NS_EPS = 1e-5
MATS = {"q": "self_attn/q_proj", "k": "self_attn/k_proj", "v": "self_attn/v_proj", "o": "self_attn/o_proj",
        "gate": "mlp/gate_proj", "up": "mlp/up_proj", "down": "mlp/down_proj"}
OUT_FIRST = {"o", "down"}   # these store [Out=embed, In...]; the rest store [Out..., In=embed]
BUFFER = "opt_state/inner_state/inner_states/muonh/inner_state/1/momentum_buffer/transformer/layers/stacked/{}/weight"


def read(ckpt, path):
    spec = {"driver": "zarr3", "kvstore": {"driver": "ocdbt", "base": {"driver": "file", "path": ckpt}, "path": path}}
    return np.asarray(tensorstore.open(spec).result().read().result(), dtype=np.float64)


def to2d(name, a):
    return a.reshape(a.shape[0], -1) if name in OUT_FIRST else a.reshape(-1, a.shape[-1])


def newton_schulz(g):
    x = g / (np.linalg.norm(g) + NS_EPS)
    transpose = x.shape[0] > x.shape[1]
    if transpose:
        x = x.T
    a, b, c = NS_COEFF
    for _ in range(5):
        A = x @ x.T
        x = a * x + (b * A + c * (A @ A)) @ x
    return x.T if transpose else x


def main():
    ckpt = ir.ts.vtk.CKPT
    init, _ = ir.models()
    w0 = {p: w for p, w in ir.leaves(init)}
    rows = []
    for name, sub in MATS.items():
        W0 = w0[f".transformer.layers.stacked.{sub.replace('/', '.')}.weight"]
        WT = read(ckpt, f"model/transformer/layers/stacked/{sub}/weight")
        M = read(ckpt, BUFFER.format(sub))
        assert W0.shape == WT.shape == M.shape, (name, W0.shape, WT.shape, M.shape)
        for layer in range(W0.shape[0]):
            x0, xT, m = to2d(name, W0[layer]), to2d(name, WT[layer]), to2d(name, M[layer])
            o = newton_schulz(m)
            n0, nT, no = np.linalg.norm(x0), np.linalg.norm(xT), np.linalg.norm(o)
            cos_init = float((xT * x0).sum() / (nT * n0))
            r_O = float(-(o * xT).sum() / (no * nT))
            q_O = float(-(o * x0).sum() / (no * n0))
            rows.append({"matrix": name, "layer": layer, "norm_init": float(n0), "norm_final": float(nT), "cos_init": cos_init,
                         "r_O": r_O, "q_O": q_O, "q_perp": q_O - r_O * cos_init, "floor": float(1 / np.sqrt(x0.size))})
            r = rows[-1]
            print(f"{name:5s} layer {layer:2d}  cos_init {cos_init:+.4f}  r_O {r_O:+.4f}  q_O {q_O:+.4f}  q_perp {r['q_perp']:+.4f}  "
                  f"floor {r['floor']:.4f}", flush=True)
    json.dump({"ckpt": ckpt, "model": ir.ts.MODEL, "seed": ir.SEED, "rows": rows}, open(os.environ["OUT"], "w"), indent=1)


if __name__ == "__main__":
    main()
