"""TREC prediction of Bergsma, Dey and Hestness (2025, arXiv 2509.25380, eq. 3): L(t) = 1 - c(t)^p * t^m, t = i/T, p = 0.5,
with c_i the weight of update i in the final weights, normalized to a maximum of 1 (shape only; c_0, the
initialization, is dropped), for three readings of our optimizer:

  fixed-norm  MuonH and AdamH keep each matrix's Frobenius norm: an update orthogonal to W of relative size eta shrinks
              the old weights by 1/sqrt(1 + eta^2) per step, so c_i = eta_i * prod_{j>i} (1 + eta_j^2)^(-1/2)
  adamw-0.1   the config's weight_decay=0.1 taken at face value (AdamW): c_i = eta_i * prod_{j>i} (1 - 0.1 * eta_j)
  schedule    no decay at all (the paper's lambda -> 0 limit, and the plain-Adam embeddings): c_i = eta_i

eta is the matrix learning rate logged in W&B (optim/learning_rate), one value per step.

    python experiments/trec/predict.py      # prints each reading's valley for a range of m (vis node)
"""

import json
import os

import numpy as np

CACHE = "/scratch/gpfs/GROUP/USER/project/trec/cache/wandb"
RUNS = {"130m-d12": "muonh-qwen3-130m-della4xh100-d12", "300m": "muonh-qwen3-300m-della4xh100-s1-rerun"}
TOTAL = {"130m-d12": 4959, "300m": 11444}


def lr_schedule(model):
    """The matrix learning rate at every step 0..T-1, from W&B (steps W&B did not log are filled from the neighbours)."""
    d = json.load(open(os.path.join(CACHE, f"{RUNS[model]}.json")))
    s, v = np.array(d["optim/learning_rate"]).T
    return np.interp(np.arange(TOTAL[model]), s, v)


def coefficients(eta, reading):
    if reading == "schedule":
        c = eta.copy()
    else:
        keep = 1 / np.sqrt(1 + eta ** 2) if reading == "fixed-norm" else 1 - 0.1 * eta
        assert reading in ("fixed-norm", "adamw-0.1"), reading
        after = np.concatenate([np.cumsum(np.log(keep)[::-1])[::-1][1:], [0.0]])   # sum over j > i
        c = eta * np.exp(after)
    return c / c.max()


def predicted(c, m, p=0.5):
    t = (np.arange(len(c)) + 1) / len(c)
    return 1 - c ** p * t ** m


def pearson(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def fit_m(c, trec, ms=np.arange(0.25, 40.01, 0.25)):
    """The training-fraction exponent with the highest Pearson r between prediction and measured TREC (paper's m*)."""
    r = [pearson(predicted(c, m), trec) for m in ms]
    k = int(np.argmax(r))
    return float(ms[k]), r[k]


if __name__ == "__main__":
    for model in RUNS:
        eta = lr_schedule(model)
        T = len(eta)
        print(model, "T", T, "peak lr", eta.max(), "sum eta^2/2", (eta ** 2).sum() / 2, "sum 0.1*eta", 0.1 * eta.sum())
        for reading in ("fixed-norm", "adamw-0.1", "schedule"):
            c = coefficients(eta, reading)
            row = []
            for m in (1, 2, 4, 6, 8, 12, 18):
                L = predicted(c, m)
                row.append(f"m={m}: {np.argmin(L) / T:.2f}")
            print(f"  {reading:10s} c(0)={c[1]:.3f} c(0.5)={c[T // 2]:.3f} c(0.9)={c[int(0.9 * T)]:.3f}  valley at", ", ".join(row))
