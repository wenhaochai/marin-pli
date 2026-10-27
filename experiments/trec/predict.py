"""TREC prediction of Bergsma, Dey and Hestness (2025, arXiv 2509.25380, eq. 3): L(t) = 1 - c(t)^p * t^m, t = i/T, p = 0.5,
with c_i the weight of update i in the final weights, normalized to a maximum of 1 (shape only; c_0, the
initialization, is dropped), for three readings of our optimizer and a control:

  momentum     MuonH and AdamH as they run (the vocab session's derivation, tmp/trec/muonh_prediction.md): every step
               W <- a_t (W - eta_t R O_t), a_t = (1 + 2 eta_t r_t + eta_t^2)^(-1/2), with r_t the outward radial part of
               the unit step O_t. Momentum makes steps point the same way, so r_t = kappa eta_t + r0 with
               kappa = (A - 1) / 2, A the two-sided sum of the step autocorrelation: A = 73 measured on the 300m
               checkpoints (momentum 0.98), A = 23 from the toy (experiments/trec/toy_sphere.py) at the 130m momentum
               0.95; r0 = 0.004 (the final checkpoints). c_i = eta_i prod_{j>=i} a_j.
  independent  the same rescaling with steps at right angles to the weights (A = 1, r = 0): a_t = (1 + eta_t^2)^(-1/2)
  adamw-0.1    the config's weight_decay=0.1 taken at face value (AdamW): c_i = eta_i prod_{j>i} (1 - 0.1 * eta_j)
  lr-only      the control: nothing shrinks, c_i = eta_i (the paper's F.3 limit of no weight decay)

eta is the matrix learning rate logged in W&B (optim/learning_rate), one value per step.

    python experiments/trec/predict.py      # prints each reading's peak and valley for a range of m (vis node)
"""

import json
import os

import numpy as np

CACHE = "/scratch/gpfs/GROUP/USER/project/trec/cache/wandb"
RUNS = {"130m-d12": "muonh-qwen3-130m-della4xh100-d12", "300m": "muonh-qwen3-300m-della4xh100-s1-rerun",
        "300m-8k": "muonh-qwen3-300m-della4xh100-v8k-s1", "300m-bytes": "muonh-qwen3-300m-della4xh100-bytes"}
TOTAL = {"130m-d12": 4959, "300m": 11444, "300m-8k": 15464, "300m-bytes": 54248}   # final checkpoint step + 1
A = {"130m-d12": 23.0, "300m": 73.0, "300m-8k": 73.0, "300m-bytes": 73.0}
R0 = 0.004


def lr_schedule(model):
    """The matrix learning rate at every step 0..T-1, from W&B (steps W&B did not log are filled from the neighbours)."""
    d = json.load(open(os.path.join(CACHE, f"{RUNS[model]}.json")))
    s, v = np.array(d["optim/learning_rate"]).T
    return np.interp(np.arange(TOTAL[model]), s, v)


def keep(eta, reading, model):
    """The factor by which step t shrinks everything already in the weights."""
    if reading == "momentum":
        r = (A[model] - 1) / 2 * eta + R0
        return 1 / np.sqrt(1 + 2 * eta * r + eta ** 2)
    if reading == "independent":
        return 1 / np.sqrt(1 + eta ** 2)
    if reading == "lr-only":
        return np.ones_like(eta)
    assert reading == "adamw-0.1", reading
    return 1 - 0.1 * eta


def coefficients(eta, reading, model):
    k = np.log(keep(eta, reading, model))
    after = np.cumsum(k[::-1])[::-1]          # sum over j >= i
    if reading == "adamw-0.1":
        after = np.concatenate([after[1:], [0.0]])   # AdamW: the decay of the steps after i
    c = eta * np.exp(after)
    return c / c.max()


def predicted(c, m, p=0.5):
    t = (np.arange(len(c)) + 1) / len(c)
    return 1 - c ** p * t ** m


def pearson(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def fit_m(c, trec, ms=np.arange(0.0, 40.01, 0.1)):
    """The training-fraction exponent with the highest Pearson r between prediction and measured TREC (paper's m*)."""
    r = [pearson(predicted(c, m), trec) for m in ms]
    k = int(np.argmax(r))
    return float(ms[k]), r[k]


def valley(c, m):
    return float(np.argmin(predicted(c, m)) + 1) / len(c)


if __name__ == "__main__":
    for model in RUNS:
        eta = lr_schedule(model)
        T = len(eta)
        print(model, "T", T, "peak lr", round(eta.max(), 4), "A", A[model])
        for reading in ("momentum", "independent", "adamw-0.1", "lr-only"):
            c = coefficients(eta, reading, model)
            row = ", ".join(f"m={m}: {valley(c, m):.3f}" for m in (0, 0.5, 1, 2, 4, 8))
            print(f"  {reading:11s} c peak {np.argmax(c) / T:.3f}  valley at {row}")
