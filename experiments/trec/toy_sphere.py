"""Toy check of the MuonH decay on one matrix: pure-noise gradients, tangent to the sphere (a scale-invariant layer),
Nesterov momentum, Newton-Schulz direction, levanter's norm_preserving_update rule (re-implemented in numpy).

Prediction: the outward radial fraction of the step is r = eta * sum_{k>=1} rho_k (rho = step autocorrelation), so the
projection shrinks W, and any perturbation stored in it, by a = (1 + 2 eta r + eta^2)^(-1/2) ~ 1 - (A/2) eta^2 per step,
with A = 1 + 2 sum_{k>=1} rho_k. Usage: python toy_sphere.py OUT.json [beta:eta ...]
Copied from the Tokenizer Size session (tmp/trec/scripts/toy_sphere.py, 2026-10-26); the TREC page adds momentum 0.95,
the 130m recipe, given on the command line as beta:eta pairs.
"""

import json
import sys

import numpy as np

NS_COEFF = (3.4445, -4.7750, 2.0315)


def ns(g, eps=1e-5):
    x = g / (np.linalg.norm(g) + eps)
    t = x.shape[0] > x.shape[1]
    if t:
        x = x.T
    a, b, c = NS_COEFF
    for _ in range(5):
        A = x @ x.T
        x = a * x + (b * A + c * (A @ A)) @ x
    return x.T if t else x


def run(beta, eta, steps, n=128, seed=0, nesterov=True, use_ns=True):
    rng = np.random.default_rng(seed)
    W0 = rng.standard_normal((n, n)).astype(np.float32)
    W0 /= np.linalg.norm(W0)
    W = W0.copy()
    delta = rng.standard_normal((n, n))
    delta -= (delta * W).sum() * W
    delta *= 1e-4 / np.linalg.norm(delta)
    Wp = W + delta
    Wp /= np.linalg.norm(Wp)
    buf = np.zeros_like(W)
    bufp = np.zeros_like(W)
    dirs, rs, log_a, pert = [], [], [], []
    for t in range(steps):
        G = rng.standard_normal((n, n)).astype(np.float32)
        g = G - (G * W).sum() * W          # tangent at W (scale-invariant layer)
        gp = G - (G * Wp).sum() * Wp       # the twin's own tangent projection of the same noise
        buf = beta * buf + g
        bufp = beta * bufp + gp
        u = beta * buf + g if nesterov else buf
        up = beta * bufp + gp if nesterov else bufp
        O = ns(u) if use_ns else u
        Op = ns(up) if use_ns else up
        O = O / np.linalg.norm(O)   # not in place: with use_ns=False, O is the momentum buffer itself
        Op = Op / np.linalg.norm(Op)
        r = -(O * W).sum()                 # |W| = 1
        new = W - eta * O
        a = 1.0 / np.linalg.norm(new)
        W = new * a
        newp = Wp - eta * Op
        Wp = newp / np.linalg.norm(newp)
        rs.append(r)
        log_a.append(np.log(a))
        pert.append(np.linalg.norm(Wp - W))
        if t % 10 == 0:
            dirs.append(O.ravel().copy())
    # step autocorrelation from the saved directions (every 10 steps) and from consecutive r
    burn = steps // 4
    rs = np.array(rs)
    la = np.array(log_a)
    out = dict(beta=beta, eta=eta, steps=steps, nesterov=nesterov, ns=use_ns,
               r_mean=float(rs[burn:].mean()), r_over_eta=float(rs[burn:].mean() / eta),
               A_from_r=float(1 + 2 * rs[burn:].mean() / eta),
               decay_per_step=float(-la[burn:].mean()),
               A_from_decay=float(2 * (-la[burn:].mean()) / eta**2),
               overlap_W0=float((W * W0).sum()),
               pred_overlap_from_a=float(np.exp(la.sum())),
               pert_ratio_end=float(pert[-1] / pert[0]),
               pert_ratio_pred=float(np.exp(la[1:].sum())))
    return out


def main():
    res = []
    runs = [(0.0, 0.003, True, True), (0.9, 0.003, True, True), (0.98, 0.003, True, True),
            (0.98, 0.003, False, False), (0.98, 0.0015, True, True)]
    if len(sys.argv) > 2:
        runs = [(float(b), float(e), True, True) for b, e in (x.split(":") for x in sys.argv[2:])]
    for beta, eta, nesterov, use_ns in runs:
        o = run(beta, eta, 1500, nesterov=nesterov, use_ns=use_ns)
        print(json.dumps(o), flush=True)
        res.append(o)
    json.dump(res, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()
