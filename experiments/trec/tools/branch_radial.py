"""A check on the Figure 10 branches (owner popup 2026-11-01, 「CPU 检查 A」; rules in docs/trec/STATUS.md "A check",
committed before the run). Does momentum 0.98 make MuonH's fixed-norm shrink stronger on these runs?

For each MuonH matrix W (7 kinds x 12 layers = 84) and its stored momentum buffer M (after step k), levanter's MuonH
(optim/muonh.py: buf <- beta buf + g, Nesterov u = beta buf + g; optim/util.py norm_preserving_update) takes the next step
W' = (W - eta |W| NS(u)/|NS(u)|) rescaled to |W|, with u = beta^2 M + (1 + beta) g'. The rescaling shrinks everything
already in W by a = (1 + 2 eta r + eta^2)^(-1/2), r = -<NS(u), W> / (|NS(u)| |W|), so -ln a = eta r + eta^2 / 2, and with
r = kappa eta + r0, -ln a = (A/2) eta^2 + r0 eta, A = 2 kappa + 1. The next gradient g' is not stored; two readings:
  r_M      u = M (the reading of tmp/trec/scripts/measure_radial.py),
  r_noise  u = beta^2 M + (1 + beta) g_hat, g_hat random and tangent to W with |g_hat| = sqrt(1 - beta^2) |M| (the
           pure-noise steady state |M|^2 = |g|^2 / (1 - beta^2)), mean of 2 seeded draws.
Newton-Schulz as in training: float32, "simple" coefficients, 5 steps, eps 1e-5 (tmp/trec/scripts/measure_radial.py).

    measure (marin venv, tensorstore):  OMP_NUM_THREADS=7 python branch_radial.py measure
    analyze (numpy):                    python3 branch_radial.py analyze   (the pre-registered rules)
    model (numpy):                      python3 branch_radial.py model     (added after the run, verification
                                        2026-11-01: per-step shrink ratio, the measured shrink as a model of the
                                        0.98 curves, the shrink's share of the 0.95 forgetting)
Output: /scratch/gpfs/GROUP/USER/project/trec/out/ck/radial/radial.json (per matrix and checkpoint).
"""
import json
import os
import sys

import numpy as np

STORE = "/scratch/gpfs/GROUP/USER/marin_store_big/speedrun"
OUT = "/scratch/gpfs/GROUP/USER/project/trec/out/ck/radial"
BASE = f"{STORE}/muonh-qwen3-130m-della4xh100-d12-ck248/2026.09.13/checkpoints/step-2480"
ARMS = [("0.5", 0.01, 2976), ("0.25", 0.005, 3472), ("0.125", 0.0025, 4464)]   # LR_MUL, lr, last kept step
MOMS = [("", 0.95), ("-mom0.98", 0.98)]
SS = (124, 248)                                                                   # plus the last kept step
NS_COEFF = (3.4445, -4.7750, 2.0315)
NS_EPS = 1e-5
MATS = {"q": "self_attn/q_proj", "k": "self_attn/k_proj", "v": "self_attn/v_proj", "o": "self_attn/o_proj",
        "gate": "mlp/gate_proj", "up": "mlp/up_proj", "down": "mlp/down_proj"}
OUT_FIRST = {"o", "down"}


def ckpts():
    """[(label, lr or None, beta list, path)] of the 19 checkpoints."""
    out = [("base", None, (0.95, 0.98), BASE)]
    for lrm, lr, last in ARMS:
        for tag, beta in MOMS:
            run = f"muonh-qwen3-130m-della4xh100-d12-constant-ck124-lrm{lrm}{tag}-br2480"
            for s in SS + (last - 2480,):
                out.append((f"{beta}/{lr}/{s}", lr, (beta,), f"{STORE}/{run}/2026.09.13/checkpoints/step-{2480 + s}"))
    return out


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


def radial(o, w):
    return float(-(o * w).sum() / (np.linalg.norm(o) * np.linalg.norm(w)))


def measure():
    import tensorstore as ts

    def read(ck, path):
        spec = {"driver": "zarr3", "kvstore": {"driver": "ocdbt", "base": {"driver": "file", "path": ck}, "path": path}}
        return np.asarray(ts.open(spec).result().read().result(), dtype=np.float32)

    os.makedirs(OUT, exist_ok=True)
    res = {}
    for label, lr, betas, ck in ckpts():
        assert json.load(open(f"{ck}/metadata.json"))["step"] == int(ck.rsplit("-", 1)[1]), ck
        rows = {}
        for kind, sub in MATS.items():
            W = read(ck, f"model/transformer/layers/stacked/{sub}/weight")
            M = read(ck, f"opt_state/inner_state/inner_states/muonh/inner_state/1/momentum_buffer/transformer/layers/stacked/{sub}/weight")
            for layer in range(W.shape[0]):
                w = W[layer].reshape(W[layer].shape[0], -1) if kind in OUT_FIRST else W[layer].reshape(-1, W[layer].shape[-1])
                m = M[layer].reshape(w.shape) if kind in OUT_FIRST else M[layer].reshape(-1, M[layer].shape[-1])
                rec = {"r_M": radial(newton_schulz(m), w), "m_over_w": float(np.linalg.norm(m) / np.linalg.norm(w))}
                for beta in betas:
                    rs = []
                    for draw in range(2):
                        rng = np.random.default_rng([layer, list(MATS).index(kind), draw, int(beta * 100)])
                        z = rng.standard_normal(w.shape).astype(np.float32)
                        z -= (z * w).sum() / (w * w).sum() * w
                        z *= np.sqrt(1 - beta ** 2) * np.linalg.norm(m) / np.linalg.norm(z)
                        rs.append(radial(newton_schulz(beta ** 2 * m + (1 + beta) * z), w))
                    rec[f"r_noise_{beta}"] = float(np.mean(rs))
                rows[f"{kind}{layer}"] = rec
        res[label] = {"ckpt": ck, "lr": lr, "mats": rows}
        med = np.median([v["r_M"] for k, v in rows.items() if not k.startswith("gate")])
        print(f"{label:<16} median r_M (no gate) {med:.5f}", flush=True)
    json.dump(res, open(f"{OUT}/radial.json", "w"), indent=0)


def analyze():
    sys.path.insert(0, "/scratch/gpfs/GROUP/USER/project/marin-trec/experiments/trec")
    import build_site_data as b
    res = json.load(open(f"{OUT}/radial.json"))
    names = list(res["base"]["mats"])
    core = [n for n in names if not n.startswith("gate")]

    def r(label, reading, beta=None):
        key = reading if reading == "r_M" else f"r_noise_{beta}"
        return np.array([res[label]["mats"][n][key] for n in names])

    sel = np.array([n in core for n in names])
    gate = ~sel
    print(f"{len(names)} matrices, {sel.sum()} without gate_proj; reading: median over those (gate apart)\n")
    for reading in ("r_noise", "r_M"):
        print(f"== {reading}")
        # dA per matrix, paired at equal lr and s
        for lrm, lr, last in ARMS:
            for s in SS + (last - 2480,):
                r95 = r(f"0.95/{lr}/{s}", reading, 0.95); r98 = r(f"0.98/{lr}/{s}", reading, 0.98)
                dA = 2 * (r98 - r95) / lr
                print(f"lr {lr:<7g} s {s:<5} median r 0.95 {np.median(r95[sel]):.5f}  0.98 {np.median(r98[sel]):.5f}"
                      f"  dA median {np.median(dA[sel]):6.1f} (IQR {np.percentile(dA[sel], 25):6.1f} to {np.percentile(dA[sel], 75):6.1f}),"
                      f" mean {np.mean(dA[sel]):6.1f}; gate {np.median(dA[gate]):6.1f}")
        # A per momentum from r = kappa lr + r0 over the three arms at equal s
        for s in SS:
            for beta in (0.95, 0.98):
                lrs = np.array([lr for _, lr, _ in ARMS])
                R = np.stack([r(f"{beta}/{lr}/{s}", reading, beta) for _, lr, _ in ARMS])     # (3, n)
                kappa, r0 = np.polyfit(lrs, R, 1)
                A = 2 * kappa + 1
                print(f"s {s:<4} momentum {beta}: A median {np.median(A[sel]):6.1f} (IQR {np.percentile(A[sel], 25):6.1f} to"
                      f" {np.percentile(A[sel], 75):6.1f}), r0 median {np.median(r0[sel]):.4f}; gate A {np.median(A[gate]):6.1f}")
        rb = r("base", reading, 0.95)
        R = np.stack([r(f"0.95/{lr}/248", reading, 0.95) for _, lr, _ in ARMS])
        kappa, r0 = np.polyfit(np.array([lr for _, lr, _ in ARMS]), R, 1)
        print(f"base step-2480 (lr 0.0125, buffer at 0.95): median r {np.median(rb[sel]):.5f};"
              f" A from the 0.95 r0 at s 248: {np.median((2 * (rb - r0) / 0.0125 + 1)[sel]):.1f}")
        # predicted growth of D from s = 248 to the end against the measured one
        c95, c98 = b.branch_curves(2480), b.branch_curves(2480, "-mom0.98")
        for (lrm, lr, last), (_, _, _, a95), (_, _, _, a98) in zip(ARMS, c95, c98):
            dr248 = np.median((r(f"0.98/{lr}/248", reading, 0.98) - r(f"0.95/{lr}/248", reading, 0.95))[sel])
            drend = np.median((r(f"0.98/{lr}/{last - 2480}", reading, 0.98) - r(f"0.95/{lr}/{last - 2480}", reading, 0.95))[sel])
            s_end = a95[-1, 0]
            pred = lr * (dr248 + drend) / 2 * (s_end - 248)
            d = lambda s: float(np.interp(s, a95[:, 0], np.log(a95[:, 1])) - np.interp(s, a98[:, 0], np.log(a98[:, 1])))
            print(f"lr {lr:<7g} growth of D from s 248 to {int(s_end)}: predicted {pred:+.3f}, measured {d(s_end) - d(248):+.3f}")
        print()


def model():
    sys.path.insert(0, "/scratch/gpfs/GROUP/USER/project/marin-trec/experiments/trec")
    sys.path.insert(0, "/scratch/gpfs/GROUP/USER/project/marin-trec/experiments/trec/tools")
    import build_site_data as b
    from branch_fit import spread
    res = json.load(open(f"{OUT}/radial.json"))
    names = list(res["base"]["mats"])
    sel = np.array([not n.startswith("gate") for n in names])
    c95, c98 = b.branch_curves(2480), b.branch_curves(2480, "-mom0.98")

    def r(label, reading, beta):
        key = reading if reading == "r_M" else f"r_noise_{beta}"
        return np.array([res[label]["mats"][n][key] for n in names])

    def steps_to(a, level):
        s, f = a[:, 0], a[:, 1]
        for i in range(1, len(f)):
            if f[i] <= level < f[i - 1]:
                t = (np.log(f[i - 1]) - np.log(level)) / (np.log(f[i - 1]) - np.log(f[i]))
                return float(s[i - 1] + t * (s[i] - s[i - 1]))
        return float("nan")

    def fit_p(arms):
        ps = np.arange(0.5, 2.51, 0.01)
        sp = np.array([spread(arms, p) for p in ps])
        k = int(np.nanargmin(sp)); ok = ps[sp <= 2 * sp[k]]
        return round(float(ps[k]), 2), round(float(ok.min()), 2), round(float(ok.max()), 2), float(spread(arms, 1.32))

    for reading in ("r_noise", "r_M"):
        print(f"== {reading}")
        ratios = []
        for lrm, lr, last in ARMS:
            for s in SS + (last - 2480,):
                q = (lr * r(f"0.98/{lr}/{s}", reading, 0.98) + lr * lr / 2) / (lr * r(f"0.95/{lr}/{s}", reading, 0.95) + lr * lr / 2)
                ratios.append(float(np.median(q[sel])))
        print(f"per-step shrink (lr r + lr^2/2), 0.98 over 0.95, median per matrix: {min(ratios):.2f} to {max(ratios):.2f}")
        # the measured extra shrink as a model of the 0.98 curves: f95(s) exp(-lr dr R(s)), dr = median r98 - r95 (mean of s 248 and last)
        for ramp in (False, True):
            pred, ends, t03, shares = [], [], [], []
            for (lrm, lr, last), (_, _, _, a95), (_, _, _, a98) in zip(ARMS, c95, c98):
                dr = np.median(((r(f"0.98/{lr}/248", reading, 0.98) - r(f"0.95/{lr}/248", reading, 0.95))
                                + (r(f"0.98/{lr}/{last - 2480}", reading, 0.98) - r(f"0.95/{lr}/{last - 2480}", reading, 0.95)))[sel] / 2)
                sv = a95[:, 0]
                R = sv - 0.98 * (1 - 0.98 ** sv) / 0.02 if ramp else sv
                q = np.stack([sv, a95[:, 1] * np.exp(-lr * dr * R)], 1)
                pred.append((lr, q)); ends.append(q[-1, 1] / a95[-1, 1]); t03.append(steps_to(q, 0.3) / steps_to(a95, 0.3))
            p, lo, hi, g = fit_p(pred)
            print(f"measured extra shrink{' with build-up' if ramp else ', at once':<15} end ratio {np.round(ends, 3).tolist()}"
                  f"  steps to 0.3 ratio {np.round(t03, 3).tolist()}  p {p} ({lo}-{hi})  gap at p 1.32 {g:.4f}")
        for (lrm, lr, last), (_, _, _, a95) in zip(ARMS, c95):
            r95 = np.median(((r(f"0.95/{lr}/248", reading, 0.95) + r(f"0.95/{lr}/{last - 2480}", reading, 0.95)) / 2)[sel])
            share = (lr * r95 + lr * lr / 2) * a95[-1, 0] / -np.log(a95[-1, 1])
            shares.append(round(float(share), 3))
        print(f"gross shrink of the 0.95 arms over each run / -ln f95 at the end: {shares}")
        early = []
        for (lrm, lr, last), (_, _, _, a95), (_, _, _, a98) in zip(ARMS, c95, c98):
            assert a95[1, 0] == 124
            dr = np.median((r(f"0.98/{lr}/124", reading, 0.98) - r(f"0.95/{lr}/124", reading, 0.95))[sel])
            d124 = float(np.log(a95[1, 1] / a98[1, 1]))
            early.append((round(d124 / (lr * dr * 124), 2), round(d124 / (lr * dr * (124 - 0.98 * (1 - 0.98 ** 124) / 0.02)), 2)))
        print(f"extra forgetting at s 124 over the measured extra shrink by then (at once, with build-up): {early}")
        print()


if __name__ == "__main__":
    {"measure": measure, "analyze": analyze, "model": model}[sys.argv[1]]()
