"""Momentum test (owner popup 2026-10-31): the three 2480 arms at momentum 0.98 (w727-w729) against the same arms at
0.95 (w658-w660), on the fraction of the imprint left f from build_site_data.branch_curves.

Measured, per arm: f after 248 steps and at the end, the end ratio f98/f95, the steps to reach f = 0.3 (first crossing,
log-linear between kept checkpoints; the half-life of the lr-0.01 arms falls before the first checkpoint at 124 steps,
so it is not used) and their ratio, and the extra forgetting D(s) = ln f95(s) - ln f98(s) at each kept checkpoint.

Shrink model: if momentum 0.98 acted only by making the fixed-norm shrink stronger (-ln a = (A/2) lr^2 per step) on the
whole imprint, f98(s) = f95(s) exp(-(dA/2) lr^2 R(s)) with R(s) = s, or R(s) = sum_k (1 - 0.98^k) when the extra shrink
builds up with the momentum buffer carried over from 0.95 (a rough stand-in for its 50-step time constant). dA = 73 - 23
(A at 0.98 from 300m checkpoints, at 0.95 from the simulation) or 65 - 23 (both from the simulation; STATUS Facts). Prints
each variant's end ratios, steps-to-0.3 ratios, best p and range (branch_fit.spread, within twice the best) and gap on
the p = 1.32 axis, next to the measured ones. Two checks that need no value of A: the dA each arm's end would need
(D_end = (dA/2) lr^2 R(s_end)), and how D after 124 steps grows per doubling of lr (a shrink acting at once: 4x).
Run on a vis node: python3 experiments/trec/tools/branch_m98.py"""
import sys
import numpy as np
sys.path.insert(0, "/scratch/gpfs/GROUP/USER/project/marin-trec/experiments/trec")
sys.path.insert(0, "/scratch/gpfs/GROUP/USER/project/marin-trec/experiments/trec/tools")
import build_site_data as b
from branch_fit import spread

c95, c98 = b.branch_curves(2480), b.branch_curves(2480, "-mom0.98")
assert c95 and c98, "an arm is not scored"
LRS = [lr for _, lr, _, _ in c95]
S = c95[0][3][:, 0]


def steps_to(a, level):
    """First crossing of f = level, log-linear between kept checkpoints (nan if f never gets there)."""
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


def summary(name, arms98):
    arms95 = [(lr, a) for _, lr, _, a in c95]
    end = [a98[-1, 1] / a95[-1, 1] for (_, a95), (_, a98) in zip(arms95, arms98)]
    t03 = [steps_to(a98, 0.3) / steps_to(a95, 0.3) for (_, a95), (_, a98) in zip(arms95, arms98)]
    p, lo, hi, g132 = fit_p(arms98)
    print(f"{name:<34} end ratio {np.round(end, 3).tolist()}  steps to 0.3 ratio {np.round(t03, 3).tolist()}"
          f"  p {p} ({lo}-{hi})  gap at p 1.32 {g132:.1e}")


arms98 = [(lr, a) for _, lr, _, a in c98]
print("lr       f@248 .95/.98   end .95/.98   steps to 0.3 .95/.98")
for (_, lr, _, a95), (_, a98) in zip(c95, arms98):
    assert np.array_equal(a95[:, 0], a98[:, 0]) and (a98[1:, 1] < a95[1:, 1]).all()
    print(f"{lr:<8g} {np.interp(248, a95[:, 0], a95[:, 1]):.3f}/{np.interp(248, a98[:, 0], a98[:, 1]):.3f}"
          f"   {a95[-1, 1]:.3f}/{a98[-1, 1]:.3f}   {steps_to(a95, 0.3):.0f}/{steps_to(a98, 0.3):.0f}")
    d = np.log(a95[1:, 1]) - np.log(a98[1:, 1])
    print("   measured extra forgetting D(s): " + ", ".join(f"{int(x)}: {v:.3f}" for x, v in zip(a95[1:, 0], d)))

print("\n0.95 arms: p %s (%s-%s), gap at p 1.32 %.1e" % fit_p([(lr, a) for _, lr, _, a in c95]))
summary("measured 0.98", arms98)
for dA in (50.0, 42.0):
    for ramp in (False, True):
        pred = []
        for _, lr, _, a95 in c95:
            s = a95[:, 0]
            R = s - 0.98 * (1 - 0.98 ** s) / 0.02 if ramp else s
            pred.append((lr, np.stack([s, a95[:, 1] * np.exp(-dA / 2 * lr ** 2 * R)], 1)))
        summary(f"shrink dA {dA:g}{', buffer ramp' if ramp else ''}", pred)
        print(f"{'':<34} extra at the ends " + str([round(float(np.log(a95[-1, 1] / q[-1, 1])), 3) for (_, _, _, a95), (_, q) in zip(c95, pred)]))

print("\nNo single shrink: dA that each arm's end would need, at once / with the build-up: "
      + str([round(float(2 * np.log(a95[-1, 1] / a98[-1, 1]) / (lr ** 2 * a95[-1, 0])), 1) for (_, lr, _, a95), (_, a98) in zip(c95, arms98)]) + " / "
      + str([round(float(2 * np.log(a95[-1, 1] / a98[-1, 1]) / (lr ** 2 * (a95[-1, 0] - 0.98 * (1 - 0.98 ** a95[-1, 0]) / 0.02))), 1) for (_, lr, _, a95), (_, a98) in zip(c95, arms98)]))
d124 = [float(np.log(a95[1, 1] / a98[1, 1])) for (_, _, _, a95), (_, a98) in zip(c95, arms98)]
assert all(a95[1, 0] == 124 for _, _, _, a95 in c95)
print(f"D at 124 steps {np.round(d124, 3).tolist()}; per doubling of lr x{d124[0] / d124[1]:.2f} and x{d124[1] / d124[2]:.2f} (a shrink acting at once: x4)")
