"""Momentum test (owner popup 2026-10-31): the three 2480 arms at momentum 0.98 (w727-w729) against the same arms at
0.95 (w658-w660), on the fraction of the imprint left f from build_site_data.branch_curves.

Prints, per arm: f after 248 steps and at the end, the end ratio f98/f95, the steps to reach f = 0.3 (first crossing,
log-linear between kept checkpoints; the half-life of the lr-0.01 arms falls before the first checkpoint at 124 steps,
so it is not used) and their ratio 0.98/0.95, and the extra forgetting D(s) = ln f95(s) - ln f98(s) at each checkpoint.
Then the shrink check: if momentum 0.98 acted through the fixed-norm shrink, -ln a = (A/2) lr^2 per step with A 23 -> 73,
the extra forgetting would be D_shrink(s) = (73 - 23)/2 * lr^2 * s, growing with the steps and with lr^2.
A = 73 is measured on 300m checkpoints at momentum 0.98 and A = 23 in a simulation at 0.95 (STATUS Facts); neither on
these branches, which also start from the momentum buffer built at 0.95.
Run on a vis node: python3 experiments/trec/tools/branch_m98.py"""
import sys
import numpy as np
sys.path.insert(0, "/scratch/gpfs/GROUP/USER/project/marin-trec/experiments/trec")
import build_site_data as b

A95, A98 = 23.0, 73.0
c95, c98 = b.branch_curves(2480), b.branch_curves(2480, "-mom0.98")
assert c95 and c98, "an arm is not scored"


def steps_to(a, level):
    """First crossing of f = level, log-linear between kept checkpoints (nan if f never gets there)."""
    s, f = a[:, 0], a[:, 1]
    for i in range(1, len(f)):
        if f[i] <= level < f[i - 1]:
            t = (np.log(f[i - 1]) - np.log(level)) / (np.log(f[i - 1]) - np.log(f[i]))
            return float(s[i - 1] + t * (s[i] - s[i - 1]))
    return float("nan")


print("lr       f@248 .95/.98   end .95/.98   end ratio   steps to 0.3 .95/.98 (ratio)")
ratios_end, ratios_03, d_end, d_pred = [], [], [], []
for (_, lr, _, a95), (_, lr98, _, a98) in zip(c95, c98):
    assert lr == lr98 and np.array_equal(a95[:, 0], a98[:, 0])
    s = a95[:, 0]
    assert (a98[1:, 1] < a95[1:, 1]).all(), "a 0.98 fraction is not below its 0.95 value"
    t95, t98 = steps_to(a95, 0.3), steps_to(a98, 0.3)
    ratios_end.append(a98[-1, 1] / a95[-1, 1]); ratios_03.append(t98 / t95)
    print(f"{lr:<8g} {np.interp(248, s, a95[:, 1]):.3f}/{np.interp(248, s, a98[:, 1]):.3f}   {a95[-1, 1]:.3f}/{a98[-1, 1]:.3f}"
          f"   {ratios_end[-1]:.3f}       {t95:.0f}/{t98:.0f} ({ratios_03[-1]:.2f})")
    d = np.log(a95[1:, 1]) - np.log(a98[1:, 1])
    dp = (A98 - A95) / 2 * lr ** 2 * s[1:]
    d_end.append(d[-1]); d_pred.append(dp[-1])
    print("   extra forgetting D(s) at s = " + ", ".join(f"{int(x)}: {v:.3f}" for x, v in zip(s[1:], d)))
    print("   shrink model 25 lr^2 s      " + ", ".join(f"{int(x)}: {v:.3f}" for x, v in zip(s[1:], dp)))

print(f"\nend ratio f98/f95 {[round(float(r), 3) for r in ratios_end]}; steps to 0.3, 0.98/0.95 {[round(float(r), 3) for r in ratios_03]}")
print(f"extra forgetting at the end: measured {[round(float(v), 3) for v in d_end]}, shrink model {[round(float(v), 3) for v in d_pred]}")
