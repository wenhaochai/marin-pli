"""Momentum test (owner popup 2026-10-31): the three 2480 arms at momentum 0.98 (w727-w729) against the same arms at
0.95 (w658-w660). Prints, per arm, the fraction of the imprint left after 248 steps and at the end and the half-life,
then the spread of the 0.98 arms on the fourth panel's axis (sum of lr^1.32), and a two-term check: fit
rate = 1/half-life = a lr + c lr^2 to the 0.95 arms (least squares in relative error) and predict the 0.98 half-lives if the
lr^2 part were the fixed-norm shrink, which momentum 0.98 makes A_98 / A_95 = 73 / 23 times as strong.
Run on a vis node: python3 experiments/trec/tools/branch_m98.py"""
import sys
import numpy as np
sys.path.insert(0, "/scratch/gpfs/GROUP/USER/project/marin-trec/experiments/trec")
sys.path.insert(0, "/scratch/gpfs/GROUP/USER/project/marin-trec/experiments/trec/tools")
import build_site_data as b
from branch_fit import spread  # noqa: E402  (branch_fit prints its own table on import)

A95, A98 = 23.0, 73.0
c95, c98 = b.branch_curves(2480), b.branch_curves(2480, "-mom0.98")
assert c95 and c98, "an arm is not scored"


def half_life(a):
    return float(np.interp(0.5, a[::-1, 1], a[::-1, 0]))


def left_at(a, s):
    return float(np.interp(s, a[:, 0], a[:, 1]))


print("\nlr       mom   left@248  left@end  half-life")
rows = []
for (_, lr, _, a95), (_, lr98, _, a98) in zip(c95, c98):
    assert lr == lr98
    h95, h98 = half_life(a95), half_life(a98)
    rows.append((lr, h95, h98))
    for mom, a, h in (("0.95", a95, h95), ("0.98", a98, h98)):
        print(f"{lr:<8g} {mom}  {left_at(a, 248):.3f}     {a[-1, 1]:.3f}     {h:.0f}")
    print(f"         half-life 0.98 / 0.95 = {h98 / h95:.2f}")

arms98 = [(lr, a) for _, lr, _, a in c98]
ps = np.arange(0.5, 2.51, 0.01)
sp = np.array([spread(arms98, p) for p in ps])
print(f"\n0.98 arms: spread at p = 1.32 (panel-4 axis) {spread(arms98, 1.32):.1e}; best {sp.min():.1e} at p {ps[sp.argmin()]:.2f}")

lr = np.array([r[0] for r in rows]); h95 = np.array([r[1] for r in rows]); h98 = np.array([r[2] for r in rows])
r95 = 1 / h95
a, c = np.linalg.lstsq(np.stack([lr / r95, lr ** 2 / r95], 1), np.ones(len(lr)), rcond=None)[0]
pred = 1 / (a * lr + c * A98 / A95 * lr ** 2)
print(f"two-term fit to 0.95: rate = {a:.3f} lr + {c:.1f} lr^2; fitted 0.95 half-lives {np.round(1 / (a * lr + c * lr ** 2)).astype(int).tolist()}"
      f" (measured {np.round(h95).astype(int).tolist()})")
print(f"shrink x{A98 / A95:.2f} in the lr^2 part predicts 0.98 half-lives {np.round(pred).astype(int).tolist()};"
      f" measured {np.round(h98).astype(int).tolist()}")
print(f"effective p from half-lives (lr 0.0025 vs 0.01): 0.95 {np.log(h95[-1] / h95[0]) / np.log(4):.2f},"
      f" predicted 0.98 {np.log(pred[-1] / pred[0]) / np.log(4):.2f}, measured 0.98 {np.log(h98[-1] / h98[0]) / np.log(4):.2f}")
