import json, numpy as np
from scipy.optimize import curve_fit
B = json.load(open("baselines.json"))["130m"]; Fb = B["backbone_flops"]
fin = np.load("bbfrozen.npz")["-pls1-sep-bbfrozen/final"]
x = np.array([(k + 1) / 6 * Fb for k in range(1, 6)]); y = fin[1:]
law = lambda c, E, A, a: E + A * (c / 1e18) ** (-a)
try:
    p, _ = curve_fit(law, x, y, p0=(y.min() - 0.3, 0.3, 0.5), bounds=([0, 0, 0], [y.min(), 10, 5]), maxfev=20000)
    print("power law", np.round(p, 3), "resid", np.round(law(x, *p) - y, 3))
except Exception as e: print("fit failed", e)
# log-log slope between neighbours: a power law has a slope that shrinks in size toward the end
print("local slopes dlogL/dlogC", np.round(np.diff(np.log(y)) / np.diff(np.log(x)), 3))
