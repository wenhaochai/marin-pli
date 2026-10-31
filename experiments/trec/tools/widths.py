"""Figure 4 curves as drawn (measured: 0 at the low point, 1 at the unread mean; predictions: min-max): the mean over
each quarter of training, the half-depth interval (value < 0.5, 500-step mean for the measured curve) and the maximum."""
import sys
import numpy as np
sys.path.insert(0, "/scratch/gpfs/GROUP/USER/project/marin-trec/experiments/trec")
import build_site_data as bsd, predict
for model, _ in bsd.PANELS:
    train, unread = bsd.scored(model)
    s = bsd.smooth(train)
    T = len(s)
    t = (np.arange(T) + 1) / T
    mu = unread.mean()
    eta = predict.lr_schedule(model)
    meas = (s - s.min()) / (mu - s.min())
    s5 = bsd.smooth(train, 500)
    curves = {"measured": (meas, (s5 - s.min()) / (mu - s.min()))}
    for r in ("momentum", "independent", "adamw-0.1", "lr-only"):
        c = predict.coefficients(eta, r, model)
        m, _ = predict.fit_m(c, s)
        v = bsd.minmax(predict.predicted(c, m))
        curves[r] = (v, v)
    for k, (v, v5) in curves.items():
        below = np.where(v5 < 0.5)[0]
        q = [v[(t > a) & (t <= a + 0.25)].mean() for a in (0, 0.25, 0.5, 0.75)]
        print(f"{model:9s} {k:12s} quarters {np.round(q, 3)}; half depth [{t[below[0]]:.3f}, {t[below[-1]]:.3f}] width {t[below[-1]] - t[below[0]]:.3f}; max {v.max():.3f}")
