"""Which measure of the time since a batch was read makes the decay of its imprint the same in d12 and rsqrtcd:
steps, sum of lr, or sum of lr^2? Fraction left = E_s(b) / E_first(b); binned against each measure, the spread
between the two runs (mean |difference| of the binned curves over common bins, divided by the mean level)."""
import json, numpy as np
R = json.load(open("/scratch/gpfs/GROUP/USER/project/trec/out/ck/imprint.json"))
T = 4959; t = np.arange(T)
LR = {"muonh-qwen3-130m-della4xh100-d12-ck248": np.where(t < 991.8, 0.02, 0.02 * np.clip((T - t) / (T - 991.8), 0, 1)),
      "muonh-qwen3-130m-della4xh100-d12-rsqrtcd-ck248": np.where(t < 3967, 0.02 / np.sqrt(1 + 0.0092 * t), 0.02 / np.sqrt(1 + 0.0092 * 3967) * np.clip((T - t) / (T - 3967), 0, 1))}
pts = {}
for run, d in R.items():
    st, bl, E = np.array(d["steps"]), np.array(d["blocks"]), np.array(d["E"], dtype=float)
    lr = LR[run]; c1, c2 = np.concatenate([[0], np.cumsum(lr)]), np.concatenate([[0], np.cumsum(lr ** 2)])
    rows = []
    for j, b in enumerate(bl):
        ok = np.flatnonzero(~np.isnan(E[:, j]))
        if len(ok) < 2 or E[ok[0], j] < 0.05: continue
        e0 = E[ok[0], j]; s0 = st[ok[0]]
        for i in ok[1:]:
            s = st[i]; rows.append((s - s0, c1[s] - c1[s0], c2[s] - c2[s0], E[i, j] / e0, b / T))
    pts[run] = np.array(rows)
names = ["steps", "sum lr", "sum lr^2"]
for k, nm in enumerate(names):
    allx = np.concatenate([p[:, k] for p in pts.values()])
    bins = np.quantile(allx[allx > 0], np.linspace(0, 1, 11))
    curves = []
    for run, p in pts.items():
        c = [np.median(p[(p[:, k] >= lo) & (p[:, k] < hi), 3]) if ((p[:, k] >= lo) & (p[:, k] < hi)).sum() >= 3 else np.nan for lo, hi in zip(bins[:-1], bins[1:])]
        curves.append(np.array(c))
    ok = ~np.isnan(curves[0]) & ~np.isnan(curves[1])
    spread = np.mean(np.abs(curves[0][ok] - curves[1][ok])) / np.mean((curves[0][ok] + curves[1][ok]) / 2)
    print(f"{nm:9s} bins {ok.sum()}  d12 {np.round(curves[0], 3)}\n          rsqrtcd {np.round(curves[1], 3)}   relative spread {spread:.2f}")
