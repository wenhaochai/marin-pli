"""Is forgetting after a branch set by how far the weights moved?

The 130m-d12 MuonH run (d12-ck248) was continued from step S (1240, 2480, 3720) at a constant lr in three arms with
lr x N = 5: lr 0.01 x 500, 0.005 x 1000, 0.0025 x 2000, a checkpoint every 124 steps. The fraction f of the imprint of
the batches read just before S that is left at each checkpoint comes from build_site_data.branch_curves(S). f lines
up across the arms against steps * lr^p with p fitted per S (tools/branch_fit.py: 1.22, 1.32, 1.40), not against p = 1,
p = 2 or steps. This script tests whether f is instead a function of the weight displacement since the branch point.

Displacement. MuonH (matrices) and AdamH (lm_head) keep every matrix at its norm, a stacked array being a stack of
matrices along axis 0 (levanter.optim.util.norm_preserving_update). Per matrix the displacement is the angle
    theta = angle(W_s, W_S) = 2 arcsin(| W_s/|W_s| - W_S/|W_S| | / 2)      (= arccos of the cosine, but exact near 0)
over the 12 layers x {q, k, v, o, gate, up, down} plus lm_head = 85 matrices; the embedding (plain Adam) and 1-D
params are skipped. D_mean / D_median are the mean / median of theta over the 85; D_attn, D_mlp, D_head are the
means over the attention matrices, the MLP matrices and lm_head. |W_s| / |W_S| - 1 is checked to 1e-3 per matrix.

Collapse. Spread as in branch_fit.py: interpolate log f of each arm on 50 points over the common range of the x axis,
then take the mean squared gap to the arms' mean; x = log(steps * lr^p) (p = 0 is steps) or x = log D.

Growth. Per S and arm, alpha_arm = slope of log D against log N over the checkpoints. Per S, a free fit
log D = c + b log lr + alpha log N over all arms; f collapsing against D then implies f collapses against N lr^(b/alpha),
which is 1/alpha when b = 1.

Data. Branch point: marin_store_big/speedrun/muonh-qwen3-130m-della4xh100-d12-ck248/2026.09.13/checkpoints/step-<S>;
arms: .../muonh-qwen3-130m-della4xh100-d12-constant-ck124-lrm<M>-br<S>/2026.09.13/checkpoints/step-<s>, M = 0.5,
0.25, 0.125 for lr 0.01, 0.005, 0.0025. Arrays are read directly with tensorstore (zarr3 on OCDBT).
Output: out/ck/drift/drift.json.

    JAX_PLATFORMS=cpu PYTHONPATH=<worktree>:<worktree>/lib/*/src python tools/branch_drift.py     # on a vis node
"""
import glob
import json
import os
import sys

import numpy as np
import tensorstore as ts

sys.path.insert(0, "/scratch/gpfs/GROUP/USER/project/marin-trec/experiments/trec")
import build_site_data as bsd  # noqa: E402

STORE = "/scratch/gpfs/GROUP/USER/marin_store_big/speedrun"
BASE = f"{STORE}/muonh-qwen3-130m-della4xh100-d12-ck248/2026.09.13/checkpoints"
ARM = STORE + "/muonh-qwen3-130m-della4xh100-d12-constant-ck124-lrm{m}-br{s}/2026.09.13/checkpoints"
ARMS = [(0.5, 0.01), (0.25, 0.005), (0.125, 0.0025)]   # BRANCH order of build_site_data
BRANCH_POINTS = (1240, 2480, 3720)
L = "model/transformer/layers/stacked/"
ATTN = [L + f"self_attn/{n}_proj/weight" for n in "qkvo"]
MLP = [L + f"mlp/{n}_proj/weight" for n in ("gate", "up", "down")]
HEAD = ["model/lm_head/weight"]
OUT = "/scratch/gpfs/GROUP/USER/project/trec/out/ck/drift/drift.json"
NORM_TOL = 1e-3


def read(ckpt, path):
    spec = {"driver": "zarr3", "kvstore": {"driver": "ocdbt", "base": f"file://{ckpt}/", "path": path}}
    return np.asarray(ts.open(spec).result().read().result(), dtype=np.float64)


def unit_matrices(ckpt):
    """{group: (norms [n], unit matrices [n, -1])}; a stacked array is a stack of matrices along axis 0."""
    out = {}
    for group, paths in (("attn", ATTN), ("mlp", MLP), ("head", HEAD)):
        ws = []
        for p in paths:
            w = read(ckpt, p)
            ws.append(w.reshape(w.shape[0], -1) if p != HEAD[0] else w.reshape(1, -1))
        norms = np.concatenate([np.linalg.norm(w, axis=1) for w in ws])
        units = [w / np.linalg.norm(w, axis=1, keepdims=True) for w in ws]
        out[group] = (norms, units)
    return out


def angles(a, b):
    """{group: angles [n]} between two checkpoints, and the worst relative norm change."""
    th, dn = {}, 0.0
    for g in a:
        na, ua = a[g]
        nb, ub = b[g]
        th[g] = np.concatenate([2 * np.arcsin(np.minimum(np.linalg.norm(x - y, axis=1) / 2, 1.0)) for x, y in zip(ua, ub)])
        dn = max(dn, float(np.max(np.abs(nb / na - 1))))
    return th, dn


def spread(cs):
    """Mean squared gap in log f between arms, on 50 points over the common range of x; cs = [(x, log f)]."""
    lo, hi = max(c[0].min() for c in cs), min(c[0].max() for c in cs)
    g = np.linspace(lo, hi, 50)
    ys = np.array([np.interp(g, x, y) for x, y in cs])
    return float(np.mean((ys - ys.mean(0)) ** 2))


def spread_p(rows, p):
    return spread([(np.log(r["steps"] * r["lr"] ** p), np.log(r["f"])) for r in rows])


def analyse(s):
    curves = bsd.branch_curves(s)
    assert curves is not None, f"branch_curves({s}) not complete"
    ref = unit_matrices(f"{BASE}/step-{s}")
    arms = []
    for (m, lr), (label, lr_c, _, fa) in zip(ARMS, curves):
        assert lr == lr_c, (lr, lr_c)
        f_of = {int(k): float(v) for k, v in fa}
        pts = []
        for d in sorted(glob.glob(ARM.format(m=m, s=s) + "/step-*"), key=lambda d: int(d.rsplit("-", 1)[1])):
            n = int(d.rsplit("-", 1)[1]) - s
            th, dn = angles(ref, unit_matrices(d))
            assert dn < NORM_TOL, f"{d}: norm changed by {dn:.2e}"
            assert n in f_of, f"{d}: no f at {n} steps"
            allth = np.concatenate([th["attn"], th["mlp"], th["head"]])
            pts.append({"steps": n, "lr": lr, "f": f_of[n], "D_mean": float(allth.mean()), "D_median": float(np.median(allth)),
                        "D_attn": float(th["attn"].mean()), "D_mlp": float(th["mlp"].mean()), "D_head": float(th["head"][0]),
                        "D_max": float(allth.max()), "norm_dev": dn})
            print(f"  {s} {label:18s} N {n:5d} f {f_of[n]:.3f} D {pts[-1]['D_mean']:.4f} (attn {pts[-1]['D_attn']:.4f}, mlp {pts[-1]['D_mlp']:.4f},"
                  f" head {pts[-1]['D_head']:.4f}) norm dev {dn:.1e}", flush=True)
        missing = sorted(set(f_of) - {0} - {p["steps"] for p in pts})
        assert not missing, f"{s} {label}: f but no checkpoint at {missing}"
        arms.append({"label": label, "lr": lr, "points": pts})

    rows = [{k: np.array([p[k] for p in a["points"]]) for k in a["points"][0]} for a in arms]
    ps = np.arange(0.5, 2.51, 0.01)
    sp = np.array([spread_p(rows, p) for p in ps])
    p_fit = float(ps[int(sp.argmin())])
    spreads = {"steps": spread_p(rows, 0), "p=1": spread_p(rows, 1), f"p={p_fit:.2f}": float(sp.min()), "p=2": spread_p(rows, 2)}
    for key in ("D_mean", "D_median", "D_attn", "D_mlp", "D_head"):
        spreads["log " + key] = spread([(np.log(r[key]), np.log(r["f"])) for r in rows])

    growth = {}
    for key in ("D_mean", "D_median", "D_attn", "D_mlp"):
        per_arm = [float(np.polyfit(np.log(r["steps"]), np.log(r[key]), 1)[0]) for r in rows]
        X = np.concatenate([np.stack([np.ones_like(r["steps"], dtype=float), np.log(r["lr"]), np.log(r["steps"])], 1) for r in rows])
        y = np.concatenate([np.log(r[key]) for r in rows])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = float(np.sqrt(np.mean((X @ coef - y) ** 2)))
        growth[key] = {"alpha_per_arm": per_arm, "free_fit": {"log_c": float(coef[0]), "b": float(coef[1]), "alpha": float(coef[2]), "rms_log_resid": resid},
                       "implied_p_b_over_alpha": float(coef[1] / coef[2]), "one_over_alpha": float(1 / coef[2])}
    return {"branch_point": s, "p_fit": p_fit, "spreads": spreads, "growth": growth, "arms": arms}


def main():
    res = []
    for s in BRANCH_POINTS:
        r = analyse(s)
        res.append(r)
        print(f"{s}: p_fit {r['p_fit']:.2f}; spreads " + ", ".join(f"{k} {v:.1e}" for k, v in r["spreads"].items()))
        for k, g in r["growth"].items():
            ff = g["free_fit"]
            print(f"   {k}: alpha per arm {[round(a, 3) for a in g['alpha_per_arm']]}; free fit b {ff['b']:.3f} alpha {ff['alpha']:.3f}"
                  f" (rms {ff['rms_log_resid']:.3f}); b/alpha {g['implied_p_b_over_alpha']:.2f}, 1/alpha {g['one_over_alpha']:.2f}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    print(OUT)


if __name__ == "__main__":
    main()
