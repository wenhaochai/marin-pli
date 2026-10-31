"""Is forgetting after a branch set by the part of the later movement that undoes the pre-branch update?

Setup, data and the fraction left f as in tools/branch_drift.py (d12-ck248 branched at S = 1240, 2480, 3720 into three
constant-lr arms with lr x N = 5). branch_drift found that the angle D between W_s and W_S does not line up the arms
(spread 5.7e-3 / 7.4e-3 / 2.7e-2) while steps * lr^p does (9.6e-4 / 7.0e-4 / 1.1e-3, p = 1.22 / 1.32 / 1.40).

Hypothesis. The batches read at steps S-248..S-1 are forgotten as far as the later movement undoes what they did. Per
matrix (the same 85 fixed-norm matrices: 12 layers x {q, k, v, o, gate, up, down} plus lm_head):
    u   = W_S - W_{S-248}, minus its component along W_S (tangent space of the sphere at W_S)
    d_s = W_s - W_S
    r_s = -<d_s, u> / <u, u>        (fraction of the pre-S update undone; 1 = fully undone)
    o_s = |d_s - (<d_s, u>/<u, u>) u| / |d_s|      (orthogonal remainder fraction)
R_mean = mean of r over the matrices, R_pool = -sum<d, u> / sum<u, u>; both also per group (attn, mlp, head). O_mean is
the mean of o, O_pool = sqrt(sum |d_perp|^2 / sum |d|^2).

Questions per S. (1) Spread of log f across the arms against x = R_pool, R_mean (spread() of branch_drift: 50 points
on the common range of x); x = log R when every R > 0, and x = R in every case (marked). (2) f against R: no-parameter
f = 1 - R, free line f = a + c R, f = exp(-c R) and log f = a - c R; RMS residuals and the per-arm mean residual.
(3) Free fit log R = c + b log lr + a log N over the arms; b/a is the exponent q for which R collapses against N lr^q.

f for every arm is recomputed from the TRAIN_RANGE scorings with build_site_data._range_score (the formula of
build_site_data.branch_curves, checked against it on the three base arms). EXTRA_ARMS = comma-separated run-id
prefixes (e.g. muonh-qwen3-130m-della4xh100-d12-constant-ck124-lrm0.5-mom0.98-br2480) adds arms; lr = 0.02 x the lrm<M>
of the name, S from -br<S>. A checkpoint without a scoring, or an arm without checkpoints, is skipped. Extra arms are
reported per arm and in "with_extra" spreads; the main spreads and fits use the three base arms only.

Output: out/ck/proj/proj.json and out/ck/proj/SUMMARY.txt (SUMMARY.txt's closing paragraph is written by hand after
the run and kept on re-runs below the marker line).

    JAX_PLATFORMS=cpu nice -n 19 <marin venv>/bin/python tools/branch_proj.py      # on a vis node
"""
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import branch_drift as bd  # noqa: E402
bsd = bd.bsd

OUTDIR = "/scratch/gpfs/GROUP/USER/project/trec/out/ck/proj"
DRIFT = "/scratch/gpfs/GROUP/USER/project/trec/out/ck/drift/drift.json"
SCORES = os.path.join(bsd.OUT, "ck", "branch")
GROUPS = (("attn", bd.ATTN), ("mlp", bd.MLP), ("head", bd.HEAD))
MARK = "--- notes (hand-written, kept on re-run) ---"


def matrices(ckpt):
    """{group: [array [n, d]]} raw weights in float64; a stacked array is a stack of matrices along axis 0."""
    out = {}
    for g, paths in GROUPS:
        ws = []
        for p in paths:
            w = bd.read(ckpt, p)
            ws.append(w.reshape(w.shape[0], -1) if g != "head" else w.reshape(1, -1))
        out[g] = ws
    return out


def tangent_update(w_pre, w_at):
    """{group: (u [n, d] list, W_S norms [n])}: u = W_S - W_{S-248} without its component along W_S."""
    out = {}
    for g in w_at:
        us, ns = [], []
        for a, b in zip(w_pre[g], w_at[g]):
            nb = np.linalg.norm(b, axis=1, keepdims=True)
            e = b / nb
            u = b - a
            us.append(u - np.sum(u * e, axis=1, keepdims=True) * e)
            ns.append(nb[:, 0])
        out[g] = (us, np.concatenate(ns))
    return out


def project(w_at, tu, w_s):
    """Per-matrix <d,u>, <u,u>, |d|^2, |d_perp|^2 per group, and the worst |W_s|/|W_S| - 1."""
    res, dn = {}, 0.0
    for g in w_at:
        du, uu, dd, nr = [], [], [], []
        for a, u, b in zip(w_at[g], tu[g][0], w_s[g]):
            d = b - a
            du.append(np.sum(d * u, axis=1))
            uu.append(np.sum(u * u, axis=1))
            dd.append(np.sum(d * d, axis=1))
            nr.append(np.linalg.norm(b, axis=1) / np.linalg.norm(a, axis=1) - 1)
        du, uu, dd = map(np.concatenate, (du, uu, dd))
        res[g] = {"du": du, "uu": uu, "dd": dd, "pp": np.maximum(dd - du ** 2 / uu, 0.0)}
        dn = max(dn, float(np.max(np.abs(np.concatenate(nr)))))
    return res, dn


def summarise(pr):
    """R_mean, R_pool, O_mean, O_pool overall and per group from project()."""
    out = {}
    for name, gs in (("", ("attn", "mlp", "head")), ("_attn", ("attn",)), ("_mlp", ("mlp",)), ("_head", ("head",))):
        du, uu, dd, pp = (np.concatenate([pr[g][k] for g in gs]) for k in ("du", "uu", "dd", "pp"))
        r = -du / uu
        out["R_mean" + name] = float(r.mean())
        out["R_pool" + name] = float(-du.sum() / uu.sum())
        out["O_mean" + name] = float(np.mean(np.sqrt(pp / dd)))
        out["O_pool" + name] = float(np.sqrt(pp.sum() / dd.sum()))
        if name == "":
            out["R_std"] = float(r.std())
            out["R_min"], out["R_max"] = float(r.min()), float(r.max())
    return out


def f_curve(run, s):
    """{steps since s: f} of one arm from its TRAIN_RANGE scorings (formula of build_site_data.branch_curves)."""
    pre = bsd._range_score(os.path.join(SCORES, "base", f"step-{s - 248}"), s)
    at = bsd._range_score(os.path.join(SCORES, "base", f"step-{s}"), s)
    assert pre and at, f"base scorings for {s} missing"
    e0 = ((at[0] - at[1]) - (pre[0] - pre[1])).mean()
    out = {}
    for d in glob.glob(os.path.join(SCORES, run, "step-*")):
        r = bsd._range_score(d, s)
        if r is not None:
            out[int(d.rsplit("-", 1)[1]) - s] = float(((r[0] - r[1]) - (pre[0] - pre[1])).mean() / e0)
    return out


def ckpt_dirs(run):
    ds = glob.glob(f"{bd.STORE}/{run}/*/checkpoints/step-*")
    return {int(d.rsplit("-", 1)[1]): d for d in ds}


def arm_list(s):
    """[(run, label, lr, extra)] for branch point s: the three base arms, then EXTRA_ARMS matching -br<s>."""
    arms = [(r.replace("-br2480", f"-br{s}"), lab, lr, False) for r, lab, lr, _ in bsd.BRANCH]
    for pre in filter(None, (p.strip() for p in os.environ.get("EXTRA_ARMS", "").split(","))):
        for d in sorted(glob.glob(f"{bd.STORE}/{pre}*")):
            run = os.path.basename(d)
            mb, ml = re.search(r"-br(\d+)", run), re.search(r"lrm([0-9.]+?)(?:-|$)", run)
            if not (mb and ml) or int(mb.group(1)) != s or run.endswith(".err") or run in {a[0] for a in arms}:
                continue
            lr = 0.02 * float(ml.group(1))
            tag = run.split(f"lrm{ml.group(1)}", 1)[1].replace(f"-br{s}", "").strip("-")
            arms.append((run, f"lr {lr:g} {tag}".strip(), lr, True))
    return arms


def cols(points):
    return {k: np.array([p[k] for p in points]) for k in points[0]}


def spread_x(rows, key, log):
    """branch_drift.spread of log f against x = R (or log R); None if log is asked and some R <= 0. Sorted by x;
    'monotone' says whether every arm's x already rose with N."""
    if log and any(np.any(r[key] <= 0) for r in rows):
        return None, None
    cs, mono = [], True
    for r in rows:
        x = np.log(r[key]) if log else r[key]
        mono &= bool(np.all(np.diff(x) > 0))
        o = np.argsort(x)
        cs.append((x[o], np.log(r["f"][o])))
    return bd.spread(cs), mono


def relation(rows, key):
    """f against R pooled over arms: f = 1 - R, f = a + c R, f = exp(-c R), log f = a - c R."""
    R = np.concatenate([r[key] for r in rows])
    f = np.concatenate([r["f"] for r in rows])
    arm = np.concatenate([np.full(len(r["f"]), i) for i, r in enumerate(rows)])

    def rep(pred, name, params):
        res = f - pred
        return {"form": name, "params": params, "rms_f": float(np.sqrt(np.mean(res ** 2))),
                "rms_logf": float(np.sqrt(np.mean((np.log(f) - np.log(np.maximum(pred, 1e-9))) ** 2))) if np.all(pred > 0) else None,
                "mean_resid_per_arm": [float(res[arm == i].mean()) for i in range(len(rows))]}

    out = [rep(1 - R, "f = 1 - R", {})]
    c1, c0 = np.polyfit(R, f, 1)
    out.append(rep(c0 + c1 * R, "f = a + c R", {"a": float(c0), "c": float(c1)}))
    c = float(-np.sum(R * np.log(f)) / np.sum(R * R))
    out.append(rep(np.exp(-c * R), "f = exp(-c R)", {"c": c}))
    k1, k0 = np.polyfit(R, np.log(f), 1)
    out.append(rep(np.exp(k0 + k1 * R), "log f = a - c R", {"a": float(k0), "c": float(-k1)}))
    return out


def growth(rows, key):
    if any(np.any(r[key] <= 0) for r in rows):
        return None
    X = np.concatenate([np.stack([np.ones(len(r["steps"])), np.log(r["lr"]), np.log(r["steps"])], 1) for r in rows])
    y = np.concatenate([np.log(r[key]) for r in rows])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return {"log_c": float(coef[0]), "b": float(coef[1]), "a": float(coef[2]), "rms_log_resid": float(np.sqrt(np.mean((X @ coef - y) ** 2))),
            "q_b_over_a": float(coef[1] / coef[2]),
            "a_per_arm": [float(np.polyfit(np.log(r["steps"]), np.log(r[key]), 1)[0]) for r in rows]}


def analyse(s, drift):
    w_pre, w_at = matrices(f"{bd.BASE}/step-{s - 248}"), matrices(f"{bd.BASE}/step-{s}")
    tu = tangent_update(w_pre, w_at)
    uu_all = np.concatenate([np.concatenate([np.sum(u * u, axis=1) for u in tu[g][0]]) for g in tu])
    u_rel = np.sqrt(uu_all) / np.concatenate([tu[g][1] for g in tu])
    print(f"{s}: |u|/|W_S| mean {u_rel.mean():.4f} (min {u_rel.min():.4f}, max {u_rel.max():.4f})", flush=True)

    curves = bsd.branch_curves(s)
    assert curves is not None, f"branch_curves({s}) not complete"
    ref = {lr: {int(a): float(b) for a, b in c} for _, lr, _, c in curves}

    arms = []
    for run, label, lr, extra in arm_list(s):
        cks, f_of = ckpt_dirs(run), f_curve(run, s)
        if not extra:
            for n, v in ref[lr].items():
                if n:
                    assert abs(f_of[n] - v) < 1e-9, (run, n, f_of[n], v)
        steps = sorted(st for st in cks if st - s in f_of and st > s)
        skipped = sorted(set(cks) - set(steps))
        if not steps:
            print(f"  skip {run}: no checkpoint with a scoring", flush=True)
            continue
        pts = []
        for st in steps:
            pr, dn = project(w_at, tu, matrices(cks[st]))
            assert dn < bd.NORM_TOL, f"{cks[st]}: norm changed by {dn:.2e}"
            p = {"steps": st - s, "lr": lr, "f": f_of[st - s], "norm_dev": dn, **summarise(pr)}
            pts.append(p)
            print(f"  {s} {label:22s} N {p['steps']:5d} f {p['f']:.3f} R_pool {p['R_pool']:.4f} R_mean {p['R_mean']:.4f}"
                  f" (attn {p['R_pool_attn']:.3f} mlp {p['R_pool_mlp']:.3f} head {p['R_pool_head']:.3f}) O_pool {p['O_pool']:.4f} norm dev {dn:.1e}", flush=True)
        arms.append({"run": run, "label": label, "lr": lr, "extra": extra, "skipped_steps": skipped, "points": pts})

    base = [cols(a["points"]) for a in arms if not a["extra"]]
    allr = [cols(a["points"]) for a in arms]
    keys = ("R_pool", "R_mean", "R_pool_attn", "R_pool_mlp", "R_pool_head", "R_mean_attn", "R_mean_mlp")
    spreads = {}
    for k in keys:
        for log in (False, True):
            v, mono = spread_x(base, k, log)
            spreads[("log " if log else "") + k] = None if v is None else {"spread": v, "monotone": mono}
    d = next(r for r in drift if r["branch_point"] == s)
    p_fit = d["p_fit"]
    ref_spreads = {"log D_mean (branch_drift)": d["spreads"]["log D_mean"], f"steps*lr^p, p={p_fit:.2f}": bd.spread_p(base, p_fit),
                   "steps*lr (p=1)": bd.spread_p(base, 1), "steps": bd.spread_p(base, 0)}
    with_extra = None
    if len(allr) > len(base):
        with_extra = {k: spread_x(allr, k, False)[0] for k in ("R_pool", "R_mean")}
        with_extra.update({f"steps*lr^p, p={p_fit:.2f}": bd.spread_p(allr, p_fit), "steps*lr (p=1)": bd.spread_p(allr, 1)})
    return {"branch_point": s, "p_fit": p_fit, "u_over_W": {"mean": float(u_rel.mean()), "min": float(u_rel.min()), "max": float(u_rel.max())},
            "spreads": spreads, "reference_spreads": ref_spreads, "spreads_with_extra": with_extra,
            "relation": {k: relation(base, k) for k in ("R_pool", "R_mean")},
            "growth": {k: growth(base, k) for k in ("R_pool", "R_mean", "R_pool_attn", "R_pool_mlp")},
            "max_norm_dev": max(p["norm_dev"] for a in arms for p in a["points"]), "arms": arms}


def fmt(v):
    return "   n/a  " if v is None else f"{v:8.1e}"


def summary(res, all_rel):
    L = ["branch_proj.py: is forgetting after a branch set by the part of the later movement that undoes the pre-branch update?",
         "r = -<W_s - W_S, u>/<u,u>, u = tangential part of W_S - W_{S-248}; R_pool = pooled over 85 matrices, R_mean = mean of r.",
         "", "1. Spread of log f across the 3 base arms (lower = better collapse). x = R is linear; 'log R' only where every R > 0.",
         f"{'x':28s}" + "".join(f"{'S=' + str(r['branch_point']):>10s}" for r in res)]
    rows = [(k, [r["reference_spreads"][k] if k in r["reference_spreads"] else None for r in res]) for k in ("steps", "steps*lr (p=1)")]
    rows.append(("steps*lr^p (fitted p)", [list(r["reference_spreads"].values())[1] for r in res]))
    rows.append(("log D_mean (branch_drift)", [r["reference_spreads"]["log D_mean (branch_drift)"] for r in res]))
    for k in ("R_pool", "log R_pool", "R_mean", "log R_mean", "R_pool_attn", "R_pool_mlp", "R_pool_head"):
        rows.append((k, [r["spreads"][k]["spread"] if r["spreads"][k] else None for r in res]))
    for k, vs in rows:
        L.append(f"{k:28s}" + "".join(f"  {fmt(v)}" for v in vs))
    L.append("fitted p: " + ", ".join(f"{r['p_fit']:.2f}" for r in res) + "; R monotone in N in every arm: "
             + ", ".join(str(r["spreads"]["R_pool"]["monotone"]) for r in res))
    if any(r["spreads_with_extra"] for r in res):
        L.append("with EXTRA_ARMS: " + "; ".join(f"S={r['branch_point']}: " + ", ".join(f"{k} {v:.1e}" for k, v in r["spreads_with_extra"].items())
                                                for r in res if r["spreads_with_extra"]))
    L += ["", "2. f against R_pool, pooled over arms (RMS residual in f; mean residual per arm lr 0.01 / 0.005 / 0.0025)."]
    for r in res + [{"branch_point": "all", "relation": {"R_pool": all_rel}}]:
        for fit in r["relation"]["R_pool"]:
            par = " ".join(f"{k} {v:.3f}" for k, v in fit["params"].items())
            L.append(f"S={str(r['branch_point']):5s} {fit['form']:16s} {par:18s} rms {fit['rms_f']:.3f}  arm means "
                     + " ".join(f"{v:+.3f}" for v in fit["mean_resid_per_arm"]))
    L += ["", "3. Free fit log R = c + b log lr + a log N over the 3 base arms; q = b/a (R collapses against N lr^q); p from f."]
    for r in res:
        for k in ("R_pool", "R_mean"):
            g = r["growth"][k]
            L.append(f"S={r['branch_point']} {k:7s} " + ("R <= 0 somewhere: no fit" if g is None else
                     f"b {g['b']:.3f} a {g['a']:.3f} q {g['q_b_over_a']:.2f} (p {r['p_fit']:.2f}) rms {g['rms_log_resid']:.3f}; a per arm "
                     + " ".join(f"{v:.3f}" for v in g["a_per_arm"])))
    L += ["", "Range of R_pool / O_pool (orthogonal remainder |d_perp|/|d|) / R per group at the last checkpoint of each arm:"]
    for r in res:
        for a in r["arms"]:
            p = a["points"][-1]
            L.append(f"S={r['branch_point']} {a['label']:22s} N {p['steps']:5d} f {p['f']:.3f} R_pool {p['R_pool']:.3f} R_mean {p['R_mean']:.3f}"
                     f" (attn {p['R_pool_attn']:.3f} mlp {p['R_pool_mlp']:.3f} head {p['R_pool_head']:.3f}) O_pool {p['O_pool']:.4f}")
        L.append(f"S={r['branch_point']} |u|/|W_S| mean {r['u_over_W']['mean']:.4f}; max | |W_s|/|W_S| - 1 | {r['max_norm_dev']:.1e}")
    return "\n".join(L) + "\n"


def main():
    with open(DRIFT) as f:
        drift = json.load(f)
    res = [analyse(s, drift) for s in bd.BRANCH_POINTS]
    base_all = [cols(a["points"]) for r in res for a in r["arms"] if not a["extra"]]
    all_rel = relation(base_all, "R_pool")
    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, "proj.json"), "w") as f:
        json.dump({"per_branch_point": res, "relation_all_S": {"R_pool": all_rel, "R_mean": relation(base_all, "R_mean")}}, f, indent=1)
    txt = summary(res, all_rel)
    path = os.path.join(OUTDIR, "SUMMARY.txt")
    notes = ""
    if os.path.exists(path):
        old = open(path).read()
        if MARK in old:
            notes = "\n" + old[old.index(MARK):]
    with open(path, "w") as f:
        f.write(txt + notes)
    print(txt)


if __name__ == "__main__":
    main()
