"""Training-monitor health card for marin-della runs (Deng 2026 via the training-monitor skill), from W&B histories only.

    .venv/bin/python health_card.py <calibration run ids...> -- <run ids...>

Derivable P0 signals: clip rate (grad/norm/total > max_grad_norm), robust spike z for loss and grad norm, co-spike
count, throughput stability (p10/p50/p90 of throughput/mfu, stall fraction). Spike cutoff is calibrated on the
calibration runs (healthy baselines): cutoff = their 99.9th percentile of z. Full-resolution histories via scan_history.
"""
import sys

import numpy as np
import wandb

MAX_GRAD_NORM = 1.0
W = 200  # robust-z window (steps)
api = wandb.Api(timeout=120)
entity = api.default_entity


def history(rid):
    r = api.run(f"{entity}/marin-della/{rid}")
    rows = [x for x in r.scan_history(keys=["_step", "train/loss", "grad/norm/total"], page_size=2000)]
    rows = sorted((x["_step"], x["train/loss"], x["grad/norm/total"]) for x in rows if x.get("train/loss") is not None and x.get("grad/norm/total") is not None)
    st = np.array([a for a, _, _ in rows]); loss = np.array([b for _, b, _ in rows]); gn = np.array([c for _, _, c in rows])
    thr = [x for x in r.scan_history(keys=["_step", "throughput/mfu", "throughput/tokens_per_second"], page_size=2000)]
    mfu = np.array([x["throughput/mfu"] for x in thr if x.get("throughput/mfu") is not None])
    tps = np.array([x["throughput/tokens_per_second"] for x in thr if x.get("throughput/tokens_per_second") is not None])
    return st, loss, gn, mfu, tps


def robust_z(x, w=W, eps=1e-8):
    u = np.log(x + eps); z = np.zeros_like(u)
    for t in range(len(u)):
        win = u[max(0, t - w):t] if t > 0 else u[:1]
        q = np.median(win); d = np.median(np.abs(win - q))
        z[t] = 0.6745 * abs(u[t] - q) / (d + eps)
    return z


def card(rid, cutoff=None):
    st, loss, gn, mfu, tps = history(rid)
    zl, zg = robust_z(loss), robust_z(gn)
    out = dict(rid=rid, n=len(st), clip_rate=float((gn > MAX_GRAD_NORM).mean()), gn_mean=float(gn.mean()), gn_p99=float(np.percentile(gn, 99)), gn_max=float(gn.max()),
               loss_final=float(loss[-50:].mean()), zl=zl, zg=zg,
               mfu_p10=float(np.percentile(mfu, 10)) if len(mfu) else float("nan"), mfu_p50=float(np.percentile(mfu, 50)) if len(mfu) else float("nan"), mfu_p90=float(np.percentile(mfu, 90)) if len(mfu) else float("nan"),
               stall_frac=float((tps < 0.5 * np.median(tps)).mean()) if len(tps) else float("nan"))
    if cutoff is not None:
        co = (zl > cutoff) & (zg > cutoff)
        out["cospike"] = int(co.sum()); out["cospike_steps"] = st[co][:10].tolist()
        out["loss_spikes"] = int((zl > cutoff).sum()); out["gn_spikes"] = int((zg > cutoff).sum())
    return out


args = sys.argv[1:]
sep = args.index("--")
calib, runs = args[:sep], args[sep + 1:]
zs = []
cards = {}
for rid in calib:
    c = card(rid); cards[rid] = c; zs.append(np.concatenate([c["zl"][W:], c["zg"][W:]]))
cutoff = float(np.percentile(np.concatenate(zs), 99.9))
print(f"calibration on {len(calib)} healthy runs: robust-z cutoff (99.9th pct) = {cutoff:.2f}; window {W} steps; clip threshold {MAX_GRAD_NORM}")
print(f"{'run':60s} {'steps':>5s} {'clip%':>6s} {'gn_mean':>7s} {'gn_p99':>6s} {'gn_max':>6s} {'co-spk':>6s} {'L-spk':>5s} {'G-spk':>5s} {'mfu p10/50/90':>16s} {'stall%':>6s}")
for rid in calib + runs:
    c = card(rid, cutoff)
    print(f"{rid:60s} {c['n']:5d} {100*c['clip_rate']:6.2f} {c['gn_mean']:7.3f} {c['gn_p99']:6.3f} {c['gn_max']:6.2f} {c['cospike']:6d} {c['loss_spikes']:5d} {c['gn_spikes']:5d} {c['mfu_p10']:5.2f}/{c['mfu_p50']:5.2f}/{c['mfu_p90']:5.2f} {100*c['stall_frac']:6.2f}"
          + (f"  co-spike steps {c['cospike_steps']}" if c["cospike"] else ""))
