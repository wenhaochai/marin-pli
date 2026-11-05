"""Depth test, live read, 2026-11-05: last-layer train CE and held-out loss of the runs without mixing and with learned mixing at 12, 24 and 48 layers. Series go to live_depth_1105.npz for the status figure. W&B state may read "crashed" for a live run.
Usage: live_depth_1105.py"""
import numpy as np, wandb
from pathlib import Path
api = wandb.Api(timeout=300)
P = "reself/marin-della/muonh-qwen3-130m-della4xh100-pls1"
RUNS = [("id12", "-sep-bbfrozen-ahc-hcid-fa4thd-vs-d12", 12), ("hc12", "-sep-bbfrozen-ahc-fa4thd-vs-d12", 12),
        ("id24", "-sep-bbfrozen-ahc-hcid-fa4thd-vs-d24", 24), ("hc24", "-sep-bbfrozen-ahc-fa4thd-vs-d24", 24),
        ("id48", "-sep-bbfrozen-ahc-hcid-fa4thd-vs-d48", 48), ("hc48", "-sep-bbfrozen-ahc-fa4thd-vs-d48", 48)]
save = {}
for name, tag, L in RUNS:
    r = api.run(P + tag)
    kt, ke = f"train/pls/L{L - 1}", f"eval/L{L - 1}/loss"
    tr = {x["_step"]: x[kt] for x in r.scan_history(keys=["_step", kt], page_size=2000)}
    ev = {x["_step"]: x[ke] for x in r.scan_history(keys=["_step", ke], page_size=2000)}
    st = np.array(sorted(tr)); y = np.array([tr[s] for s in st], float)
    es = np.array(sorted(ev)); ey = np.array([ev[s] for s in es], float)
    save[name + "/step"], save[name + "/ce"], save[name + "/es"], save[name + "/ey"] = st, y, es, ey
    last = y[-50:].mean()
    print(f"{name} {tag} state {r.state} last step {st[-1]} rows {len(st)} last-50 train {last:.4f}; eval steps {es.tolist()} values {[round(float(v), 3) for v in ey]}", flush=True)
np.savez(Path(__file__).parent / "live_depth_1105.npz", **save)
for L in (12, 24, 48):
    a, b = save[f"id{L}/step"], save[f"hc{L}/step"]
    ya, yb = save[f"id{L}/ce"], save[f"hc{L}/ce"]
    print(f"-- D({L}) train, no mixing minus learned mixing, 500-step bins")
    for lo in range(0, 5000, 500):
        hi = lo + 500
        ma, mb = (a >= lo) & (a < hi), (b >= lo) & (b < hi)
        if ma.sum() > 0.8 * 500 and mb.sum() > 0.8 * 500:
            print(f"  {lo:4d}-{hi - 1:<4d} no mixing {ya[ma].mean():.3f} learned {yb[mb].mean():.3f} D {ya[ma].mean() - yb[mb].mean():+.3f}")
