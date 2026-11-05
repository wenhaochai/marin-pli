"""Depth test, final read, 2026-11-05: rc, last step, missing logged steps, final train (mean of the last 50 logged steps) and final held-out loss of no mixing and learned mixing at 12, 24 and 48 layers, all seeds at 48, plus the one-path references (Pre-LN) at the three depths. Writes final_depth_1105.npz."""
import numpy as np, wandb
from pathlib import Path
api = wandb.Api(timeout=300)
P = "reself/marin-della/muonh-qwen3-130m-della4xh100"
RUNS = [("id12", "-pls1-sep-bbfrozen-ahc-hcid-fa4thd-vs-d12", 12), ("hc12", "-pls1-sep-bbfrozen-ahc-fa4thd-vs-d12", 12),
        ("id24", "-pls1-sep-bbfrozen-ahc-hcid-fa4thd-vs-d24", 24), ("hc24", "-pls1-sep-bbfrozen-ahc-fa4thd-vs-d24", 24),
        ("id48", "-pls1-sep-bbfrozen-ahc-hcid-fa4thd-vs-d48", 48), ("id48s1", "-pls1-sep-bbfrozen-ahc-hcid-fa4thd-vs-d48-s1", 48),
        ("hc48", "-pls1-sep-bbfrozen-ahc-fa4thd-vs-d48", 48), ("hc48old", "-pls1-sep-bbfrozen-ahc-vs-d48-g4", 48), ("hc48s1", "-pls1-sep-bbfrozen-ahc-vs-d48-s1", 48)]
save = {}
for name, tag, L in RUNS:
    try:
        r = api.run(P + tag)
    except Exception as e:
        print(name, tag, "NOT FOUND", str(e)[:100]); continue
    kt, ke = f"train/pls/L{L - 1}", f"eval/L{L - 1}/loss"
    tr = {x["_step"]: x[kt] for x in r.scan_history(keys=["_step", kt], page_size=2000)}
    ev = {x["_step"]: x[ke] for x in r.scan_history(keys=["_step", ke], page_size=2000)}
    st = np.array(sorted(tr)); y = np.array([tr[s] for s in st], float)
    es = np.array(sorted(ev)); ey = np.array([ev[s] for s in es], float)
    miss = sorted(set(range(int(st[0]), int(st[-1]) + 1)) - set(st.tolist()))
    save[name + "/step"], save[name + "/ce"], save[name + "/es"], save[name + "/ey"] = st, y, es, ey
    cfg = r.config
    print(f"{name} {tag} state {r.state} steps {st[0]}..{st[-1]} missing {miss[:8]} finite {np.isfinite(y).all()} max last 500 {y[-500:].max():.3f} final train {y[-50:].mean():.4f} final held-out {ey[-1]:.4f} evals {[(int(a), round(float(b), 3)) for a, b in zip(es, ey)]}", flush=True)
np.savez(Path(__file__).parent / "final_depth_1105.npz", **save)
print("-- one-path and other references, every run whose tag names the depth")
for tag, L in (("-d12", 12), ("-d24", 24), ("-pls1-sep-bbfrozen-apreln-vs-d12", 12), ("-pls1-sep-bbfrozen-apreln-vs-d24", 24), ("-pls1-sep-bbfrozen-apreln-vs-d48-g4", 48), ("-pls1-sep-bbfrozen-apreln-fa4thd-vs-d48", 48), ("-pls1-sep-bbfrozen-ahc-vs-d12", 12), ("-pls1-sep-bbfrozen-ahc-vs-d24", 24)):
    try:
        r = api.run(P + tag)
    except Exception as e:
        print(tag, "not found"); continue
    s = r.summary
    print(tag, r.state, "step", s.get("_step"), "eval last layer", s.get(f"eval/L{L - 1}/loss"), "eval/loss", s.get("eval/loss"), "train", s.get(f"train/pls/L{L - 1}"), "train/loss", s.get("train/loss"), flush=True)
