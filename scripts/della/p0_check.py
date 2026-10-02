"""P0 training alerts (training-monitor skill) for the W&B runs listed in p0_runs (next to this file), one line per new alert.
Run it from a watcher loop every few minutes; it keeps its state in state/p0/ beside it.

Per running run, from the most recent logged points (W&B logs train metrics every 10 steps):
  NONFINITE  train/loss or grad/norm/total is NaN/inf
  LOSS-SPIKE robust spike score z(log loss) > 8 over the trailing 100 points
  CO-SPIKE   z(log loss) > 6 and z(log grad norm) > 6 at the same step
  CLIP       share of the last 300 points with grad/norm/total > max_grad_norm above 5%
  SLOW       median throughput/duration of the last 30 points > 1.25x the run's own median
Thresholds from six healthy runs (2026-10-02; 130m-1.2B, OV, OV+ss, baseline, ss): z(log loss) max 4.5, no step with
both z > 6, clip rate <= 0.5%, step-time p99 <= 1.28x median (checkpoint stalls). State per run in state/p0/.
"""
import json
import math
import os
import re
import sys
import urllib.request
from pathlib import Path
from statistics import median

D = Path(__file__).resolve().parent
STATE = D / "state" / "p0"; STATE.mkdir(parents=True, exist_ok=True)
W, Z_LOSS, Z_CO, CLIP_RATE, SLOW = 100, 8.0, 6.0, 0.05, 1.25
KEYS = ["_step", "train/loss", "grad/norm/total", "throughput/duration"]


def auth():
    netrc = (Path.home() / ".netrc").read_text()
    key = re.search(r"machine\s+api\.wandb\.ai\s+login\s+\S+\s+password\s+(\S+)", netrc).group(1)
    import base64
    return "Basic " + base64.b64encode(f"api:{key}".encode()).decode()


def query(run_path, min_step, with_config):
    entity, project, name = run_path.split("/")
    spec = json.dumps({"keys": KEYS, "samples": 2000, "minStep": max(0, min_step)})
    q = ("query($e:String!,$p:String!,$n:String!,$s:[JSONString!]!){project(name:$p,entityName:$e){run(name:$n){state "
         + ("config " if with_config else "") + "sampledHistory(specs:$s)}}}")
    body = json.dumps({"query": q, "variables": {"e": entity, "p": project, "n": name, "s": [spec]}}).encode()
    req = urllib.request.Request("https://api.wandb.ai/graphql", body, {"Content-Type": "application/json", "Authorization": AUTH})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())["data"]["project"]["run"]


def zscore(u, t):
    w = u[t - W:t]; q = median(w); d = median(abs(x - q) for x in w)
    return 0.6745 * abs(u[t] - q) / (d + 1e-12)


AUTH = auth()
for run_path in [l.split()[0] for l in (D / "p0_runs").read_text().splitlines() if l.strip() and not l.startswith("#")]:
    name = run_path.split("/")[-1]
    sf = STATE / f"{name}.json"
    st = json.loads(sf.read_text()) if sf.exists() else {"last": -1, "alerted": []}
    try:
        first = "clip" not in st
        run = query(run_path, 0 if first else st["last"] - 10 * (W + 300), first)
        if run is None or run.get("state") != "running":
            continue
        if first:
            cfg = json.loads(run["config"]) if isinstance(run["config"], str) else run["config"]
            st["clip"] = (cfg.get("optimizer", {}).get("value") or {}).get("max_grad_norm")
        rows = sorted((r for r in run["sampledHistory"][0] if all(k in r for k in KEYS)), key=lambda r: r["_step"])
        if not rows:
            continue
        if first:
            st["med_ms"] = median(r["throughput/duration"] for r in rows) * 1e3
        steps = [r["_step"] for r in rows]
        loss = [r["train/loss"] for r in rows]; gn = [r["grad/norm/total"] for r in rows]
        new = [i for i, s in enumerate(steps) if s > st["last"]]
        alerts = []
        for i in new:
            if not (math.isfinite(loss[i]) and math.isfinite(gn[i])):
                alerts.append(("NONFINITE", steps[i], f"loss {loss[i]} grad norm {gn[i]}"))
        ul = [math.log(max(x, 1e-12)) if math.isfinite(x) else float("nan") for x in loss]
        ug = [math.log(max(x, 1e-12)) if math.isfinite(x) else float("nan") for x in gn]
        for i in new:
            if i >= W and steps[i] >= 200 and all(math.isfinite(x) for x in ul[i - W:i + 1] + ug[i - W:i + 1]):
                zl, zg = zscore(ul, i), zscore(ug, i)
                if zl > Z_LOSS:
                    alerts.append(("LOSS-SPIKE", steps[i], f"z(loss) {zl:.1f}, loss {loss[i]:.3f}"))
                if zl > Z_CO and zg > Z_CO:
                    alerts.append(("CO-SPIKE", steps[i], f"z(loss) {zl:.1f}, z(grad norm) {zg:.1f}"))
        if st.get("clip") and len(gn) >= 300:
            rate = sum(g > st["clip"] for g in gn[-300:]) / 300
            if rate > CLIP_RATE:
                alerts.append(("CLIP", steps[-1] // 1000 * 1000, f"{rate:.0%} of the last 300 logged steps above max_grad_norm {st['clip']}"))
        if len(rows) >= 30 and st.get("med_ms"):
            recent = median(r["throughput/duration"] for r in rows[-30:]) * 1e3
            if recent > SLOW * st["med_ms"]:
                alerts.append(("SLOW", steps[-1] // 1000 * 1000, f"median step {recent:.0f} ms vs run median {st['med_ms']:.0f} ms"))
        for kind, step, msg in alerts:
            tag = f"{kind}@{step}"
            if tag not in st["alerted"]:
                print(f"P0 {name}: {kind} at step {step}: {msg}", flush=True)
                st["alerted"].append(tag)
        st["last"] = steps[-1]
    except Exception as e:  # a W&B hiccup must not kill the watcher; say so once per kind
        tag = f"ERR:{type(e).__name__}"
        if tag not in st["alerted"]:
            print(f"P0 {name}: check failed ({type(e).__name__}: {str(e)[:120]})", flush=True)
            st["alerted"].append(tag)
    sf.write_text(json.dumps(st))
