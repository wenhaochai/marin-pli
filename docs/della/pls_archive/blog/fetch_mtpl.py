"""Main-model held-out next-token loss of the MTP ladder (Figure 17, fig-mtpl): NTP and NTP + one DeepSeek-V3 MTP module
(weight 0.3) at 130m (the Figure 16 pair, job 15523669, whose detached probes do not touch the backbone) and at 300m, 520m
and 1_2b (pls_mtp_ladder_submit.sh, baseline training). y = eval/loss, levanter's held-out (Paloma) next-token loss of the
model's own head; the MTP loss is a training loss only, so eval/loss of an MTP run is its next-token loss (checked
2026-10-16: 300m 3.5837 NTP, 3.5839 MTP; the 4-GPU 300m baseline 3.5848). x = parameters of the NTP model without probe
heads or MTP module: W&B parameter_count of the ladder NTP runs (PLS_W=0 adds no parameters), and for 130m the same count
computed (embeddings and lm_head 2 * 128256 * 512, six layers 4h^2 + 3h*inter + 4h + 2*64, final norm 512 = 154154240;
the 300m formula gives 299669760, W&B's count). A run counts only when W&B says finished and its last step is
num_train_steps - 1. W&B reself/marin-della -> mtpl.json {size: {params, ntp, mtp, probes}} (a missing arm is absent).
Figure 18 (fig-mtpl-depth): the post-hoc probes of each finished model (PROBE_FROM runs, -probes<steps>at<step>): per arm
next (train/pls/L{k}, the next-token probe of layer k+1; the last layer is the model's own frozen head) and after
(train/pls2/L{k}, the probe for the token after next), each the mean of the last 50 logged steps, as everywhere on the
page. At 130m these reproduce the in-training probes of Figure 16 (MTP - NTP: layer 5 next +0.085 vs +0.080, layer 6
after -0.185 vs -0.183; checked 2026-10-16), so the probe budget (2000 steps at batch 128, 1B tokens) holds.
Usage: python fetch_mtpl.py. Exits 2 on a W&B error."""
import json, sys, wandb
from pathlib import Path
OUT = Path(__file__).resolve().parent / "mtpl.json"
P = "reself/marin-della/muonh-qwen3-"
RUNS = {"130m": ("130m-della4xh100-pls1-sep-bbfrozen-os-fut", "130m-della4xh100-pls1-sep-bbfrozen-os-mtp0.3-fut"),
        **{s: (f"{s}-della8xh100-pls0-noev", f"{s}-della8xh100-pls0-noev-mtp0.3") for s in ("300m", "520m", "1_2b")}}
def nl(h, inter): return 4 * h * h + 3 * h * inter + 4 * h + 2 * 64
P130 = 2 * 128256 * 512 + 6 * nl(512, 1792) + 512
assert 2 * 128256 * 768 + 12 * nl(768, 2688) + 768 == 299669760
api = wandb.Api(timeout=300)
out, errors = {}, 0
for size, (ntp, mtp) in RUNS.items():
    for arm, rid in (("ntp", ntp), ("mtp", mtp)):
        try:
            r = api.run(P + rid)
        except Exception as e:
            if "Could not find run" in str(e) or "404" in str(e):
                print(size, arm, "no run yet"); continue
            print(size, arm, "W&B ERROR", type(e).__name__, str(e)[:200]); errors += 1; continue
        n, last = int(r.config["trainer"]["num_train_steps"]), int(r.summary.get("_step", -1))
        if r.state != "finished" or last != n - 1 or r.summary.get("eval/loss") is None:
            print(size, arm, r.state, "step", last, "of", n, "skipped"); continue
        d = out.setdefault(size, {})
        d[arm] = round(float(r.summary["eval/loss"]), 5)
        if arm == "ntp":
            d["params"] = P130 if size == "130m" else int(r.summary["parameter_count"])
        print(size, arm, d[arm], "params", d.get("params"))
PROBES = {"130m": ("130m-della8xh100-pls1-sep-bbfrozen-os-fut-probes2000at4958", "130m-della8xh100-pls1-sep-bbfrozen-os-mtp0.3-fut-probes2000at4958", 6),
          "300m": ("300m-della8xh100-pls0-noev-probes2000at11443", "300m-della8xh100-pls0-noev-mtp0.3-probes2000at11443", 12),
          "520m": ("520m-della8xh100-pls0-noev-vs-probes1000at9917", "520m-della8xh100-pls0-noev-mtp0.3-vs-probes1000at9917", 24),   # PARAM_MAP=vocab rerun (15952439/40)
          "1_2b": ("1_2b-della8xh100-pls0-noev-probes1000at22887", "1_2b-della8xh100-pls0-noev-mtp0.3-probes1000at22887", 16)}
for size, (ntp, mtp, L) in PROBES.items():
    got = {}
    for arm, rid in (("ntp", ntp), ("mtp", mtp)):
        try:
            r = api.run(P + rid)
        except Exception as e:
            print(size, arm, "probes: no run yet" if "Could not find run" in str(e) or "404" in str(e) else f"probes: W&B ERROR {e}"[:200]); continue
        n, last = int(r.config["trainer"]["num_train_steps"]), int(r.summary.get("_step", -1))
        if r.state != "finished" or last != n - 1:
            print(size, arm, "probes", r.state, "step", last, "of", n, "skipped"); continue
        keys = [f"train/pls/L{k}" for k in range(L)] + [f"train/pls2/L{k}" for k in range(L)]
        h = list(r.scan_history(keys=["_step", *keys], min_step=max(0, n - 200), max_step=n, page_size=1000))
        m = {k: sum(v := [x[k] for x in h if x.get(k) is not None][-50:]) / len(v) for k in keys}
        nxt = [round(m[f"train/pls/L{k}"], 4) for k in range(L)]
        # The last next readout is the checkpoint's own frozen head, so it beats every probe below it. The 520m probe runs
        # fail this from step 2 (frozen head 9.8 nats against eval 3.34; checked 2026-10-21): their backbone path is broken.
        if nxt[-1] > min(nxt[:-1]):
            print(size, arm, "probes: frozen head", nxt[-1], "above the best probe", min(nxt[:-1]), "skipped"); continue
        got[arm] = {"next": nxt, "after": [round(m[f"train/pls2/L{k}"], 4) for k in range(L)]}
    if len(got) == 2 and size in out:
        out[size]["probes"] = got
        print(size, "probes MTP - NTP next", [round(a - b, 3) for a, b in zip(got["mtp"]["next"], got["ntp"]["next"])],
              "after", [round(a - b, 3) for a, b in zip(got["mtp"]["after"], got["ntp"]["after"])])
if errors:
    sys.exit(2)
out = {s: d for s, d in out.items() if "params" in d}
OUT.write_text(json.dumps(out, indent=1) + "\n")
print("wrote", OUT, out)
