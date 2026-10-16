"""Main-model held-out next-token loss of the MTP ladder (Figure 17, fig-mtpl): NTP and NTP + one DeepSeek-V3 MTP module
(weight 0.3) at 130m (the Figure 16 pair, job 15523669, whose detached probes do not touch the backbone) and at 300m, 520m
and 1_2b (pls_mtp_ladder_submit.sh, baseline training). y = eval/loss, levanter's held-out (Paloma) next-token loss of the
model's own head; the MTP loss is a training loss only, so eval/loss of an MTP run is its next-token loss (checked
2026-10-16: 300m 3.5837 NTP, 3.5839 MTP; the 4-GPU 300m baseline 3.5848). x = parameters of the NTP model without probe
heads or MTP module: W&B parameter_count of the ladder NTP runs (PLS_W=0 adds no parameters), and for 130m the same count
computed (embeddings and lm_head 2 * 128256 * 512, six layers 4h^2 + 3h*inter + 4h + 2*64, final norm 512 = 154154240;
the 300m formula gives 299669760, W&B's count). A run counts only when W&B says finished and its last step is
num_train_steps - 1. W&B reself/marin-della -> mtpl.json {size: {params, ntp, mtp}} (a missing arm is absent).
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
if errors:
    sys.exit(2)
out = {s: d for s, d in out.items() if "params" in d}
OUT.write_text(json.dumps(out, indent=1) + "\n")
print("wrote", OUT, out)
