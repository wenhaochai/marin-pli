"""Experiment record of the website page exps/trec.html, written to <site>/assets/data/trec-record.js as
window.TREC_RECORD: one row per scored checkpoint, and the page's GPU cost.

Rows: the training run each final checkpoint comes from (W&B reself/marin-della) and the wq job that scored it. A row's
state, hours and time segments come from that job's wq record (/scratch/gpfs/GROUP/USER/wq/jobs/<n>/state.json);
when one job scores several checkpoints, its hours are split by the scoring seconds each checkpoint's chunks recorded.
Cost: H100-hours = GPUs x the wall time of every run of every wq job of the page; main = the jobs of the rows, other =
the rest (smoke tests). A job with no record fails the build rather than being guessed.

    python experiments/trec/build_record.py <site checkout>     # vis node
"""

import datetime
import glob
import json
import os
import sys
import zoneinfo

import numpy as np

WQ = "/scratch/gpfs/GROUP/USER/wq/jobs"
OUT = "/scratch/gpfs/GROUP/USER/project/trec/out"
NY = zoneinfo.ZoneInfo("America/New_York")
ROWS = [
    dict(model="130m-d12", run="muonh-qwen3-130m-della4xh100-d12", setup={"en": "WSD to zero, lr 0.02", "zh": "WSD 降到零，lr 0.02"},
         size="130m", layers=12, steps=4959, batches=4959 + 512, jobs=["w465"], figs="1, 2, 3, 4"),
    dict(model="300m", run="muonh-qwen3-300m-della4xh100-s1-rerun", setup={"en": "Cosine to zero, lr 0.01", "zh": "cosine 降到零，lr 0.01"},
         size="300m", layers=12, steps=11444, batches=11444 + 512, jobs=["w465"], figs="1, 2, 3, 4, 5"),
] + [   # scored by the vocab session's wq w438 and w457 for its page (192 training blocks and 16 unread ones); their cost is on that page
    dict(model=None, run=f"muonh-qwen3-300m-della4xh100{tag}", setup={"en": f"{en}, seed {seed}", "zh": f"{zh}，种子 {seed}"}, size="300m", layers=12,
         steps=steps, batches=208, jobs=[], figs="5", external=True)
    for tag, en, zh, seed, steps in [
        ("-rep8-s1", "Marin tokenizer, 8 passes over an eighth", "Marin tokenizer，一个八分之一子集读 8 遍", 1, 11444),
        ("-v8k-s1", "8K truncation, unique tokens", "8K 截短版，不重复 token", 1, 15464),
        ("-v8k-rep8-s1", "8K truncation, 8 passes over an eighth", "8K 截短版，一个八分之一子集读 8 遍", 1, 15464),
        ("-v8k", "8K truncation, unique tokens", "8K 截短版，不重复 token", 0, 15464),
        ("-v8k-rep8", "8K truncation, 8 passes over an eighth", "8K 截短版，一个八分之一子集读 8 遍", 0, 15464),
        ("-bytes", "Bytes, unique tokens", "字节，不重复 token", 0, 54248),
        ("-bytes-rep8", "Bytes, 8 passes over an eighth", "字节，一个八分之一子集读 8 遍", 0, 54248),
        ("-bytes-s1", "Bytes, unique tokens", "字节，不重复 token", 1, 54248),
        ("-bytes-rep8-s1", "Bytes, 8 passes over an eighth", "字节，一个八分之一子集读 8 遍", 1, 54248)]]
OTHER = ["w439", "w447"]   # smokes of trec_score.py (w439: replay without the mixture permutation, failed its check; w447 passed)
CODE = "https://github.com/wenhaochai/marin-pli/tree/09e8580bd6be1240572b858130a2df0411f68bef"   # the public commit (wenhaochai/marin-pli, branch trec), scrubbed and scanned 2026-10-27 (Figure 5 second seed)


def job(w):
    d = os.path.join(WQ, w[1:])
    spec, state = json.load(open(os.path.join(d, "spec.json"))), json.load(open(os.path.join(d, "state.json")))
    return spec, state


def iso(ts):
    return datetime.datetime.fromtimestamp(ts, NY).strftime("%Y-%m-%dT%H:%M%z")


def runs_of(state):
    return [(h["start"], h.get("end")) for h in state.get("history", []) if h.get("start")]


def gpu_hours(w):
    spec, state = job(w)
    now = datetime.datetime.now().timestamp()
    return int(spec["gpus"]) * sum(((e or now) - s) for s, e in runs_of(state)) / 3600


def scoring_seconds(run):
    return sum(float(np.load(f)["seconds"]) for f in glob.glob(os.path.join(OUT, run, "[0-9]*.npz")))


def main():
    site = sys.argv[1]
    rows, main_jobs = [], set()
    for r in ROWS:
        row = {k: r[k] for k in ("setup", "size", "layers", "steps", "batches", "figs")}
        row["url"] = f"https://wandb.ai/reself/marin-della/runs/{r['run']}"
        row["code"], row["gpus"], row["hours"], row["segments"], row["submitted"] = CODE, 0, 0.0, [], ""
        if r.get("external"):
            row["state"] = ""
        elif not r["jobs"]:
            row["state"] = "planned"
        else:
            states = []
            for w in r["jobs"]:
                main_jobs.add(w)
                spec, state = job(w)
                states.append(state["state"])
                row["gpus"] = int(spec["gpus"])
                row["submitted"] = row["submitted"] or iso(spec["submitted"])
                row["segments"] += [[iso(s), iso(e) if e else None] for s, e in runs_of(state)]
                share = [scoring_seconds(q["run"]) for q in ROWS if w in q["jobs"]]
                mine = scoring_seconds(r["run"])
                row["hours"] += gpu_hours(w) / int(spec["gpus"]) * (mine / sum(share) if sum(share) else 1 / len(share))
            bad = [s for s in states if s in ("failed", "cancelled")]
            if bad:
                raise SystemExit(f"{r['run']}: job state {states}; fix the record before building")
            row["state"] = "" if all(s == "done" for s in states) else "running" if "running" in states else "queued"
        row["hours"] = round(row["hours"], 2)
        rows.append(row)
    other = [w for w in OTHER if w not in main_jobs]
    cost = {"main": {"h100_hours": round(sum(gpu_hours(w) for w in main_jobs), 1), "jobs": len(main_jobs)},
            "other": {"h100_hours": round(sum(gpu_hours(w) for w in other), 1), "jobs": len(other)}}
    path = os.path.join(site, "assets/data/trec-record.js")
    with open(path, "w") as f:
        f.write("/* Generated by experiments/trec/build_record.py (marin branch trec); regenerate rather than hand-edit. */\n")
        f.write("window.TREC_RECORD = " + json.dumps({"rows": rows, "cost": cost}, ensure_ascii=False, separators=(",", ":")) + ";\n")
    print(path, [(r["size"], r["state"], r["hours"]) for r in rows], cost)


if __name__ == "__main__":
    main()
