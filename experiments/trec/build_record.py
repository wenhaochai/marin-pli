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
         size="130m", layers=12, steps=4959, batches=4959 + 512, jobs=["w465"], figs="1, 2, 3, 4, 6"),
    dict(model="300m", run="muonh-qwen3-300m-della4xh100-s1-rerun", setup={"en": "Cosine to zero, lr 0.01", "zh": "cosine 降到零，lr 0.01"},
         size="300m", layers=12, steps=11444, batches=11444 + 512, jobs=["w465"], figs="1, 2, 3, 4, 5, 6"),
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
P = "muonh-qwen3-"
# Figures 6 and 7 (owner popups 2026-10-27): the other MuonH checkpoints, 1 in 8 training batches (blocks of 8 steps every
# 64, plus the last 8) and 512 unread ones, none for the runs that read their data more than once; wq w513 and w514
# (scripts/della/trec-runs-*.sbatch). Layers and steps come from the W&B config (experiments/trec/fetch_wandb.py).
NEW = [(f"{P}130m-della4xh100-d{n}", "", f"{n} layers, WSD to zero, lr 0.02", f"{n} 层，WSD 降到零，lr 0.02", "130m", "w513", "6")
       for n in (1, 2, 3, 4, 5, 24, 36)] + [
    (f"{P}300m-della8xh100-pls0-noev", "", "Cosine to zero, lr 0.01, seed 0", "cosine 降到零，lr 0.01，种子 0", "300m", "w513", "6"),
    (f"{P}520m-della8xh100-pls0-noev", "", "1.04 passes over the data", "数据读 1.04 遍", "520m", "w513", "7"),
    (f"{P}1_2b-della8xh100-pls0-noev", "", "2.40 passes over the data", "数据读 2.40 遍", "1.2b", "w513", "7"),
    (f"{P}300m-della4xh100-v8k", "", "8K truncation, seed 0", "8K 截短版，种子 0", "300m", "w514", "6"),
    (f"{P}300m-della4xh100-v8k-s1", "", "8K truncation, seed 1", "8K 截短版，种子 1", "300m", "w514", "6"),
    (f"{P}300m-della4xh100-lrx2-fp2-v8k", "", "8K, learning rate x2 from step T/8", "8K，从第 T/8 步起学习率 ×2", "300m", "w514", "6"),
    (f"{P}300m-della4xh100-lrx3-fp2-v8k", "", "8K, learning rate x3 from step T/8", "8K，从第 T/8 步起学习率 ×3", "300m", "w514", "6"),
    (f"{P}300m-della4xh100-lrx4-fp2-v8k", "", "8K, learning rate x4 from step T/8", "8K，从第 T/8 步起学习率 ×4", "300m", "w514", "6"),
    (f"{P}300m-della4xh100-ema0.999", "", "EMA 0.999, raw weights", "EMA 0.999，原始权重", "300m", "w514", "6"),
    (f"{P}300m-della4xh100-ema0.999", "-ema", "EMA 0.999, EMA weights", "EMA 0.999，EMA 权重", "300m", "w514", "6"),
    (f"{P}300m-della4xh100-ema0.9998", "", "EMA 0.9998, raw weights", "EMA 0.9998，原始权重", "300m", "w514", "6"),
    (f"{P}300m-della4xh100-ema0.9998", "-ema", "EMA 0.9998, EMA weights", "EMA 0.9998，EMA 权重", "300m", "w514", "6"),
    (f"{P}300m-della4xh100-b96", "", "Batch 96", "batch 96", "300m", "w514", "6"),
    (f"{P}300m-della4xh100-v16k", "", "16K truncation", "16K 截短版", "300m", "w514", "6"),
    (f"{P}300m-della4xh100-v32k", "", "32K truncation", "32K 截短版", "300m", "w514", "6"),
    (f"{P}300m-della4xh100-v64k", "", "64K truncation", "64K 截短版", "300m", "w514", "6"),
    (f"{P}300m-della4xh100-bytes", "", "Bytes, seed 0", "字节，种子 0", "300m", "w514", "6"),
    (f"{P}130m-della4xh100-tpp200", "", "10x steps, 2.60 passes, seed 0", "10 倍步数，读 2.60 遍，种子 0", "130m", "w514", "7"),
    (f"{P}130m-della4xh100-tpp200-s1", "", "10x steps, 2.60 passes, seed 1", "10 倍步数，读 2.60 遍，种子 1", "130m", "w514", "7")]
N_SEQ = 2441406   # sequences of fineweb-edu-10B under the Marin tokenizer (w465 log)


def sampled(steps, tail):
    t = np.arange(steps)
    return len(np.union1d(t[t % 64 < 8], t[-8:])) + tail


def new_rows():
    rows = []
    for run, sfx, en, zh, size, w, figs in NEW:
        cfg = json.load(open(f"/scratch/gpfs/GROUP/USER/project/trec/cache/wandb/{run}.json"))["config"]
        steps, batch = cfg["trainer"]["num_train_steps"], cfg["trainer"]["train_batch_size"]
        once = "-v" in run or "-bytes" in run or steps * batch <= N_SEQ   # the K-token streams hold more sequences
        rows.append(dict(model=None, run=run, out=f"runs/{run}{sfx}", setup={"en": en, "zh": zh}, size=size, layers=cfg["model"]["num_layers"],
                         steps=steps, batches=sampled(steps, 512 if once else 0), jobs=[w], figs=figs))
    return rows


ROWS += new_rows()
# Figure 8 (owner popup 2026-10-28): the 130m d12 control trained again under the equal-weight schedule of
# kexue.fm/archives/11459 (branch trec-rsqrt) and scored on every training batch and 512 unread ones, in one wq job each
# (training and scoring); code: the public commit once the runs finish (empty until then; queued rows show no code link)
ROWS += [dict(model=None, run=f"muonh-qwen3-130m-della4xh100-d12-{tag}", out=f"muonh-qwen3-130m-della4xh100-d12-{tag}", setup={"en": en, "zh": zh},
              size="130m", layers=12, steps=4959, batches=4959 + 512, jobs=[w], figs="8", code="")
         for tag, en, zh, w in [("rsqrt", "lr 0.02 / \u221a(1 + 0.0092 t)", "lr 0.02 / \u221a(1 + 0.0092 t)", "w590"),
                                ("rsqrtcd", "lr 0.02 / \u221a(1 + 0.0092 t), linear to zero over the last fifth", "lr 0.02 / \u221a(1 + 0.0092 t)，最后五分之一线性降到零", "w591")]]
# Figure 9 (owner popup 2026-10-29): the same two schedules with a checkpoint every 248 steps (branch trec-rsqrt
# 31a2bb26e0); checkpoints 248..4712 score a 1-in-16 sample (blocks of 8 every 128, plus the last 8) and 256 unread
# batches, the final one every batch and 512 unread ones
_CK_SCORED = 19 * (len(np.union1d(np.arange(4959)[np.arange(4959) % 128 < 8], np.arange(4951, 4959))) + 256) + 4959 + 512
ROWS += [dict(model=None, run=f"muonh-qwen3-130m-della4xh100-d12{tag}-ck248", out=f"ck/muonh-qwen3-130m-della4xh100-d12{tag}-ck248", setup={"en": en, "zh": zh},
              size="130m", layers=12, steps=4959, batches=_CK_SCORED, jobs=[w], figs="9", code="")
         for tag, en, zh, w in [("", "WSD to zero, lr 0.02, a checkpoint every 248 steps", "WSD 降到零，lr 0.02，每 248 步一个 checkpoint", "w642"),
                                ("-rsqrtcd", "lr 0.02 / \u221a(1 + 0.0092 t) with cooldown, a checkpoint every 248 steps", "lr 0.02 / \u221a(1 + 0.0092 t) 加冷却，每 248 步一个 checkpoint", "w643")]]
# Figure 9 at 300m (owner popup 2026-10-29): the 300m reference run trained again with a checkpoint every 572 steps
# (branch trec-rsqrt 2aaa9d3507)
_CK300 = 20 * (len(np.union1d(np.arange(11444)[np.arange(11444) % 128 < 8], np.arange(11436, 11444))) + 256) + 11444 + 512
ROWS += [dict(model=None, run="muonh-qwen3-300m-della4xh100-ck572-s1", out="ck/muonh-qwen3-300m-della4xh100-ck572-s1",
              setup={"en": "Cosine to zero, lr 0.01, seed 1, a checkpoint every 572 steps", "zh": "cosine 降到零，lr 0.01，种子 1，每 572 步一个 checkpoint"},
              size="300m", layers=12, steps=11444, batches=_CK300, jobs=["w667"], figs="9", code="")]
# Figure 10 (owner popup 2026-10-29): d12-ck248 continued from step 2480 at a constant lr, lr x steps = 5 (branch
# trec-rsqrt dc5c92b4c9); checkpoints every 124 steps score batches 2232-2479 and 256 unread ones (arm a also scores
# the d12-ck248 checkpoints 2232 and 2480)
ROWS += [dict(model=None, run=f"muonh-qwen3-130m-della4xh100-d12-constant-ck124-lrm{m}-br2480", out=f"ck/branch/muonh-qwen3-130m-della4xh100-d12-constant-ck124-lrm{m}-br2480",
              setup={"en": f"From step 2480 of the checkpointed WSD run, lr {lr} for {n} steps", "zh": f"从每 248 步存 checkpoint 的 WSD run 第 2480 步起，lr {lr} 训练 {n} 步"},
              size="130m", layers=12, steps=n, batches=(n // 124 + 1 + (2 if w == "w658" else 0)) * (248 + 256), jobs=[w], figs="10", code="")
         for m, lr, n, w in [("0.5", 0.01, 500, "w658"), ("0.25", 0.005, 1000, "w659"), ("0.125", 0.0025, 2000, "w660")]]
# Figure 10 at branch points 1240 and 3720 (owner popup 2026-10-29: is p the same early and late?), trec-rsqrt 8970c90296
ROWS += [dict(model=None, run=f"muonh-qwen3-130m-della4xh100-d12-constant-ck124-lrm{m}-br{bp}", out=f"ck/branch/muonh-qwen3-130m-della4xh100-d12-constant-ck124-lrm{m}-br{bp}",
              setup={"en": f"From step {bp} of the checkpointed WSD run, lr {lr} for {n} steps", "zh": f"从每 248 步存 checkpoint 的 WSD run 第 {bp} 步起，lr {lr} 训练 {n} 步"},
              size="130m", layers=12, steps=n, batches=(n // 124 + 1 + (2 if arm == "a" else 0)) * (248 + 256), jobs=[w], figs="10", code="")
         for bp, ws in [(1240, ["w671", "w672", "w673"]), (3720, ["w674", "w675", "w676"])]
         for (m, lr, n, arm), w in zip([("0.5", 0.01, 500, "a"), ("0.25", 0.005, 1000, "b"), ("0.125", 0.0025, 2000, "c")], ws)]
OTHER = ["w439", "w447", "w512", "w651", "w652", "w653"]   # smokes of trec_score.py (w439: replay without the mixture permutation, failed its check; w447 passed; w512 the trec-runs smoke); w651-w653: Figure 10 branches whose good smokes failed a gate that looked for a loss line a 10-step run does not print (rerun as w658-w660)
CODE = "https://github.com/wenhaochai/marin-pli/tree/2347279edabd36dcdf1ac0097d335d3d76130ccc"   # the public commit (wenhaochai/marin-pli, branch trec), scrubbed and scanned 2026-10-28 (Figures 6 and 7)


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
        row["code"], row["gpus"], row["hours"], row["segments"], row["submitted"] = r.get("code", CODE), 0, 0.0, [], ""
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
                share = [scoring_seconds(q.get("out", q["run"])) for q in ROWS if w in q["jobs"]]
                mine = scoring_seconds(r.get("out", r["run"]))
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
