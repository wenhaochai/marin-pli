"""Experiment record of wenhaochai.com/blogs/per-layer-contribution.html -> assets/data/plc-record.js (generated; never
hand-edit). One row per training run behind the page: the setup in words, size, layers, tokens, GPUs, hours, state and
the figure that uses it, the W&B link once the run exists, the public code commit it ran (W&B's commit mapped through the scrub's commit map). Finished runs: state and wall time (_runtime) from W&B reself/marin-della;
queued or running jobs: state and elapsed time from sacct (job ids below; run on a node with the Slurm client, it fails rather than guess); runs not yet submitted: "Planned".
Smoke tests and failed jobs get no rows (template rule, 2026-10-05). The table ends with two lines counted from sacct over
every GPU job the project started (working directory under project/marin-pls, the pls branch and its worktrees), H100-hours
= GPUs x elapsed: "smoke tests and failed jobs" (job names with "smoke" or "diag"; FAILED, CANCELLED, OUT_OF_MEMORY,
NODE_FAIL; a TIMEOUT no later job of the same name continued; the completed runs in RERUN, which a corrected run replaces)
and "main jobs" (the rest: completed or running, and TIMEOUT segments of resumed chains). The two add up to everything.
A row whose job failed or is in RERUN shows as planned until its replacement is submitted.
Run as: build_record.py OUT (on a vis node, which has sacct)."""
import json, subprocess, sys
from pathlib import Path
import wandb

OUT = Path(sys.argv[1])
P = "reself/marin-della/muonh-qwen3-"
S = lambda en, zh: {"en": en, "zh": zh}  # noqa: E731
BASE, SHARED, SG, SEP, PROBE, DEPTH = S("Baseline", "基线"), S("Shared head", "共享头"), S("Shared head, stop-grad", "共享头 stop-grad"), S("Separate heads", "独立头"), S("Probes only", "只加探针"), S("Ordinary model", "普通模型")
# (setup, size, layers, tokens in B, GPUs, source, figures); source: ("wandb", run) | ("slurm", jobid) | ("planned", None)
RUNS = [
    (BASE, "130m", 6, 2.6, 4, ("wandb", "130m-della4xh100"), "1, 2, 3"),
    (SHARED, "130m", 6, 2.6, 4, ("wandb", "130m-della4xh100-pls1"), "1, 3"),
    (SG, "130m", 6, 2.6, 4, ("wandb", "130m-della4xh100-pls1-dh"), "1, 3"),
    (SEP, "130m", 6, 2.6, 4, ("wandb", "130m-della4xh100-pls1-sep"), "1, 2, 3, 8"),
    (PROBE, "130m", 6, 2.6, 4, ("wandb", "130m-della4xh100-pls1-sep-bbfrozen"), "3, 8"),
    *[(DEPTH, "130m", d, 2.6, 4, ("wandb", f"130m-della4xh100-d{d}"), "2, 3, 8") for d in range(1, 6)],
    (BASE, "300m", 12, 6.0, 4, ("wandb", "300m-della4xh100"), "4, 5"),
    (SHARED, "300m", 12, 6.0, 4, ("wandb", "300m-della4xh100-pls1"), "5"),
    (SEP, "300m", 12, 6.0, 4, ("wandb", "300m-della4xh100-pls1-sep"), "4, 5"),
    (PROBE, "300m", 12, 6.0, 4, ("wandb", "300m-della4xh100-pls1-sep-bbfrozen"), "5"),
    (SEP, "130m", 48, 2.6, 8, ("slurm", 14969859, "130m-della4xh100-pls1-sep-d48"), "6"),
    (PROBE, "130m", 48, 2.6, 8, ("slurm", 14969860, "130m-della4xh100-pls1-sep-bbfrozen-d48"), "6"),
    *[(S(f"{a}: {n['en'].lower()}", f"{a}：{n['zh']}"), "130m", 48, 2.6, 8, ("slurm", j, f"130m-della4xh100-pls1-sep{bb}-a{tag}-d48"), "7")
      for a, tag, (js, jp) in [("Pre-LN", "preln", (14986868, 14986869)), ("LayerNorm Scaling", "lns", (14986870, 14986871)), ("DeepNorm", "deepnorm", (15011493, 15011494)), ("KEEL", "keel", (15011495, 15011496)),
                               ("Hyper-Connections", "hc", (15011497, 15011498)), ("mHC", "mhc", (15011499, 15011501)), ("AttnRes (Full)", "attnres", (15011503, 15011504)),
                               ("AttnRes (Block)", "attnres_block", (15011505, 15011506)), ("MoDA", "moda", (15011724, 15011508))]
      for n, j, bb in ((SEP, js, ""), (PROBE, jp, "-bbfrozen"))],
    (S("Separate heads, layer losses summing to 1", "独立头，各层损失权重合计为 1"), "130m", 6, 2.6, 4, ("slurm", 14998751, "130m-della4xh100-pls0.2-sep"), "8"),
    (S("Separate heads, each head trains its own layer", "独立头，每个头只训练自己那一层"), "130m", 6, 2.6, 4, ("slurm", 14998752, "130m-della4xh100-pls1-sep-local"), "8"),
    (S("Separate heads, both", "独立头，两者都用"), "130m", 6, 2.6, 4, ("slurm", 14998753, "130m-della4xh100-pls0.2-sep-local"), "8"),
    (S("Separate heads, heads retrained (frozen backbone)", "独立头，重训读出头（冻结骨干）"), "300m", 12, 1.0, 4, ("slurm", 15011512, "300m-della4xh100-pls1-sep-headft2000"), "9"),
    (S("Probes only, heads retrained (frozen backbone)", "只加探针，重训读出头（冻结骨干）"), "300m", 12, 1.0, 4, ("slurm", 15011513, "300m-della4xh100-pls1-sep-bbfrozen-headft2000"), "9"),
]
# Completed jobs whose run is replaced by a corrected one: the 48-layer Sandwich-LN pair stalled near 6.3 nats on the 130m
# learning rate (2026-10-05; rerun after the depth_lr_diag sweep).
RERUN = {14969859, 14969860}
STATE = {"finished": S("Done", "完成"), "running": S("Running", "运行中"), "crashed": S("Failed", "失败"), "failed": S("Failed", "失败"),
         "COMPLETED": S("Done", "完成"), "RUNNING": S("Running", "运行中"), "PENDING": S("Queued", "排队中"), "planned": S("Planned", "计划中")}
api = wandb.Api(timeout=300)


def sacct(jid):
    """State and elapsed hours of a Slurm job; fails loudly when sacct gives nothing (never a silent default)."""
    res = subprocess.run(["sacct", "-n", "-X", "-P", "-j", str(jid), "-o", "State,ElapsedRaw"], capture_output=True, text=True, timeout=120)
    out = res.stdout.strip().splitlines()
    if res.returncode != 0 or not out:
        raise RuntimeError(f"sacct gave nothing for job {jid}: rc={res.returncode} stderr={res.stderr.strip()[:200]}")
    state, secs = out[0].split("|")[:2]
    return state.split()[0], float(secs or 0) / 3600


rows = []
WB = "https://wandb.ai/reself/marin-della/runs/muonh-qwen3-"
CODE = "https://github.com/wenhaochai/marin-pli/tree/"
# private commit -> public commit, from the last scrub of the pls branch (tmp/pls_scan/scrub.sh, git filter-repo)
CMAP = dict(line.split()[:2] for line in open("/scratch/gpfs/GROUP/USER/tmp/pls_scan/pls_public/.git/filter-repo/commit-map").read().splitlines()[1:])


def wandb_run(rid):
    """The run on W&B, or None before it starts."""
    try:
        r = api.run(P + rid)
        r.state
        return r
    except Exception:
        return None


def code_url(run):
    """The public tree of the commit the run used (W&B records it); empty if that commit is not public yet."""
    c = getattr(run, "commit", None) if run is not None else None
    pub = CMAP.get(c or "")
    return CODE + pub if pub else ""


for setup, size, layers, tok, gpus, src, figs in RUNS:
    kind, ref = src[0], src[1]
    rid = ref if kind == "wandb" else src[2]
    run = wandb_run(rid) if kind != "planned" else None
    if kind == "wandb":
        r = run
        state, hours = r.state, float(r.summary.get("_runtime", 0)) / 3600
    elif kind == "slurm":
        state, hours = sacct(ref)
        if state not in ("COMPLETED", "RUNNING", "PENDING") or ref in RERUN:   # counted under failed jobs; the row waits for its replacement
            state, hours, run = "planned", 0.0, None
    else:
        state, hours = "planned", 0.0
    if state in ("PENDING", "planned"):   # no links before a job starts (a W&B page made ahead of it holds nothing)
        run = None
    url = WB + rid if run is not None else ""
    code = code_url(run)
    rows.append(dict(setup=setup, size=size, layers=layers, tokens=tok, gpus=gpus, hours=round(hours, 1), state=STATE.get(state, S(state, state)), figs=figs, url=url, code=code))
    print(f"{setup['en']:52s} {size:10s} L{layers:<3d} {state:9s} {hours:6.1f} h x {gpus} GPUs")


def project_jobs():
    """Every GPU job the project started (sacct, working directory under project/marin-pls): (jobid, name, state, GPUs, hours)."""
    res = subprocess.run(["sacct", "-n", "-X", "-P", "-S", "2026-09-01", "-E", "now", "-o", "JobID,JobName,State,ElapsedRaw,AllocTRES,WorkDir"], capture_output=True, text=True, timeout=300)
    if res.returncode != 0 or not res.stdout.strip():
        raise RuntimeError(f"sacct gave nothing for the project's jobs: rc={res.returncode} stderr={res.stderr.strip()[:200]}")
    jobs = []
    for line in res.stdout.strip().splitlines():
        jid, name, state, secs, tres, wd = line.split("|")[:6]
        if "/project/marin-pls" not in wd or "gres/gpu=" not in tres or state.startswith("PENDING"):
            continue
        gpus = int(tres.split("gres/gpu=")[1].split(",")[0])
        jobs.append((jid, name, state.split()[0], gpus, float(secs or 0) / 3600))
    return jobs


def bucket(jobs):
    """'smoke' (smoke tests and failed jobs) or 'main' per job, by the rules in the module docstring."""
    out = []
    for jid, name, state, gpus, h in jobs:
        base = int(jid.split("_")[0])
        later_ok = any(n == name and int(j.split("_")[0]) > base and s in ("COMPLETED", "RUNNING") for j, n, s, _, _ in jobs)
        failed = ("smoke" in name or "diag" in name or base in RERUN or state in ("FAILED", "CANCELLED", "OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL", "DEADLINE", "PREEMPTED")
                  or (state == "TIMEOUT" and not later_ok) or state not in ("COMPLETED", "RUNNING", "TIMEOUT"))
        out.append(("smoke" if failed else "main", jid, name, state, gpus, h))
    return out


B = bucket(project_jobs())
tot = {k: dict(h100_hours=round(sum(g * h for b, _, _, _, g, h in B if b == k)), jobs=sum(b == k for b, *_ in B)) for k in ("smoke", "main")}
for b, jid, name, state, g, h in sorted(B, key=lambda x: (x[0], x[1])):
    print(f"  {b:5s} {jid:12s} {name:28s} {state:14s} {g} x {h:6.2f} h = {g * h:7.1f} H100-h")
rec = dict(rows=rows, total=tot)
OUT.write_text("/* Generated by build_record.py (pls branch, docs/della/pls_archive/blog); regenerate rather than hand-edit. */\nwindow.PLC_RECORD = " + json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + ";\n")
print("rows", len(rows), "| smoke tests and failed jobs", tot["smoke"], "| main jobs", tot["main"], "->", OUT)
