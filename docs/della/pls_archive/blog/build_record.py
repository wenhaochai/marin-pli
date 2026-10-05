"""Experiment record of wenhaochai.com/blogs/per-layer-contribution.html -> assets/data/plc-record.js (generated; never
hand-edit). One row per training run behind the page: the setup in words, size, layers, tokens, GPUs, hours, state and
the figure that uses it, the W&B link once the run exists, the public code commit it ran (W&B's commit mapped through the scrub's commit map), plus totals (GPU-hours only in the total). Finished runs: state and wall time (_runtime) from W&B reself/marin-della;
queued or running jobs: state and elapsed time from sacct (job ids below; run on a node with the Slurm client, it fails rather than guess); runs not yet submitted: "Planned".
GPU-hours = GPUs x hours. Smoke tests are not counted. Run as: build_record.py OUT (on a vis node, which has sacct)."""
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
      for a, tag, (js, jp) in [("Pre-LN", "preln", (14986868, 14986869)), ("LayerNorm Scaling", "lns", (14986870, 14986871)), ("DeepNorm", "deepnorm", (14986872, 14986873)), ("KEEL", "keel", (14986874, 14986875))]
      for n, j, bb in ((SEP, js, ""), (PROBE, jp, "-bbfrozen"))],
    *[(S(f"{a}: {n['en'].lower()}", f"{a}：{n['zh']}"), "130m", 48, 2.6, 8, ("planned", None, f"130m-della4xh100-pls1-sep{bb}-a{tag}-d48"), "7")
      for a, tag in (("Hyper-Connections", "hc"), ("mHC", "mhc"), ("AttnRes (Full)", "attnres"), ("AttnRes (Block)", "attnres_block"), ("MoDA", "moda")) for n, bb in ((SEP, ""), (PROBE, "-bbfrozen"))],
    (S("Separate heads, layer losses summing to 1", "独立头，各层损失权重合计为 1"), "130m", 6, 2.6, 4, ("slurm", 14998751, "130m-della4xh100-pls0.2-sep"), "8"),
    (S("Separate heads, each head trains its own layer", "独立头，每个头只训练自己那一层"), "130m", 6, 2.6, 4, ("slurm", 14998752, "130m-della4xh100-pls1-sep-local"), "8"),
    (S("Separate heads, both", "独立头，两者都用"), "130m", 6, 2.6, 4, ("slurm", 14998753, "130m-della4xh100-pls0.2-sep-local"), "8"),
]
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


rows, tot_h, tot_done = [], 0.0, 0
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
    run = wandb_run(rid)
    if kind == "wandb":
        r = run
        state, hours = r.state, float(r.summary.get("_runtime", 0)) / 3600
    elif kind == "slurm":
        state, hours = sacct(ref)
    else:
        state, hours = "planned", 0.0
    url = WB + rid if run is not None else ""
    code = code_url(run)
    gh = gpus * hours
    tot_h += gh
    tot_done += state in ("finished", "COMPLETED")
    rows.append(dict(setup=setup, size=size, layers=layers, tokens=tok, gpus=gpus, hours=round(hours, 1), gpu_hours=round(gh), state=STATE.get(state, S(state, state)), figs=figs, url=url, code=code))
    print(f"{setup['en']:52s} {size:10s} L{layers:<3d} {state:9s} {hours:6.1f} h x {gpus} GPUs")
rec = dict(rows=rows, total=dict(runs=len(rows), done=tot_done, gpu_hours=round(tot_h)))
OUT.write_text("/* Generated by build_record.py (pls branch, docs/della/pls_archive/blog); regenerate rather than hand-edit. */\nwindow.PLC_RECORD = " + json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + ";\n")
print("runs", len(rows), "done", tot_done, "GPU-hours", round(tot_h), "->", OUT)
