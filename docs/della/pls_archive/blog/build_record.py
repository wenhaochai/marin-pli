"""Experiment record of wenhaochai.com/blogs/per-layer-contribution.html -> assets/data/plc-record.js (generated; never
hand-edit). One row per training run behind the page: the setup in words, size, layers, tokens, GPUs, hours, state and
the figure that uses it, the W&B link once the run exists, the public code commit it ran (W&B's commit mapped through the scrub's commit map). Finished runs: state and wall time (_runtime) from W&B reself/marin-della;
queued or running jobs: state and elapsed time from sacct (job ids below; run on a node with the Slurm client, it fails rather than guess); runs not yet submitted: "Planned".
Smoke tests and jobs outside the rows get no rows. The table ends with two lines counted from sacct over every GPU job the
project started (working directory under project/marin-pls, the pls branch and its worktrees), H100-hours = GPUs x
elapsed (template rule, owner, 2026-10-05; packed jobs split by their task ledger logs/packed/JOBID.tsv, template dcfd5a1, 2026-10-07): "main runs" are the jobs that belong to a run with a row, every resume segment
included (a slurm row's own job, plus the jobs listed in JOBS for runs launched before the record kept job ids; each
listed job is checked against its log's W&B run id when the log still exists), and "other compute" is everything else:
smoke tests, debugging, superseded versions (RERUN), abandoned runs, probes not on the page. The two add up to everything
the project ran. A row whose job did not finish, or whose run is in RERUN, shows as planned until its replacement is
submitted; its jobs are other compute.
48-layer rows (Figures 6 and 7, the -vs-d48-g4 runs, two 4-GPU runs per 8-GPU pair job) name their whole afterany chain
("chain" source, audit 2026-10-11): Done when W&B says finished and the history reaches the last step; Running once a
training task of the run has run and a chain job is still queued or running; Queued before that; Failed when the chain
has ended short. Their hours are the wall time of every training task of the run in the ledgers.
Training tasks count as main when their run has a row that is not Planned and they exited 0, 124 (the pair script's
training limit, before the 24 h wall: the next chain job resumes) or 143 (stopped by TERM, resumed); crashed tasks are
other (audit 2026-10-11: a resume segment belongs to its run). A packed job counts as a main job only if it gave main
some hours.
Run as: build_record.py OUT (on a vis node, which has sacct)."""
import json, os, re, subprocess, sys
from pathlib import Path
import wandb

OUT = Path(sys.argv[1])
P = "reself/marin-della/muonh-qwen3-"
S = lambda en, zh: {"en": en, "zh": zh}  # noqa: E731
BASE, SHARED, SG, SEP, PROBE, DEPTH = S("Baseline", "基线"), S("Shared head", "共享头"), S("Shared head, stop-grad", "共享头 stop-grad"), S("Separate heads", "独立头"), S("Probes only", "只加探针"), S("Ordinary model", "普通模型")
# (setup, size, layers, tokens in B, GPUs, source, figures); source: ("wandb", run) | ("slurm", jobid) | ("planned", None)
RUNS = [
    (BASE, "130m", 6, 2.6, 4, ("wandb", "130m-della4xh100"), "1, 2, 3, 12, 13, 14"),
    (SHARED, "130m", 6, 2.6, 4, ("wandb", "130m-della4xh100-pls1"), "1, 3"),
    (SG, "130m", 6, 2.6, 4, ("wandb", "130m-della4xh100-pls1-dh"), "1, 3"),
    (SEP, "130m", 6, 2.6, 4, ("wandb", "130m-della4xh100-pls1-sep"), "1, 2, 3, 12, 13, 14"),
    (PROBE, "130m", 6, 2.6, 4, ("wandb", "130m-della4xh100-pls1-sep-bbfrozen"), "3, 12, 14"),
    *[(DEPTH, "130m", d, 2.6, 4, ("wandb", f"130m-della4xh100-d{d}"), "2, 3, 12, 13, 14") for d in range(1, 6)],
    (BASE, "300m", 12, 6.0, 4, ("wandb", "300m-della4xh100"), "4, 5"),
    (SHARED, "300m", 12, 6.0, 4, ("wandb", "300m-della4xh100-pls1"), "5"),
    (SEP, "300m", 12, 6.0, 4, ("wandb", "300m-della4xh100-pls1-sep"), "4, 5"),
    (PROBE, "300m", 12, 6.0, 4, ("wandb", "300m-della4xh100-pls1-sep-bbfrozen"), "5"),
    # 48 layers (-vs-d48-g4): each design's pair-job chain; the earlier 8-GPU FSDP runs (-d48) are superseded, so other
    *[(n if tag == "" else S(f"{a}: {n['en'].lower()}", f"{a}：{n['zh']}"), "130m", 48, 2.6, 4,
       ("chain", jobs, f"130m-della4xh100-pls1-sep{bb}{'-a' + tag if tag else ''}-vs-d48-g4"), "6, 9, 10, 11" if tag == "" else ("7, 8" if tag == "attnres" else ("7, 9, 10" if tag in ("attnres_block", "moda") else "7, 9, 10, 11")))
      for a, tag, jobs in [("Sandwich-LN", "", [15266143, 15266144, 15273924]), ("Pre-LN", "preln", [15266145, 15266146, 15273925]),
                           ("LayerNorm Scaling", "lns", [15266147, 15266148]), ("DeepNorm", "deepnorm", [15266149, 15266150]), ("KEEL", "keel", [15266151, 15266152]),
                           ("Hyper-Connections", "hc", [15266153, 15266154, 15341053, 15341054]), ("mHC", "mhc", [15266155, 15266156, 15341055, 15341056]),
                           ("AttnRes (Full)", "attnres", [15266159, 15266160, 15266161]), ("AttnRes (Block)", "attnres_block", [15266157, 15266158]),
                           ("MoDA", "moda", [15266162, 15266163, 15266164, 15341589])]
      for n, bb in ((SEP, ""), (PROBE, "-bbfrozen"))],
    # AttnRes (Full), second init seed (Figure 8; owner popup 2026-10-13)
    *[(S(f"AttnRes (Full), seed 1: {n['en'].lower()}", f"AttnRes (Full)，种子 1：{n['zh']}"), "130m", 48, 2.6, 4,
       ("chain", [15520124, 15520125, 15520126, 15538869], f"130m-della4xh100-pls1-sep{bb}-aattnres-vs-d48-s1-g4"), "8") for n, bb in ((SEP, ""), (PROBE, "-bbfrozen"))],
    # HC with positive writes (Figure 11; owner popup 2026-10-14): the causal test of HC's half last-layer cost
    *[(S(f"HC with positive writes: {n['en'].lower()}", f"写入为正的 HC：{n['zh']}"), "130m", 48, 2.6, 4,
       ("chain", [15536266, 15536267], f"130m-della4xh100-pls1-sep{bb}-ahc_pos-vs-d48-g4"), "11") for n, bb in ((SEP, ""), (PROBE, "-bbfrozen"))],
    # ordinary models of 12, 24 and 36 layers, the same-depth references for Figure 6 (owner popup 2026-10-11)
    *[(DEPTH, "130m", d, 2.6, 4, ("slurm", f"15519767_{d}", f"130m-della4xh100-d{d}"), "6") for d in (12, 24, 36)],
    # multi-token prediction (Figure 16; owner popup 2026-10-13): one pair job, both runs with probes for x_{t+1} and x_{t+2}
    (S("NTP, probes for the next two tokens", "NTP，预测后两个词的探针"), "130m", 6, 2.6, 4, ("chain", [15523669], "130m-della4xh100-pls1-sep-bbfrozen-os-fut"), "17, 18"),
    (S("NTP + MTP, probes for the next two tokens", "NTP + MTP，预测后两个词的探针"), "130m", 6, 2.6, 4, ("chain", [15523669], "130m-della4xh100-pls1-sep-bbfrozen-os-mtp0.3-fut"), "17, 18"),
    # MTP across model sizes (Figures 17 and 18; owner popup 2026-10-14). The 130m point is the Figure 16 pair above. Each
    # larger size trains NTP and NTP + MTP 0.3 as the baseline's training, 8 GPUs per run, one afterany chain each (the
    # first 520m NTP job failed its smoke and the first 1_2b NTP job crashed in training, both out of GPU memory: other).
    *[(S(f"{a} as the baseline", f"{a}，按基线训练"), size, layers, tok, 8, ("chain", jobs, f"{size}-della8xh100-pls0-noev{tag}"), "18")
      for a, tag, runs in [("NTP", "", [("300m", 12, 6.0, [15591470, 15591471]), ("520m", 24, 10.4, [15591472, 15591473, 15664536]), ("1_2b", 16, 24.0, [15591482, 15591483, 15591484])]),
                           ("NTP + MTP", "-mtp0.3", [("300m", 12, 6.0, [15591485, 15591486]), ("520m", 24, 10.4, [15591490, 15591491]), ("1_2b", 16, 24.0, [15591492, 15591506, 15591507])])]
      for size, layers, tok, jobs in runs],
    # post-hoc probes of the finished models (Figure 18): fresh next-token and x_{t+2} heads on the frozen final checkpoint,
    # 1B tokens each. The 520m and 1_2b probes are submitted when their training runs finish.
    *[(S(f"Probes on the finished {a} model", f"在训练完的 {a} 模型上加探针"), size, layers, 1.0, 8,
       ("slurm", jid, rid) if jid else ("planned", None, rid), "19")
      for a, size, layers, jid, rid in [
          ("NTP", "130m", 6, 15591508, "130m-della8xh100-pls1-sep-bbfrozen-os-fut-probes2000at4958"),
          ("NTP + MTP", "130m", 6, 15591509, "130m-della8xh100-pls1-sep-bbfrozen-os-mtp0.3-fut-probes2000at4958"),
          ("NTP", "300m", 12, 15714847, "300m-della8xh100-pls0-noev-probes2000at11443"),
          ("NTP + MTP", "300m", 12, 15690439, "300m-della8xh100-pls0-noev-mtp0.3-probes2000at11443"),
          ("NTP", "520m", 24, 15716064, "520m-della8xh100-pls0-noev-probes1000at9917"),
          ("NTP + MTP", "520m", 24, 15699612, "520m-della8xh100-pls0-noev-mtp0.3-probes1000at9917"),
          ("NTP", "1_2b", 16, None, "1_2b-della8xh100-pls0-noev-probes1000at22887"),
          ("NTP + MTP", "1_2b", 16, None, "1_2b-della8xh100-pls0-noev-mtp0.3-probes1000at22887")]],
    (S("Separate heads, layer losses summing to 1", "独立头，各层损失权重合计为 1"), "130m", 6, 2.6, 4, ("slurm", 14998751, "130m-della4xh100-pls0.2-sep"), "12, 13"),
    (S("Separate heads, each head trains its own layer", "独立头，每个头只训练自己那一层"), "130m", 6, 2.6, 4, ("slurm", 14998752, "130m-della4xh100-pls1-sep-local"), "12"),
    (S("Separate heads, both", "独立头，两者都用"), "130m", 6, 2.6, 4, ("slurm", 14998753, "130m-della4xh100-pls0.2-sep-local"), "12"),
    (S("Separate heads, layer-loss weight 0.1", "独立头，逐层损失权重 0.1"), "130m", 6, 2.6, 4, ("slurm", 15064965, "130m-della4xh100-pls0.1-sep"), "12"),
    (S("Separate heads, layer-loss weight 0.05", "独立头，逐层损失权重 0.05"), "130m", 6, 2.6, 4, ("slurm", 15064966, "130m-della4xh100-pls0.05-sep"), "12"),
    (S("Separate heads, weights rising with depth (summing to 1)", "独立头，权重随深度增加（合计为 1）"), "130m", 6, 2.6, 4, ("slurm", 15055614, "130m-della4xh100-pls0.2-sep-shdepth"), "13"),
    (S("Separate heads, weights summing to 1, gradient surgery", "独立头，权重合计为 1，处理梯度冲突"), "130m", 6, 2.6, 4, ("slurm", 15055616, "130m-della4xh100-pls0.2-sep-pcg"), "13"),
    (S("Separate heads, weights rising with depth, gradient surgery", "独立头，权重随深度增加，处理梯度冲突"), "130m", 6, 2.6, 4, ("slurm", 15055618, "130m-della4xh100-pls0.2-sep-shdepth-pcg"), "13"),
    (S("Separate heads, weight 1, gradient surgery", "独立头，权重 1，处理梯度冲突"), "130m", 6, 2.6, 4, ("slurm", 15083130, "130m-della4xh100-pls1-sep-pcg"), "13"),
    (S("Separate heads from step 300 (probes before)", "第 300 步起用独立头（之前为探针）"), "130m", 6, 2.6, 4, ("slurm", 15168919, "130m-della4xh100-pls1-sep-on300-400"), "14"),
    (S("Separate heads from step 1,500 (probes before)", "第 1,500 步起用独立头（之前为探针）"), "130m", 6, 2.6, 4, ("slurm", 15168919, "130m-della4xh100-pls1-sep-on1500-1600"), "14"),
    (S("Separate heads, heads retrained (frozen backbone)", "独立头，重训读出头（冻结骨干）"), "300m", 12, 1.0, 4, ("slurm", 15011512, "300m-della4xh100-pls1-sep-headft2000"), "16"),
    (S("Probes only, heads retrained (frozen backbone)", "只加探针，重训读出头（冻结骨干）"), "300m", 12, 1.0, 4, ("slurm", 15011513, "300m-della4xh100-pls1-sep-bbfrozen-headft2000"), "16"),
    # ten more attempts at the last-layer cost (Figure 15; owner popup 2026-10-16), branch pls-auto10: two 130m runs per
    # 8-GPU pair job (pls_auto10_pair.sbatch, fetchable); the job ids join as the pairs are submitted
    *[(S(en, zh), "130m", 6, 2.6, 4, ("slurm", jid, rid) if jid else ("planned", None, rid), "15") for en, zh, jid, rid in [
        ("Separate heads, the readouts clipped on their own", "独立头，读出头单独裁剪", 15698932, "130m-della4xh100-pls1-sep-os"),
        ("Separate heads, normalise then mix at 0.2", "独立头，先正交化再按 0.2 混合", 15700045, "130m-della4xh100-pls1-sep-mix0.2"),
        ("Separate heads learning the final layer's distribution", "独立头学最终层的分布", 15699791, "130m-della4xh100-pls1-sep-kd"),
        ("Separate heads weighted by the final layer's confidence", "独立头按最终层置信度加权", 15698936, "130m-della4xh100-pls1-sep-conf"),
        ("Separate heads with exit adapters", "独立头加出口适配器", 15699791, "130m-della4xh100-pls1-sep-ad1024"),
        ("Separate heads training MLPs only", "独立头只更新 MLP", 15700045, "130m-della4xh100-pls1-sep-mlpo"),
        ("Separate heads on the first 256 positions", "独立头只用前 256 个位置", 15698936, "130m-della4xh100-pls1-sep-first256"),
        ("Separate heads, one random layer per step", "独立头，每步随机一层", 15698932, "130m-della4xh100-pls1-sep-randl"),
        ("Hyper-Connections: separate heads", "Hyper-Connections：独立头", 15700214, "130m-della4xh100-pls1-sep-ahc"),
        ("Hyper-Connections: probes only", "Hyper-Connections：只加探针", 15700214, "130m-della4xh100-pls1-sep-bbfrozen-ahc"),
        ("AttnRes (Full): separate heads", "AttnRes (Full)：独立头", 15699797, "130m-della4xh100-pls1-sep-aattnres"),
        ("AttnRes (Full): probes only", "AttnRes (Full)：只加探针", 15699797, "130m-della4xh100-pls1-sep-bbfrozen-aattnres")]],
]
# Completed jobs whose run is replaced by a corrected one: the 48-layer Sandwich-LN pair stalled near 6.3 nats on the 130m
# learning rate (2026-10-05; rerun after the depth_lr_diag sweep).
RERUN = {14969859, 14969860}
# Slurm jobs of the runs whose rows read W&B (launched before the record kept job ids), from sacct's submit lines:
# job name and --export knobs (2026-10-05). Resume segments included (300m shared head: 14732139 timed out, 14732140
# continued it). The 130m and 300m baselines ran in project/marin for another page and are reused here, so no job of
# this project belongs to them.
JOBS = {"130m-della4xh100-pls1": [14732138], "130m-della4xh100-pls1-dh": [14769012], "130m-della4xh100-pls1-sep": [14769391],
        "130m-della4xh100-pls1-sep-bbfrozen": [14888104], **{f"130m-della4xh100-d{d}": [f"14886384_{d}"] for d in range(1, 6)},
        "300m-della4xh100-pls1": [14732139, 14732140], "300m-della4xh100-pls1-sep": [14940604], "300m-della4xh100-pls1-sep-bbfrozen": [14888822]}
STATE = {"finished": S("Done", "完成"), "running": S("Running", "运行中"), "crashed": S("Failed", "失败"), "failed": S("Failed", "失败"),
         "COMPLETED": S("Done", "完成"), "RUNNING": S("Running", "运行中"), "PENDING": S("Queued", "排队中"), "planned": S("Planned", "计划中")}
OK_EXIT = {"0", "124", "143"}   # training-task exits that count for the run (see the docstring)
api = wandb.Api(timeout=300)


def sacct(jid):
    """State and elapsed hours of a Slurm job; fails loudly when sacct gives nothing (never a silent default)."""
    res = subprocess.run(["sacct", "-n", "-X", "-P", "-j", str(jid), "-o", "State,ElapsedRaw"], capture_output=True, text=True, timeout=120)
    out = res.stdout.strip().splitlines()
    if res.returncode != 0 or not out:
        raise RuntimeError(f"sacct gave nothing for job {jid}: rc={res.returncode} stderr={res.stderr.strip()[:200]}")
    state, secs = out[0].split("|")[:2]
    return state.split()[0], float(secs or 0) / 3600


PACKED = Path(os.environ.get("PLS_PACKED_DIR", "/scratch/gpfs/GROUP/USER/project/marin-pls/logs/packed"))   # env: tests only


LEDGER_COLS = ["task", "run", "kind", "gpus", "start", "end", "exit"]


def ledger(jid: str):
    """A packed job's task ledger (tools/packed_task.sh): [{task, run, kind, gpus, start, end, exit}], [] when the file
    holds no task yet (every hour then goes to other), or None for a plain job. Header lines are skipped wherever they
    stand (two tasks starting together can both write one); any other line that is not a full, numeric task line
    raises (a truncated write is a loss to look at, not to guess around)."""
    f = PACKED / f"{jid}.tsv"
    if not f.exists():
        return None
    out = []
    for n, line in enumerate(f.read_text().splitlines(), 1):
        cols = line.split("\t")
        if not line.strip() or cols == LEDGER_COLS:
            continue
        ok = len(cols) == 7 and cols[2] in ("main", "other") and all(c.lstrip("-").isdigit() for c in cols[3:])
        if not ok or int(cols[5]) < int(cols[4]):
            raise RuntimeError(f"ledger {f} line {n} is not a task line: {line!r}")
        out.append(dict(zip(LEDGER_COLS, cols)))
    return out


# fetched.tsv (tools/fetch_pending.sh, 2026-10-16): one line per job of this project that a host job ran in its leftover
# time: fetched_id host_id start end exit. The fetched job itself ends CANCELLED in sacct with no elapsed time; it ran
# inside the host, from start to end, on the host's GPUs. Its own tasks (smoke, train) write their ledger lines to
# logs/packed/<host>.tsv here, since pls_h100x8_single.sbatch keeps PACKED_LEDGER_DIR and SLURM_JOB_ID is the host's.
# A row whose job was fetched takes its state from the exit code and its hours from those task lines. H100-hours: a host
# of this project already counts the fetched run through its own ledger; a host outside it (another project's job) adds
# GPUs x (end - start) to the closing lines, split by the task lines in that window, and the job counts once.
def fetched():
    f = PACKED / "fetched.tsv"
    out = {}
    for n, line in enumerate(f.read_text().splitlines() if f.exists() else [], 1):
        cols = line.split("\t")
        if not line.strip():
            continue
        if len(cols) != 5 or not all(c.lstrip("-").isdigit() for c in cols):
            raise RuntimeError(f"{f} line {n} is not a fetched line: {line!r}")
        fid, host, start, end, ex = cols
        out[fid] = dict(host=host, start=int(start), end=int(end), exit=ex)
    return out


FETCH = fetched()
TASKS = [dict(t, jid=f.stem) for f in sorted(PACKED.glob("*.tsv")) if f.name != "fetched.tsv" for t in (ledger(f.stem) or [])]


import time
NOW = time.time()


def train_start(jid):
    """For a pair job still in training: when its smokes ended (its training tasks have no ledger line until they end);
    None if the job has no ledger, no finished smoke, or already-ended training tasks."""
    L = ledger(str(jid))
    if not L or any(t["kind"] == "main" for t in L) or not [t for t in L if t["task"].startswith("smoke")]:
        return None
    return max(int(t["end"]) for t in L if t["task"].startswith("smoke"))


def run_hours(rid, chain=()):
    """Wall hours of every training task of a run that counts (OK_EXIT), over all ledgers, plus the time so far of a
    chain job that is training now."""
    done = sum(int(t["end"]) - int(t["start"]) for t in TASKS if t["run"] == rid and t["kind"] == "main" and t["exit"] in OK_EXIT)
    live = sum(NOW - ts for ts in (train_start(j) for j in chain if sacct(j)[0] == "RUNNING") if ts is not None)
    return (done + live) / 3600


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
    elif kind == "chain":
        done = run is not None and run.state == "finished" and int(run.summary.get("_step", -1)) == int(run.config["trainer"]["num_train_steps"]) - 1
        hours = run_hours(rid, ref); live = [st for st in (sacct(j)[0] for j in ref) if st in ("RUNNING", "PENDING")]
        # Running: a chain job trains now, or the run is past 10% of its steps and its next job waits (between 24 h jobs)
        under_way = run is not None and int(run.summary.get("_step", 0)) >= 0.1 * int(run.config["trainer"]["num_train_steps"])
        state = "COMPLETED" if done else ("RUNNING" if "RUNNING" in live or under_way else "PENDING") if live else "failed"
    elif kind == "slurm" and str(ref) in FETCH:   # run by a host job (fetch_pending.sh)
        F = FETCH[str(ref)]
        state = "COMPLETED" if F["exit"] == "0" else "planned"
        hours = sum(int(t["end"]) - int(t["start"]) for t in TASKS if t["run"] == rid and t["kind"] == "main" and t["exit"] in OK_EXIT) / 3600 if state == "COMPLETED" else 0.0
        if state == "planned":
            run = None
    elif kind == "slurm":
        try:
            state, hours = sacct(ref)
        except RuntimeError:   # an array task still pending has no sacct line of its own
            if "_" not in str(ref) or str(ref).split("_")[0] not in subprocess.run(["squeue", "-h", "-r", "-o", "%i"], capture_output=True, text=True).stdout.replace("_", " ").split():
                raise
            state, hours = "PENDING", 0.0
        if state not in ("COMPLETED", "RUNNING", "PENDING") or ref in RERUN:   # did not finish, or replaced: the row waits for its replacement
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


LOGS = Path("/scratch/gpfs/GROUP/USER/project/marin-pls/logs/slurm")


def log_runs(jid: str) -> set:
    """W&B run ids named in a job's Slurm log (empty when the log is gone)."""
    a, _, t = jid.partition("_")
    files = list(LOGS.glob(f"*_{a}_{t}.out")) if t else list(LOGS.glob(f"*_{a}.out")) + list(LOGS.glob(f"*_{a}_[ab].log"))   # a pair job logs each run apart
    return {m for f in files for m in re.findall(r"wandb\.ai/reself/marin-della/runs/([A-Za-z0-9_.-]+)", f.read_text(errors="replace"))}


MAIN = {}   # job id -> run id of its row (a pair job serves two rows; any one of them names it)
for (setup, size, layers, tok, gpus, src, figs), row in zip(RUNS, rows):
    if row["state"]["en"] in ("Planned", "Queued"):   # no job has run for it yet (a queued array task has no sacct line)
        continue
    rid = src[1] if src[0] == "wandb" else src[2]
    chain = [j for j in src[1] if sacct(j)[0] != "PENDING"] if src[0] == "chain" else []
    for j in ([src[1]] if src[0] == "slurm" else []) + chain + JOBS.get(rid, []):
        MAIN[str(j)] = rid
for j, rid in MAIN.items():
    seen = log_runs(j)
    if seen and P.split("/")[-1] + rid not in seen:
        raise RuntimeError(f"job {j} is listed for run {rid}, but its log names {sorted(seen)}")
missing = [rid for (_, _, _, _, _, src, _), row in zip(RUNS, rows) if row["state"]["en"] == "Done" and src[0] == "wandb" and src[1] not in JOBS and src[1] not in ("130m-della4xh100", "300m-della4xh100")]
if missing:
    raise RuntimeError(f"finished rows without their Slurm jobs in JOBS: {missing}")
jobs = project_jobs()
absent = sorted(set(MAIN) - {j for j, *_ in jobs} - set(FETCH))
if [j for j in absent if sacct(j.split("_")[0])[0] != "PENDING"]:
    raise RuntimeError(f"jobs listed for rows are not among the project's GPU jobs: {absent}")
# H100-hours by line (template dcfd5a1, owner 2026-10-07). A plain job goes whole to main (it belongs to a row) or other.
# A packed job's sacct GPU-hours split by its ledger: each main task that exited 0 gives GPUs x wall time to main; smoke
# copies, failed tasks and the remainder (startup, gaps, idle GPUs) go to other; the job counts once, as main if any of
# its tasks is main.
ROWRUNS = {(src[1] if src[0] == "wandb" else src[2]) for (*_, src, _), row in zip(RUNS, rows) if row["state"]["en"] != "Planned"}
hours = {"main": 0.0, "other": 0.0}; njobs = {"main": 0, "other": 0}; total = 0.0
for jid, name, state, g, h in jobs:
    gh = g * h; total += gh; L = ledger(jid)
    if L is None:
        k = "main" if jid in MAIN else "other"; hours[k] += gh; njobs[k] += 1
        continue
    main_gh = sum(int(t["gpus"]) * (int(t["end"]) - int(t["start"])) / 3600 for t in L if t["kind"] == "main" and t["exit"] in OK_EXIT and t["run"] in ROWRUNS)
    ts = train_start(jid) if state == "RUNNING" and jid in MAIN else None
    if ts is not None:   # a pair job training now: both runs since the smokes ended (at most what sacct has counted)
        main_gh += min(g * (NOW - ts) / 3600, gh - main_gh)
    if main_gh > gh + 1e-6:
        raise RuntimeError(f"packed job {jid}: ledger main tasks {main_gh:.3f} GPU-h exceed sacct {gh:.3f}")
    hours["main"] += main_gh; hours["other"] += gh - main_gh
    njobs["main" if main_gh > 0 else "other"] += 1
    print(f"packed job {jid} {name}: sacct {gh:.2f} GPU-h -> main {main_gh:.2f}, other {gh - main_gh:.2f}")
if abs(hours["main"] + hours["other"] - total) > 1e-6:
    raise RuntimeError(f"main + other {hours['main'] + hours['other']:.6f} != sacct total {total:.6f}")
# fetched runs whose host is another project's job: their GPU-hours are not in this project's sacct total
own = {j for j, *_ in jobs}; ext = 0.0
for fid, F in sorted(FETCH.items()):
    if F["host"] in own:
        continue
    req = subprocess.run(["sacct", "-n", "-X", "-P", "-j", fid, "-o", "ReqTRES"], capture_output=True, text=True, timeout=120).stdout
    if "gres/gpu=" not in req:
        raise RuntimeError(f"fetched job {fid}: no GPU count in sacct ReqTRES {req!r}")
    g = int(req.split("gres/gpu=")[1].split(",")[0].split()[0])
    gh = g * (F["end"] - F["start"]) / 3600
    main_gh = sum(int(t["gpus"]) * (int(t["end"]) - int(t["start"])) / 3600 for t in TASKS if t["jid"] == F["host"] and t["kind"] == "main" and t["exit"] in OK_EXIT
                  and t["run"] in ROWRUNS and F["start"] <= int(t["start"]) and int(t["end"]) <= F["end"] + 60)
    main_gh = min(main_gh, gh)
    hours["main"] += main_gh; hours["other"] += gh - main_gh; ext += gh
    njobs["main" if main_gh > 0 else "other"] += 1
    print(f"fetched job {fid} in host {F['host']} (another project), exit {F['exit']}: {gh:.3f} GPU-h -> main {main_gh:.3f}, other {gh - main_gh:.3f}")
total += ext
tot = {k: dict(h100_hours=round(hours[k]), jobs=njobs[k]) for k in ("main", "other")}
print(f"check: main {hours['main']:.3f} + other {hours['other']:.3f} = {hours["main"] + hours["other"]:.3f} GPU-h = sacct total plus fetched runs in other projects' hosts {total:.3f}")
for jid, name, state, g, h in sorted(jobs, key=lambda x: x[0]):
    print(f"  {'packed' if ledger(jid) is not None else 'main' if jid in MAIN else 'other':10s} {jid:12s} {name:28s} {state:14s} {g} x {h:6.2f} h = {g * h:7.1f} H100-h")
# a row whose job is packed takes its own task's state and wall time, not the whole job's (audit 2026-10-07: a pair job
# that ends FAILED or exits 3 can hold one finished run)
for (setup, size, layers, tok, gpus, src, figs), row in zip(RUNS, rows):
    L = ledger(str(src[1])) if src[0] == "slurm" else None
    if L is None:
        continue
    mine = [t for t in L if t["run"] == src[2]]
    if mine:
        row["hours"] = round(sum(int(t["end"]) - int(t["start"]) for t in mine) / 3600, 1)
        if mine[-1]["exit"] == "0":   # finished: links as for any finished run
            run = wandb_run(src[2])
            row.update(state=STATE["COMPLETED"], url=WB + src[2] if run is not None else "", code=code_url(run))
        else:                         # stopped or failed: the row waits for its rerun; its time went to other
            row.update(state=STATE["planned"], url="", code="", hours=0.0)
    elif str(src[1]) not in FETCH and sacct(src[1])[0] not in ("PENDING", "RUNNING"):   # the job ended before this task did: rerun
        row.update(state=STATE["planned"], url="", code="", hours=0.0)
rec = dict(rows=rows, total=tot)
OUT.write_text("/* Generated by build_record.py (pls branch, docs/della/pls_archive/blog); regenerate rather than hand-edit. */\nwindow.PLC_RECORD = " + json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + ";\n")
print("rows", len(rows), "| main runs", tot["main"], "| other compute", tot["other"], "->", OUT)
