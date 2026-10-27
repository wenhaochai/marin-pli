"""Experiment record of wenhaochai.com/exps/per-layer-contribution.html -> assets/data/plc-record.js (generated; never
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
    (PROBE, "130m", 6, 2.6, 4, ("wandb", "130m-della4xh100-pls1-sep-bbfrozen"), "3, 12, 14, 16"),
    *[(DEPTH, "130m", d, 2.6, 4, ("wandb", f"130m-della4xh100-d{d}"), "2, 3, 12, 13, 14") for d in range(1, 6)],
    (BASE, "300m", 12, 6.0, 4, ("wandb", "300m-della4xh100"), "4, 5"),
    (SHARED, "300m", 12, 6.0, 4, ("wandb", "300m-della4xh100-pls1"), "5"),
    (SEP, "300m", 12, 6.0, 4, ("wandb", "300m-della4xh100-pls1-sep"), "4, 5"),
    (PROBE, "300m", 12, 6.0, 4, ("wandb", "300m-della4xh100-pls1-sep-bbfrozen"), "5"),
    # 48 layers (-vs-d48-g4): each design's pair-job chain; the earlier 8-GPU FSDP runs (-d48) are superseded, so other
    *[(n if tag == "" else S(f"{a}: {n['en'].lower()}", f"{a}：{n['zh']}"), "130m", 48, 2.6, 4,
       ("chain", jobs, f"130m-della4xh100-pls1-sep{bb}{'-a' + tag if tag else ''}-vs-d48-g4"), "6, 9, 10, 11, 16" if tag == "" else ("7, 8, 17, 44" if tag == "attnres" else ("7, 9, 10, 17, 44" if tag == "attnres_block" else ("7, 9, 10, 17" if tag == "moda" else ("7, 9, 10, 11, 17, 43" if tag == "hc" else "7, 9, 10, 11, 17")))))
      for a, tag, jobs in [("Sandwich-LN", "", [15266143, 15266144, 15273924]), ("Pre-LN", "preln", [15266145, 15266146, 15273925]),
                           ("LayerNorm Scaling", "lns", [15266147, 15266148]), ("DeepNorm", "deepnorm", [15266149, 15266150]), ("KEEL", "keel", [15266151, 15266152]),
                           ("Hyper-Connections", "hc", [15266153, 15266154, 15341053, 15341054]), ("mHC", "mhc", [15266155, 15266156, 15341055, 15341056]),
                           ("AttnRes (Full)", "attnres", [15266159, 15266160, 15266161]), ("AttnRes (Block)", "attnres_block", [15266157, 15266158]),
                           ("MoDA", "moda", [15266162, 15266163, 15266164, 15341589])]
      for n, bb in ((SEP, ""), (PROBE, "-bbfrozen"))],
    # AttnRes (Full), second init seed (Figure 8; owner popup 2026-10-13)
    *[(S(f"AttnRes (Full), seed 1: {n['en'].lower()}", f"AttnRes (Full)，种子 1：{n['zh']}"), "130m", 48, 2.6, 4,
       ("chain", [15520124, 15520125, 15520126, 15538869], f"130m-della4xh100-pls1-sep{bb}-aattnres-vs-d48-s1-g4"), "8, 44") for n, bb in ((SEP, ""), (PROBE, "-bbfrozen"))],
    # HC with positive writes (Figure 11; owner popup 2026-10-14): the causal test of HC's half last-layer cost
    *[(S(f"HC with positive writes: {n['en'].lower()}", f"写入为正的 HC：{n['zh']}"), "130m", 48, 2.6, 4,
       ("chain", [15536266, 15536267], f"130m-della4xh100-pls1-sep{bb}-ahc_pos-vs-d48-g4"), "11") for n, bb in ((SEP, ""), (PROBE, "-bbfrozen"))],
    # ordinary models of 12, 24 and 36 layers, the same-depth references for Figure 6 (owner popup 2026-10-11)
    *[(DEPTH, "130m", d, 2.6, 4, ("slurm", f"15519767_{d}", f"130m-della4xh100-d{d}"), "6, 16") for d in (12, 24, 36)],
    # multi-token prediction (Figure 16; owner popup 2026-10-13): one pair job, both runs with probes for x_{t+1} and x_{t+2}
    (S("NTP, probes for the next two tokens", "NTP，预测后两个词的探针"), "130m", 6, 2.6, 4, ("chain", [15523669], "130m-della4xh100-pls1-sep-bbfrozen-os-fut"), "19, 20"),
    (S("NTP + MTP, probes for the next two tokens", "NTP + MTP，预测后两个词的探针"), "130m", 6, 2.6, 4, ("chain", [15523669], "130m-della4xh100-pls1-sep-bbfrozen-os-mtp0.3-fut"), "19, 20"),
    # MTP across model sizes (Figures 17 and 18; owner popup 2026-10-14). The 130m point is the Figure 16 pair above. Each
    # larger size trains NTP and NTP + MTP 0.3 as the baseline's training, 8 GPUs per run, one afterany chain each (the
    # first 520m NTP job failed its smoke and the first 1_2b NTP job crashed in training, both out of GPU memory: other).
    *[(S(f"{a} as the baseline", f"{a}，按基线训练"), size, layers, tok, 8, ("chain", jobs, f"{size}-della8xh100-pls0-noev{tag}"), "20")
      for a, tag, runs in [("NTP", "", [("300m", 12, 6.0, [15591470, 15591471]), ("520m", 24, 10.4, [15591472, 15591473, 15664536]), ("1_2b", 16, 24.0, [15591482, 15591483, 15591484])]),
                           ("NTP + MTP", "-mtp0.3", [("300m", 12, 6.0, [15591485, 15591486]), ("520m", 24, 10.4, [15591490, 15591491]), ("1_2b", 16, 24.0, [15591492, 15591506, 15591507])])]
      for size, layers, tok, jobs in runs],
    # post-hoc probes of the finished models (Figure 18): fresh next-token and x_{t+2} heads on the frozen final checkpoint,
    # 1B tokens each. The 520m and 1_2b probes are submitted when their training runs finish.
    *[(S(f"Probes on the finished {a} model", f"在训练完的 {a} 模型上加探针"), size, layers, 1.0, 8,
       ("chain", jid, rid) if isinstance(jid, list) else ("wq", jid, rid) if str(jid).startswith("w") else ("slurm", jid, rid) if jid else ("planned", None, rid), "21")
      for a, size, layers, jid, rid in [
          ("NTP", "130m", 6, 15591508, "130m-della8xh100-pls1-sep-bbfrozen-os-fut-probes2000at4958"),
          ("NTP + MTP", "130m", 6, 15591509, "130m-della8xh100-pls1-sep-bbfrozen-os-mtp0.3-fut-probes2000at4958"),
          ("NTP", "300m", 12, 15714847, "300m-della8xh100-pls0-noev-probes2000at11443"),
          ("NTP + MTP", "300m", 12, 15690439, "300m-della8xh100-pls0-noev-mtp0.3-probes2000at11443"),
          # 520m: the first probe runs (15716064 -> 15738658, 15699612 -> 15738657) read the frozen head at 9.8 nats under
          # FSDP (2026-10-21); their jobs are other compute, and the reruns keep every layer whole on each GPU (-vs)
          ("NTP", "520m", 24, "w12", "520m-della8xh100-pls0-noev-vs-probes1000at9917"),   # wq (was Slurm 15952439)
          ("NTP + MTP", "520m", 24, "w13", "520m-della8xh100-pls0-noev-mtp0.3-vs-probes1000at9917"),   # wq (was 15952440)
          ("NTP", "1_2b", 16, [15760472, 15760473, 15832439, 15907475], "1_2b-della8xh100-pls0-noev-probes1000at22887"),
          ("NTP + MTP", "1_2b", 16, [15779949, 15779950, 15832441, 15907476, 15907916], "1_2b-della8xh100-pls0-noev-mtp0.3-probes1000at22887")]],
    # Lean follow-ups at 48 layers (owner popups 2026-10-21): one new arm each beside the -vs-d48-g4 ordinary controls.
    # hc and attnres run on 4 GPUs like their controls: on 8 their first step aborted in XLA (w16, w17, other compute).
    *[(S(en, zh), "130m", 48, 2.6, g, ("wq", jids, f"130m-della{g}xh100-pls1-sep-bbfrozen-{tag}-vs-d48"), fig) for en, zh, g, jids, tag, fig in [
        ("Pre-LN, pre-norm scales start at 1/\u221al: probes only", "Pre-LN，pre-norm 增益初始化为 1/\u221al：只加探针", 8, "w15", "apreln-gilns", "36"),
        ("Pre-LN with LNS's start, step size and clip: probes only", "Pre-LN，取 LNS 的初始值、步长和裁剪：只加探针", 8, "w61", "apreln-gilnsbf-slns", "36"),
        ("Pre-LN with LNS's step size and clip, scales start at 1: probes only", "Pre-LN，取 LNS 的步长和裁剪，增益初始化为 1：只加探针", 8, "w62", "apreln-slns", "36"),
        ("Hyper-Connections, static routing: probes only", "Hyper-Connections，静态路由：只加探针", 4, "w27", "ahc-rstatic", "37"),
        ("AttnRes (Full), static routing: probes only", "AttnRes (Full)，静态路由：只加探针", 4, "w28", "aattnres-rstatic", "37")]],
    # owner popups 2026-10-22: design variants (Figure 38) and the HC depth sweep (Figure 39), pls_arch3_fetchable.sbatch
    *[(S(en + (": separate heads" if bb == "" else ": probes only"), zh + ("：独立头" if bb == "" else "：只加探针")), "130m", L, 2.6, 4,
       ("wq", jid, f"130m-della4xh100-pls1-sep{bb}-{tag}-vs-d{L}"), fig) for en, zh, L, tag, fig, pair in [
        ("mHC, residual mixing fixed to I", "mHC，残差混合固定为 I", 48, "amhc-mhcid", "38, 47", ("w76", "w77")),
        ("AttnRes (Full), pseudo-query scale clipped to 10", "AttnRes (Full)，伪查询尺度裁剪到 10", 48, "aattnres-qclip10", "38", ("w78", "w79")),
        ("Hyper-Connections, 24 layers", "Hyper-Connections，24 层", 24, "ahc", "39", ("w80", "w81")),
        ("Hyper-Connections, 12 layers", "Hyper-Connections，12 层", 12, "ahc", "39", ("w82", "w83")),
        # owner popup 2026-10-24: sources at unit RMS in every mix (Figure 41)
        ("AttnRes (Full), sources at unit RMS", "AttnRes (Full)，源归一化到单位 RMS", 48, "aattnres-vnorm", "41", ("w122", "w123")),
        # owner popup 2026-10-25: Pre-LN at 12 and 24 layers (Figure 39)
        ("Pre-LN, 12 layers", "Pre-LN，12 层", 12, "apreln", "39", ("w147", "w148")),
        ("Pre-LN, 24 layers", "Pre-LN，24 层", 24, "apreln", "39", ("w149", "w150"))] for bb, jid in (("", pair[0]), ("-bbfrozen", pair[1]))],
    # owner popup 2026-10-25: separate heads with static routing; their probes runs are w27 / w28 (Figure 37)
    *[(S(en, zh), "130m", 48, 2.6, 4, ("wq", jid, f"130m-della4xh100-pls1-sep-{tag}-vs-d48"), "37") for en, zh, jid, tag in [
        ("Hyper-Connections, static routing: separate heads", "Hyper-Connections，静态路由：独立头", "w151", "ahc-rstatic"),
        ("AttnRes (Full), static routing: separate heads", "AttnRes (Full)，静态路由：独立头", "w152", "aattnres-rstatic")]],
    # owner popup 2026-10-25: every intermediate loss at weight 1/47 (Figure 42); control: the Pre-LN probes run
    (S("Pre-LN, separate heads, intermediate weight 1/47", "Pre-LN，独立头，中间层权重 1/47"), "130m", 48, 2.6, 4, ("wq", "w154", "130m-della4xh100-pls0.0212766-sep-apreln-vs-d48"), "42"),
    # owner popup 2026-10-25 (13:40): HC 48 layers seed 1 (Figure 39) and Pre-LN readouts that see 1/8 (Figure 40)
    (S("Hyper-Connections, seed 1: separate heads", "Hyper-Connections，种子 1：独立头"), "130m", 48, 2.6, 4, ("wq", "w178", "130m-della4xh100-pls1-sep-ahc-vs-d48-s1"), "39"),
    (S("Hyper-Connections, seed 1: probes only", "Hyper-Connections，种子 1：只加探针"), "130m", 48, 2.6, 4, ("wq", "w179", "130m-della4xh100-pls1-sep-bbfrozen-ahc-vs-d48-s1"), "39"),
    (S("Pre-LN, separate heads; intermediate readouts see 1/8 of the coordinates", "Pre-LN，独立头；中间层读出只看 1/8 的坐标"), "130m", 48, 2.6, 4, ("wq", "w180", "130m-della4xh100-pls1-sep-apreln-vis0.125-vs-d48"), "40"),
    # owner popup 2026-10-25 (16:30): HC 48 layers with 1 / 2 residual streams instead of 4 (Figure 43)
    (S("Hyper-Connections, 1 residual stream: separate heads", "Hyper-Connections，1 条残差流：独立头"), "130m", 48, 2.6, 4, ("wq", "w194", "130m-della4xh100-pls1-sep-ahc-hcs1-vs-d48"), "43"),
    (S("Hyper-Connections, 1 residual stream: probes only", "Hyper-Connections，1 条残差流：只加探针"), "130m", 48, 2.6, 4, ("wq", "w195", "130m-della4xh100-pls1-sep-bbfrozen-ahc-hcs1-vs-d48"), "43"),
    (S("Hyper-Connections, 2 residual streams: separate heads", "Hyper-Connections，2 条残差流：独立头"), "130m", 48, 2.6, 4, ("wq", "w196", "130m-della4xh100-pls1-sep-ahc-hcs2-vs-d48"), "43"),
    (S("Hyper-Connections, 2 residual streams: probes only", "Hyper-Connections，2 条残差流：只加探针"), "130m", 48, 2.6, 4, ("wq", "w197", "130m-della4xh100-pls1-sep-bbfrozen-ahc-hcs2-vs-d48"), "43"),
    # queue fill (owner 2026-10-25 via terminal, no popup for these four; placeholders: owner popup 2026-10-25): HC with 8
    # streams (Figure 43) and AttnRes Block with 24 blocks (Figure 44), each against its own probes run
    (S("Hyper-Connections, 8 residual streams: separate heads", "Hyper-Connections，8 条残差流：独立头"), "130m", 48, 2.6, 4, ("wq", "w221", "130m-della4xh100-pls1-sep-ahc-hcs8-vs-d48"), "43"),
    (S("Hyper-Connections, 8 residual streams: probes only", "Hyper-Connections，8 条残差流：只加探针"), "130m", 48, 2.6, 4, ("wq", "w222", "130m-della4xh100-pls1-sep-bbfrozen-ahc-hcs8-vs-d48"), "43"),
    (S("AttnRes (Block), 24 blocks: separate heads", "AttnRes (Block)，24 个块：独立头"), "130m", 48, 2.6, 4, ("wq", "w223", "130m-della4xh100-pls1-sep-aattnres_block-arb24-vs-d48"), "44"),
    (S("AttnRes (Block), 24 blocks: probes only", "AttnRes (Block)，24 个块：只加探针"), "130m", 48, 2.6, 4, ("wq", "w224", "130m-della4xh100-pls1-sep-bbfrozen-aattnres_block-arb24-vs-d48"), "44"),
    # owner popup 2026-10-26 (~06:40): the speed stack (FA4 THD attention, lse kernel) checked on Pre-LN 48 layers before new
    # pairs use it; its cost must match the JAX-flash 0.089 / 0.092 within seed noise (replaces the fa4 pair w261/w262)
    (S("Pre-LN, new speed stack (FA4 attention, fused CE kernel): separate heads", "Pre-LN，新加速组合（FA4 attention、融合 CE kernel）：独立头"), "130m", 48, 2.6, 4, ("wq", "w333", "130m-della4xh100-pls1-sep-apreln-fa4thd-vs-d48"), ""),
    (S("Pre-LN, new speed stack (FA4 attention, fused CE kernel): probes only", "Pre-LN，新加速组合（FA4 attention、融合 CE kernel）：只加探针"), "130m", 48, 2.6, 4, ("wq", "w334", "130m-della4xh100-pls1-sep-bbfrozen-apreln-fa4thd-vs-d48"), ""),
    # owner popup 2026-10-26 (~16:40): HC on the new CE path (Figure 39) and HC with mHC's read / write maps (Figure 38)
    (S("Hyper-Connections, new CE path: separate heads", "Hyper-Connections，新 CE 路径：独立头"), "130m", 48, 2.6, 4, ("wq", "w387", "130m-della4xh100-pls1-sep-ahc-fa4thd-vs-d48"), "39"),
    (S("Hyper-Connections, new CE path: probes only", "Hyper-Connections，新 CE 路径：只加探针"), "130m", 48, 2.6, 4, ("wq", "w388", "130m-della4xh100-pls1-sep-bbfrozen-ahc-fa4thd-vs-d48"), "39"),
    (S("Hyper-Connections, read/write = sigmoid: separate heads", "Hyper-Connections，读写 = sigmoid：独立头"), "130m", 48, 2.6, 4, ("wq", "w389", "130m-della4xh100-pls1-sep-ahc-hcrwsig-vs-d48"), "38"),
    (S("Hyper-Connections, read/write = sigmoid: probes only", "Hyper-Connections，读写 = sigmoid：只加探针"), "130m", 48, 2.6, 4, ("wq", "w390", "130m-della4xh100-pls1-sep-bbfrozen-ahc-hcrwsig-vs-d48"), "38"),
    # owner popups 2026-10-27: reruns of the Pre-LN probes model on both attention kernels (Figure 45); separate heads whose
    # intermediate losses stop at 50% / 80% of training (Figure 46)
    (S("Pre-LN, rerun of seed 0 on JAX-flash attention: probes only", "Pre-LN，种子 0 在 JAX flash 注意力上重跑：只加探针"), "130m", 48, 2.6, 4, ("wq", "w500", "130m-della4xh100-pls1-sep-bbfrozen-apreln-vs-d48-rerun"), "45"),
    (S("Pre-LN, rerun of seed 0 on fa4thd attention: probes only", "Pre-LN，种子 0 在 fa4thd 注意力上重跑：只加探针"), "130m", 48, 2.6, 4, ("wq", "w501", "130m-della4xh100-pls1-sep-bbfrozen-apreln-fa4thd-vs-d48-rerun"), "45"),
    (S("Pre-LN, separate heads stopped at 50% of training", "Pre-LN，独立头损失在训练 50% 处停掉"), "130m", 48, 2.6, 4, ("wq", "w503", "130m-della4xh100-pls1-sep-off2480-apreln-vs-d48"), "46"),
    (S("Pre-LN, separate heads stopped at 80% of training", "Pre-LN，独立头损失在训练 80% 处停掉"), "130m", 48, 2.6, 4, ("wq", "w504", "130m-della4xh100-pls1-sep-off3967-apreln-vs-d48"), "46"),
    (S("Hyper-Connections, separate heads stopped at 80% of training", "Hyper-Connections，独立头损失在训练 80% 处停掉"), "130m", 48, 2.6, 4, ("wq", "w505", "130m-della4xh100-pls1-sep-off3967-ahc-vs-d48"), "46"),
    # owner popups 2026-10-23: readouts that see part of the state, and Pre-LN seed 1 (Figure 40)
    *[(S(en, zh), "130m", 48, 2.6, 4, ("wq", jid, f"130m-della4xh100-pls1-sep{rid}"), "40") for en, zh, jid, rid in [
        ("Pre-LN, separate heads; intermediate readouts see 1/2 of the coordinates", "Pre-LN，独立头；中间层读出只看 1/2 的坐标", "w87", "-apreln-vis0.5-vs-d48"),
        ("Pre-LN, separate heads; intermediate readouts see 1/4 of the coordinates", "Pre-LN，独立头；中间层读出只看 1/4 的坐标", "w88", "-apreln-vis0.25-vs-d48"),
        ("Pre-LN, seed 1: separate heads", "Pre-LN，种子 1：独立头", "w89", "-apreln-vs-d48-s1"),
        ("Pre-LN, seed 1: probes only", "Pre-LN，种子 1：只加探针", "w90", "-bbfrozen-apreln-vs-d48-s1")]],
    (S("Probes on the finished NTP model, weights kept", "在训练完的 NTP 模型上加探针，保留权重"), "130m", 6, 1.0, 8, ("wq", "w14", "130m-della8xh100-pls1-sep-bbfrozen-os-fut-probes2000at4958-keep"), "22"),
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
    (S("Separate heads, heads retrained (frozen backbone)", "独立头，重训读出头（冻结骨干）"), "300m", 12, 1.0, 4, ("slurm", 15011512, "300m-della4xh100-pls1-sep-headft2000"), "18"),
    (S("Probes only, heads retrained (frozen backbone)", "只加探针，重训读出头（冻结骨干）"), "300m", 12, 1.0, 4, ("slurm", 15011513, "300m-della4xh100-pls1-sep-bbfrozen-headft2000"), "18"),
    # the 130m separate heads at weight 1 again, for its final checkpoint (Figure 16, the logit lens; owner popups 2026-10-17)
    (S("Separate heads, rerun for its checkpoint", "独立头，为保留 checkpoint 重跑"), "130m", 6, 2.6, 8, ("slurm", 15720165, "130m-della8xh100-pls1-sep"), "16"),
    # the logit lens (Figures 16, 17; owner popups 2026-10-17, 18) and the Kimi K3 lens (Figure 22): eval-only jobs, one row
    # per job; hours from the task ledgers (run column = the group); the link opens one of the job's runs, its group's others
    # sit beside it on W&B (-lensat<step> in the name)
    *[(S(en, zh), size, layers, 0.0, 8, ("lens", jid, grp, link), figs) for en, zh, size, layers, jid, grp, link, figs in [
        ("Logit lens of 20 finished 6-layer models (eval)", "20 个训练完的 6 层模型的 logit lens（评估）", "130m", 6, 15720392, "lens-130m", "130m-della8xh100-pls1-sep-bbfrozen-lensat4958", "16"),
        ("Logit lens of the 6-layer separate-heads model (eval)", "6 层独立头模型的 logit lens（评估）", "130m", 6, 15787177, "lens-130sep", "130m-della8xh100-pls1-sep-lensat4958", "16"),
        ("Logit lens, 48 layers: five designs, both setups, and ordinary models of 12-36 layers (eval)", "logit lens，48 层：五种架构的两种设置，以及 12-36 层普通模型（评估）", "130m", 48, 15720393, "lens-48a", "130m-della8xh100-pls1-sep-vs-d48-lensat4958-g4", "16, 17"),
        ("Logit lens, 48 layers: Hyper-Connections and mHC, both setups (eval)", "logit lens，48 层：Hyper-Connections 和 mHC 的两种设置（评估）", "130m", 48, 15720394, "lens-48b", "130m-della8xh100-pls1-sep-ahc-vs-d48-lensat4958-g4", "17"),
        ("Logit lens, 48 layers: AttnRes (Full), AttnRes (Block), MoDA, both setups (eval)", "logit lens，48 层：AttnRes (Full)、AttnRes (Block)、MoDA 的两种设置（评估）", "130m", 48, 15795008, "lens-48c", "130m-della8xh100-pls1-sep-aattnres-vs-d48-lensat4958-g4", "17"),
        ("Logit lens of the 300m models and the 300m and 520m NTP / MTP models (eval)", "300m 模型以及 300m、520m NTP / MTP 模型的 logit lens（评估）", "300m", 12, 15720395, "lens-300m", "300m-della8xh100-pls1-sep-lensat11443", "16"),
        ("Kimi K3 logit lens (eval, 64 x 1,024 tokens)", "Kimi K3 的 logit lens（评估，64 x 1,024 个 token）", "K3", 93, 15736820, "k3-lens", "", "23"),
        ("Kimi K3 AttnRes check-up: smoke (eval)", "Kimi K3 AttnRes 体检：冒烟测试（评估）", "K3", 93, 15817062, "k3-checkup-smoke", "", "24-29"),
        ("Kimi K3 AttnRes check-up (eval, Paloma C4, 64 x 1,024 tokens, 12 ablations)", "Kimi K3 AttnRes 体检（评估，Paloma C4，64 x 1,024 个 token，12 种消融）", "K3", 93, 15817869, "k3-checkup", "", "24-29"),
        ("Kimi K3 AttnRes check-up v2: smoke, one read per layer (eval)", "Kimi K3 AttnRes 体检 v2：冒烟测试，每层只读一次（评估）", "K3", 93, 15844464, "k3-checkup-v2-smoke", "", "24-30"),
        ("Kimi K3 AttnRes check-up v2 (eval, Paloma C4, 128 x 1,024 tokens, 12 ablations)", "Kimi K3 AttnRes 体检 v2（评估，Paloma C4，128 x 1,024 个 token，12 种消融）", "K3", 93, 15844650, "k3-checkup-v2", "", "24-30"),
        ("Kimi K3 embedding split (eval, Paloma C4, 128 x 1,024 tokens, 11 variants)", "Kimi K3 embedding 拆分（评估，Paloma C4，128 x 1,024 个 token，11 种变体）", "K3", 93, 15855315, "k3-checkup-v3", "", "30"),
        ("48 layers, ten designs: internal check-up (eval, Paloma C4)", "48 层十种结构：内部体检（评估，Paloma C4）", "130m", 48, 15844436, "depthmed", "", "31-35"),
        ("48 layers, AttnRes (Full): internal check-up (eval)", "48 层 AttnRes (Full)：内部体检（评估）", "130m", 48, 15849760, "depthmed-attnres", "", "31-35"),
        ("48 layers, ten designs: true layer deletion (eval)", "48 层十种结构：真正删层（评估）", "130m", 48, 15855981, "depthmed-delete", "", "31")]],
    # ten more attempts at the last-layer cost (Figure 15; owner popup 2026-10-16), branch pls-auto10: two 130m runs per
    # 8-GPU pair job (pls_auto10_pair.sbatch, fetchable); the job ids join as the pairs are submitted
    *[(S(en, zh), "130m", 6, 2.6, 4, ("chain", jid, rid) if isinstance(jid, list) else ("slurm", jid, rid) if jid else ("planned", None, rid), "15") for en, zh, jid, rid in [
        ("Separate heads, the readouts clipped on their own", "独立头，读出头单独裁剪", 15698932, "130m-della4xh100-pls1-sep-os"),
        ("Separate heads, normalise then mix at 0.2", "独立头，先正交化再按 0.2 混合", 15700045, "130m-della4xh100-pls1-sep-mix0.2"),
        ("Separate heads learning the final layer's distribution", "独立头学最终层的分布", [15699791, 15755719], "130m-della4xh100-pls1-sep-kd"),
        ("Separate heads weighted by the final layer's confidence", "独立头按最终层置信度加权", 15698936, "130m-della4xh100-pls1-sep-conf"),
        ("Separate heads with exit adapters", "独立头加出口适配器", 15699791, "130m-della4xh100-pls1-sep-ad1024"),
        ("Separate heads training MLPs only", "独立头只更新 MLP", 15700045, "130m-della4xh100-pls1-sep-mlpo"),
        ("Separate heads on the first 256 positions", "独立头只用前 256 个位置", 15698936, "130m-della4xh100-pls1-sep-first256"),
        ("Separate heads, one random layer per step", "独立头，每步随机一层", 15698932, "130m-della4xh100-pls1-sep-randl"),
        ("Hyper-Connections: separate heads", "Hyper-Connections：独立头", 15700214, "130m-della4xh100-pls1-sep-ahc"),
        ("Hyper-Connections: probes only", "Hyper-Connections：只加探针", 15700214, "130m-della4xh100-pls1-sep-bbfrozen-ahc"),
        ("AttnRes (Full): separate heads", "AttnRes (Full)：独立头", 15699797, "130m-della4xh100-pls1-sep-aattnres"),
        ("AttnRes (Full): probes only", "AttnRes (Full)：只加探针", 15699797, "130m-della4xh100-pls1-sep-bbfrozen-aattnres"),
        ("Separate heads, only layers 1-3 reach the backbone", "独立头，只有第 1-3 层的梯度进入主干", 15772582, "130m-della4xh100-pls1-sep-upto3"),
        ("Separate heads, only layers 1-2 reach the backbone", "独立头，只有第 1-2 层的梯度进入主干", 15772582, "130m-della4xh100-pls1-sep-upto2"),
        ("Separate heads, only layer 1 reaches the backbone", "独立头，只有第 1 层的梯度进入主干", 15805655, "130m-della4xh100-pls1-sep-upto1"),
        ("Separate heads, only layers 1-2 reach the backbone, seed 1", "独立头，只有第 1-2 层的梯度进入主干，种子 1", 15805655, "130m-della4xh100-pls1-sep-upto2-s1")]],
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
         "COMPLETED": S("Done", "完成"), "RUNNING": S("Running", "运行中"), "PENDING": S("Queued", "排队中"), "planned": S("Planned", "计划中"),
         "STOPPED": S("Stopped", "已停止"),
         "PAUSED": S("Paused, resume queued", "暂停，续跑已排队"), "PARTLY": S("Partly done", "部分完成")}
OK_EXIT = {"0", "124", "143"}   # training-task exits that count for the run (see the docstring)
api = wandb.Api(timeout=300)


WQ = "/scratch/gpfs/GROUP/USER/tools/wq/wq"   # GPU jobs go to wq since 2026-10-21 (owner): ids w<n>, not in sacct


# wq jobs cancelled on the owner's decision, with no rerun: w79 (ordinary AttnRes, query clip) diverged at step 660
# (owner popup 2026-10-24, "取消 w79").
STOPPED = {"w79"}


def wq_state(wid):
    """A wq job's state as the Slurm word this file uses (queued -> PENDING, running -> RUNNING, done -> COMPLETED; failed,
    cancelled or lost -> failed); fails loudly when wq gives nothing."""
    res = subprocess.run([WQ, "state", str(wid)], capture_output=True, text=True, timeout=120)
    st = res.stdout.strip()
    if res.returncode != 0 or not st:
        raise RuntimeError(f"wq gave nothing for job {wid}: rc={res.returncode} stderr={res.stderr.strip()[:200]}")
    return {"queued": "PENDING", "running": "RUNNING", "done": "COMPLETED"}.get(st, "failed")


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
        if len(cols) != 5 or not all(c.lstrip("-").isdigit() for c in cols[1:]) or not re.fullmatch(r"w?\d+", cols[0]):
            raise RuntimeError(f"{f} line {n} is not a fetched line: {line!r}")
        if cols[0].startswith("w"):   # wq jobs: WQ_FETCH below
            continue
        fid, host, start, end, ex = cols
        out[fid] = dict(host=host, start=int(start), end=int(end), exit=ex)
    return out


FETCH = fetched()
# wq jobs (w<n>): fetch_pending.sh writes each run of one (a job can run several times) to fetched.tsv under the job's
# working directory, which was /home/USER for w15-w17. Every run of a wq job of project pls, from both files; the GPU
# count comes from its wq spec (no sacct for wq ids).
WQJOBS = Path("/scratch/gpfs/GROUP/USER/wq/jobs")
def wq_fetched():
    seen = []
    for f in (PACKED / "fetched.tsv", Path("/home/USER/logs/packed/fetched.tsv")):
        for line in (f.read_text().splitlines() if f.exists() else []):
            cols = line.split("\t")
            if len(cols) == 5 and re.fullmatch(r"w\d+", cols[0]) and line not in seen:
                spec = json.loads((WQJOBS / cols[0][1:] / "spec.json").read_text())
                if spec.get("project") == "pls":
                    seen.append(line)
    return [dict(fid=c[0], host=c[1], start=int(c[2]), end=int(c[3]), exit=c[4], gpus=int(json.loads((WQJOBS / c[0][1:] / "spec.json").read_text())["gpus"]))
            for c in (l.split("\t") for l in seen)]
WQ_FETCH = wq_fetched()
LOCKS = Path("/scratch/gpfs/GROUP/USER/tmp/fetch_locks")


def fetching(ref):
    """True while a host job runs this job: fetch_pending.sh cancelled it from the queue and keeps its lock dir with
    batch.sh (no UNCONFIRMED or RESUBMIT marker), and fetched.tsv has no line for it yet."""
    d = LOCKS / str(ref)
    return (str(ref) not in FETCH and (d / "batch.sh").exists() and not (d / "UNCONFIRMED").exists()
            and not (d / "RESUBMIT").exists() and sacct(ref)[0].startswith("CANCELLED"))
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


# Timeline (owner 2026-10-25, "主页exp record记录每条实验的时间线"): every row gets "submitted" and "segments"
# ([[start, end | null], ...], ISO 8601 in America/New_York, minute precision) for assets/record-timeline.js. Job ids per
# row: its wq id, its Slurm job (array task) or chain jobs, or JOBS for the early W&B rows; tools/job_segments.py reads wq's
# run history and sacct. A Slurm job that a host ran (fetched.tsv) is CANCELLED in sacct, so its segment is the host's
# start and end from fetched.tsv. The two baselines reused from project/marin have no job here: their segment is the W&B
# run's creation to its last heartbeat. Planned rows (no job yet, or one that did not finish) get none.
SEGTOOL = "/scratch/gpfs/GROUP/USER/tools/job_segments.py"
import calendar


def row_jobs(kind, ref, rid):
    if kind == "wandb":
        return [str(j) for j in JOBS.get(rid, [])]
    if kind == "chain":
        return [str(j) for j in ref]
    if kind in ("wq", "slurm", "lens"):
        return [str(ref)]
    return []


ALLJOBS = sorted({j for setup, size, layers, tok, gpus, src, figs in RUNS for j in row_jobs(src[0], src[1], src[1] if src[0] == "wandb" else src[2])})
_seg = subprocess.run(["python3", "-I", SEGTOOL, *ALLJOBS], capture_output=True, text=True, timeout=1800)
if _seg.returncode != 0:
    raise RuntimeError(f"job_segments.py failed: {_seg.stderr.strip()[:300]}")
SEGS = json.loads(_seg.stdout)
os.environ["TZ"] = "America/New_York"; time.tzset()
ny = lambda t: time.strftime("%Y-%m-%dT%H:%M%z", time.localtime(t))   # noqa: E731


def row_timeline(kind, ref, rid, run, state):
    """(submitted, segments) of a row; a row that ends Planned drops them below (its jobs are other compute)."""
    subs, segs = [], []
    for j in row_jobs(kind, ref, rid):
        if j in FETCH:   # run inside a host job: the host's window
            segs.append([ny(FETCH[j]["start"]), ny(FETCH[j]["end"])])
        info = SEGS.get(j, {})
        if info.get("error"):
            raise RuntimeError(f"job_segments.py: {j}: {info['error']}")
        if info.get("submitted"):
            subs.append(info["submitted"])
        if j not in FETCH:
            segs += [list(s) for s in info.get("segments", [])]
    if kind == "wandb" and not segs and run is not None:   # the reused baselines: W&B creation to last heartbeat
        utc = lambda u: calendar.timegm(time.strptime(u.rstrip("Z")[:19], "%Y-%m-%dT%H:%M:%S")) if u else None   # noqa: E731  (W&B times are UTC)
        c, h = utc(run.created_at), utc(getattr(run, "heartbeat_at", None) or run._attrs.get("heartbeatAt"))
        if c:
            segs = [[ny(c), ny(h) if h else None]]
    segs.sort(key=lambda s: s[0])
    return (min(subs) if subs else None), segs


rows = []
WB = "https://wandb.ai/reself/marin-della/runs/muonh-qwen3-"
CODE = "https://github.com/wenhaochai/marin-pli/tree/"
# private commit -> public commit, from the last scrub of the pls branch (tmp/pls_scan/scrub.sh, git filter-repo)
# The pls-mtp branch (w12-w14 ran from it) has its own scrub (scrub_mtp.sh), public branch per-layer-contribution-mtp;
# filter-repo maps the shared history to the same public commits, so the two maps merge.
CMAP = {k: v for f in ("pls_public", "pls_mtp_public")
        for k, v in (line.split()[:2] for line in open(f"/scratch/gpfs/GROUP/USER/tmp/pls_scan/{f}/.git/filter-repo/commit-map").read().splitlines()[1:])}


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
        hours = run_hours(rid, ref); live = [st for st in ("RUNNING" if fetching(j) else sacct(j)[0] for j in ref) if st in ("RUNNING", "PENDING")]   # a host job may run one
        # Running: a chain job trains now, or the run is past 10% of its steps and its next job waits (between 24 h jobs)
        under_way = run is not None and int(run.summary.get("_step", 0)) >= 0.1 * int(run.config["trainer"]["num_train_steps"])
        # Paused: under way, no chain job training now, the next one queued (controller audit 2026-10-18)
        state = "COMPLETED" if done else ("RUNNING" if "RUNNING" in live else "PAUSED" if under_way else "PENDING") if live else "failed"
    elif kind == "lens":   # an eval-only job of several checkpoints: ("lens", jobid, group = the ledger's run name, run id linked)
        st = ("COMPLETED" if FETCH[str(ref)]["exit"] == "0" else "FAILED") if str(ref) in FETCH else sacct(ref)[0]
        mine = [t for t in TASKS if t["run"] == rid and t["kind"] == "main"]
        hours = sum(int(t["end"]) - int(t["start"]) for t in mine if t["exit"] == "0") / 3600
        ok = any(t["exit"] == "0" for t in mine)
        if not mine and str(ref) not in FETCH:   # a plain job without a ledger (the K3 lens): the whole job is the row
            st, hours = sacct(ref); ok = st == "COMPLETED"
            hours = hours if st != "PENDING" else 0.0
        state = st if st in ("COMPLETED", "RUNNING", "PENDING") else ("PARTLY" if ok else "failed")
        run = wandb_run(src[3]) if src[3] and ok else None
        rid = src[3] or rid
    elif kind == "wq":   # a wq job (run by a pilot host): its training tasks sit in the host's ledger under the run id
        state = wq_state(ref)
        hours = sum(int(t["end"]) - int(t["start"]) for t in TASKS if t["run"] == rid and t["kind"] == "main" and t["exit"] in OK_EXIT) / 3600
        if state == "failed" and ref in STOPPED:   # cancelled on the owner's decision, not to be rerun: its hours and run stay
            state = "STOPPED"
        elif state == "failed":
            state, hours, run = "planned", 0.0, None
    elif kind == "slurm" and str(ref) in FETCH:   # run by a host job (fetch_pending.sh)
        F = FETCH[str(ref)]
        state = "COMPLETED" if F["exit"] == "0" else "planned"
        hours = sum(int(t["end"]) - int(t["start"]) for t in TASKS if t["run"] == rid and t["kind"] == "main" and t["exit"] in OK_EXIT) / 3600 if state == "COMPLETED" else 0.0
        if state == "planned":
            run = None
    elif kind == "slurm" and fetching(ref):   # a host job runs it now: no fetched.tsv line until it ends
        state, hours = "RUNNING", 0.0
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
    submitted, segments = row_timeline(kind, ref, rid, run, state)
    rows.append(dict(setup=setup, size=size, layers=layers, tokens=tok, gpus=gpus, hours=round(hours, 1), state=STATE.get(state, S(state, state)), figs=figs, url=url, code=code,
                     submitted=submitted, segments=segments))
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
    for j in ([src[1]] if src[0] in ("slurm", "lens") else []) + chain + JOBS.get(rid, []):
        MAIN[str(j)] = rid
LENSJOBS = {str(src[1]) for (*_, src, _) in RUNS if src[0] == "lens"}
for j, rid in MAIN.items():
    if j in LENSJOBS:
        continue
    seen = log_runs(j)
    if seen and P.split("/")[-1] + rid not in seen:
        raise RuntimeError(f"job {j} is listed for run {rid}, but its log names {sorted(seen)}")
missing = [rid for (_, _, _, _, _, src, _), row in zip(RUNS, rows) if row["state"]["en"] == "Done" and src[0] == "wandb" and src[1] not in JOBS and src[1] not in ("130m-della4xh100", "300m-della4xh100")]
if missing:
    raise RuntimeError(f"finished rows without their Slurm jobs in JOBS: {missing}")
jobs = project_jobs()
absent = sorted(set(MAIN) - {j for j, *_ in jobs} - set(FETCH) - {j for j in MAIN if fetching(j)})   # a fetched job still running has no sacct time yet
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
for F in WQ_FETCH:   # every run of a wq job; the hosts are pilots, never this project's jobs; its tasks sit in logs/packed/w<n>.tsv
    gh = F["gpus"] * (F["end"] - F["start"]) / 3600
    main_gh = sum(int(t["gpus"]) * (int(t["end"]) - int(t["start"])) / 3600 for t in TASKS if t["jid"] in (F["host"], F["fid"]) and t["kind"] == "main" and t["exit"] in OK_EXIT
                  and t["run"] in ROWRUNS and F["start"] <= int(t["start"]) and int(t["end"]) <= F["end"] + 60)
    main_gh = min(main_gh, gh)
    hours["main"] += main_gh; hours["other"] += gh - main_gh; ext += gh
    njobs["main" if main_gh > 0 else "other"] += 1
    print(f"wq job {F['fid']} run in host {F['host']}, exit {F['exit']}: {F['gpus']} GPUs, {gh:.3f} GPU-h -> main {main_gh:.3f}, other {gh - main_gh:.3f}")
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
# eval jobs that write no W&B run (the K3 jobs read a public model with their own scripts): the code link is the public
# commit of the script they ran (scrubbed marin-pli history, per-layer-contribution branch)
CODE_NO_WANDB = {15736820: "aa402e9c51347df74acecb31e042c928d2ec4495", 15817062: "8abc839e279992f660f7fc7765a3ec001d64b49b",
                 15817869: "8abc839e279992f660f7fc7765a3ec001d64b49b", 15795008: "7252c0f1148d25f06984a3854d5d8c4f8dbf8189",
                 15844464: "0b6172eea78e696a5261821c1fc528c43f060a51", 15844650: "0b6172eea78e696a5261821c1fc528c43f060a51",
                 15855315: "f5496bab4d52faffe1abf38f5b52a60d48ed2370", 15844436: "f9837f1f760d8e96334951354468b1fa4cbb1834",
                 15849760: "f9837f1f760d8e96334951354468b1fa4cbb1834", 15855981: "8e4ae34d48226eac341c48a4f165d8e25fb5905e"}
for (setup, size, layers, tok, gpus, src, figs), row in zip(RUNS, rows):
    if src[0] == "lens" and src[1] in CODE_NO_WANDB and not row["code"] and row["state"] != STATE["planned"]:
        row["code"] = CODE + CODE_NO_WANDB[src[1]]
    if src[0] == "lens" and src[1] in CODE_NO_WANDB and not row["url"] and row["state"] != STATE["planned"]:
        row["url"] = "none"   # no W&B run by design; the page shows a dash, the premerge check accepts it
def alloc_gpus(jid):
    """GPUs the job held (sacct AllocTRES), None while it has none (pending)."""
    out = subprocess.run(["sacct", "-n", "-X", "-P", "-j", str(jid), "-o", "AllocTRES"], capture_output=True, text=True, timeout=120).stdout
    for part in out.strip().split(","):
        if part.startswith("gres/gpu="):
            return int(part.split("=")[1])
    return None


# eval jobs: the row shows the GPUs the job held (the eval rows default to 8; the check-ups ran on 1 or 2)
for (setup, size, layers, tok, gpus, src, figs), row in zip(RUNS, rows):
    if src[0] == "lens":
        row["gpus"] = alloc_gpus(src[1]) or row["gpus"]
# a plain eval job (no ledger) whose sacct state is FAILED although most of its work finished: 15844436 wrote 18 of its
# 20 models before both AttnRes (Full) models ran out of memory (rerun 15849760)
PARTLY_JOBS = {15844436}
for (setup, size, layers, tok, gpus, src, figs), row in zip(RUNS, rows):
    if src[0] == "lens" and src[1] in PARTLY_JOBS and row["state"] == STATE["failed"]:
        row["state"] = STATE["PARTLY"]
# a row that ends Planned (no job yet, or one that did not finish) shows no timeline: its jobs are other compute
for row in rows:
    if row["state"] == STATE["planned"]:
        row["submitted"], row["segments"] = None, []
rec = dict(rows=rows, total=tot)
OUT.write_text("/* Generated by build_record.py (pls branch, docs/della/pls_archive/blog); regenerate rather than hand-edit. */\nwindow.PLC_RECORD = " + json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + ";\n")
print("rows", len(rows), "| main runs", tot["main"], "| other compute", tot["other"], "->", OUT)
