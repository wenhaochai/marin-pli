"""Spec check of packed-job plans (logs/plans/*.txt): every training task's own VAR=value set, given to the scaling
script's dry run, must give exactly the run id the plan declares (muonh-qwen3-<RUN>); smoke lines are skipped. Also
checks the line format (8+ fields, mode smoke|seq|par, kind main|other, GPUs 4 or 8, a timeout(1) duration) and that
two par tasks of one block use different GPUs. Run on a vis node: nice -n 19 .venv/bin/python scripts/della/check_plans.py [plan ...]"""
import glob
import os
import re
import subprocess
import sys

plans = sys.argv[1:] or sorted(glob.glob("logs/plans/*.txt"))
bad = 0
for plan in plans:
    block = []
    for n, line in enumerate(open(plan), 1):
        f = line.split()
        if not f or f[0].startswith("#"):
            continue
        if f[0] == "wait":
            gpus = [g for t in block for g in t]
            if len(gpus) != len(set(gpus)):
                print(f"BAD {plan}:{n}: par tasks share GPUs {gpus}"); bad += 1
            block = []
            continue
        mode, task, run, kind, ngpu, limit, script, *vars_ = f
        env = dict(v.replace("+", " ").split("=", 1) for v in vars_)
        if mode not in ("smoke", "seq", "par") or kind not in ("main", "other") or ngpu not in ("4", "8") or not re.fullmatch(r"\d+[smhd]?(\d+m)?", limit.replace("h", "h", 1)) and not re.fullmatch(r"\d+h\d+m", limit):
            print(f"BAD {plan}:{n}: fields {f[:7]}"); bad += 1
        if not os.path.exists(script):
            print(f"BAD {plan}:{n}: no script {script}"); bad += 1
        if mode == "par":
            block.append(env.get("CUDA_VISIBLE_DEVICES", "all").split(","))
        if mode == "smoke":
            continue
        out = subprocess.run([".venv/bin/python", "-m", "experiments.references.della_muonh_qwen3_scaling"], env={**os.environ, **env, "DRY_RUN": "1"},
                             capture_output=True, text=True, timeout=600).stdout
        got = re.search(r"^output: speedrun/([^/\s]+)/", out, re.M)
        got = got.group(1) if got else None
        ok = got == f"muonh-qwen3-{run}" and kind == "main"
        print(("ok  " if ok else "BAD ") + f"{os.path.basename(plan)}:{task}: declared {run}, dry run {got}")
        bad += not ok
print("ALL PLANS OK" if not bad else f"{bad} PROBLEMS")
sys.exit(1 if bad else 0)
