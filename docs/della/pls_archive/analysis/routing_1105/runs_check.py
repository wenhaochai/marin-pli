"""Run the top of build_record.py up to its figure sets (no W&B, no sacct) and print the run ids the routing figures name that the record lacks."""
import sys; sys.argv = ["build_record.py", "/dev/null"]
src = open("/scratch/gpfs/GROUP/USER/tmp/pls_blog/build_record.py", encoding="utf-8").read()
cut = src.index("_have = {")
ns = {"__name__": "check", "__file__": "/scratch/gpfs/GROUP/USER/tmp/pls_blog/build_record.py"}
exec(compile(src[:cut], "build_record_top", "exec"), ns)
have = {(src_[1] if src_[0] == "wandb" else src_[2]) for (*_, src_, _) in ns["RUNS"]}
for n, st in ns["FIGX"].items():
    print(n, "missing:", sorted(x.split("bbfrozen-")[1] for x in st - have))
import re
print([h for h in sorted(have) if re.search(r"bbfrozen-(ahc|amhc|aattnres|apreln|alns|vs-d48)", h) and ("g4" in h or "s1" in h or "vs-d48" == h[-6:])][:40])
