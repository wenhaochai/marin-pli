import re
D = "/scratch/gpfs/GROUP/USER/tmp/pls/survey_notes/pdftxt/"
def load(f):
    return re.sub(r"\s+", " ", open(D + f + ".txt", encoding="utf-8", errors="replace").read().replace("\x00", ""))
t = load("2410.20672")
for p in ["13.24", "50.2 ", "early-exit", "Early-exit", "coefficient", "0.1"]:
    for m in list(re.finditer(re.escape(p), t))[:6]:
        print(f"[RRT] {p!r}: ...{t[max(0, m.start()-220): m.start()+160]}...\n")
h = load("2505.16792")
i = h.find("Table 3")
print("[HASTE] Table 3 context:", h[max(0, i-1400): i+300])
