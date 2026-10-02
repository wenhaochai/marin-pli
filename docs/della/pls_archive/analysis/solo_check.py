import re
t = re.sub(r"\s+", " ", open("/scratch/gpfs/GROUP/USER/tmp/pls/survey_notes/pdftxt/2609.35440.txt", encoding="utf-8", errors="replace").read().replace("\x00", ""))
for p in ["read-only", "frozen", "stop-gradient", "stop gradient", "detach", "PRIV", "shared readout", "private"]:
    for m in list(re.finditer(re.escape(p), t))[:2]:
        print(f"{p!r}: ...{t[max(0, m.start()-260): m.start()+220]}...\n")
