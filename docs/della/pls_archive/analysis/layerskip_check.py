import re
t = re.sub(r"\s+", " ", open("/scratch/gpfs/GROUP/USER/tmp/pls/survey_notes/pdftxt/2404.16710.txt", encoding="utf-8", errors="replace").read().replace("\x00", ""))
print("ABSTRACT:", t[t.find("Abstract"): t.find("Abstract") + 1300], "\n")
for p in ["rotational", "gradual", "e_{scale}", "escale", "layer dropout rate", "speedups of up to", "same LM head", "shared exit", "normalization"]:
    for m in list(re.finditer(re.escape(p), t))[:2]:
        print(f"{p!r}: ...{t[max(0, m.start()-250): m.start()+250]}...\n")
