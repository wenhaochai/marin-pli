import re
D = "/scratch/gpfs/GROUP/USER/tmp/pls/survey_notes/pdftxt/"
h = re.sub(r"\s+", " ", open(D + "2505.16792.txt", encoding="utf-8", errors="replace").read().replace("\x00", ""))
i = h.find("8.1 5.20 126.1")
print(h[max(0, i - 1300): i + 450])
