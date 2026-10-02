import re, sys
D = "/scratch/gpfs/GROUP/USER/tmp/pls/survey_notes/pdftxt/"
checks = {
    "1808.04444": ["1.062", "1.158", "2n"],
    "2404.16710": ["reduces the accuracy of the last layer", "46.0", "43.1", "scale"],
    "2410.20672": ["51.7", "50.2", "12.85", "13.24"],
    "2407.14320": ["71.61", "68.39"],
    "2505.16792": ["5.3", "8.1", "250K", "400K", "7.4"],
    "2609.35440": ["70.62", "75.34", "67.08", "16%", "51%"],
    "2312.04916": ["1/4", "0.25"],
    "1908.10118": ["34.87"],
    "1910.10073": ["36.2", "35.9"],
}
for f, pats in checks.items():
    t = open(D + f + ".txt", encoding="utf-8", errors="replace").read().replace("\x00", "")
    t = re.sub(r"\s+", " ", t)
    for p in pats:
        idx = [m.start() for m in re.finditer(re.escape(p), t)]
        snip = t[max(0, idx[0] - 110): idx[0] + 90] if idx else ""
        print(f"[{f}] {p!r}: {len(idx)} hits | {snip}")
    print()
