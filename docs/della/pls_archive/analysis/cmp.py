import re, sys
def norm(s):
    s = re.sub(r'<https?://[^>]*>', ' ', s)
    s = re.sub(r'\$[^$]*\$', ' ', s)
    s = s.lower()
    s = re.sub(r'[^a-z0-9%.\-+ ]+', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()
notion = open(sys.argv[1]).read(); blog = norm(open(sys.argv[2]).read())
missing = []
for line in notion.split('\n'):
    for sent in re.split(r'(?<=[.!?])\s+', line):
        n = norm(sent)
        if len(n) < 25: continue
        # check with sliding 6-word shingles: fraction of shingles found in blog
        w = n.split()
        sh = [' '.join(w[i:i+5]) for i in range(max(1, len(w)-4))]
        hit = sum(1 for x in sh if x in blog) / len(sh)
        if hit < 0.6: missing.append((round(hit, 2), sent.strip()[:400]))
print(len(missing), 'notion sentences not (fully) in blog copy')
for h, s in missing: print(f'[{h}] {s}')
