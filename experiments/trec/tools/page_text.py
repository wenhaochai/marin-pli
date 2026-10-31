"""English text of exps/trec.html for the anti-autoresearch ledger: page.md and paper.txt in the cwd (tmp/trec_audit).
Run in tmp/trec_audit after copying the site's exps/trec.html there: it reads ./trec.html and writes ./paper.txt."""
import html, re
s = open("trec.html").read()
s = re.sub(r"<svg.*?</svg>", "", s, flags=re.S)
s = re.sub(r'<pre class="diff.*?</pre>', "[diff of the core change; see code/trec_score.py]", s, flags=re.S)
def txt(x):
    x = re.sub(r"<sub>(.*?)</sub>", r"_\1", x)
    x = re.sub(r'<a href="([^"]+)">(.*?)</a>', r"\2 [\1]", x)
    return html.unescape(re.sub(r"<[^>]+>", "", x)).strip()
out = ["# Training Re-evaluation Curves (exps/trec.html, English text)", ""]
for m in re.finditer(r'<p class="ar-standfirst" lang="en">(.*?)</p>', s, re.S):
    out += ["Standfirst: " + txt(m.group(1)), ""]
body = s[s.index('<section id="setup">'):]
for m in re.finditer(r'<h2>(.*?)</h2>|<h3 class="hyp" id="([^"]+)">(.*?)</h3>|<p class="hyp-answer" lang="en">(.*?)</p>|<p lang="en">(.*?)</p>|<div class="setup-name">(.*?)</div><p>(.*?)</p>|data-title-en="([^"]*)"|data-cap-en="([^"]*)"|<ul lang="en">(.*?)</ul>|data-more-en="([^"]*)"', body, re.S):
    g = m.groups()
    if g[0]: out += ["", "## " + txt(re.sub(r'<span lang="zh">.*?</span>', "", g[0])), ""]
    elif g[1]: out += ["", "### " + txt(re.sub(r'<span lang="zh">.*?</span>', "", g[2])), ""]
    elif g[3]: out += ["Answer: " + txt(g[3]), ""]
    elif g[4]: out += [txt(g[4]), ""]
    elif g[5]: out += ["Setup card: " + txt(re.sub(r'<span lang="zh">.*?</span>', "", g[5])) + ": " + txt(re.sub(r'<span lang="zh">.*?</span>', "", g[6]))]
    elif g[7]: out += ["", "Figure title: " + html.unescape(g[7])]
    elif g[8]: out += ["Caption: " + html.unescape(g[8]), ""]
    elif g[9]: out += [txt(li) for li in re.findall(r"<li>(.*?)</li>", g[9], re.S)] + [""]
    elif g[10]: out += ["Details: " + txt(html.unescape(g[10])), ""]   # long captions (added 2026-10-31: most numbers are here)
open("page.md", "w").write("\n".join(out) + "\n")
open("paper.txt", "w").write("\n".join(out) + "\n")
