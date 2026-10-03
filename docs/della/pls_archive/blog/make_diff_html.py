"""Core diff of an experiment against the baseline, as syntax-highlighted HTML for a Pretrain Practice page.
Each section compares a passage of a baseline file with a passage of ours (or adds ours alone): lines that match are
context, the baseline's other lines are red, ours green. Every line is a real line of its file at the given commit; only
trailing comments are removed (the code itself is unchanged). Writes the <pre class="diff code"> block between the
<!-- diff:SLUG --> markers of the page.
Usage: make_diff_html.py PAGE SLUG REPO SPEC.json
SPEC: {"base_commit": ..., "ours_commit": ..., "sections": [{"base": [path, first, last, "what"] or null,
                                                            "ours": [path, first, last]}, ...]}"""
import difflib, io, json, subprocess, sys, tokenize
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer
page, slug, repo, spec_path = sys.argv[1:5]
spec = json.load(open(spec_path))
def lines(commit, path, a, b):
    src = subprocess.run(["git", "-C", repo, "show", f"{commit}:{path}"], capture_output=True, text=True, check=True).stdout.split("\n")
    return src[a - 1:b]
def strip_comment(line):  # drop a trailing comment found by the tokenizer, never touch code
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(line.strip() + "\n").readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return line
    for t in toks:
        if t.type == tokenize.COMMENT:
            off = len(line) - len(line.lstrip())
            return line[:off + t.start[1]].rstrip()
    return line
fmt = HtmlFormatter(nowrap=True)
def hl(code):
    return highlight(code, PythonLexer(), fmt).rstrip("\n") if code.strip() else ""
def row(sign, cls, code):
    return f'<span class="{cls}">{sign}{hl(code)}</span>'
out = []
for sec in spec.get("sections", []):
    op, oa, ob = sec["ours"]
    # blank lines are left out of the comparison; every shown line keeps its real line number
    ours = [(oa + i, strip_comment(l)) for i, l in enumerate(lines(spec["ours_commit"], op, oa, ob)) if l.strip()]
    if sec["base"]:
        bp, ba, bb, what = sec["base"]
        base = [(ba + i, strip_comment(l)) for i, l in enumerate(lines(spec["base_commit"], bp, ba, bb)) if l.strip()]
        out += [f'<span class="hd">--- {bp}, baseline {what}</span>', f'<span class="hd">+++ {op}, comments omitted</span>']
        sm = difflib.SequenceMatcher(None, [t for _, t in base], [t for _, t in ours], autojunk=False)
        for group in sm.get_grouped_opcodes(1):   # changed lines with one line of context; longer unchanged runs collapse
            i0, j0 = group[0][1], group[0][3]
            out.append(f'<span class="hk">@@ -{base[min(i0, len(base) - 1)][0]} +{ours[min(j0, len(ours) - 1)][0]} @@</span>')
            for tag, i1, i2, j1, j2 in group:
                if tag == "equal":
                    out += [row(" ", "ctx", t) for _, t in ours[j1:j2]]
                else:
                    out += [row("-", "del", t) for _, t in base[i1:i2]] + [row("+", "add", t) for _, t in ours[j1:j2]]
    else:
        out += [f'<span class="hk">@@ +{oa} @@ {op}, added, comments omitted</span>']
        out += [row("+", "add", t) for _, t in ours]
pre = '<pre class="diff code">' + "".join(out) + "</pre>"
if "inline" in spec:   # rewritten code for the clearest presentation, comments in each language; the Code link has the real code
    pre = ""
    for lang, d in spec["inline"].items():
        rows = [f'<span class="hd">--- {d["base_title"]}</span>', f'<span class="hd">+++ {d["ours_title"]}</span>']
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, d["base"], d["ours"], autojunk=False).get_opcodes():
            if tag == "equal":
                rows += [row(" ", "ctx", l) for l in d["ours"][j1:j2]]
            else:
                rows += [row("-", "del", l) for l in d["base"][i1:i2]] + [row("+", "add", l) for l in d["ours"][j1:j2]]
        pre += f'<pre class="diff code" lang="{lang}">' + "".join(rows) + "</pre>"
s = open(page).read(); a, b = f"<!-- diff:{slug} -->", f"<!-- /diff:{slug} -->"
i, j = s.index(a) + len(a), s.index(b)
open(page, "w").write(s[:i] + pre + s[j:])
print("wrote", len(out), "lines into", page)
