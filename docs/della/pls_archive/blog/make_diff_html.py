"""Core diff of an experiment against the baseline, as syntax-highlighted HTML for a Pretrain Practice page.
Red lines are real lines of the baseline file, green lines real lines of ours; only trailing comments are removed
(the code itself is unchanged). Writes the <pre class="diff code"> block between <!-- diff:SLUG --> markers.
Usage: make_diff_html.py PAGE SLUG REPO BASE_COMMIT HEAD_COMMIT"""
import html, io, re, subprocess, sys, tokenize
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer
page, slug, repo, base, head = sys.argv[1:6]
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
def block(sign, cls, code_lines):
    return [f'<span class="{cls}">{sign}{hl(strip_comment(l))}</span>' for l in code_lines]
BASE_FILE = "lib/levanter/src/levanter/models/lm_model.py"; OURS = "experiments/references/per_layer_qwen3.py"
out = [f'<span class="hd">--- {BASE_FILE}, baseline loss</span>', f'<span class="hd">+++ {OURS}, comments omitted</span>',
       '<span class="hk">@@ -337,18 +327,10 @@</span>']
out += block("-", "del", lines(base, BASE_FILE, 337, 354))
out += block("+", "add", lines(head, OURS, 327, 336))
pre = f'<pre class="diff code">' + "".join(out) + "</pre>"
s = open(page).read(); a, b = f"<!-- diff:{slug} -->", f"<!-- /diff:{slug} -->"
i, j = s.index(a) + len(a), s.index(b)
open(page, "w").write(s[:i] + pre + s[j:])
print("wrote", len(out), "lines into", page)
