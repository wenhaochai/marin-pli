"""Merge the polished unit files (Argon pass, owner 2026-11-01: 「你让argon重新给你整体润色一下中英语写作，再push一次」)
back into exps/per-layer-contribution.html and check every unit against its old text.

Usage: apply_units.py <units.json> <parts dir> <out.html> <review.md> [--accept id,...] [--revert id,...]
                      [--override overrides.json]

units.json comes from extract_units.py (offsets into the page it names). The parts dir holds the edited part_<k>.txt.
overrides.json (optional) maps a unit id to a replacement text that wins over the part file (manual fixes after the audit).
Checks; a unit that fails keeps its old text unless its id is in --accept, and --revert forces the old text:
  numbers  figure captions: the numbers of the short and the long caption together, per figure and language (a number
           may move from the short into the long caption); other units: the numbers of the unit. Tags are removed and
           entities decoded before numbers are read; a number is a digit run with an optional decimal part.
  tags     the HTML tags of the unit with their attributes, as a multiset.
  form     one text line; no em dash added; no new ⟦ ⟧; no '@@' or '##' inside the text.
Warnings, printed and not enforced: a short caption over 200 English or 90 Chinese characters; a long caption, paragraph
or answer more than 10 percent longer than before; 主干, or 付 used for a cost, in the Chinese.
review.md lists every changed unit, old and new, for the meaning audit."""
import html, json, re, sys
from collections import Counter
from pathlib import Path

a = sys.argv[1:]
uj, pdir, outp, revp = Path(a[0]), Path(a[1]), Path(a[2]), Path(a[3])
opt = lambda name: set(a[a.index(name) + 1].split(",")) if name in a else set()
accept, revert = opt("--accept"), opt("--revert")
overrides = json.load(open(a[a.index("--override") + 1])) if "--override" in a else {}
data = json.load(open(uj)); units = data["units"]; page = Path(data["page"]).read_text()
byid = {u["id"]: u for u in units}

# 1. parse the edited part files
lines_of = {}
for f in sorted(pdir.glob("part_*.txt")):
    cur, buf = None, []
    for line in f.read_text().split("\n") + ["@@ END"]:
        if line.startswith("@@ "):
            if cur is not None: lines_of[cur] = buf
            cur, buf = line[3:].split("|")[0].strip(), []
        elif line.startswith("##") or not line.strip():
            continue
        elif cur is not None:
            buf.append(line.strip())
lines_of.pop("END", None)
missing = [u["id"] for u in units if u["id"] not in lines_of]
extra = [k for k in lines_of if k not in byid]

TAG = re.compile(r"<[^>]+>")
NUM = re.compile(r"\d+(?:\.\d+)?")
nums = lambda t: Counter(NUM.findall(html.unescape(TAG.sub(" ", t))))
tags = lambda t: Counter(TAG.findall(t))
fail, warn, new = {}, {}, {}
for u in units:
    i = u["id"]
    ls = lines_of.get(i, [u["shown"]])
    t = (" " if u["lang"] == "en" else "").join(ls)
    if i in overrides: t = overrides[i]
    new[i] = t
    f = []
    if len(ls) != 1 and i not in overrides: f.append(f"{len(ls)} lines")
    if t.count("—") > u["shown"].count("—"): f.append("em dash added")
    if ("⟦" in t or "⟧" in t) and "⟦" not in u["shown"]: f.append("⟦⟧ added")
    if "@@" in t or "##" in t: f.append("@@/## in text")
    if not t.strip(): f.append("empty")
    if tags(t) != tags(u["shown"]): f.append(f"tags {dict(tags(u['shown']) - tags(t))} -> {dict(tags(t) - tags(u['shown']))}")
    if not u["fig"] and nums(t) != nums(u["shown"]):
        f.append(f"numbers lost {dict(nums(u['shown']) - nums(t))} added {dict(nums(t) - nums(u['shown']))}")
    if f: fail[i] = f
# figure-level number check (short + long caption, per language)
figs = {}
for u in units:
    if u["fig"]: figs.setdefault((u["fig"], u["lang"]), []).append(u)
for (fig, lang), us in figs.items():
    old = sum((nums(u["shown"]) for u in us), Counter()); nw = sum((nums(new[u["id"]]) for u in us), Counter())
    if old != nw:
        for u in us: fail.setdefault(u["id"], []).append(f"figure numbers lost {dict(old - nw)} added {dict(nw - old)}")
for u in units:
    i, t = u["id"], new[u["id"]]
    w = []
    if u["kind"] == "short caption" and len(t) > (200 if u["lang"] == "en" else 90): w.append(f"short caption {len(t)} chars")
    if u["kind"] in ("long caption", "paragraph", "answer") and len(t) > 1.1 * len(u["shown"]):
        w.append(f"longer {len(u['shown'])} -> {len(t)}")
    if u["lang"] == "zh" and "主干" in t: w.append("主干")
    if u["lang"] == "zh" and re.search(r"付(?:了|出)?\s*(?:\d|约|全部|得)|付的", t): w.append("付 for a cost")
    if w: warn[i] = w

# 2. choose the final text and rebuild the page
final, status = {}, {}
for u in units:
    i = u["id"]
    if i in revert: final[i], status[i] = u["shown"], "reverted"
    elif i in fail and i not in accept: final[i], status[i] = u["shown"], "failed, old text kept"
    else:
        final[i] = new[i]
        status[i] = ("accepted despite: " + "; ".join(fail[i])) if i in fail else ("changed" if new[i] != u["shown"] else "unchanged")
out = page
for u in sorted(units, key=lambda u: -u["start"]):
    t = final[u["id"]]
    if t == u["shown"]: raw = u["raw"]
    elif u["attr"]: raw = html.escape(t, quote=True)
    else: raw = t
    out = out[:u["start"]] + raw + out[u["end"]:]
outp.write_text(out)

# 3. report
c = Counter(status[u["id"]].split(",")[0].split(" despite")[0] for u in units)
print(f"units {len(units)}; parsed {len(lines_of)}; missing {missing}; extra {extra}")
print("status:", dict(c))
for u in units:
    i = u["id"]
    if i in fail: print(f"FAIL {i} part {u['part']} {u['fig'] or u['group']} {u['kind']}: {'; '.join(fail[i])} [{status[i]}]")
for u in units:
    i = u["id"]
    if i in warn: print(f"warn {i} part {u['part']} {u['fig'] or u['group']} {u['kind']}: {'; '.join(warn[i])}")
L = ["# Polish pass: changed units, old and new", ""]
for u in units:
    i = u["id"]
    if final[i] != u["shown"] or i in fail:
        L += [f"## {i} | part {u['part']} | {u['fig'] or u['group']} | {u['kind']} | {status[i]}",
              "", f"OLD: {u['shown']}", "", f"NEW: {new[i]}", ""]
revp.write_text("\n".join(L))
print(f"wrote {outp} and {revp}")
