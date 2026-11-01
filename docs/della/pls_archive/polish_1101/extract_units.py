"""Extract the prose of exps/per-layer-contribution.html into editable unit files for the Argon polish pass
(owner 2026-11-01: 「你让argon重新给你整体润色一下中英语写作，再push一次」).

Usage: extract_units.py <page.html> <outdir> <split: comma-separated group or figure ids that start parts 2..n>
Writes <outdir>/units.json (every unit: id, group, kind, lang, figure, span offsets, raw text, shown text),
<outdir>/part_<k>.txt (the units of part k, EN then ZH per pair, for the worker to edit in place), and prints a
per-group summary. Units: figure short and long captions (data-cap-*, data-more-*), question answers
(p.hyp-answer), paragraphs (p lang=en/zh, Details folds included), setup-panel sentences and lean-link notes
(p > span lang), related-work items. Not units (left as they are): titles (h1, h2, question h3, contents), figure
titles, setup-panel names, Acknowledgements, Citation, the video's aria-label, the record.
Attribute values are HTML-escaped with html.escape(quote=True); extraction asserts that round trip for every unit."""
import html, json, re, sys
from pathlib import Path

page, outdir, split = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3].split(",")
s = page.read_text()
outdir.mkdir(parents=True, exist_ok=True)

# groups: Setup, each question (h3.hyp), Related work
marks = [(m.start(), m.group(1)) for m in re.finditer(r'<section id="(setup|related)">', s)]
marks += [(m.start(), m.group(1)) for m in re.finditer(r'<h3 class="hyp" id="([^"]+)">', s)]
marks.sort()
qtext = {m.group(1): (html.unescape(m.group(2)), m.group(3))
         for m in re.finditer(r'<h3 class="hyp" id="([^"]+)">.*<span lang="en">([^<]*)</span><span lang="zh">([^<]*)</span></h3>', s)}
stop = s.index('<section id="ack">')


def group_of(pos):
    g = None
    for p, name in marks:
        if p <= pos: g = name
    return g


units = []


def add(kind, lang, start, end, fig=None, title=None, attr=False):
    raw = s[start:end]
    if attr:
        shown = html.unescape(raw)
        esc = html.escape(shown, quote=True)
        assert raw in (esc, esc.replace("&#x27;", "'")), (fig, kind, lang, raw[:80])
    else:
        shown = raw.replace("&#x27;", "'").replace("&quot;", '"')
    assert "\n" not in shown, (kind, lang, start)
    units.append(dict(kind=kind, lang=lang, start=start, end=end, fig=fig, title=title, attr=attr, raw=raw,
                      shown=shown, group=group_of(start)))


# figure captions
for m in re.finditer(r'<figure class="viz"[^>]*>', s):
    tag, t0 = m.group(0), m.start()
    fid = re.search(r' id="([^"]+)"', tag).group(1)
    ten = html.unescape(re.search(r'data-title-en="([^"]*)"', tag).group(1))
    for kind in ("cap", "more"):
        for lang in ("en", "zh"):
            a = re.search(rf'data-{kind}-{lang}="([^"]*)"', tag)
            add({"cap": "short caption", "more": "long caption"}[kind], lang, t0 + a.start(1), t0 + a.end(1), fid, ten, True)
# paragraphs with lang (answers, paragraphs, Details folds), before Acknowledgements
for m in re.finditer(r'<p( class="hyp-answer")? lang="(en|zh)">(.*?)</p>', s[:stop], re.S):
    add("answer" if m.group(1) else "paragraph", m.group(2), m.start(3), m.end(3))
# setup-panel sentences and lean-link notes: <p( class=..)?><span lang="en">..</span><span lang="zh">..</span></p>
for m in re.finditer(r'<p( class="lean-link")?><span lang="en">(.*?)</span><span lang="zh">(.*?)</span></p>', s[:stop], re.S):
    kind = "note" if m.group(1) else "setup panel"
    name = None
    if kind == "setup panel":
        nm = list(re.finditer(r'<div class="setup-name"><span lang="en">([^<]*)</span>', s[:m.start()]))[-1]
        name = html.unescape(nm.group(1))
    add(kind, "en", m.start(2), m.end(2), title=name)
    add(kind, "zh", m.start(3), m.end(3), title=name)
# related work
for lang in ("en", "zh"):
    ul = re.search(rf'<ul lang="{lang}">(.*?)</ul>', s, re.S)
    for li in re.finditer(r'<li>(.*?)</li>', ul.group(1), re.S):
        add("related work item", lang, ul.start(1) + li.start(1), ul.start(1) + li.end(1))

units.sort(key=lambda u: u["start"])
# pair EN and ZH: figure attrs by (fig, kind); others by order of appearance
en = [u for u in units if u["lang"] == "en"]; zh = [u for u in units if u["lang"] == "zh"]
assert len(en) == len(zh), (len(en), len(zh))
def pkey(u): return (u["fig"], u["kind"]) if u["fig"] else None
zfig = {pkey(u): u for u in zh if u["fig"]}
zrest = [u for u in zh if not u["fig"]]
pairs = []
ri = 0
for u in en:
    if u["fig"]:
        pairs.append((u, zfig[pkey(u)]))
    else:
        z = zrest[ri]; ri += 1
        assert z["kind"] == u["kind"] and z["group"] == u["group"], (u["kind"], z["kind"], u["group"], z["group"])
        pairs.append((u, z))
for i, (a, b) in enumerate(pairs):
    a["id"], b["id"] = f"u{i:03d}en", f"u{i:03d}zh"

# parts: a new part starts at each split point (a group id or a figure id), in page order
part_of, k, prev_g, prev_f = {}, 1, None, None
for a, b in pairs:
    if (a["group"] in split and a["group"] != prev_g) or (a["fig"] in split and a["fig"] != prev_f): k += 1
    part_of[a["id"]] = part_of[b["id"]] = k
    prev_g, prev_f = a["group"], a["fig"]
for u in units: u["part"] = part_of[u["id"]]

def words(t): return len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'.\-]*", re.sub(r"<[^>]+>", " ", t)))
def cjk(t): return len(re.findall(r"[\u4e00-\u9fff]", t))
print(f"{'part':5s} {'units':>6s} {'EN words':>9s} {'ZH chars':>9s}  groups")
for k in sorted(set(part_of.values())):
    us = [u for u in units if u["part"] == k]
    gs = []
    for u in us:
        if u["group"] not in gs: gs.append(u["group"])
    print(f"{k:5d} {len(us):6d} {sum(words(u['shown']) for u in us if u['lang'] == 'en'):9d} "
          f"{sum(cjk(u['shown']) for u in us if u['lang'] == 'zh'):9d}  {' '.join(gs)}")
parts = None
for k in sorted(set(part_of.values())):
    lines = [f"## Part {k} of the polish pass over exps/per-layer-contribution.html. Edit only the text lines below each @@ line.", ""]
    last_g, last_fig = None, None
    for a, b in pairs:
        if a["part"] != k: continue
        if a["group"] != last_g:
            last_g = a["group"]
            q = qtext.get(last_g)
            lines += ["", f"## ===== {'Question: ' + q[0] + ' / ' + q[1] if q else 'Section: ' + last_g} =====", ""]
        if a["fig"] and a["fig"] != last_fig:
            last_fig = a["fig"]
            lines += [f"## --- Figure {a['fig']}, title: {a['title']}", ""]
        for u in (a, b):
            extra = f" | panel: {u['title']}" if u["kind"] == "setup panel" else ""
            lines += [f"@@ {u['id']} | {u['fig'] or u['group']} | {u['kind']}{extra} | {u['lang'].upper()}", u["shown"], ""]
    (outdir / f"part_{k}.txt").write_text("\n".join(lines))
json.dump(dict(page=str(page), units=units), open(outdir / "units.json", "w"), ensure_ascii=False, indent=0)
print(len(units), "units;", len(pairs), "pairs")
