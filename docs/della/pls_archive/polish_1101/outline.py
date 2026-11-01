"""Outline of exps/per-layer-contribution.html for the Argon polish pass (2026-11-01): sections, headings, figures,
figure attributes, and the amount of English and Chinese text. Usage: outline.py <page>"""
import re, sys, html
from html.parser import HTMLParser
src = open(sys.argv[1]).read()

class P(HTMLParser):
    def __init__(s):
        super().__init__(); s.stack=[]; s.out=[]; s.attrs_seen={}; s.figs=[]; s.cur=None; s.text={'en':[], 'zh':[]}
        s.langstack=[]
    def handle_starttag(s, tag, attrs):
        a=dict(attrs)
        if tag=='figure':
            s.figs.append((s.getpos()[0], a.get('id'), a.get('class'), sorted(k for k in a if k.startswith('data-'))))
            for k in a:
                if k.startswith('data-'): s.attrs_seen[k]=s.attrs_seen.get(k,0)+1
            for k,v in a.items():
                if k.endswith('-en') and v: s.text['en'].append(v)
                if k.endswith('-zh') and v: s.text['zh'].append(v)
        if tag in ('section','h1','h2','h3'):
            s.out.append((s.getpos()[0], tag, a.get('id'), a.get('class')))
        s.langstack.append(a.get('lang'))
    def handle_endtag(s, tag):
        if s.langstack: s.langstack.pop()
    def handle_data(s, d):
        langs=[l for l in s.langstack if l]
        if langs and d.strip(): s.text[langs[-1]].append(d)
p=P(); p.feed(src)
print("sections/headings:")
for ln,tag,i,c in p.out: print(f"  {ln:5d} {tag} id={i} class={c}")
print("figure data- attributes:", p.attrs_seen)
print("n figures:", len(p.figs))
for ln,i,c,ks in p.figs: print(f"  {ln:5d} {i} {c} {' '.join(ks)}")
en=' '.join(p.text['en']); zh=''.join(p.text['zh'])
print("EN words (lang=en text + *-en attributes):", len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'.\-]*", en)))
print("ZH chars (CJK in lang=zh text + *-zh attributes):", len(re.findall(r'[一-鿿]', zh)))
