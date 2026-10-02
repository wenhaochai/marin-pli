import sys, re
from html.parser import HTMLParser

class P(HTMLParser):
    def __init__(self):
        super().__init__()
        self.out = []; self.skip = 0; self.links = []; self.imgs = []
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in ('script', 'style', 'svg'): self.skip += 1
        if tag in ('p', 'div', 'br', 'li', 'tr', 'h1', 'h2', 'h3', 'h4', 'section', 'figure', 'figcaption', 'pre', 'blockquote'): self.out.append('\n')
        if tag in ('h1','h2','h3','h4'): self.out.append('#' * int(tag[1]) + ' ')
        if tag == 'li': self.out.append('- ')
        if tag in ('td', 'th'): self.out.append(' | ')
        if tag == 'a' and a.get('href'): self.links.append(a['href'])
        if tag == 'img':
            src = a.get('src', '')
            self.imgs.append((src[:120], a.get('alt', '')))
            self.out.append(f"[IMG alt={a.get('alt','')!r} src={src[:80]!r}]")
        if tag == 'annotation': self.skip += 1
    def handle_endtag(self, tag):
        if tag in ('script', 'style', 'svg', 'annotation'): self.skip = max(0, self.skip - 1)
        if tag in ('p', 'div', 'h1', 'h2', 'h3', 'h4', 'tr', 'pre', 'figcaption', 'blockquote'): self.out.append('\n')
    def handle_data(self, d):
        if not self.skip: self.out.append(d)

p = P(); p.feed(open(sys.argv[1], encoding='utf-8').read())
txt = ''.join(p.out)
txt = re.sub(r'[ \t]+', ' ', txt)
txt = re.sub(r'\n\s*\n+', '\n\n', txt)
print(txt)
print('\n## LINKS'); [print(l) for l in dict.fromkeys(p.links)]
print('\n## IMGS'); [print(i) for i in p.imgs]
