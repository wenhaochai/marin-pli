"""Schematics of the ten DepthBench designs (arXiv 2609.32534) for wenhaochai.com/blogs/per-layer-contribution.html,
in the cover style (Anthropic illustration: ivory ground, slate outlines, wobble filter, offset fills, no text). Writes
the figure between <!-- archs --> and <!-- /archs --> in the page given as argv[1]. One layer per panel:
  stream = thick horizontal line; norm = small kraft pill; branch F = manilla box; add = ringed plus on the stream;
  scale = clay trapezoid (narrowing = scaled down, widening = scaled up); several streams = thin parallel lines with a
  mixing grid; earlier layers = small boxes or dots with arcs into the layer that reads them.
Update rules (from the paper): Pre-LN h + F(LN h); Sandwich-LN h + LN(F(LN h)); LNS h + F(LN(h)/sqrt(l));
DeepNorm LN(a h + F(h)), a = (2L)^(1/4); KEEL RMSNorm(a h + F(RMSNorm h)), a = number of sublayers; HC 4 streams,
H A + F(LN(H alpha)) beta^T; mHC the same with Sinkhorn(A) doubly stochastic; AttnRes (Full) softmax mix of all earlier
sublayer outputs and the embedding; AttnRes (Block) the mix over 8 frozen block outputs and the running sum; MoDA
attention over sequence KV and the same token's earlier-layer KV."""
import random, re, sys
INK, IVORY, OAT, MANILLA, KRAFT, CLAY = "#141413", "#F0EEE6", "#E3DACC", "#EBDBBC", "#D4A27F", "#D97757"
W, H, D = 520, 420, 10


class P:
    def __init__(self, fid):
        self.fid, self.o = fid, []

    def f(self): return f'filter="url(#{self.fid})"'

    def path(self, d, w=6, dash=None):
        da = f' stroke-dasharray="{dash}"' if dash else ""
        self.o.append(f'<path d="{d}" stroke="{INK}" stroke-width="{w}" stroke-linecap="round" stroke-linejoin="round" fill="none"{da} {self.f()}/>')

    def line(self, pts, w=6, dash=None): self.path("M" + " L".join(f"{x} {y}" for x, y in pts), w, dash)

    def box(self, x, y, w, h, fill, rx=18, dash=None):
        da = f' stroke-dasharray="{dash}"' if dash else ""
        self.o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{IVORY}"/>')
        if fill != "none": self.o.append(f'<rect x="{x + D}" y="{y + D}" width="{w}" height="{h}" rx="{rx}" fill="{fill}"/>')
        self.o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="none" stroke="{INK}" stroke-width="6" stroke-linejoin="round"{da} {self.f()}/>')

    def norm(self, cx, cy, horiz=False):
        w, h = (80, 40) if horiz else (40, 84)
        self.box(cx - w / 2, cy - h / 2, w, h, KRAFT, 18)

    def add(self, cx, cy, r=22):
        self.o.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{IVORY}"/>')
        self.o.append(f'<circle cx="{cx + 6}" cy="{cy + 6}" r="{r}" fill="{OAT}"/>')
        self.o.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{INK}" stroke-width="5" {self.f()}/>')
        self.line([(cx - 10, cy), (cx + 10, cy)], 5); self.line([(cx, cy - 10), (cx, cy + 10)], 5)

    def dot(self, cx, cy, r, fill):
        self.o.append(f'<circle cx="{cx + 5}" cy="{cy + 5}" r="{r}" fill="{fill}"/><circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{INK}" stroke-width="5" {self.f()}/>')

    def trap(self, x, cy, w, h_in, h_out):            # flow left to right; h_out < h_in means scaled down
        pts = [(x, cy - h_in / 2), (x + w, cy - h_out / 2), (x + w, cy + h_out / 2), (x, cy + h_in / 2)]
        d = "M" + " L".join(f"{a} {b}" for a, b in pts) + " Z"
        self.o.append(f'<path d="{d}" fill="{IVORY}"/>')
        self.o.append(f'<path d="' + "M" + " L".join(f"{a + 8} {b + 8}" for a, b in pts) + f' Z" fill="{CLAY}"/>')
        self.o.append(f'<path d="{d}" fill="none" stroke="{INK}" stroke-width="6" stroke-linejoin="round" {self.f()}/>')

    def grid(self, x, y, n, cell, shades):
        self.box(x - 8, y - 8, n * cell + 16, n * cell + 16, "none", 12)
        for i in range(n):
            for j in range(n):
                self.o.append(f'<rect x="{x + j * cell + 3}" y="{y + i * cell + 3}" width="{cell - 6}" height="{cell - 6}" rx="4" fill="{shades[i][j]}"/>')

    def svg(self, seed):
        return (f'<svg viewBox="0 0 {W} {H}" role="img" style="width:100%;height:auto;display:block"><defs><filter id="{self.fid}" filterUnits="userSpaceOnUse" x="0" y="0" width="{W}" height="{H}">'
                f'<feTurbulence type="fractalNoise" baseFrequency="0.012" numOctaves="2" seed="{seed}"/><feDisplacementMap in="SourceGraphic" scale="4"/></filter></defs>'
                f'<rect width="{W}" height="{H}" fill="{IVORY}"/>' + "".join(self.o) + "</svg>")


SY = 320                                               # stream height


def branch(p, x0, x1, top=170):                         # up from the stream at x0, across, down into the add at x1
    p.line([(x0, SY), (x0, top), (x1, top), (x1, SY - 22)])


def pre_ln(p):
    p.line([(40, SY), (480, SY)], 9); branch(p, 100, 400); p.norm(160, 170); p.box(215, 105, 130, 130, MANILLA); p.add(400, SY)


def sandwich(p):
    p.line([(40, SY), (480, SY)], 9); branch(p, 90, 420); p.norm(140, 170); p.box(185, 105, 120, 130, MANILLA); p.norm(360, 170); p.add(420, SY)


def lns(p):
    p.line([(40, SY), (480, SY)], 9); branch(p, 90, 420); p.norm(135, 170); p.trap(180, 170, 50, 90, 40); p.box(260, 105, 120, 130, MANILLA); p.add(420, SY)


def deepnorm(p, pre=False):
    p.line([(40, SY), (150, SY)], 9); p.line([(220, SY), (470, SY)], 15)
    p.trap(150, SY, 70, 40, 90)                         # residual scaled up
    branch(p, 80, 370); (p.norm(140, 170), p.box(200, 105, 120, 130, MANILLA)) if pre else p.box(165, 105, 130, 130, MANILLA)
    p.add(370, SY); p.norm(435, SY, horiz=False)


def hc(p, balanced):
    ys = [290, 315, 340, 365]
    for y in ys: p.line([(40, y), (485, y)], 5)
    rnd = random.Random(5)
    shades = [[OAT] * 4 for _ in range(4)]
    if balanced:                                        # doubly stochastic: every row and column carries the same weight
        for i in range(4): shades[i][i] = KRAFT; shades[i][(i + 2) % 4] = KRAFT
    else:
        for i in range(4):
            for j in rnd.sample(range(4), rnd.choice([1, 2, 3])): shades[i][j] = rnd.choice([KRAFT, INK])
    p.grid(62, 276, 4, 26, shades)
    for y in ys: p.line([(205, y), (240, 165)], 4)       # read: the streams into one input
    p.line([(240, 165), (255, 165)]); p.norm(275, 165); p.box(310, 105, 100, 120, MANILLA); p.line([(410, 165), (430, 165)])
    for y in ys: p.line([(430, 165), (470, y)], 4); p.dot(470, y, 8, CLAY)


def attnres(p, block):
    xs = [70, 150, 230, 310]
    p.dot(40, 330, 14, KRAFT)                            # the embedding
    if block:
        p.box(60, 268, 140, 120, "none", 22, dash="14 11"); p.box(220, 268, 140, 120, "none", 22, dash="14 11")
    for x in xs: p.box(x + (6 if block else 0), 290, 52 if block else 60, 80, OAT, 14)
    p.box(400, 270, 90, 120, CLAY, 18)                   # the layer that reads
    src = [(40, 316), (130, 268), (290, 268)] if block else [(40, 316)] + [(x + 30, 290) for x in xs]
    for k, (x, y) in enumerate(src):
        h = 60 + 30 * k if not block else 70 + 50 * k
        p.path(f"M{x} {y} C{x} {y - h - 60} {445} {270 - h - 40} {445} 268", 5)


def moda(p):
    p.line([(40, SY), (480, SY)], 9)
    for x in (70, 125, 180): p.dot(x, SY, 12, OAT)
    branch(p, 230, 440); p.norm(270, 170); p.box(310, 105, 110, 130, CLAY)
    for k, x in enumerate((70, 125, 180)): p.path(f"M{x} {SY - 14} C{x} {120 - 20 * k} {330} {60} {350} 105", 4, dash="10 9")
    p.add(440, SY)


ARCH = [
    ("Pre-LN", pre_ln, None, "Normalize, transform, add back to the stream.", "先归一化，再变换，加回残差流。"),
    ("Sandwich-LN", sandwich, ("Baseline", "基线"), "A second normalization on the branch output before it is added: the baseline's own design.", "分支输出加回之前再归一化一次：基线自己的设计。"),
    ("LayerNorm Scaling", lns, None, "The normalized input is scaled by 1/√ℓ, more in deeper layers.", "归一化后的输入乘以 1/√ℓ，越深的层缩得越多。"),
    ("DeepNorm", lambda p: deepnorm(p), None, "No norm before the branch; the stream is scaled up by (2L)^¼ and normalized after the add.", "分支前不归一化；残差流乘以 (2L)^¼，相加后再归一化。"),
    ("KEEL", lambda p: deepnorm(p, pre=True), None, "Norms before and after, with a residual gain equal to the number of sublayers.", "前后都归一化，残差增益等于子层数。"),
    ("Hyper-Connections", lambda p: hc(p, False), None, "Four residual streams, mixed by learned weights, read into the branch and written back.", "四条残差流，由可学习的权重混合，读入分支再写回。"),
    ("mHC", lambda p: hc(p, True), None, "Hyper-connections with the stream mixing kept doubly stochastic.", "超连接，但流之间的混合矩阵保持双随机。"),
    ("AttnRes (Full)", lambda p: attnres(p, False), None, "A layer's input is a learned softmax mix of every earlier output and the embedding.", "每层的输入是此前所有层输出和 embedding 的 softmax 加权混合。"),
    ("AttnRes (Block)", lambda p: attnres(p, True), None, "The same mix over 8 block outputs and the running sum of the current block.", "同样的混合，只在 8 个块的输出和当前块的累加和上做。"),
    ("MoDA", moda, None, "Attention also reads the same token's keys and values from earlier layers.", "注意力还读取同一 token 在之前各层的 key 和 value。"),
]
cells = []
for i, (name, draw, tag, en, zh) in enumerate(ARCH):
    p = P(f"sch-arch-{i}"); draw(p)
    t = f' <span class="setup-tag"><span lang="en">{tag[0]}</span><span lang="zh">{tag[1]}</span></span>' if tag else ""
    cells.append(f'    <div class="setup">{p.svg(20 + i)}<div class="setup-name">{name}{t}</div><p><span lang="en">{en}</span><span lang="zh">{zh}</span></p></div>\n')
fig = '<!-- archs -->\n  <figure class="setups" id="archs">\n' + "".join(cells) + "  </figure>\n  <!-- /archs -->"
page = sys.argv[1]; s = open(page).read()
if "<!-- archs -->" in s:
    s = re.sub(r"<!-- archs -->.*?<!-- /archs -->", lambda m: fig, s, flags=re.S)
else:
    i = s.index('  <figure class="viz" id="fig-d48-arch"'); s = s[:i] + "  " + fig + "\n" + s[i:]
open(page, "w").write(s); print("wrote", len(cells), "schematics into", page)
