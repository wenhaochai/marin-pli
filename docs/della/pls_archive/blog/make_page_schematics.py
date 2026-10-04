"""Schematics for the subsections of wenhaochai.com/blogs/per-layer-contribution.html, in the cover style (Anthropic
illustration: ivory ground, slate outlines, wobble filter, offset fills, no text). Writes two figures into the page given
as argv[1], each between its markers:
  <!-- deep --> ... <!-- /deep -->    a 48-layer narrow network with separate heads, and with probes (stop marks on the head
                                      stems), many thin layers standing for depth
  <!-- fixes --> ... <!-- /fixes -->  the 2 x 2 of Figure 7: separate heads; layer losses summing to 1 (smaller heads on thin
                                      stems); each head trains its own layer (stop marks on the stream before every layer,
                                      where the layer losses' gradient ends); both
Names and one-line descriptions are bilingual HTML under each panel; the setup tag classes match the page's."""
import re, sys
INK, IVORY, OAT, MANILLA, KRAFT, CLAY = "#141413", "#F0EEE6", "#E3DACC", "#EBDBBC", "#D4A27F", "#D97757"
W, H, D = 520, 420, 10


class P:
    def __init__(self, fid, seed):
        self.fid, self.seed, self.o = fid, seed, []

    def f(self): return f'filter="url(#{self.fid})"'

    def line(self, x1, y1, x2, y2, w=6):
        self.o.append(f'<path d="M{x1} {y1} L{x2} {y2}" stroke="{INK}" stroke-width="{w}" stroke-linecap="round" fill="none" {self.f()}/>')

    def box(self, x, y, w, h, fill, rx=16, sw=6, d=D):
        self.o.append(f'<rect x="{x + d}" y="{y + d}" width="{w}" height="{h}" rx="{rx}" fill="{fill}"/>')
        self.o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="none" stroke="{INK}" stroke-width="{sw}" stroke-linejoin="round" {self.f()}/>')

    def dot(self, cx, cy, r, fill, sw=5):
        self.o.append(f'<circle cx="{cx + 6}" cy="{cy + 6}" r="{r}" fill="{fill}"/><circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{INK}" stroke-width="{sw}" {self.f()}/>')

    def stop(self, x, y, half=20, w=5, vertical=False):  # a T-bar: the gradient ends here
        if vertical:
            self.line(x, y - half, x, y + half, w)
        else:
            self.line(x - half, y, x + half, y, w)

    def svg(self):
        return (f'<svg viewBox="0 0 {W} {H}" role="img" style="width:100%;height:auto;display:block"><defs><filter id="{self.fid}" filterUnits="userSpaceOnUse" x="0" y="0" width="{W}" height="{H}">'
                f'<feTurbulence type="fractalNoise" baseFrequency="0.012" numOctaves="2" seed="{self.seed}"/><feDisplacementMap in="SourceGraphic" scale="4"/></filter></defs>'
                f'<rect width="{W}" height="{H}" fill="{IVORY}"/>' + "".join(self.o) + "</svg>")


def three_layers(p, small_heads=False, local=False, probes=False):
    """The page's 3-layer vocabulary: stream at y 320, layers at x 82/212/342, heads above; the last head is the model's."""
    p.line(70, 320, 480, 320)
    p.dot(48, 320, 22, KRAFT)
    for i, x in enumerate((82, 212, 342)):
        p.box(x, 265, 96, 110, MANILLA if i != 1 else OAT, 20)
        cx = x + 48
        last = i == 2
        if local and not last:
            p.stop(x - 17, 320, half=22, vertical=True)  # the layer losses' gradient stops before this layer
        if small_heads and not last:
            p.line(cx, 265, cx, 196, 3)
            p.box(cx - 28, 154, 56, 42, CLAY, 12, sw=5, d=7)
            p.line(cx, 154, cx, 120, 3)
            p.dot(cx, 106, 12, CLAY, sw=4)
        else:
            p.line(cx, 265, cx, 178)
            if probes and not last:
                p.stop(cx, 222)
            p.box(cx - 40, 120, 80, 58, INK if last else CLAY, 16)
            p.line(cx, 120, cx, 92)
            p.dot(cx, 74, 17, INK if last else CLAY)


def deep(p, probes):
    """Many thin layers: 12 drawn for 48."""
    p.line(50, 330, 490, 330)
    p.dot(34, 330, 16, KRAFT)
    xs = [62 + 36 * i for i in range(12)]
    for i, x in enumerate(xs):
        p.box(x, 290, 24, 80, MANILLA if i % 2 == 0 else OAT, 8, sw=5, d=6)
        cx, last = x + 12, i == len(xs) - 1
        p.line(cx, 290, cx, 222, 4)
        if probes and not last:
            p.stop(cx, 256, half=11, w=4)
        p.box(cx - 13, 190, 26, 32, INK if last else CLAY, 8, sw=4, d=5)
        p.line(cx, 190, cx, 166, 4)
        p.dot(cx, 154, 9, INK if last else CLAY, sw=4)


def cell(svg, en_name, zh_name, en, zh, tag=None):
    t = f' <span class="setup-tag{" main" if tag and tag[0] == "Main" else ""}"><span lang="en">{tag[0]}</span><span lang="zh">{tag[1]}</span></span>' if tag else ""
    return f'    <div class="setup">{svg}<div class="setup-name"><span lang="en">{en_name}</span><span lang="zh">{zh_name}</span>{t}</div><p><span lang="en">{en}</span><span lang="zh">{zh}</span></p></div>\n'


def figure(fid, cells):
    return f'<!-- {fid} -->\n  <figure class="setups" id="{fid}">\n' + "".join(cells) + f"  </figure>\n  <!-- /{fid} -->"


cells_deep = []
for k, (probes, en_n, zh_n, en, zh, tag) in enumerate([
        (False, "Separate heads, 48 layers", "独立头，48 层", "The 130m layer stacked 48 deep, every layer predicting through a head of its own.", "130m 的层堆到 48 层，每一层都用自己的头预测。", ("Main", "主做法")),
        (True, "Probes only, 48 layers", "只加探针，48 层", "The same network with heads that take no gradient back: it trains as an ordinary 48-layer model.", "同样的网络，头的梯度不回传：它和普通的 48 层模型训练得完全一样。", ("Baseline", "基线"))]):
    p = P(f"sch-deep-{k}", 40 + k); deep(p, probes); cells_deep.append(cell(p.svg(), en_n, zh_n, en, zh, tag))
cells_fix = []
for k, (kw, en_n, zh_n, en, zh) in enumerate([
        (dict(), "Separate heads", "独立头", "Every layer loss has weight 1 and its gradient reaches every layer below it.", "每个逐层损失的权重是 1，梯度传到它下面的每一层。"),
        (dict(small_heads=True), "Layer losses summing to 1", "各层损失权重合计为 1", "Each layer loss is weighted 0.2, so the five sum to the final loss's weight.", "每个逐层损失的权重是 0.2，五个加起来等于最终损失的权重。"),
        (dict(local=True), "Each head trains its own layer", "每个头只训练自己那一层", "A layer loss trains its head and its own layer; its gradient stops before reaching the layers below.", "逐层损失只训练它的头和它所在的那一层；梯度不再传到下面的层。"),
        (dict(small_heads=True, local=True), "Both", "两者都用", "Layer losses weighted 0.2, each training only its own layer.", "逐层损失权重为 0.2，并且各自只训练自己那一层。")]):
    p = P(f"sch-fix-{k}", 50 + k); three_layers(p, **kw); cells_fix.append(cell(p.svg(), en_n, zh_n, en, zh))
page = sys.argv[1]; s = open(page).read()
for fid, cells in (("deep", cells_deep), ("fixes", cells_fix)):
    fig = figure(fid, cells)
    if f"<!-- {fid} -->" in s:
        s = re.sub(rf"<!-- {fid} -->.*?<!-- /{fid} -->", lambda m: fig, s, flags=re.S)
    else:
        anchor = '  <figure class="viz" id="fig-d48"' if fid == "deep" else '  <figure class="viz" id="fig-fix"'
        i = s.index(anchor); s = s[:i] + "  " + fig + "\n" + s[i:]
open(page, "w").write(s); print("schematics written:", len(cells_deep), "deep,", len(cells_fix), "fixes")
