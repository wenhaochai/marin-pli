"""Setup schematics of exps/trec.html in the site's illustration style (tools/exp-art/style.js colours), written between
the <!-- setups:trec-q1 --> and <!-- setups:trec-q2 --> markers of the page.

q1: the two runs, each a learning-rate schedule above its batches read in order, and the final model that scores them.
q2: the readings of one optimizer step on a weight vector (owner popup 2026-10-27: four, two by two): fixed norm with
    momentum (MuonH as it runs: the update tilts outward, the rescaling to the old norm shrinks more), fixed norm with
    independent steps (an update at right angles, then the rescaling), weight decay (shrink by 1 - eta*lambda, then add
    the update), and the control, learning rate only (add the update).

    python experiments/trec/setups_svg.py <page>
"""

import math
import re
import sys

PAPER, INK, LINE, WIRE, REST, RULE, TINT, BLUE, BLUE2 = "#FFFFFF", "#202124", "#5F6368", "#9AA0A6", "#F1F3F4", "#DADCE0", "#E8F0FE", "#1A73E8", "#8AB4F8"
SW = 3.6


def svg(body):
    return ('<svg viewBox="0 0 520 420" role="img" style="width:100%;height:auto;display:block"><rect width="520" height="420" fill="#FFFFFF"/>'
            + "".join(body) + "</svg>")


def run_panel(sched, n=8):
    x0, x1, base, h = 60, 400, 190, 110
    pts = [(x0 + (x1 - x0) * k / 80, base - h * sched(k / 80)) for k in range(81)]
    d = " L".join(f"{x:.1f} {y:.1f}" for x, y in pts)
    body = [f'<path d="M{x0} {base} L{d} L{x1} {base} Z" fill="{REST}"/>',
            f'<path d="M{d}" fill="none" stroke="{LINE}" stroke-width="{SW}" stroke-linecap="round" stroke-linejoin="round"/>']
    gap = (x1 - x0) / n
    ts = min(30, 0.62 * gap)
    for i in range(n):
        cx = x0 + gap * (i + 0.5)
        body.append(f'<rect x="{cx - ts / 2:.1f}" y="{250 - ts / 2:.1f}" width="{ts:.1f}" height="{ts:.1f}" rx="{min(7, ts / 4):.1f}" fill="{TINT}" stroke="{LINE}" stroke-width="{min(SW, ts / 8):.2f}"/>')
        body.append(f'<circle cx="{cx:.1f}" cy="300" r="{min(6, ts / 5):.1f}" fill="{BLUE}"/>')
    body.append(f'<rect x="430" y="228" width="44" height="44" rx="10" fill="{INK}"/>')
    return svg(body)


def vec(x1, y1, x2, y2, color, w=SW):
    return f'<path d="M{x1:.1f} {y1:.1f} L{x2:.1f} {y2:.1f}" stroke="{color}" stroke-width="{w}" stroke-linecap="round" fill="none"/>'


def pt(x, y, fill, r=9):
    stroke = "" if fill in (INK, BLUE, BLUE2) else f' stroke="{LINE}" stroke-width="{SW}"'
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}"{stroke}/>'


def step_panel(kind):
    cx, cy, R = 250, 250, 140
    a = math.radians(105)
    ax, ay = cx + R * math.cos(a), cy - R * math.sin(a)
    tx, ty = math.sin(a), math.cos(a)        # tangent pointing clockwise (to the right at the top)
    U = 86                                    # the update's length
    body = [f'<circle cx="{cx}" cy="{cy}" r="{R}" fill="none" stroke="{RULE}" stroke-width="{SW}"/>', pt(cx, cy, INK, 7)]
    if kind in ("fixed", "momentum"):
        tilt = math.radians(32 if kind == "momentum" else 0)   # momentum: part of the update points outward
        rx, ry = (ax - cx) / R, (ay - cy) / R
        bx, by = ax + U * (tx * math.cos(tilt) + rx * math.sin(tilt)), ay + U * (ty * math.cos(tilt) + ry * math.sin(tilt))
        n = math.hypot(bx - cx, by - cy)
        nx, ny = cx + (bx - cx) * R / n, cy + (by - cy) * R / n
        body += [vec(cx, cy, ax, ay, LINE), vec(ax, ay, bx, by, BLUE2), vec(bx, by, nx, ny, WIRE, 3),
                 vec(cx, cy, nx, ny, BLUE), pt(ax, ay, REST), pt(bx, by, BLUE2, 7), pt(nx, ny, BLUE)]
    elif kind == "decay":
        s = 0.72
        sx, sy = cx + s * (ax - cx), cy + s * (ay - cy)
        bx, by = sx + U * tx, sy + U * ty
        body += [vec(cx, cy, ax, ay, LINE), vec(ax, ay, sx, sy, WIRE, 3), vec(sx, sy, bx, by, BLUE2),
                 vec(cx, cy, bx, by, BLUE), pt(ax, ay, REST), pt(sx, sy, REST, 7), pt(bx, by, BLUE)]
    else:
        bx, by = ax + U * tx, ay + U * ty
        body += [vec(cx, cy, ax, ay, LINE), vec(ax, ay, bx, by, BLUE2), vec(cx, cy, bx, by, BLUE), pt(ax, ay, REST), pt(bx, by, BLUE)]
    return svg(body)


def L(en, zh):
    return f'<span lang="en">{en}</span><span lang="zh">{zh}</span>'


def setup(pic, name, text, tag=None):
    t = f' <span class="setup-tag main">{L("Main", "主做法")}</span>' if tag == "main" else ""
    return f'<div class="setup">{pic}<div class="setup-name">{name}{t}</div><p>{text}</p></div>'


WSD = lambda s: 1.0 if s < 0.2 else 1 - (s - 0.2) / 0.8
COS = lambda s: s / 0.087 if s < 0.087 else 0.5 * (1 + math.cos(math.pi * (s - 0.087) / 0.913))
Q1 = [setup(run_panel(WSD), L("130m, 12 layers, WSD", "130m，12 层，WSD"),
            L("The learning rate stays at 0.02 for the first fifth of 4959 steps and falls linearly to zero; the final model scores every batch.",
              "学习率在 4959 步的前五分之一保持 0.02，之后线性降到零；最终模型给每一批打分。")),
      setup(run_panel(COS), L("300m, cosine", "300m，cosine"),
            L("The learning rate warms up to 0.01 over 1000 steps and follows a cosine to zero at step 11444; the final model scores every batch.",
              "学习率用 1000 步预热到 0.01，之后按 cosine 在第 11444 步降到零；最终模型给每一批打分。"))]
Q2 = [setup(step_panel("momentum"), L("Fixed norm, momentum", "固定范数，带动量"),
            L("MuonH as it runs: momentum tilts each update outward, so rescaling to the old norm shrinks the old weights by about 1 − Aη²/2, with A = 23 at 130m and 73 at 300m.",
              "MuonH 的实际做法：动量让每次更新带一点向外的分量，缩放回原来的范数时，旧权重约乘以 1 − Aη²/2；130m 上 A = 23，300m 上 A = 73。"), tag="main"),
      setup(step_panel("fixed"), L("Fixed norm, independent steps", "固定范数，各步不相关"),
            L("The same rescaling if every update were at right angles to the weights: the old weights shrink by 1/√(1 + η²).",
              "同样的缩放，但假设每次更新都与权重垂直：旧权重乘以 1/√(1 + η²)。")),
      setup(step_panel("decay"), L("Weight decay 0.1", "权重衰减 0.1"),
            L("The config's unused weight decay read as AdamW: the old weights shrink by 1 − 0.1η, then the update is added.",
              "把配置里没被使用的权重衰减当作 AdamW：旧权重先乘以 1 − 0.1η，再加上更新。")),
      setup(step_panel("plain"), L("Learning rate only", "只看学习率"),
            L("The control: nothing shrinks, so each update keeps the weight its learning rate gave it.",
              "对照：什么都不缩小，每次更新保留学习率给它的权重。"))]


Q3 = [setup(run_panel(COS, 6), L("Marin tokenizer, 128K", "Marin tokenizer，128K"),
            L("The text takes 11444 steps of 128 sequences of 4096 tokens.", "这段文本切成 11444 步，每步 128 条长 4096 个 token 的序列。")),
      setup(run_panel(COS, 8), L("8K truncation", "8K 截短版"),
            L("Smaller tokens: the same text takes 15464 steps.", "词更短：同样的文本切成 15464 步。")),
      setup(run_panel(COS, 28), L("Bytes", "字节"),
            L("One byte per token: the same text takes 54248 steps.", "每个字节一个 token：同样的文本切成 54248 步。"))]


def main():
    page = sys.argv[1]
    s = open(page).read()
    for key, items, cls in (("trec-q1", Q1, "setups two"), ("trec-q2", Q2, "setups two"), ("trec-q3", Q3, "setups")):
        a, b = f"<!-- setups:{key} -->", f"<!-- /setups:{key} -->"
        assert s.count(a) == 1 and s.count(b) == 1, key
        s = re.sub(re.escape(a) + ".*?" + re.escape(b), lambda m: a + f'<figure class="{cls}">' + "".join(items) + "</figure>" + b, s, flags=re.S)
    open(page, "w").write(s)
    print("setups written into", page)


if __name__ == "__main__":
    main()
