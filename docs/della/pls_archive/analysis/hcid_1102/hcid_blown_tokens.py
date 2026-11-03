"""w810's blown tokens at the final checkpoint.
Left: state norm over embedding norm by layer, median and 99th percentile over 4 x 1024 Paloma c4_en validation tokens, for w810 (HC mixing = I, separate
heads) and HC separate heads (../../pls_hcid/blown_state.json, blown_fig_data.py).
Right: w810 only. By layer, the part of the training-loss gradient (down_proj and o_proj) carried by 134 tokens, (134 / 4092) |mean gradient of those tokens|, on the
same text: the blown tokens (solid) and 134 random other tokens (dashed). ../../pls_hcid/blown_coherence.json (blown_coherence.py, two halves of each set).
Output: hcid_blown_tokens.{pdf,png}."""
import json
from pathlib import Path
import numpy as np
from style import *
HERE = Path(__file__).parent
apply_style()
D = HERE.parents[1] / "pls_hcid"
st = json.load(open(D / "blown_state.json")); gr = json.load(open(D / "blown_grad.json"))
W, H = "-sep-ahc-hcid-fa4thd-vs-d48", "-sep-ahc-fa4thd-vs-d48"
fig, axes = canvas(1, 2, width=WIDTH_POST, panel_height=2.0, title=None,
                   legend=[("HC, mixing = I, separate heads", BLUE), ("HC, separate heads", GREY)], title_pt=9.2, tick_pt=TEXT_PT)
ax, bx = np.ravel(axes)
x = np.arange(1, 49)
ax.set_yscale("log")
ax.plot(x, st[W]["p99"], color=BLUE, lw=1.6); ax.plot(x, st[W]["p50"], color=BLUE, lw=1.6, ls=(0, (3, 2)))
ax.plot(x, st[H]["p99"], color=GREY, lw=1.2); ax.plot(x, st[H]["p50"], color=GREY, lw=1.2, ls=(0, (3, 2)))
ax.set_xlim(1, 48); ax.set_xticks([1, 12, 24, 36, 48]); ax.set_ylim(1, 1e8)
class Pow:
    def format(self, v):
        return "1" if v == 1 else "10\u00b2" if v == 1e2 else "10\u2074" if v == 1e4 else "10\u2076"
y_values(ax, [1, 1e2, 1e4, 1e6], Pow()); room(ax)
panel_label(ax, "Relative state norm")
co = json.load(open(D / "blown_coherence.json"))
L = sorted(map(int, co))
def part(r, u, v, cuv):   # (n / N) |mean over the two halves|, n = 134, N = 4092: the two halves are equal in size
    a, b = r[u], r[v]
    return 134 / 4092 * 0.5 * float(np.sqrt(a * a + b * b + 2 * a * b * r[cuv]))
blown = [part(co[str(l)], "A", "B", "cos_AB") for l in L]; other = [part(co[str(l)], "C", "D", "cos_CD") for l in L]
bx.plot(L, blown, color=BLUE, lw=1.6); bx.plot(L, other, color=BLUE, lw=1.6, ls=(0, (3, 2)))
bx.set_xlim(0, 25); bx.set_xticks([1, 8, 16, 24])
nice_y(bx, 0, 20, zero=True, fmt="{:.0f}"); room(bx)
panel_label(bx, "Gradient norm")
save(fig, HERE / "hcid_blown_tokens")
print("blown", [round(v, 1) for v in blown]); print("other", [round(v, 1) for v in other])
