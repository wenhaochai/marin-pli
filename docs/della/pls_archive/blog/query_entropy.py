"""A1 of the 2026-10-21 survey: is how sharply each AttnRes sublayer reads its sources set by the length of its query?
Source: the depth check-ups in checkup/ (attnres_checkup.py, k3_checkup_v3.py of marin-pls-auto10), one file per model:
full / full-sep (Full AttnRes, probes / separate heads), block / block-sep (Block AttnRes), k3 (Kimi K3).
Per sublayer (L<i>.attn, L<i>.mlp): the query norm |w| (queries.norm), the logit scale s = |w| sqrt(H) (a source v enters
the softmax as (v . w) / RMS(v) = s cos(v, w), in our models and in K3; H = 512 ours, 7168 K3), the read's entropy over its sources (attn_in_entropy,
mlp_in_entropy, in nats, mean over tokens), its normalised entropy H / log(n_sources) and its largest mean weight.
Prints, per model, the medians, the share of near-binary reads (largest weight > 0.9) and the Spearman correlation of |w|
with the normalised entropy; then the same correlation pooled over the models with block sources (block, block-sep, k3:
at most 9 sources each), with s in place of |w|. Run on a vis node: nice -n 19 python3 query_entropy.py
"""
import json
import math
from pathlib import Path

HERE = Path(__file__).parent / "checkup"
WIDTH = {"k3": 7168}


def rank(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = rank(a), rank(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    return cov / math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))


def med(v):
    s = sorted(v)
    return s[len(s) // 2]


def rows(d, model, width=512):
    norm = dict(zip(d["queries"]["names"], d["queries"]["norm"]))
    out = []
    for L in d["layers"]:
        for kind in ("attn", "mlp"):
            p = L[f"{kind}_in_probs"]
            if len(p) < 2:
                continue  # one source: nothing to choose
            h = L[f"{kind}_in_entropy"]
            out.append(dict(name=f"L{L['layer']}.{kind}", layer=L["layer"], norm=norm[f"L{L['layer']}.{kind}"], s=norm[f"L{L['layer']}.{kind}"] * math.sqrt(width), h=h,
                            hn=h / math.log(len(p)), pmax=max(p), n=len(p), model=model, depth=L["layer"] / d["n_layers"]))
    return out


if __name__ == "__main__":
    pooled = []
    for f in ("full", "full-sep", "block", "block-sep", "k3"):
        d = json.load(open(HERE / f"{f}.json"))
        r = rows(d, f, WIDTH.get(f, 512))
        if f in ("block", "block-sep", "k3"):
            pooled += r
        nm, hn, pm = [x["norm"] for x in r], [x["hn"] for x in r], [x["pmax"] for x in r]
        print(f"{f:10s} {d['model']:20s} {len(r):3d} reads  |w| median {med(nm):6.3f}  H/logn median {med(hn):.3f}  "
              f"s median {med([x['s'] for x in r]):5.1f}  max-weight median {med(pm):.3f}  binary(>0.9) {100 * sum(p > 0.9 for p in pm) / len(pm):5.1f}%  "
              f"spearman(|w|, H/logn) {spearman(nm, hn):+.2f}")
        for k, (lo, hi) in enumerate(((0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1))):
            s = [x for x in r if lo <= x["depth"] < hi]
            if s:
                print(f"    depth third {k + 1} (layers {s[0]['layer']}-{s[-1]['layer']}): |w| {med([x['norm'] for x in s]):6.3f}  s {med([x['s'] for x in s]):5.1f}  H/logn {med([x['hn'] for x in s]):.3f}  "
                      f"max-weight {med([x['pmax'] for x in s]):.3f}")
    print(f"pooled block-source reads ({len(pooled)}): spearman(s, H/logn) {spearman([x['s'] for x in pooled], [x['hn'] for x in pooled]):+.2f}, "
          f"spearman(|w|, H/logn) {spearman([x['norm'] for x in pooled], [x['hn'] for x in pooled]):+.2f}")
    for lo, hi in ((0, 5), (5, 10), (10, 20), (20, 40), (40, 1e9)):
        b = [x for x in pooled if lo <= x["s"] < hi]
        if b:
            print(f"    s in [{lo}, {hi}): {len(b):3d} reads (k3 {sum(x['model'] == 'k3' for x in b)})  H/logn median {med([x['hn'] for x in b]):.3f}")
