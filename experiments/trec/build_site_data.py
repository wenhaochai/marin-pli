"""Figure data of the website page exps/trec.html (Training Re-evaluation Curves), written to
<site>/assets/data/trec-figs.js as window.TREC_FIGS, one entry per figure, drawn by C.epoch (assets/article-charts.js).

Sources: the learning rates logged in W&B (cache/wandb/<run>.json, fetch_wandb.py) and the scored batches of
trec_score.py (project/trec/out/<run>/<first index>.npz). A figure whose runs have not finished is a placeholder: its
axes, legend and a status word.

  fig-trec      the final model's loss on each training batch (sum of per-sequence loss over the summed loss weights of
                the batch, 100-step centred moving average) against the training fraction t/T, and the mean loss of the
                unread batches, mean +- 2 standard errors over batches
  fig-coef      c_i, the weight of update i in the final weights, for three readings of the optimizer (predict.py), each
                divided by its maximum
  fig-predict   measured TREC and the prediction 1 - c^0.5 t^m of each reading with its own best m (highest Pearson r),
                each min-max scaled over the run

    python experiments/trec/build_site_data.py <site checkout>     # vis node
"""

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import predict  # noqa: E402

OUT = "/scratch/gpfs/GROUP/USER/project/trec/out"
SMOOTH = 100
N_PLOT = 400   # points per drawn curve
INK, BLUE, RED, GREY, BAND = "#202124", "#1A73E8", "#D93025", "#9AA0A6", "#DADCE0"
PANELS = [("130m-d12", {"en": "130m, WSD", "zh": "130m，WSD"}), ("300m", {"en": "300m, cosine", "zh": "300m，cosine"})]
READINGS = [("fixed-norm", {"en": "Fixed norm", "zh": "固定范数"}, BLUE),
            ("adamw-0.1", {"en": "Weight decay 0.1", "zh": "权重衰减 0.1"}, RED),
            ("schedule", {"en": "Schedule only", "zh": "只看学习率调度"}, GREY)]
QUEUED, PLANNED = {"en": "Queued", "zh": "排队中"}, {"en": "Planned", "zh": "计划中"}
XLABEL = {"en": "Training fraction", "zh": "训练进度"}
XTICKS = [[v, f"{v:g}"] for v in (0, 0.25, 0.5, 0.75, 1)]


def ticks(lo, hi, step, digits):
    return [[round(v, 6), f"{v:.{digits}f}"] for v in np.arange(lo, hi + step / 2, step)]


def scored(model):
    """(training-step losses [T] or None, unread-batch losses [n]) of a finished TREC, else None."""
    files = sorted(glob.glob(os.path.join(OUT, predict.RUNS[model], "[0-9]*.npz")))
    if not files:
        return None
    idx, loss = [], []
    for f in files:
        z = np.load(f)
        idx.append(z["index"])
        loss.append(z["seq_loss"].sum(1) / z["seq_weight"].sum(1))
    idx, loss = np.concatenate(idx), np.concatenate(loss)
    T = predict.TOTAL[model]
    train = loss[idx < T]
    if not np.array_equal(np.sort(idx[idx < T]), np.arange(T)):
        return None   # not every training step scored yet
    order = np.argsort(idx[idx < T])
    return train[order], loss[idx >= T]


def smooth(v, w=SMOOTH):
    """Centred moving average; the window shrinks at both ends."""
    c = np.concatenate([[0.0], np.cumsum(v)])
    i = np.arange(len(v))
    lo, hi = np.maximum(i - w // 2, 0), np.minimum(i + w - w // 2, len(v))
    return (c[hi] - c[lo]) / (hi - lo)


def thin(t, v, n=N_PLOT):
    k = np.unique(np.linspace(0, len(t) - 1, n).round().astype(int))
    return [[round(float(t[j]), 5), round(float(v[j]), 5)] for j in k]


def minmax(v):
    return (v - v.min()) / (v.max() - v.min())


def fig_trec(data):
    panels = []
    for model, label in PANELS:
        d = data[model]
        if d is None:
            lo = {"130m-d12": 3.0, "300m": 2.75}[model]   # placeholder spans, 0.5 nats around the final train loss
            panels.append({"label": label, "xlim": [0, 1], "ylim": [lo, lo + 0.5], "xticks": XTICKS, "yticks": ticks(lo, lo + 0.5, 0.1, 1),
                           "empty": QUEUED, "marks": []})
            continue
        train, unread = d
        s = smooth(train)
        t = (np.arange(len(s)) + 1) / len(s)
        m, se = unread.mean(), unread.std(ddof=1) / np.sqrt(len(unread))
        panels.append({"label": label, "xlim": [0, 1], "marks": [
            {"type": "band", "lo": round(float(m - 2 * se), 5), "hi": round(float(m + 2 * se), 5), "color": BAND},
            {"type": "line", "pts": thin(t, s), "color": INK, "width": 1.8, "alpha": 0.85}], "s": s, "unread": m})
    have = [p for p in panels if "s" in p]
    if have:   # one span for every panel, each centred on its own data (writing:plot rule 7)
        span = max(max(p["s"].max(), p["unread"]) - p["s"].min() for p in have) * 1.15
        step = 0.02 if span < 0.12 else 0.05 if span < 0.3 else 0.1
        span = np.ceil(span / step) * step
        for p in have:
            mid = (max(p["s"].max(), p["unread"]) + p["s"].min()) / 2
            lo = np.floor((mid - span / 2) / step) * step
            p.update(ylim=[round(lo, 4), round(lo + span, 4)], xticks=XTICKS, yticks=ticks(lo, lo + span, step, 2))
            del p["s"], p["unread"]
    return {"title": {"en": "TRECs of the 130m and 300m runs", "zh": "130m 和 300m 两个 run 的 TREC"},
            "quantity": {"en": "Loss (nats)", "zh": "损失（nats）"}, "xlabel": XLABEL, "cols": 2, "height": 220,
            "legend": [[{"en": "Final-model loss", "zh": "最终模型的损失"}, INK, "line"], [{"en": "Unread batches", "zh": "没读过的批次"}, BAND, "box"]],
            "panels": panels}


def fig_coef():
    panels = []
    for model, label in PANELS:
        eta = predict.lr_schedule(model)
        t = (np.arange(len(eta)) + 1) / len(eta)
        marks = [{"type": "line", "pts": thin(t, predict.coefficients(eta, r)), "color": col, "width": 1.8, "alpha": 0.85}
                 for r, _, col in reversed(READINGS)]   # the fixed-norm line on top
        panels.append({"label": label, "xlim": [0, 1], "ylim": [0, 1.15], "xticks": XTICKS, "yticks": ticks(0, 1, 0.25, 2), "marks": marks})
    return {"title": {"en": "Weight of each update in the final model", "zh": "每次更新在最终模型里的权重"},
            "quantity": {"en": "Share of the largest update", "zh": "相对最大一次更新的份额"}, "xlabel": XLABEL, "cols": 2, "height": 200,
            "legend": [[lab, col, "line"] for _, lab, col in READINGS], "panels": panels}


def fig_predict(data):
    panels, fits = [], {}
    for model, label in PANELS:
        base = {"label": label, "xlim": [0, 1], "ylim": [0, 1.15], "xticks": XTICKS, "yticks": ticks(0, 1, 0.25, 2)}
        if data[model] is None:
            panels.append(dict(base, empty=PLANNED, marks=[]))
            continue
        s = smooth(data[model][0])
        t = (np.arange(len(s)) + 1) / len(s)
        eta = predict.lr_schedule(model)
        marks = [{"type": "line", "pts": thin(t, minmax(s)), "color": INK, "width": 1.8, "alpha": 0.85}]
        for r, _, col in READINGS[:2]:
            c = predict.coefficients(eta, r)
            m, rp = predict.fit_m(c, s)
            fits[f"{model}/{r}"] = {"m": m, "pearson": round(rp, 4)}
            marks.append({"type": "line", "pts": thin(t, minmax(predict.predicted(c, m))), "color": col, "width": 1.8, "alpha": 0.85})
        panels.append(dict(base, marks=marks))
    return {"title": {"en": "Predicted and measured TREC shapes", "zh": "预测与实测的 TREC 形状"},
            "quantity": {"en": "Normalized loss", "zh": "归一化损失"}, "xlabel": XLABEL, "cols": 2, "height": 200,
            "legend": [[{"en": "Measured", "zh": "实测"}, INK, "line"]] + [[lab, col, "line"] for _, lab, col in READINGS[:2]],
            "panels": panels, "fits": fits}


def main():
    site = sys.argv[1]
    data = {m: scored(m) for m, _ in PANELS}
    figs = {"fig-trec": fig_trec(data), "fig-coef": fig_coef(), "fig-predict": fig_predict(data)}
    path = os.path.join(site, "assets/data/trec-figs.js")
    with open(path, "w") as f:
        f.write("/* Generated by experiments/trec/build_site_data.py (marin branch trec); regenerate rather than hand-edit. */\n")
        f.write("window.TREC_FIGS = " + json.dumps(figs, ensure_ascii=False, separators=(",", ":")) + ";\n")
    print(path, {k: [("empty" if "empty" in p else "data") for p in v["panels"]] for k, v in figs.items()}, figs["fig-predict"]["fits"])


if __name__ == "__main__":
    main()
