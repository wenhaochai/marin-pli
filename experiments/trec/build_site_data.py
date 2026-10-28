"""Figure data of the website page exps/trec.html (Training Re-evaluation Curves), written to
<site>/assets/data/trec-figs.js as window.TREC_FIGS, one entry per figure, drawn by C.epoch (assets/article-charts.js).

Sources: the learning rates logged in W&B (cache/wandb/<run>.json, fetch_wandb.py) and the scored batches of
trec_score.py (project/trec/out/<run>/<first index>.npz). A figure whose runs have not finished is a placeholder: its
axes, legend and a status word.

  fig-trec      the final model's loss on each training batch (sum of per-sequence loss over the summed loss weights of
                the batch, 100-step centred moving average) against the training fraction t/T, and the mean loss of the
                unread batches, mean +- 2 standard errors over batches
  fig-coef      c_i, the weight of update i in the final weights, for three readings of the optimizer and the control
                without averaging (predict.py), each divided by its maximum
  fig-predict   measured TREC and the prediction 1 - c^0.5 t^m of each reading with its own best m (highest Pearson r),
                each scaled to 0 at its low point and 1 for a forgotten batch: the unread mean for the measured curve,
                c = 0 for a prediction (min-max over the run)

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
INK, BLUE, RED, GREY, CYAN, BAND = "#202124", "#1A73E8", "#D93025", "#9AA0A6", "#007B83", "#DADCE0"
PANELS = [("130m-d12", {"en": "130m, WSD", "zh": "130m，WSD"}), ("300m", {"en": "300m, cosine", "zh": "300m，cosine"})]
READINGS = [("momentum", {"en": "Fixed norm, momentum", "zh": "固定范数，带动量"}, BLUE),
            ("independent", {"en": "Fixed norm, independent steps", "zh": "固定范数，各步不相关"}, GREY),
            ("adamw-0.1", {"en": "Weight decay 0.1", "zh": "权重衰减 0.1"}, RED),
            ("lr-only", {"en": "Learning rate only", "zh": "只看学习率"}, CYAN)]   # the control (owner popup 2026-10-27)
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
        marks = [{"type": "line", "pts": thin(t, predict.coefficients(eta, r, model)), "color": col, "width": 1.8, "alpha": 0.85}
                 for r, _, col in reversed(READINGS)]   # the momentum line on top
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
        unread = data[model][1].mean()   # 1 = a forgotten batch, as c = 0 is for a prediction; the noisy maximum is not
        marks = [{"type": "line", "pts": thin(t, (s - s.min()) / (unread - s.min())), "color": INK, "width": 1.8, "alpha": 0.85}]
        for r, _, col in READINGS:
            c = predict.coefficients(eta, r, model)
            m, rp = predict.fit_m(c, s)
            fits[f"{model}/{r}"] = {"m": m, "pearson": round(rp, 4)}
            marks.append({"type": "line", "pts": thin(t, minmax(predict.predicted(c, m))), "color": col, "width": 1.8, "alpha": 0.85})
        panels.append(dict(base, marks=marks))
    return {"title": {"en": "Predicted and measured TREC shapes", "zh": "预测与实测的 TREC 形状"},
            "quantity": {"en": "Normalized loss", "zh": "归一化损失"}, "xlabel": XLABEL, "cols": 2, "height": 200,
            "legend": [[{"en": "Measured", "zh": "实测"}, INK, "line"]] + [[lab, col, "line"] for _, lab, col in READINGS],
            "panels": panels, "fits": fits}


INIT = "/scratch/gpfs/GROUP/USER/project/trec/out/init/{}.json"   # experiments/trec/init_retention.py
MATS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
ATTN_C, MLP_C = "#E37400", "#1E8E3E"   # GM2 yellow 900, green 600


def fig_init():
    """cos(W_T, W_0) of every attention and MLP matrix by layer (init_retention.py), and the fixed-norm predictions
    prod_t a_t with independent steps (A = 1) and with momentum (A of predict.A)."""
    panels = []
    for model, label in PANELS:
        rows = json.load(open(INIT.format(model)))["rows"]
        eta = predict.lr_schedule(model)
        pred = {r: float(np.exp(np.log(predict.keep(eta, r, model)).sum())) for r in ("independent", "momentum")}
        attn, mlp = [], []
        for r in rows:
            name = next((m for m in MATS if r["param"].endswith(f".{m}.weight")), None)
            if name is None:
                continue
            x = r["slice"] + (MATS.index(name) - 3) * 0.09
            (attn if MATS.index(name) < 4 else mlp).append([round(x, 3), round(r["cos"], 4)])
        L = max(p[0] for p in attn + mlp)
        marks = [{"type": "line", "pts": [[-0.5, round(pred["independent"], 4)], [L + 0.5, round(pred["independent"], 4)]], "color": GREY, "width": 1.8, "alpha": 0.85},
                 {"type": "line", "pts": [[-0.5, round(pred["momentum"], 6)], [L + 0.5, round(pred["momentum"], 6)]], "color": BLUE, "width": 1.8, "alpha": 0.85},
                 {"type": "dots", "pts": attn, "color": ATTN_C}, {"type": "dots", "pts": mlp, "color": MLP_C}]
        n = int(round(L))
        panels.append({"label": label, "xlim": [-0.6, n + 0.6], "ylim": [-0.05, 1.1], "xticks": [[k, str(k)] for k in range(0, n + 1, 1 if n < 8 else 2)],
                       "yticks": ticks(0, 1, 0.25, 2), "marks": marks, "pred": pred})
    fits = {PANELS[i][0]: p.pop("pred") for i, p in enumerate(panels)}
    return {"title": {"en": "Initialization kept in the final weights", "zh": "最终权重保留的初始化"},
            "quantity": {"en": "Cosine of the final and initial matrix", "zh": "最终矩阵与初始矩阵的余弦"},
            "xlabel": {"en": "Layer", "zh": "层"}, "cols": 2, "height": 200,
            "legend": [[{"en": "Attention", "zh": "注意力"}, ATTN_C, "dot"], [{"en": "MLP", "zh": "MLP"}, MLP_C, "dot"],
                       [{"en": "Independent steps", "zh": "各步不相关"}, GREY, "line"], [{"en": "Momentum", "zh": "带动量"}, BLUE, "line"]],
            "panels": panels, "pred": fits}


W438 = "/scratch/gpfs/GROUP/USER/project/marin/logs/vocab_trec"   # the vocab session's wq w438 and w457, read-only
BYTES = "/scratch/gpfs/GROUP/USER/tmp_plots/vocab/trec/trec_bytes_{}.npz"   # exact UTF-8 bytes of each target token
TOKENIZERS = [("300m", "128K", [("-s1-rerun", "-rep8-s1")], 128000, "#5F6368"),   # 128K seed 0 has no unique-token run
              ("300m-8k", "8K", [("-v8k-s1", "-v8k-rep8-s1"), ("-v8k", "-v8k-rep8")], 8000, "#681DA8"),
              ("300m-bytes", {"en": "Bytes", "zh": "字节"}, [("-bytes", "-bytes-rep8"), ("-bytes-s1", "-bytes-rep8-s1")], 256, "#B80672")]
# the vocab page's colours; (unique, 8-pass) checkpoint pairs per training seed, the second seed scored by wq w457 (owner, 2026-10-27)


def tokenizer_curve(model, u, r, k):
    """d(t) of one (unique, 8-pass) pair: (block-centre training fraction, smoothed difference minus the unread level)."""
    P = "muonh-qwen3-300m-della4xh100"
    a, b = np.load(f"{W438}/{P}{u}.npz"), np.load(f"{W438}/{P}{r}.npz")
    byf = np.load(BYTES.format(k))
    by = byf["seq_bytes"]
    steps, T = a["steps"], int(a["total_steps"])
    assert np.array_equal(steps, b["steps"]) and np.array_equal(steps, byf["steps"]) and T == predict.TOTAL[model], (model, T)
    bpb = lambda z: z["seq_loss"].sum(1) / np.log(2) / by.sum(1)
    d = bpb(a) - bpb(b)
    tr = steps < T
    unread = (steps >= T) & (steps != steps[steps >= T].min())   # the first tail block shares mixture block with step T-1
    x = steps[tr] / T
    sm = np.convolve(d[tr], np.ones(16) / 16, mode="valid") - d[unread].mean()
    xm = np.convolve(x, np.ones(16) / 16, mode="valid")
    fresh = xm > 0.125 + 8 / len(x)
    return xm[fresh], sm[fresh]


def fig_tokenizers():
    """The vocab session's recipe (tmp_plots/vocab/trec/trec_check_pred.py): on 192 blocks spread over each unique run,
    d(t) = bits per byte of the unique model minus that of its 8-pass partner, which never read the blocks after the
    first eighth, so batch difficulty cancels there; a valid moving mean over 16 sampled blocks, kept after the subset;
    minus d on the unread tail, the level of a batch the unique model has forgotten. One line per training seed (data
    seed 42 for all, so every seed scores the same blocks). The low points come from the momentum reading,
    1 - sqrt(c) t^m, with m fitted to the 128K curve and kept for 8K and bytes; each ring sits on the seeds' mean curve."""
    curves, lows = [], []
    for model, label, pairs, k, col in TOKENIZERS:
        curves.append((model, col, [(u, *tokenizer_curve(model, u, r, k)) for u, r in pairs]))
    # m from the 128K curve: the prediction's shape against the measured one on the same t
    m128 = None
    for model, col, seeds in curves:
        c = predict.coefficients(predict.lr_schedule(model), "momentum", model)
        T = len(c)
        xm = seeds[0][1]
        mean = np.mean([sm for _, _, sm in seeds], axis=0)
        if m128 is None:
            idx = np.clip((xm * T).round().astype(int) - 1, 0, T - 1)
            ms = np.arange(0.0, 20.01, 0.1)
            rs = [predict.pearson(predict.predicted(c, m)[idx], mean) for m in ms]
            m128 = float(ms[int(np.argmax(rs))])
        v = predict.valley(c, m128)
        lows.append((model, col, v, float(np.interp(v, xm, mean))))
    marks = [{"type": "line", "pts": [[round(float(p), 4), round(float(q), 5)] for p, q in zip(xm, sm)], "color": col, "width": 1.8, "alpha": 0.85}
             for _, col, seeds in curves for _, xm, sm in seeds]
    marks += [{"type": "rings", "pts": [[round(v, 4), round(y, 5)]], "color": col} for _, col, v, y in lows]
    lo = min(float(sm.min()) for _, _, seeds in curves for _, _, sm in seeds)
    step = 0.01
    ylo = np.floor(lo * 1.15 / step) * step
    yhi = 0.01
    panel = {"xlim": [0, 1], "ylim": [round(ylo, 4), yhi], "xticks": XTICKS, "yticks": ticks(ylo, yhi, step, 2), "marks": marks}
    meas = {u: [round(float(xm[np.argmin(sm)]), 3), round(float(sm.min()), 4)] for _, _, seeds in curves for u, xm, sm in seeds}
    return {"title": {"en": "TRECs of three tokenizers at 300m", "zh": "300m 上三种 tokenizer 的 TREC"},
            "quantity": {"en": "Loss minus unread level (bits per byte)", "zh": "损失减去未读水平（每字节比特数）"}, "xlabel": XLABEL,
            "cols": 1, "height": 240,
            "legend": [[lab, col, "line"] for _, lab, _, _, col in TOKENIZERS] + [[{"en": "Predicted low point", "zh": "预测的最低点"}, "#5F6368", "ring"]],
            "panels": [panel], "fit": {"m_from_128k": m128, "predicted_low": {m: round(v, 3) for m, _, v, _ in lows}, "measured_low": meas}}


# Figures 6 and 7 (owner popups 2026-10-27): the other MuonH checkpoints, scored by wq w513 and w514
# (scripts/della/trec-runs-*.sbatch); placeholders until those jobs end.
SETTINGS = [({"en": "Depth, 130m", "zh": "深度，130m"}, ["1", "2", "3", "4", "5", "12", "24", "36"]),
            ({"en": "Learning rate after T/8, 300m 8K", "zh": "T/8 之后的学习率，300m 8K"}, ["×1", "×2", "×3", "×4"]),
            ({"en": "EMA weights, 300m", "zh": "EMA 权重，300m"}, [{"en": "0.999 raw", "zh": "0.999 原始"}, {"en": "0.999 EMA", "zh": "0.999 EMA"},
                                                              {"en": "0.9998 raw", "zh": "0.9998 原始"}, {"en": "0.9998 EMA", "zh": "0.9998 EMA"}]),
            ({"en": "Batch, 300m", "zh": "batch，300m"}, ["128", "96"]),
            ({"en": "Tokenizer, 300m", "zh": "tokenizer，300m"}, ["8K", "16K", "32K", "64K", "128K", {"en": "bytes", "zh": "字节"}]),
            ({"en": "Training seed, 300m 128K then 8K", "zh": "训练种子，300m，先 128K 后 8K"}, ["0", "1", "0 ", "1 "])]
MULTIPASS = [{"en": "520m, 1.04 passes", "zh": "520m，1.04 遍"}, {"en": "1.2b, 2.40 passes", "zh": "1.2b，2.40 遍"},
             {"en": "130m, 10x steps, 2.60 passes", "zh": "130m，10 倍步数，2.60 遍"}]


def fig_settings():
    panels = [{"xlim": [-0.6, len(cats) - 0.4], "ylim": [0.5, 1.0], "yticks": ticks(0.5, 0.9, 0.1, 1), "cats": cats, "label": label,
               "empty": QUEUED, "marks": []} for label, cats in SETTINGS]
    return {"title": {"en": "TREC low points", "zh": "TREC 最低点"},
            "quantity": {"en": "Low point (training fraction)", "zh": "最低点（训练进度）"}, "cols": 2, "height": 170,
            "legend": [[{"en": "Measured", "zh": "实测"}, INK, "dot"], [{"en": "Predicted", "zh": "预测"}, INK, "ring"]],
            "panels": panels}


def fig_multipass():
    panels = [{"xlim": [0, 1], "ylim": [2.5, 3.5], "xticks": XTICKS, "yticks": ticks(2.5, 3.25, 0.25, 2), "label": label,
               "empty": QUEUED, "marks": []} for label in MULTIPASS]
    return {"title": {"en": "TRECs of runs that read their data more than once", "zh": "数据读了不止一遍的 run 的 TREC"},
            "quantity": {"en": "Loss (nats)", "zh": "损失（nats）"}, "xlabel": XLABEL, "cols": 1, "height": 150,
            "legend": [[{"en": "Seed 0", "zh": "种子 0"}, BLUE, "line"], [{"en": "Seed 1", "zh": "种子 1"}, "#8AB4F8", "line"]],
            "panels": panels}


def main():
    site = sys.argv[1]
    data = {m: scored(m) for m, _ in PANELS}
    figs = {"fig-trec": fig_trec(data), "fig-coef": fig_coef(), "fig-init": fig_init(), "fig-predict": fig_predict(data), "fig-tokenizers": fig_tokenizers(),
            "fig-settings": fig_settings(), "fig-multipass": fig_multipass()}
    path = os.path.join(site, "assets/data/trec-figs.js")
    with open(path, "w") as f:
        f.write("/* Generated by experiments/trec/build_site_data.py (marin branch trec); regenerate rather than hand-edit. */\n")
        f.write("window.TREC_FIGS = " + json.dumps(figs, ensure_ascii=False, separators=(",", ":")) + ";\n")
    print(path, {k: [("empty" if "empty" in p else "data") for p in v["panels"]] for k, v in figs.items()}, figs["fig-predict"]["fits"], figs["fig-tokenizers"]["fit"], figs["fig-init"]["pred"])


if __name__ == "__main__":
    main()
