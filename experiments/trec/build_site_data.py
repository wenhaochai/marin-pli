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
# (scripts/della/trec-runs-*.sbatch): 1 training batch in 8 (blocks of 8 steps every 64, plus the last 8), out/runs/<run>[-ema].
# The two fully scored runs (w465) enter through the same 1-in-8 blocks.
P3, P1 = "muonh-qwen3-300m-della4xh100", "muonh-qwen3-130m-della4xh100"
R128 = "muonh-qwen3-300m-della8xh100-pls0-noev"   # 300m 128K, training seed 0: the reference of the batch and tokenizer panels
SETTINGS = [({"en": "Depth, 130m", "zh": "深度，130m"}, ["1", "2", "3", "4", "5", "12", "24", "36"],
             [f"{P1}-d{k}" for k in (1, 2, 3, 4, 5, 12, 24, 36)]),
            ({"en": "Learning rate after T/8, 300m 8K", "zh": "T/8 之后的学习率，300m 8K"}, ["×1", "×2", "×3", "×4"],
             [f"{P3}-v8k"] + [f"{P3}-lrx{k}-fp2-v8k" for k in (2, 3, 4)]),
            ({"en": "EMA weights, 300m", "zh": "EMA 权重，300m"}, [{"en": "0.999 raw", "zh": "0.999 原始"}, {"en": "0.999 EMA", "zh": "0.999 EMA"},
                                                              {"en": "0.9998 raw", "zh": "0.9998 原始"}, {"en": "0.9998 EMA", "zh": "0.9998 EMA"}],
             [f"{P3}-ema0.999", f"{P3}-ema0.999-ema", f"{P3}-ema0.9998", f"{P3}-ema0.9998-ema"]),
            ({"en": "Batch, 300m", "zh": "batch，300m"}, ["128", "96"], [R128, f"{P3}-b96"]),
            ({"en": "Tokenizer, 300m", "zh": "tokenizer，300m"}, ["8K", "16K", "32K", "64K", "128K", {"en": "bytes", "zh": "字节"}],
             [f"{P3}-v8k", f"{P3}-v16k", f"{P3}-v32k", f"{P3}-v64k", R128, f"{P3}-bytes"]),
            ({"en": "Training seed, 300m 128K then 8K", "zh": "训练种子，300m，先 128K 后 8K"}, ["0", "1", "0 ", "1 "],
             [R128, f"{P3}-s1-rerun", f"{P3}-v8k", f"{P3}-v8k-s1"])]
M_FIG1 = {0.95: 4.2, 0.98: 3.5}   # m of the momentum reading fitted in fig-predict: 130m (momentum 0.95), 300m (0.98)
BLOCK, EVERY, WIN = 8, 64, 8      # sampled blocks; the low point is the minimum of a moving mean over WIN blocks (512 steps)


def sampled(name):
    """(block-centre training fractions, block mean losses, unread losses) of one checkpoint on the 1-in-8 blocks."""
    d = os.path.join(OUT, "runs", name)
    if not os.path.isdir(d):
        d = os.path.join(OUT, name)   # fully scored by w465
    idx, loss = [], []
    for f in sorted(glob.glob(os.path.join(d, "[0-9]*.npz"))):
        z = np.load(f)
        idx.append(z["index"])
        loss.append(z["seq_loss"].sum(1) / z["seq_weight"].sum(1))
        T = int(z["total_steps"])
    if not idx:
        return None
    idx, loss = np.concatenate(idx), np.concatenate(loss)
    tr = idx < T
    keep = tr & ((idx % EVERY < BLOCK) | (idx >= T - BLOCK))
    b = idx[keep] // BLOCK
    ub = np.unique(b)
    x = np.array([idx[keep][b == k].mean() for k in ub]) / T
    y = np.array([loss[keep][b == k].mean() for k in ub])
    return x, y, loss[~tr], T


def low_point(x, y):
    xm, ym = np.convolve(x, np.ones(WIN) / WIN, mode="valid"), np.convolve(y, np.ones(WIN) / WIN, mode="valid")
    return float(xm[np.argmin(ym)])
N_SEQ = 2441406   # sequences of the fineweb-edu-10B stream under the Marin tokenizer
MULTIPASS = [({"en": "520m, 1.04 passes", "zh": "520m，1.04 遍"}, ["muonh-qwen3-520m-della8xh100-pls0-noev"]),
             ({"en": "1.2b, 2.40 passes", "zh": "1.2b，2.40 遍"}, ["muonh-qwen3-1_2b-della8xh100-pls0-noev"]),
             ({"en": "130m, 10x steps, 2.60 passes", "zh": "130m，10 倍步数，2.60 遍"}, [f"{P1}-tpp200", f"{P1}-tpp200-s1"])]
LAST = {"en": "by last read", "zh": "按最后一次读到的时间"}
SEEDS = [BLUE, "#8AB4F8"]


def fig_settings():
    """Per checkpoint: the measured low point (minimum of the moving mean over 8 sampled blocks) and the momentum
    reading's low point, 1 - sqrt(c) t^m with the run's own learning rates and m from fig-predict at its size."""
    panels, lows = [], {}
    for label, cats, names in SETTINGS:
        meas, pred = [], []
        for j, name in enumerate(names):
            s = sampled(name)
            if s is None:
                break
            x, y, unread, T = s
            base = name[:-len("-ema")] if name.endswith("-ema") else name
            cfg = json.load(open(os.path.join(predict.CACHE, f"{base}.json")))["config"]
            beta = cfg["trainer"]["model_averaging"]["beta"] if name.endswith("-ema") else None
            c = predict.run_coefficients(base, T, beta)
            lm, lp = low_point(x, y), predict.valley(c, M_FIG1[cfg["optimizer"]["momentum"]])
            meas.append([j, round(lm, 4)])
            pred.append([j, round(lp, 4)])
            lows[name] = {"measured": round(lm, 3), "predicted": round(lp, 3)}
        p = {"xlim": [-0.6, len(cats) - 0.4], "ylim": [0.6, 1.0], "yticks": ticks(0.6, 0.9, 0.1, 1), "cats": cats, "label": label}
        if len(meas) < len(names):
            panels.append(dict(p, empty=QUEUED, marks=[]))
            continue
        panels.append(dict(p, marks=[{"type": "dots", "pts": meas, "color": INK, "dx": -0.12}, {"type": "rings", "pts": pred, "color": INK, "dx": 0.12}]))
    return {"title": {"en": "TREC low points", "zh": "TREC 最低点"},
            "quantity": {"en": "Low point (training fraction)", "zh": "最低点（训练进度）"}, "cols": 2, "height": 170,
            "legend": [[{"en": "Measured", "zh": "实测"}, INK, "dot"], [{"en": "Predicted", "zh": "预测"}, INK, "ring"]],
            "panels": panels, "lows": lows}


def last_read(name, idx, T):
    """(training fraction of the last read, read count) of each batch: batch t holds stream positions B t .. B t + B - 1,
    read as sequence q = p mod N (restart strategy; the 2048-sequence mixture permutation moves a position by under
    2048 / B steps), and q is read again at every step (q + k N) / B < T."""
    B = json.load(open(os.path.join(predict.CACHE, f"{name}.json")))["config"]["trainer"]["train_batch_size"]
    q = (B * idx + B // 2) % N_SEQ
    k = (T * B - 1 - q) // N_SEQ
    return (q + k * N_SEQ) / B / T, k + 1


def fig_multipass():
    """Per run, the final model's loss on its 1-in-8 training blocks: left against the training fraction of the batch
    (moving mean over 8 blocks); right against the time the batch's data was last read, batches sorted by that time
    and averaged over 64 batches (8 blocks) within each read count."""
    rows = []
    for label, names in MULTIPASS:
        left, right, lo, hi = [], [], [], []
        for name, col in zip(names, SEEDS):
            d = os.path.join(OUT, "runs", name)
            idx, loss = [], []
            for f in sorted(glob.glob(os.path.join(d, "[0-9]*.npz"))):
                z = np.load(f)
                idx.append(z["index"])
                loss.append(z["seq_loss"].sum(1) / z["seq_weight"].sum(1))
                T = int(z["total_steps"])
            idx, loss = np.concatenate(idx), np.concatenate(loss)
            x, y, _, _ = sampled(name)
            xm, ym = np.convolve(x, np.ones(WIN) / WIN, mode="valid"), np.convolve(y, np.ones(WIN) / WIN, mode="valid")
            left.append({"type": "line", "pts": thin(xm, ym), "color": col, "width": 1.8, "alpha": 0.85})
            tl, reads = last_read(name, idx, T)
            for r in np.unique(reads):
                o = np.flatnonzero(reads == r)
                o = o[np.argsort(tl[o])]
                w = BLOCK * WIN
                if len(o) < w:
                    continue
                xr, yr = np.convolve(tl[o], np.ones(w) / w, mode="valid"), np.convolve(loss[o], np.ones(w) / w, mode="valid")
                right.append({"type": "line", "pts": thin(xr, yr), "color": col, "width": 1.8, "alpha": 0.85})
                lo.append(yr.min()); hi.append(yr.max())
            lo.append(ym.min()); hi.append(ym.max())
        rows.append((label, left, right, min(lo), max(hi)))
    rng = max(h - l for *_, l, h in rows)
    step = 0.02 if rng < 0.1 else 0.05
    span = np.ceil(rng * 1.45 / step) * step   # one span for every row (writing:plot rule 7), headroom for the panel name
    panels = []
    for label, left, right, l, h in rows:
        ylo = min(np.floor((l - 0.08 * rng) / step), np.round(((l + h) / 2 - 0.45 * span) / step)) * step   # centred, unless the name would cover the data
        y = {"ylim": [round(ylo, 4), round(ylo + span, 4)], "yticks": ticks(ylo, ylo + span - step, step, 2), "xticks": XTICKS, "xlim": [0, 1]}
        panels.append(dict(y, label=label, marks=left))
        panels.append(dict(y, label={k: label[k].split("，" if k == "zh" else ",")[0] + ("，" if k == "zh" else ", ") + LAST[k] for k in label}, marks=right))
    return {"title": {"en": "TRECs of runs that read their data more than once", "zh": "数据读了不止一遍的 run 的 TREC"},
            "quantity": {"en": "Loss (nats)", "zh": "损失（nats）"}, "xlabel": XLABEL, "cols": 2, "height": 150,
            "legend": [[{"en": "Seed 0", "zh": "种子 0"}, BLUE, "line"], [{"en": "Seed 1", "zh": "种子 1"}, "#8AB4F8", "line"]],
            "panels": panels}


RSQRT = [("muonh-qwen3-130m-della4xh100-d12", {"en": "d12", "zh": "d12"}, GREY),
         ("muonh-qwen3-130m-della4xh100-d12-rsqrt", {"en": "1/√t", "zh": "1/√t"}, BLUE),
         ("muonh-qwen3-130m-della4xh100-d12-rsqrtcd", {"en": "1/√t + cooldown", "zh": "1/√t + 冷却"}, "#8AB4F8")]


def rsqrt_lr(name, T=4959, eta=0.02, A=23.0):
    """The learning rate of a Figure 8 run: logged in W&B once cached, else its configured schedule (branch trec-rsqrt):
    eta / sqrt(1 + A eta^2 t), with rsqrtcd linear to 0 from step round(0.8 T)."""
    f = os.path.join(predict.CACHE, f"{name}.json")
    if os.path.exists(f):
        st, v = np.array(json.load(open(f))["optim/learning_rate"]).T
        return np.interp(np.arange(T), st, v)
    t = np.arange(T)
    lr = eta / np.sqrt(1 + A * eta ** 2 * t)
    if name.endswith("rsqrtcd"):
        t0 = round(0.8 * T)
        lr = np.where(t < t0, lr, lr[t0] * np.clip((T - t) / (T - t0), 0, 1))
    return lr


def fig_rsqrt():
    """Figure 8 (owner popup 2026-10-28): the 130m d12 control and the same run under the equal-weight schedule of
    kexue.fm/archives/11459, without and with a cooldown. Left: learning rate. Right: TREC (100-step moving mean of the
    final model's loss on every training batch), queued until both new runs are scored."""
    T = 4959
    lr_marks, trec_marks, lo, hi = [], [], [], []
    for name, label, col in RSQRT:
        t = (np.arange(T) + 1) / T
        lr_marks.append({"type": "line", "pts": thin(t, rsqrt_lr(name)), "color": col, "width": 1.8, "alpha": 0.85})
        files = sorted(glob.glob(os.path.join(OUT, name, "[0-9]*.npz")))
        if files:
            idx = np.concatenate([np.load(f)["index"] for f in files])
            loss = np.concatenate([np.load(f)["seq_loss"].sum(1) / np.load(f)["seq_weight"].sum(1) for f in files])
            if np.array_equal(np.sort(idx[idx < T]), np.arange(T)):
                v = smooth(loss[idx < T][np.argsort(idx[idx < T])])
                trec_marks.append({"type": "line", "pts": thin(t, v), "color": col, "width": 1.8, "alpha": 0.85})
                lo.append(v.min()); hi.append(v.max())
    # the 1/sqrt(t) line drawn last: until the cooldown starts it lies under the cooldown run's line
    lrp = {"xlim": [0, 1], "ylim": [0, 0.0225], "xticks": XTICKS, "yticks": ticks(0, 0.02, 0.005, 3), "marks": [lr_marks[0], lr_marks[2], lr_marks[1]]}
    tp = {"xlim": [0, 1], "xticks": XTICKS}
    if len(trec_marks) < len(RSQRT):
        tp.update(ylim=[3.0, 3.45], yticks=ticks(3.0, 3.4, 0.1, 1), empty=QUEUED, marks=[])
    else:
        step = 0.05 if max(hi) - min(lo) < 0.3 else 0.1
        y0 = np.floor(min(lo) / step) * step
        y1 = np.ceil((max(hi) + 0.15 * (max(hi) - min(lo))) / step) * step
        tp.update(ylim=[round(y0, 4), round(y1, 4)], yticks=ticks(y0, y1 - step, step, 2), marks=[trec_marks[0], trec_marks[2], trec_marks[1]])
    return {"title": {"en": "The 1/√t schedule", "zh": "1/√t 学习率"},
            "quantity": [{"en": "Learning rate", "zh": "学习率"}, {"en": "Loss (nats)", "zh": "损失（nats）"}],
            "xlabel": XLABEL, "cols": 2, "height": 200,
            "legend": [[label, col, "line"] for _, label, col in RSQRT], "panels": [lrp, tp]}


CK = [("muonh-qwen3-130m-della4xh100-d12-ck248", {"en": "130m, WSD", "zh": "130m\uff0cWSD"}, 4959, [992, 1984, 2976, 3968, 4958]),
      ("muonh-qwen3-130m-della4xh100-d12-rsqrtcd-ck248", {"en": "130m, 1/\u221at + cooldown", "zh": "130m\uff0c1/\u221at + \u51b7\u5374"}, 4959, [992, 1984, 2976, 3968, 4958]),
      ("muonh-qwen3-300m-della4xh100-ck572-s1", {"en": "300m, cosine", "zh": "300m\uff0ccosine"}, 11444, [2288, 4576, 6864, 9152, 11443])]   # wq w667
CK_FRACS = [0.2, 0.4, 0.6, 0.8, 1.0]   # each run's checkpoints at these training fractions
CK_COLORS = ["#AECBFA", "#669DF6", "#1A73E8", "#185ABC", "#174EA6"]   # GM2 blue 200-900: the order is the message


def fig_imprint():
    """Figure 9 (owner popup 2026-10-29): runs of Figure 8 trained again with a checkpoint every 248 steps (wq w642,
    w643). Per checkpoint, the unread level minus the loss on 1 in 16 training batches (blocks of 8 steps every 128; the
    final checkpoint scores every batch and is cut to the same blocks), as a moving mean over 3 blocks, against the
    training fraction at which the batch was read. Queued until every checkpoint of a run is scored."""
    panels = []
    for run, label, T, ck_steps in CK:
        marks = []
        for step, col in zip(ck_steps, CK_COLORS):
            files = sorted(glob.glob(os.path.join(OUT, "ck", run, f"step-{step}", "[0-9]*.npz")))
            if not files:
                break
            idx = np.concatenate([np.load(f)["index"] for f in files])
            loss = np.concatenate([np.load(f)["seq_loss"].sum(1) / np.load(f)["seq_weight"].sum(1) for f in files])
            tr = (idx < T) & (idx % 128 < 8)
            b = idx[tr] // 128
            ub = np.unique(b)
            if len(ub) < T // 128:
                break
            x = np.array([idx[tr][b == k].mean() for k in ub]) / T
            y = loss[idx >= T].mean() - np.array([loss[tr][b == k].mean() for k in ub])
            w = 3
            marks.append({"type": "line", "pts": thin(np.convolve(x, np.ones(w) / w, "valid"), np.convolve(y, np.ones(w) / w, "valid")),
                          "color": col, "width": 1.6, "alpha": 0.9})
        p = {"label": label, "xlim": [0, 1], "xticks": XTICKS}
        if len(marks) < len(ck_steps):
            p.update(ylim=[0, 0.12], yticks=ticks(0, 0.1, 0.02, 2), empty=QUEUED, marks=[])
        else:
            p.update(marks=marks)
        panels.append(p)
    have = [p for p in panels if "empty" not in p]
    if have:   # one range for both panels: the same quantity, from the same unread batches
        ys = [v for p in have for m in p["marks"] for _, v in m["pts"]]
        lo, hi = min(ys), max(ys)
        step = 0.02 if hi - lo < 0.15 else 0.05
        y0, y1 = np.floor(lo / step) * step, np.ceil((hi + 0.15 * (hi - lo)) / step) * step
        for p in have:
            p.update(ylim=[round(y0, 4), round(y1, 4)], yticks=ticks(y0, y1 - step, step, 2))
    return {"title": {"en": "Imprint of each batch over training", "zh": "\u6bcf\u4e2a\u6279\u6b21\u5728\u8bad\u7ec3\u4e2d\u7684\u5370\u8bb0"},
            "quantity": {"en": "Below the unread level (nats)", "zh": "\u4f4e\u4e8e\u672a\u8bfb\u6c34\u5e73\u7684\u91cf\uff08nats\uff09"},
            "xlabel": {"en": "Training fraction when read", "zh": "\u8bfb\u5230\u65f6\u7684\u8bad\u7ec3\u8fdb\u5ea6"}, "cols": 2, "height": 200,
            "legend": [[{"en": f"Checkpoint at {f:g}", "zh": f"\u8bad\u7ec3\u8fdb\u5ea6 {f:g} \u5904\u7684 checkpoint"} if i == 0 else {"en": f"{f:g}", "zh": f"{f:g}"}, c, "line"]
                       for i, (f, c) in enumerate(zip(CK_FRACS, CK_COLORS))], "panels": panels}


BRANCH = [("muonh-qwen3-130m-della4xh100-d12-constant-ck124-lrm0.5-br2480", "lr 0.01 \u00d7 500", 0.01, "#8AB4F8"),
          ("muonh-qwen3-130m-della4xh100-d12-constant-ck124-lrm0.25-br2480", "lr 0.005 \u00d7 1000", 0.005, "#1A73E8"),
          ("muonh-qwen3-130m-della4xh100-d12-constant-ck124-lrm0.125-br2480", "lr 0.0025 \u00d7 2000", 0.0025, "#174EA6")]


def _range_score(d):
    """(unread mean, per-batch losses of training batches 2232..2479) of one TRAIN_RANGE scoring, else None."""
    files = sorted(glob.glob(os.path.join(d, "[0-9]*.npz")))
    if not files:
        return None
    idx = np.concatenate([np.load(f)["index"] for f in files])
    loss = np.concatenate([np.load(f)["seq_loss"].sum(1) / np.load(f)["seq_weight"].sum(1) for f in files])
    tr = (idx >= 2232) & (idx < 2480)
    if tr.sum() != 248 or (idx >= 4959).sum() != 256:
        return None
    return loss[idx >= 4959].mean(), loss[tr][np.argsort(idx[tr])]


def fig_branch():
    """Figure 10 (owner popup 2026-10-29): d12-ck248 continued from step 2480 at a constant lr for N steps, lr x N = 5
    (wq w651-w653). Fraction of the imprint of the batches read at steps 2232-2479 that is left at each kept checkpoint:
    E_s = mean_b [(U_s - L_s(b)) - (U_2232 - L_2232(b))] over those batches, divided by E at step 2480; against the sum
    of lr, the number of steps and the sum of lr^2 since step 2480. Queued until every arm is scored."""
    base = os.path.join(OUT, "ck", "branch")
    pre, at = _range_score(os.path.join(base, "base", "step-2232")), _range_score(os.path.join(base, "base", "step-2480"))
    curves = []
    if pre and at:
        e0 = ((at[0] - at[1]) - (pre[0] - pre[1])).mean()
        for run, label, lr, col in BRANCH:
            steps = sorted(int(d.split("-")[-1]) for d in glob.glob(os.path.join(base, run, "step-*")))
            pts = [(0, 1.0)]
            for st in steps:
                r = _range_score(os.path.join(base, run, f"step-{st}"))
                if r is None:
                    break
                pts.append((st - 2480, float(((r[0] - r[1]) - (pre[0] - pre[1])).mean() / e0)))
            n = round(5 / lr)
            if pts[-1][0] < n - 1:
                break
            curves.append((label, lr, col, np.array(pts)))
    axes = [("Sum of lr since step 2480", "\u7b2c 2480 \u6b65\u4ee5\u6765\u7684 \u03a3lr", lambda a, lr: a * lr, [0, 5.25], [0, 1, 2, 3, 4, 5]),
            ("Steps since step 2480", "\u7b2c 2480 \u6b65\u4ee5\u6765\u7684\u6b65\u6570", lambda a, lr: a, [0, 2100], [0, 500, 1000, 1500, 2000]),
            ("Sum of lr\u00b2 since step 2480", "\u7b2c 2480 \u6b65\u4ee5\u6765\u7684 \u03a3lr\u00b2", lambda a, lr: a * lr * lr, [0, 0.0525], [0, 0.01, 0.02, 0.03, 0.04, 0.05]),
            # owner 2026-10-30: a fourth panel on the axis where the arms meet (p fitted by tools/branch_fit.py, 1.32)
            ("Sum of lr\u00b9\u00b7\u00b3 since step 2480", "\u7b2c 2480 \u6b65\u4ee5\u6765\u7684 \u03a3lr\u00b9\u00b7\u00b3", lambda a, lr: a * lr ** 1.32, [0, 1.2], [0, 0.25, 0.5, 0.75, 1.0])]
    panels = []
    for en, zh, fx, xlim, xt in axes:
        p = {"label": {"en": en, "zh": zh}, "xlim": xlim, "xticks": [[v, f"{v:g}"] for v in xt], "ylim": [0, 1.3], "yticks": ticks(0, 1.0, 0.25, 2)}
        if len(curves) < len(BRANCH):
            p.update(empty=QUEUED, marks=[])
        else:
            p["marks"] = [{"type": "line", "pts": [[round(float(fx(a, lr)), 6), round(float(v), 4)] for a, v in c], "color": col, "width": 1.8, "alpha": 0.9}
                          for _, lr, col, c in curves]
        panels.append(p)
    return {"title": {"en": "Forgetting after a branch", "zh": "\u5206\u53c9\u4e4b\u540e\u7684\u9057\u5fd8"},
            "quantity": {"en": "Fraction of the imprint left", "zh": "\u5370\u8bb0\u5269\u4e0b\u7684\u6bd4\u4f8b"}, "cols": 1, "height": 150,
            "legend": [[{"en": label, "zh": label}, col, "line"] for _, label, _, col in BRANCH], "panels": panels}


def main():
    site = sys.argv[1]
    data = {m: scored(m) for m, _ in PANELS}
    figs = {"fig-trec": fig_trec(data), "fig-coef": fig_coef(), "fig-init": fig_init(), "fig-predict": fig_predict(data), "fig-tokenizers": fig_tokenizers(),
            "fig-settings": fig_settings(), "fig-multipass": fig_multipass(), "fig-rsqrt": fig_rsqrt(), "fig-imprint": fig_imprint(), "fig-branch": fig_branch()}
    path = os.path.join(site, "assets/data/trec-figs.js")
    with open(path, "w") as f:
        f.write("/* Generated by experiments/trec/build_site_data.py (marin branch trec); regenerate rather than hand-edit. */\n")
        f.write("window.TREC_FIGS = " + json.dumps(figs, ensure_ascii=False, separators=(",", ":")) + ";\n")
    print(path, {k: [("empty" if "empty" in p else "data") for p in v["panels"]] for k, v in figs.items()}, figs["fig-predict"]["fits"], figs["fig-tokenizers"]["fit"], figs["fig-init"]["pred"])


if __name__ == "__main__":
    main()
