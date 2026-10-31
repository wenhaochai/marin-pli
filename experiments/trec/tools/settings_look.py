"""First look at w513/w514: per run, block means of the sampled batches, a moving mean over 8 blocks (512 steps),
its low point, the unread level, and the momentum prediction's low point."""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, "/scratch/gpfs/GROUP/USER/project/marin-trec/experiments/trec")
import predict

OUT = "/scratch/gpfs/GROUP/USER/project/trec/out"
CACHE = predict.CACHE


def load(d):
    idx, loss = [], []
    for f in sorted(glob.glob(os.path.join(d, "[0-9]*.npz"))):
        z = np.load(f)
        idx.append(z["index"]); loss.append(z["seq_loss"].sum(1) / z["seq_weight"].sum(1))
        T = int(z["total_steps"])
    idx, loss = np.concatenate(idx), np.concatenate(loss)
    o = np.argsort(idx)
    return idx[o], loss[o], T


def blocks(idx, loss, T, sub=None):
    tr = idx < T
    i, l = idx[tr], loss[tr]
    if sub is not None:   # 1-in-8 subsample of a fully scored run, offset sub
        keep = (i % 64 >= sub) & (i % 64 < sub + 8)
        i, l = i[keep], l[keep]
    b = i // 8
    ub = np.unique(b)
    x = np.array([i[b == k].mean() for k in ub]) / T
    y = np.array([l[b == k].mean() for k in ub])
    return x, y


def low(x, y, w=8):
    ym = np.convolve(y, np.ones(w) / w, mode="valid")
    xm = np.convolve(x, np.ones(w) / w, mode="valid")
    k = int(np.argmin(ym))
    return float(xm[k]), float(ym[k]), xm, ym


def lr(run, T):
    d = json.load(open(os.path.join(CACHE, f"{run}.json")))
    s, v = np.array(d["optim/learning_rate"]).T
    return np.interp(np.arange(T), s, v)


def pred_low(run, T, A, m, beta=None):
    eta = lr(run, T)
    r = (A - 1) / 2 * eta + predict.R0
    a = 1 / np.sqrt(1 + 2 * eta * r + eta ** 2)
    if beta is None:
        k = np.log(a); after = np.cumsum(k[::-1])[::-1]
        c = eta * np.exp(after)
    else:   # EMA of the weights: c_i = eta_i sum_{s>=i} w_s prod_{j=i..s} a_j, w_s = (1-beta) beta^(T-1-s)
        w = (1 - beta) * beta ** (T - 1 - np.arange(T))
        G = np.zeros(T + 1)
        for i in range(T - 1, -1, -1):
            G[i] = a[i] * (w[i] + G[i + 1])
        c = eta * G[:T]
    c = c / c.max()
    return predict.valley(c, m), float(eta.max())


if __name__ == "__main__":
    runs = sorted(os.listdir(f"{OUT}/runs")) + ["muonh-qwen3-130m-della4xh100-d12", "muonh-qwen3-300m-della4xh100-s1-rerun"]
    for run in runs:
        d = f"{OUT}/runs/{run}" if os.path.isdir(f"{OUT}/runs/{run}") else f"{OUT}/{run}"
        idx, loss, T = load(d)
        base = run[:-4] if run.endswith("-ema") else run
        cfg = json.load(open(os.path.join(CACHE, f"{base}.json")))["config"]
        mom = cfg["optimizer"]["momentum"]; beta = cfg["trainer"].get("model_averaging") or {}
        beta = beta.get("beta") if run.endswith("-ema") else None
        A, m = (23.0, 4.2) if mom == 0.95 else (73.0, 3.5)
        tail = loss[idx >= T]
        full = d.endswith(("d12", "s1-rerun"))
        x, y = blocks(idx, loss, T, sub=0 if full else None)
        lx, ly, _, _ = low(x, y)
        pl, pk = pred_low(base, T, A, m, beta)
        s = f"{run.replace('muonh-qwen3-', ''):34s} T {T:6d} n {int((idx < T).sum()):5d} low {lx:.3f} ({ly:.4f})"
        if len(tail):
            s += f" unread {tail.mean():.4f}+-{tail.std(ddof=1) / np.sqrt(len(tail)):.4f} depth {ly - tail.mean():+.4f}"
        s += f" pred {pl:.3f} peak-lr {pk:.4f}"
        if full:
            subs = [low(*blocks(idx, loss, T, sub=o))[0] for o in range(0, 64, 8)]
            s += f" | 8 offsets: {np.mean(subs):.3f} sd {np.std(subs, ddof=1):.3f} [{min(subs):.3f},{max(subs):.3f}]"
        print(s, flush=True)
