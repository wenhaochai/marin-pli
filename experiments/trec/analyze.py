"""Numbers behind the answers of exps/trec.html, from the scored batches of trec_score.py (w465) and predict.py.

For each run: the unread level (mean loss of the 512 unread batches), the low point of the TREC (100- and 500-step
centred moving averages) and its depth below the unread level, the mean loss of the batches of each quarter of training
minus the unread level (+- 1 standard error, batches i.i.d. from one shuffled stream), and for each reading of the
optimizer, plus a control without any averaging (c proportional to the learning rate, the paper's F.3 limit), the m with
the highest Pearson r, that r, and the r and low point at the paper's m* = 1.4 TPP^0.164 tau^-0.45 (its Fig. 6, AdamW,
linear decay to zero), with tau = 1 / (alpha_peak T) and alpha_peak = 1 - keep(peak learning rate).

    python experiments/trec/analyze.py      # vis node
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_site_data as bsd  # noqa: E402
import predict  # noqa: E402

# parameters counted as the paper does (embedding and head included); D = steps x 128 x 4096 tokens
N_PARAMS = {"130m-d12": 45.6e6 + 65.7e6 + 65.7e6, "300m": 299.6e6}


def schedule_only(eta):
    return eta / eta.max()


def main():
    for model, _ in bsd.PANELS:
        train, unread = bsd.scored(model)
        T = len(train)
        eta = predict.lr_schedule(model)
        mu, se_u = unread.mean(), unread.std(ddof=1) / np.sqrt(len(unread))
        print(f"{model}: T {T}, unread {mu:.4f} +- {se_u:.4f} nats (n {len(unread)}, batch sd {unread.std(ddof=1):.4f}), "
              f"training-batch sd {train.std(ddof=1):.4f}")
        for w in (100, 500):
            s = bsd.smooth(train, w)
            k = int(np.argmin(s))
            print(f"  low point ({w}-step mean): t {(k + 1) / T:.3f}, {s[k]:.4f} nats, {s[k] - mu:+.4f} below unread")
        t = (np.arange(T) + 1) / T
        for a, b in [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0), (0.99, 1.0)]:
            x = train[(t > a) & (t <= b)]
            d, se = x.mean() - mu, np.sqrt(x.var(ddof=1) / len(x) + se_u ** 2)
            print(f"  batches in ({a}, {b}]: {d:+.4f} +- {se:.4f} nats against unread (n {len(x)})")
        s = bsd.smooth(train)
        tpp = T * 128 * 4096 / N_PARAMS[model]
        print(f"  TPP {tpp:.1f}; readings (m fitted on the 100-step mean, then the paper's m*):")
        for r in ("momentum", "independent", "adamw-0.1", "schedule-only"):
            c = schedule_only(eta) if r == "schedule-only" else predict.coefficients(eta, r, model)
            m, rp = predict.fit_m(c, s)
            line = f"    {r:13s} c peak {(np.argmax(c) + 1) / T:.3f}; low at m=0 {predict.valley(c, 0):.3f}; fit m {m:.1f}, r {rp:.4f}, low {predict.valley(c, m):.3f}"
            if r != "schedule-only":
                alpha = 1 - predict.keep(np.array([eta.max()]), r, model)[0]
                tau = 1 / (alpha * T)
                ms = 1.4 * tpp ** 0.164 * tau ** -0.45
                line += f"; tau {tau:.3f}, m* {ms:.2f}, r {predict.pearson(predict.predicted(c, ms), s):.4f}, low {predict.valley(c, ms):.3f}"
            print(line)


if __name__ == "__main__":
    main()
