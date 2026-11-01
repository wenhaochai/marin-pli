"""Post-hoc check of the A check (verification 2026-11-01, made after the decision rules): does old content shrink by a?

Under the shrink model (later steps do not touch content already in W), the coefficient of the branch-point weights
in W_s obeys c(k+1) = a_k c(k) exactly, so ln cos(W_s, W_2480) falls by sum_k (eta r_k + eta^2/2). Here cos(W_s, W_2480)
is read per MuonH matrix (84) on the 18 branch checkpoints of branch_radial.ckpts() and compared, from s = 248 to the
last kept step, with the shrink that radial.json's r readings predict, at each momentum and for the 0.98 - 0.95 gap.
Read-only; output /scratch/gpfs/GROUP/USER/project/trec/out/ck/radial/cos.json. Measure with the marin venv (tensorstore),
analyze with python3: python branch_cos.py measure | analyze
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/scratch/gpfs/GROUP/USER/project/marin-trec/experiments/trec/tools")
import branch_radial as br  # noqa: E402

HERE = "/scratch/gpfs/GROUP/USER/project/trec/out/ck/radial"


def measure():
    import tensorstore as ts

    def read(ck, path):
        spec = {"driver": "zarr3", "kvstore": {"driver": "ocdbt", "base": {"driver": "file", "path": ck}, "path": path}}
        return np.asarray(ts.open(spec).result().read().result(), dtype=np.float64)

    cks = br.ckpts()
    base = {kind: read(br.BASE, f"model/transformer/layers/stacked/{sub}/weight") for kind, sub in br.MATS.items()}
    out = {}
    for label, lr, betas, ck in cks[1:]:
        rows = {}
        for kind, sub in br.MATS.items():
            W = read(ck, f"model/transformer/layers/stacked/{sub}/weight")
            for layer in range(W.shape[0]):
                w, w0 = W[layer].ravel(), base[kind][layer].ravel()
                n, n0 = np.linalg.norm(w), np.linalg.norm(w0)
                rows[f"{kind}{layer}"] = {"cos": float(w @ w0 / (n * n0)), "norm_dev": float(n / n0 - 1)}
        out[label] = rows
        c = np.median([v["cos"] for k, v in rows.items() if not k.startswith("gate")])
        print(f"{label:<16} median cos to step-2480 (no gate) {c:.5f}  max |norm dev| {max(abs(v['norm_dev']) for v in rows.values()):.1e}", flush=True)
    json.dump(out, open(os.path.join(HERE, "cos.json"), "w"), indent=0)


def analyze():
    cos = json.load(open(os.path.join(HERE, "cos.json")))
    rad = json.load(open(os.path.join(br.OUT, "radial.json")))
    names = list(rad["base"]["mats"])
    sel = np.array([not n.startswith("gate") for n in names])

    def c(label):
        return np.array([cos[label][n]["cos"] for n in names])

    def r(label, reading, beta):
        key = "r_M" if reading == "r_M" else f"r_noise_{beta}"
        return np.array([rad[label]["mats"][n][key] for n in names])

    for lrm, lr, last in br.ARMS:
        s1, s2 = 248, last - 2480
        n = s2 - s1
        meas = {}
        for beta in (0.95, 0.98):
            meas[beta] = -np.log(c(f"{beta}/{lr}/{s2}")) + np.log(c(f"{beta}/{lr}/{s1}"))
            line = f"lr {lr:<7g} mom {beta}: -dln cos(W_s,W_2480) s {s1}->{s2}: median {np.median(meas[beta][sel]):.4f}"
            line += f" (A_eff = {2 * np.median(meas[beta][sel]) / (lr * lr * n):5.1f})"
            for reading in ("r_noise", "r_M"):
                rr = (r(f"{beta}/{lr}/{s1}", reading, beta) + r(f"{beta}/{lr}/{s2}", reading, beta)) / 2
                pred = n * (lr * rr + lr * lr / 2)
                line += f" | {reading} pred {np.median(pred[sel]):.4f}, meas/pred median {np.median((meas[beta] / pred)[sel]):.2f}"
            print(line)
        gap = meas[0.98] - meas[0.95]
        line = f"lr {lr:<7g} extra (0.98 - 0.95): measured median {np.median(gap[sel]):+.4f} (IQR {np.percentile(gap[sel], 25):+.4f} to {np.percentile(gap[sel], 75):+.4f})"
        for reading in ("r_noise", "r_M"):
            dr = (r(f"0.98/{lr}/{s1}", reading, 0.98) - r(f"0.95/{lr}/{s1}", reading, 0.95)
                  + r(f"0.98/{lr}/{s2}", reading, 0.98) - r(f"0.95/{lr}/{s2}", reading, 0.95)) / 2
            pred = n * lr * dr
            line += f" | {reading} pred {np.median(pred[sel]):+.4f}"
        print(line)
        for s in (124, 248, s2):
            print(f"    s {s:<5} median cos 0.95 {np.median(c(f'0.95/{lr}/{s}')[sel]):.4f}  0.98 {np.median(c(f'0.98/{lr}/{s}')[sel]):.4f}")


if __name__ == "__main__":
    {"measure": measure, "analyze": analyze}[sys.argv[1]]()
