"""Does the mixing's scale change compound with depth? From res_gain tables (median over 8,192 FineWeb-Edu tokens, final checkpoint): for each design, range of the cumulative log10 gain of the stream sum through the mixing, median norm of the stream sum at the last layer, the last write's share of the new sum, and the participation of the writes in the final sum: c_l = |write_l| x product of later mixing gains, P = (sum c)^2 / sum c^2, shown as a fraction of the sublayers. Printed next to held-out loss minus no mixing at the same depth."""
import numpy as np
H = "/scratch/gpfs/GROUP/USER/tmp/pls_hcid/"
RUNS = [("HC", 12, "res_gain_depth-sep-bbfrozen-ahc-fa4thd-vs-d12", 3.7677, 3.7837), ("I", 12, "res_gain_depth-sep-bbfrozen-ahc-hcid-fa4thd-vs-d12", 3.7837, 3.7837),
        ("HC", 24, "res_gain_depth-sep-bbfrozen-ahc-fa4thd-vs-d24", 3.6799, 3.6599), ("I", 24, "res_gain_depth-sep-bbfrozen-ahc-hcid-fa4thd-vs-d24", 3.6599, 3.6599),
        ("HC", 48, "res_gain-sep-bbfrozen-ahc-fa4thd-vs-d48", 3.638, 3.580), ("I", 48, "res_gain-sep-bbfrozen-ahc-hcid-fa4thd-vs-d48", 3.579, 3.580),
        ("HC scalar", 48, "res_gain-sep-bbfrozen-ahc-hcscalar-fa4thd-vs-d48", 3.565, 3.580), ("HC static", 48, "res_gain-sep-bbfrozen-ahc-rstatic-vs-d48", 3.609, 3.580),
        ("HC 2 streams", 48, "res_gain-sep-bbfrozen-ahc-hcs2-vs-d48", 3.632, 3.580), ("HC 8 streams", 48, "res_gain-sep-bbfrozen-ahc-hcs8-vs-d48", 3.643, 3.580),
        ("HC only shrinks", 48, "res_gain_arms-sep-bbfrozen-ahc-hcshrink-fa4thd-vs-d48", 3.687, 3.580), ("HC half dose", 48, "res_gain_arms-sep-bbfrozen-ahc-hcshrink-sp0.5-fa4thd-vs-d48", 3.676, 3.580),
        ("HC moves", 48, "res_gain_arms-sep-bbfrozen-ahc-hcmove-fa4thd-vs-d48", 3.578, 3.580), ("HC keeps sum", 48, "res_gain_arms-sep-bbfrozen-ahc-hcsum-hcf32-fa4thd-vs-d48", 3.591, 3.580)]
print("design | layers | held-out | no mixing - design | log10 range | |sum| last | write share last sublayer | writes that reach the sum, fraction")
out = {}
for name, L, f, ho, ref in RUNS:
    z = np.load(H + f + ".npz"); k = [x[:-2] for x in z.files if x.endswith("/s")][0]
    ratio, share, s, wf = z[k + "/ratio"], z[k + "/share"], z[k + "/s"], z[k + "/wf"]
    cum = np.cumsum(np.log10(np.maximum(ratio, 1e-30)))
    rng = float(max(cum.max(), 0) - min(cum.min(), 0))
    later = np.concatenate([np.cumsum(np.log10(np.maximum(ratio[::-1], 1e-30)))[::-1][1:], [0.0]])   # log10 of the product of the gains after each sublayer
    c = wf * 10.0 ** later
    P = c.sum() ** 2 / (c ** 2).sum() / len(c)
    print(f"{name:15s} | {L:2d} | {ho:.3f} | {ref - ho:+.3f} | {rng:5.2f} | {s[-1]:9.3g} | {np.sqrt(share[-1]):.3f} | {P:.3f}")
    out[f"{name}/{L}"] = dict(cum=cum.tolist(), s=s.tolist(), share=np.sqrt(share).tolist(), c=c.tolist(), orders=rng, heldout=ho)
import json; json.dump(out, open(H + "scale_depth.json", "w"))
