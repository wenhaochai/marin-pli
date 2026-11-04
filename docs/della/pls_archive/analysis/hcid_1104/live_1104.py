"""Interim read of the unfinished pls runs on W&B (train CE by layer, 100-step means at fixed step windows) next to the finished references at the same steps (d48g4.npz).
Saves every fetched series to live_1104.npz for the figure. W&B state may read "crashed" for a live run; the job state comes from wq.
Also prints the L48 500-step means and the grad norm of w893 (HC mixing = I, seed 1, separate heads) next to w810 (seed 0) and HC, separate heads."""
import numpy as np, wandb
from pathlib import Path
api = wandb.Api(timeout=300)
P = "reself/marin-della/muonh-qwen3-130m-della4xh100-pls1"
z = np.load("/scratch/gpfs/GROUP/USER/tmp/pls_blog/d48g4.npz", allow_pickle=True)
LAY = [2, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48]
K = {l: f"train/pls/L{l - 1}" for l in LAY}
RUNS = {"w889 hcrwj": "-sep-bbfrozen-ahc-hcrwj-fa4thd-vs-d48", "w890 HC s1": "-sep-bbfrozen-ahc-fa4thd-vs-d48-s1", "w891 hcid s1": "-sep-bbfrozen-ahc-hcid-fa4thd-vs-d48-s1", "w960 hcscalar": "-sep-bbfrozen-ahc-hcscalar-fa4thd-vs-d48",
        "w1000 move": "-sep-bbfrozen-ahc-hcmove-fa4thd-vs-d48", "w1002 half": "-sep-bbfrozen-ahc-hcshrink-sp0.5-fa4thd-vs-d48",
        "w893 hcid s1 sep": "-sep-ahc-hcid-fa4thd-vs-d48-s1", "w1001 move sep": "-sep-ahc-hcmove-fa4thd-vs-d48", "w1003 half sep": "-sep-ahc-hcshrink-sp0.5-fa4thd-vs-d48"}
ORD = {"HC": "-sep-bbfrozen-ahc-fa4thd-vs-d48", "hcid": "-sep-bbfrozen-ahc-hcid-fa4thd-vs-d48"}
SEP = {"HC": "-sep-ahc-fa4thd-vs-d48", "hcid": "-sep-ahc-hcid-fa4thd-vs-d48"}
WIN = [(100, 200), (300, 400), (500, 600), (700, 800), (1000, 1100), (1500, 1600), (2100, 2197), (2700, 2800), (3200, 3300), (3600, 3677), (4400, 4500)]
save = {}
data = {}
for name, tag in RUNS.items():
    r = api.run(P + tag)
    by = {}
    for x in r.scan_history(keys=["_step", *K.values()], page_size=2000):
        by[x["_step"]] = x
    st = np.array(sorted(by))
    ce = np.array([[by[s][K[l]] for l in LAY] for s in st], float)
    data[name] = (st, ce)
    save[name + "/step"] = st; save[name + "/ce"] = ce
    print(f"{name}: W&B state {r.state}, rows {len(st)}, last step {st[-1]}", flush=True)
np.savez(Path(__file__).parent / "live_1104.npz", lay=np.array(LAY), **save)
def row(st, ce, lo, hi, idx):
    m = (st >= lo) & (st < hi)
    return ce[m][:, idx].mean(0) if m.sum() >= 0.5 * (hi - lo) else None
idx = [LAY.index(l) for l in (2, 12, 24, 48)]
print("\nmeans over the window at L2, L12, L24, L48; references at the same steps")
for setup, refs, names in (("ordinary", ORD, ["w889 hcrwj", "w890 HC s1", "w891 hcid s1", "w960 hcscalar", "w1000 move", "w1002 half"]), ("separate heads", SEP, ["w893 hcid s1 sep", "w1001 move sep", "w1003 half sep"])):
    print(f"-- {setup}")
    for lo, hi in WIN:
        lines = []
        for n in names:
            st, ce = data[n]
            v = row(st, ce, lo, hi, idx)
            if v is not None:
                lines.append(f"    {n:18s}" + " ".join(f"{a:6.3f}" for a in v))
        if not lines:
            continue
        print(f"  steps {lo}-{hi - 1}")
        for n, t in refs.items():
            rs, rc = z[t + "/step"], z[t + "/ce"]
            m = (rs >= lo) & (rs < hi)
            print(f"    {n:18s}" + " ".join(f"{rc[m, l - 1].mean():6.3f}" for l in (2, 12, 24, 48)))
        print("\n".join(lines), flush=True)
print("\nL48 500-step means, separate heads: w893 (seed 1), w810 (seed 0), HC; and the grad norm (median, max) from W&B")
gr = {}
for n, t in (("w893", "-sep-ahc-hcid-fa4thd-vs-d48-s1"), ("w810", "-sep-ahc-hcid-fa4thd-vs-d48"), ("HC", "-sep-ahc-fa4thd-vs-d48")):
    a = np.array(sorted((x["_step"], x["grad/norm/total"]) for x in api.run(P + t).scan_history(keys=["_step", "grad/norm/total"], page_size=2000)))
    gr[n] = a
    save[n + "/grad"] = a
st, ce = data["w893 hcid s1 sep"]
for lo in range(1000, 5000, 500):
    hi = lo + 500
    v = row(st, ce, lo, hi, [LAY.index(48)])
    o = f"  {lo}-{hi - 1}: w893 " + (f"{v[0]:6.3f}" if v is not None else "   -  ")
    for n, t in (("w810", SEP["hcid"]), ("HC", SEP["HC"])):
        rs, rc = z[t + "/step"], z[t + "/ce"]
        m = (rs >= lo) & (rs < hi)
        o += f" | {n} {rc[m, 47].mean():6.3f}"
    for n, a in gr.items():
        q = (a[:, 0] >= lo) & (a[:, 0] < hi)
        o += f" | {n} grad {np.median(a[q, 1]):7.2f} {a[q, 1].max():8.1f}" if q.any() else f" | {n} grad -"
    print(o, flush=True)
np.savez(Path(__file__).parent / "live_1104.npz", lay=np.array(LAY), **save)
