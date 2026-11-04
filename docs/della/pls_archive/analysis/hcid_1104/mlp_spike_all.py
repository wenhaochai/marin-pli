"""Does the MLP's isolated high-gain direction (mlp_spectrum.py: sigma(gate x ln_2) sigma(up x ln_2) sigma(down), about 10-30 in healthy layers, 4,517 at w810's layer 5, 40,574 at w893's layer 12) track the separate-heads cost across every 48-layer design? Weights only, no model.
For each run of the 130m 48-layer family at its newest checkpoint: the largest gain over the layers, the number of layers above 200, the layers, the largest coordinate of ln_2. Separate-heads run and ordinary run of each design, and the cost: final layer-48 train CE, separate heads minus ordinary (d48g4.npz "final"). Output: table and mlp_spike_all.json.
    nice -n 19 $PY mlp_spike_all.py"""
import glob, json, os, re, numpy as np, tensorstore as ts
ROOT = "/scratch/gpfs/GROUP/USER/marin_store_big"
SP = ROOT + "/speedrun/"
TP = ROOT + "/tmp/checkpoints-temp" + SP
G4 = np.load("/scratch/gpfs/GROUP/USER/tmp/pls_blog/d48g4.npz", allow_pickle=True)
keys = set(G4.files)
def final(tag):
    return float(G4[f"{tag}/final"][47]) if f"{tag}/final" in keys else None
dirs = sorted(os.path.basename(p) for p in glob.glob(SP + "muonh-qwen3-130m-della*-pls1-sep-*") + glob.glob(TP + "muonh-qwen3-130m-della*-pls1-sep-*"))
dirs = sorted({d for d in dirs if "vs-d48" in d and "smoke" not in d and "probes" not in d})
def newest(d):
    c = glob.glob(f"{SP}{d}/*/checkpoints/step-*") + glob.glob(f"{TP}{d}/*/checkpoints/step-*")
    return sorted(c, key=lambda p: int(p.rsplit("-", 1)[1]))[-1] if c else None
def rd(ck, key):
    s = ts.open({"driver": "zarr3", "kvstore": {"driver": "ocdbt", "base": {"driver": "file", "path": ck.rstrip("/") + "/"}, "path": f"model/transformer/layers/stacked/{key}/"}}).result()
    return np.asarray(s.read().result(), np.float64)
out = {}
for d in dirs:
    tag = d.split("-pls1", 1)[1]
    ck = newest(d)
    if ck is None:
        continue
    try:
        g, u, dn = rd(ck, "mlp/gate_proj/weight"), rd(ck, "mlp/up_proj/weight"), rd(ck, "mlp/down_proj/weight")
        try:
            ln = rd(ck, "ln_2/weight")
        except Exception:
            ln = np.ones((g.shape[0], g.shape[2]))
        L = g.shape[0]
        gain = np.zeros(L)
        for l in range(L):
            gain[l] = np.linalg.norm(g[l] * ln[l], 2) * np.linalg.norm(u[l] * ln[l], 2) * np.linalg.norm(dn[l], 2)
    except Exception as ex:
        print(tag, "skipped:", type(ex).__name__, str(ex)[:90], flush=True)
        continue
    out[tag] = dict(step=int(ck.rsplit("-", 1)[1]), gains=gain.round(1).tolist(), lnmax=float(ln.max()), L=L, final=final(tag))
    high = [int(i) + 1 for i in np.argsort(-gain)[:6] if gain[i] > 200]
    print(f"{tag:58s} step {out[tag]['step']:>5d} L {L:>2d} max gain {gain.max():9.0f} at L{int(gain.argmax()) + 1:<3d} median {np.median(gain):6.1f} above 200: {int((gain > 200).sum()):2d} {high} max ln_2 {ln.max():6.2f} final {out[tag]['final'] if out[tag]['final'] is None else round(out[tag]['final'], 3)}", flush=True)
json.dump(out, open("mlp_spike_all.json", "w"))
print("\ndesign: gain max separate heads, ordinary; cost")
for tag, r in sorted(out.items()):
    if tag.startswith("-sep-bbfrozen-"):
        continue
    o = out.get("-sep-bbfrozen-" + tag[len("-sep-"):])
    if o is None:
        continue
    cost = r["final"] - o["final"] if (r["final"] is not None and o["final"] is not None) else None
    print(f"{tag[5:]:50s} {max(r['gains']):9.0f} {max(o['gains']):9.0f}  {'-' if cost is None else format(cost, '+.3f')}", flush=True)
