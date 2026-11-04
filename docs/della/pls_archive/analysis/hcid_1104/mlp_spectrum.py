"""MLP of sublayer 1: F = down(act(gate(z)) * up(z)), z = RN(x) with the learned scale ln_2. Is there a high-gain direction at the layers where tokens first blow up (w810: L5, L10, L11, L14; w893: L12, L13)?
Reads only the weights from the checkpoints (no model): per layer, top singular values of gate x diag(ln_2), up x diag(ln_2), down; max coordinate of ln_2. For the runs w810, w893, HC separate heads, w811 and HC ordinary, mHC mixing = I + linear separate heads.
    nice -n 19 $PY mlp_spectrum.py"""
import glob, sys, numpy as np, tensorstore as ts
SP = "/scratch/gpfs/GROUP/USER/marin_store_big/speedrun/muonh-qwen3-130m-della4xh100-pls1"
TP = "/scratch/gpfs/GROUP/USER/marin_store_big/tmp/checkpoints-temp" + SP
R = {"w810": "-sep-ahc-hcid-fa4thd-vs-d48", "w893": "-sep-ahc-hcid-fa4thd-vs-d48-s1", "HC sep": "-sep-ahc-fa4thd-vs-d48", "w811 ord": "-sep-bbfrozen-ahc-hcid-fa4thd-vs-d48", "HC ord": "-sep-bbfrozen-ahc-fa4thd-vs-d48", "mHC I lin sep": "-sep-amhc-mhcid-mhcrwlin-fa4thd-vs-d48"}
def rd(ck, key):
    s = ts.open({"driver": "zarr3", "kvstore": {"driver": "ocdbt", "base": {"driver": "file", "path": ck.rstrip("/") + "/"}, "path": f"model/transformer/layers/stacked/{key}/"}}).result()
    return np.asarray(s.read().result(), np.float64)
LAY = list(range(0, 18)) + [23, 35, 47]
for name, tag in R.items():
    ck = sorted(glob.glob(f"{SP}{tag}/*/checkpoints/step-*") + glob.glob(f"{TP}{tag}/*/checkpoints/step-*"), key=lambda p: int(p.rsplit("-", 1)[1]))[-1]
    g, u, d, ln = rd(ck, "mlp/gate_proj/weight"), rd(ck, "mlp/up_proj/weight"), rd(ck, "mlp/down_proj/weight"), rd(ck, "ln_2/weight")
    if name == "w810":
        print("shapes gate", g.shape, "up", u.shape, "down", d.shape, "ln_2", ln.shape, flush=True)
    print(f"\n{name} {ck.rsplit('/', 1)[1]}: layer: gate x ln top 3 sv; up x ln top 3 sv; down top 2 sv; max ln_2 (coordinate), median ln_2; product of the top sv")
    for l in LAY:
        # weights map the embed axis to the hidden axis; orient so that the 512 embed axis multiplies ln_2
        G = g[l] if g[l].shape[1] == ln.shape[1] else g[l].T
        U = u[l] if u[l].shape[1] == ln.shape[1] else u[l].T
        sg = np.linalg.svd(G * ln[l], compute_uv=False)[:3]
        su = np.linalg.svd(U * ln[l], compute_uv=False)[:3]
        sd = np.linalg.svd(d[l], compute_uv=False)[:2]
        print(f"  L{l + 1:<3d} {sg[0]:7.1f} {sg[1]:7.1f} {sg[2]:7.1f} ; {su[0]:7.1f} {su[1]:7.1f} {su[2]:7.1f} ; {sd[0]:6.2f} {sd[1]:6.2f} ; {ln[l].max():5.2f} ({int(ln[l].argmax()):3d}) {np.median(ln[l]):5.2f} ; {sg[0] * su[0] * sd[0]:9.0f}", flush=True)
