"""HC's static mixing A0 = I + res split in three orthogonal parts (Frobenius): T = (g - 1) / m 11' (uniform shrink of the sum, g = mean row sum); U = the spread of the row sums, u_i = r_i - g spread evenly over the row, u_i / m per entry (moves stream differences into the sum); R = the rest, every row sums to 0 (w1006 has only this part). Shares of |A0 - I|^2 by layer group, final checkpoints. Usage: row_sums2.py TAG ..."""
import glob, sys, numpy as np, tensorstore as ts
ROOT = "/scratch/gpfs/GROUP/USER/marin_store_big"
SP = ROOT + "/speedrun/muonh-qwen3-130m-della4xh100-pls1"
def rd(ck, key):
    s = ts.open({"driver": "zarr3", "kvstore": {"driver": "ocdbt", "base": {"driver": "file", "path": ck.rstrip("/") + "/"}, "path": f"model/transformer/layers/stacked/{key}/"}}).result()
    return np.asarray(s.read().result(), np.float64)
for tag in sys.argv[1:]:
    ck = sorted(glob.glob(f"{SP}{tag}/*/checkpoints/step-*"), key=lambda p: int(p.rsplit("-", 1)[1]))[-1]
    R0 = np.stack([rd(ck, f"hyper/{j}/res") for j in (0, 1)], 1)
    m = R0.shape[-1]
    print(tag, ck.rsplit("/", 1)[1], "streams", m)
    print(" layers  |A-I|^2   T share  U share  R share   mean |u_i|")
    for lo, hi in ((1, 1), (2, 6), (7, 11), (12, 12), (13, 24), (25, 36), (37, 48)):
        E = R0[lo - 1:hi].reshape(-1, m, m)
        r = E.sum(-1); g = r.mean(-1, keepdims=True)             # row sums of A - I; g is the mean
        T = np.broadcast_to((g / m)[..., None], E.shape)         # (g - 1) / m in every entry, with g of A - I
        U = np.broadcast_to(((r - g) / m)[..., None], E.shape)   # u_i / m across row i
        Rr = E - T - U
        a, t, u, rr = (E ** 2).sum(), (T ** 2).sum(), (U ** 2).sum(), (Rr ** 2).sum()
        print(f"  L{lo}-{hi:<3d} {a / len(E):8.3f}  {t / a:6.2f}  {u / a:6.2f}  {rr / a:6.2f}   {np.abs(r - g).mean():6.3f}   check {(t + u + rr) / a:.4f}")
