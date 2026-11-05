"""Does one sentence hold on every finished pair? "A next-token loss after every layer lowers every layer's loss except the last." For each pair in d48g4.npz (-sep-X with separate heads, -sep-bbfrozen-X ordinary with probes): the last layer's cost, train and eval; the mean gap over the other layers, train and eval; the layer where the gap turns positive for good."""
import numpy as np
z = np.load("/scratch/gpfs/GROUP/USER/tmp/pls_blog/d48g4.npz", allow_pickle=True)
rows = []
for k in z.files:
    if k.startswith("-sep-bbfrozen") and k.endswith("/final"):
        t = k[len("-sep-bbfrozen"):-6]; s = "-sep" + t + "/final"
        if s not in z.files:
            continue
        o, a = z[k], z[s]
        L = len(o)
        d = a - o
        eo, ea = z["-sep-bbfrozen" + t + "/eval"], z["-sep" + t + "/eval"]
        e = ea - eo
        pos = np.where(d < 0)[0]
        cross = int(pos.max()) + 2 if len(pos) else 1        # first layer after the last negative gap
        rows.append((t, L, float(o[-1]), float(a[-1]), float(d[-1]), float(np.nanmean(d[:-1])), float(d[:-1].max()), cross, float(e[-1]), float(np.nanmean(e[:-1])), int((d[:-1] > 0).sum()), int((e[:-1] > 0).sum())))
print(len(rows), "pairs")
print("design                                     L  ord   sep   cost  mean gap other layers  worst other  gap turns + at layer | eval: cost  mean gap | other layers above 0: train, eval")
for r in sorted(rows, key=lambda r: (r[1], r[4])):
    print(f"{r[0][1:]:40s} {r[1]:3d} {r[2]:.3f} {r[3]:.3f} {r[4]:+.3f}  {r[5]:+.3f}  {r[6]:+.3f}  L{r[7]:<3d} | {r[8]:+.3f} {r[9]:+.3f} | {r[10]:3d} {r[11]:3d}")
c = np.array([r[4] for r in rows if r[1] == 48]); e = np.array([r[8] for r in rows if r[1] == 48])
print(f"48-layer pairs {len(c)}: cost train min {c.min():+.3f} median {np.median(c):+.3f} max {c.max():+.3f}; positive {int((c > 0).sum())}; eval min {np.nanmin(e):+.3f} median {np.nanmedian(e):+.3f} max {np.nanmax(e):+.3f}; positive {int((e > 0).sum())}")
