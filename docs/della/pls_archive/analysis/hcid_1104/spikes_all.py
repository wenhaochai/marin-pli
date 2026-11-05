"""Stability of the last layer across the 48-layer ordinary runs: events where the layer 48 train CE sits more than 0.3 nats above the 101-step running median (one event per 50 steps), and the largest excess, step 200 onward."""
import numpy as np
z = np.load("/scratch/gpfs/GROUP/USER/tmp/pls_blog/d48g4.npz", allow_pickle=True)
G = {"no routing": ["ahc-hcid-fa4thd-vs-d48", "ahc-hcid-fa4thd-vs-d48-s1", "amhc-mhcid-vs-d48", "amhc-mhcid-mhcrwlin-fa4thd-vs-d48", "amhc-mhcid-mhcrwlin-fa4thd-vs-d48-s1", "ahc-hcscalar-fa4thd-vs-d48", "aattnres-rstatic-vs-d48", "ahc-rstatic-vs-d48"],
     "routing": ["ahc-vs-d48-g4", "ahc-vs-d48-s1", "ahc-fa4thd-vs-d48", "ahc-fa4thd-vs-d48-s1", "ahc-hcmove-fa4thd-vs-d48", "amhc-vs-d48-g4", "amhc-mhcrwlin-vs-d48", "aattnres-vs-d48-g4", "aattnres-vs-d48-s1-g4", "aattnres-vnorm-vs-d48", "aattnres_block-vs-d48-g4"],
     "one path": ["apreln-vs-d48-g4", "apreln-vs-d48-s1", "vs-d48-g4", "alns-vs-d48-g4", "akeel-vs-d48-g4"]}
for g, ts in G.items():
    print(g)
    for t in ts:
        k = f"-sep-bbfrozen-{t}"
        s, c = z[k + "/step"], z[k + "/ce"][:, 47]
        m = np.array([np.median(c[max(0, i - 50):i + 51]) for i in range(len(c))])
        ex = c - m; ex[s < 200] = 0
        idx = np.where(ex > 0.3)[0]; ev = 0; last = -100
        for i in idx:
            if s[i] - last > 50:
                ev += 1
            last = s[i]
        print(f"   {t:44s} train {z[k + '/final'][47]:.3f} held-out {z[k + '/eval'][47]:.3f} events {ev:3d} largest excess {ex.max():6.2f}")
