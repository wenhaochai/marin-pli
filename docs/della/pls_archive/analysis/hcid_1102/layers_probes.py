import numpy as np
z = np.load("/scratch/gpfs/GROUP/USER/tmp/pls_blog/d48g4.npz", allow_pickle=True)
T = ["ahc-fa4thd-vs-d48", "ahc-vs-d48-g4", "ahc-vs-d48-s1", "ahc-rstatic-vs-d48", "ahc-hcid-fa4thd-vs-d48", "amhc-vs-d48-g4", "amhc-mhcrwlin-vs-d48", "amhc-mhcid-vs-d48", "amhc-mhcid-mhcrwlin-fa4thd-vs-d48", "apreln-fa4thd-vs-d48"]
L = [0, 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 45, 46, 47]
print(" " * 34, " ".join(f"{l+1:6d}" for l in L))
for what in ("final", "eval"):
    print(what)
    for t in T:
        print(f"{t:34s}", " ".join(f"{v:6.3f}" for v in z[f"-sep-bbfrozen-{t}/{what}"][L]))
print("separate heads final")
for t in T:
    k = f"-sep-{t}/final"
    if k in z.files: print(f"{t:34s}", " ".join(f"{v:6.3f}" for v in z[k][L]))
