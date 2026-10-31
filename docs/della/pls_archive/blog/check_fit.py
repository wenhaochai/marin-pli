import json, numpy as np
D = dict(np.load("/scratch/gpfs/GROUP/USER/tmp/pls_blog/flops.npz")); M = json.load(open("/scratch/gpfs/GROUP/USER/tmp/pls_blog/eval_macro.json"))
TOK, V = 128 * 4096, 128256
for size, L, hid in (("130", 6, 512), ("300", 12, 768)):
    F = D[f"{size}-base/gflops"][-1] * 1e9 / (D[f"{size}-base/step"][-1] * TOK) / 3 - 2 * hid * V
    st = M[f"{size}m-shared"]["step"][-1]; mac = np.array(M[f"{size}m-shared"]["macro"][-1])
    ks = np.arange(1, L); x = st * TOK * 3 * (ks + 1) / L * F; y = mac[1:] - 1.5
    b, a = np.polyfit(np.log(x), np.log(y), 1); fit = np.exp(a) * x ** b
    print(f"{size}m slope {b:.4f}; mean log-resid {np.mean(np.log(y) - np.log(fit)):+.2e}")
    for k, xi, yi, fi in zip(ks, x, y, fit):
        print(f"  L{k}: C={xi:.3g}  L-1.5={yi:.3f}  fit={fi:.3f}  resid={yi - fi:+.3f}")
    # free-E fit for comparison
    from scipy.optimize import curve_fit
    (E, A, al), _ = curve_fit(lambda c, E, A, al: E + A * (c / 1e18) ** (-al), x, mac[1:], p0=(mac.min() - 0.3, 0.3, 0.5), bounds=([0, 0, 0], [mac[1:].min(), 50, 5]), maxfev=20000)
    print(f"  free-E fit on Paloma macro: E={E:.3f} A={A:.3f} alpha={al:.3f}")
