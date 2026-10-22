"""Cost of repetition against z = (R-1) N / U (Chai & Xiong 2026, arXiv 2610.05591: excess loss = 0.0203 bpb x z^1.07,
N all parameters including both embedding tables, U the unique training targets). Our cost: exact Paloma macro bpb of
the R-pass run minus the unique-token run, seed 0, from wandb_cache.json (bpb_exact_tables.json). N from W&B
parameter_count (body + 2 (K+256) d); U = round(steps / R) batches x batch x 4096. Writes collapse_z.json."""
import json, math
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
TB = {int(k): {int(S): v for S, v in d.items()} for k, d in json.loads((HERE / "bpb_exact_tables.json").read_text()).items()}
C = json.loads((HERE / "wandb_cache.json").read_text())
SUBS = sorted(TB[128000][4096])
RATIO = {128000: 1.0}
RATIO.update({k: json.load(open(f"/scratch/gpfs/GROUP/USER/marin_store_big/tokenized/convert_v{k}.json"))["fineweb-edu-10B"]["ratio"] for k in (64000, 32000, 16000, 8000, 256)})
TAG = {128000: "", 64000: "-v64k", 32000: "-v32k", 16000: "-v16k", 8000: "-v8k", 256: "-bytes"}
SIZE = {"130m": (512, 22820096, 128, 4959), "300m": (768, 102584064, 128, 11444), "520m": (1024, 339834880, 256, 9918)}   # d, body params, batch, steps (128K)
def final(rid, k):
    c = C.get(rid)
    if not c: return None
    st, vals = c["h"][-1]
    return sum(v * TB[k][4096][s][0] / (math.log(2) * TB[k][4096][s][1]) for v, s in zip(vals, SUBS)) / 16
rows = []
for size, (d, body, batch, steps0) in SIZE.items():
    for k in TAG:
        N = body + 2 * (k + 256) * d
        steps = round(steps0 * RATIO[k])
        full = final(f"{size}-della4xh100{TAG[k]}", k)
        for R in (2.4, 2.6, 8):
            rep = final(f"{size}-della4xh100{TAG[k]}-rep{R:g}", k)
            if full is None or rep is None: continue
            U = round(steps / R) * batch * 4096
            z = (R - 1) * N / U
            rows.append(dict(size=size, k=k, R=R, N=N, U=U, z=round(z, 4), cost=round(rep - full, 4), pred=round(0.0203 * z ** 1.07, 4),
                             z_text=round((R - 1) * N / (U / RATIO[k]), 4)))
for r in rows: print(r)
pos = [r for r in rows if r["cost"] > 0]
for key in ("z", "z_text"):
    x, y = np.log([r[key] for r in pos]), np.log([r["cost"] for r in pos])
    p, a = np.polyfit(x, y, 1); res = y - (p * x + a); r2 = 1 - res.var() / y.var()
    print(f"fit on {key}: cost = {math.exp(a):.4f} z^{p:.2f}, R2 {r2:.3f}, n {len(pos)}")
xs = np.log([r["z"] for r in pos]); ys = np.log([r["cost"] for r in pos]); pr = np.log(0.0203) + 1.07 * xs
print("paper formula on z: R2", round(1 - ((ys - pr) ** 2).sum() / ((ys - ys.mean()) ** 2).sum(), 3), "median ratio measured/pred", round(float(np.median(np.exp(ys - pr))), 2))
(HERE / "collapse_z.json").write_text(json.dumps(rows))
