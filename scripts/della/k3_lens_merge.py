"""Add up k3_lens.py's per-rank files (token-weighted means per layer) -> OUT/merged.json, and print a table."""
import json, os, sys
out = sys.argv[1]
ranks = [json.load(open(os.path.join(out, f))) for f in sorted(os.listdir(out)) if f.startswith("rank") and f.endswith(".json")]
n = min(len(r["layers"]) for r in ranks)
w = [r["nseq"] * (r["T"] - 1) for r in ranks]
merged = dict(ranks=len(ranks), tokens=sum(w), layers=[])
for i in range(n):
    rec = {"layer": i}
    for k in ("next_ce", "next_acc", "exit_ce", "exit_acc", "sum_ce", "sum_acc"):
        rec[k] = sum(r["layers"][i][k] * wi for r, wi in zip(ranks, w)) / sum(w)
    for k in ("attn_in_probs", "exit_probs"):
        rec[k] = [sum(r["layers"][i][k][j] * wi for r, wi in zip(ranks, w)) / sum(w) for j in range(len(ranks[0]["layers"][i][k]))]
    merged["layers"].append(rec)
    print(f"L{i:2d} next {rec['next_ce']:.3f} exit {rec['exit_ce']:.3f} sum {rec['sum_ce']:.3f} | exit probs {[round(p, 2) for p in rec['exit_probs']]}")
json.dump(merged, open(os.path.join(out, "merged.json"), "w"), indent=1)
