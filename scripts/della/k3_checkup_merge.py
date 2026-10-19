"""Merge k3_checkup.py's per-rank sums -> OUT/merged.json (the shared AttnRes check-up schema) and OUT/timing.json.
CE and acc = sum over ranks of the per-token sums / number of predicted tokens (positions 0..T-2); AttnRes weights,
entropies, write and source RMS = token sums / tokens (all T positions); exit_emb_by_pos = bin sums / bin counts.
Usage: python k3_checkup_merge.py OUT_DIR "TEXT DESCRIPTION"
"""
import json
import os
import sys

out, text = sys.argv[1], sys.argv[2]
ranks = [json.load(open(os.path.join(out, f))) for f in sorted(os.listdir(out)) if f.startswith("rank") and f.endswith(".json")]
ranks.sort(key=lambda r: r["rank"])
assert [r["rank"] for r in ranks] == list(range(len(ranks))), "missing rank files"
n = min(len(r["layers"]) for r in ranks)
if n < max(len(r["layers"]) for r in ranks) or n < ranks[0]["max_layers"]:
    print(f"WARNING: only {n} layers in every rank (asked for {ranks[0]['max_layers']})")
n_pred = sum(r["n_pred"] for r in ranks)
n_tok = sum(r["n_tok"] for r in ranks)


def tot(i, k):
    return sum(r["layers"][i][k] for r in ranks)


def vtot(i, k):
    vals = [r["layers"][i][k] for r in ranks]
    assert len({len(v) for v in vals}) == 1, (i, k)
    return [sum(v[j] for v in vals) for j in range(len(vals[0]))]


layers = []
for i in range(n):
    assert len({r["layers"][i]["n_sources"] for r in ranks}) == 1
    rec = {"layer": i, "n_sources": ranks[0]["layers"][i]["n_sources"]}
    for nm in ("next", "exit", "sum"):
        rec[f"{nm}_ce"] = tot(i, f"{nm}_ce_sum") / n_pred
        rec[f"{nm}_acc"] = tot(i, f"{nm}_acc_sum") / n_pred
    for nm in ("attn_in", "mlp_in", "exit"):
        rec[f"{nm}_probs"] = [s / n_tok for s in vtot(i, f"{nm}_probs_sum")]
    for nm in ("attn_in", "mlp_in", "exit"):
        rec[f"{nm}_entropy"] = tot(i, f"{nm}_entropy_sum") / n_tok
    rec["exit_emb_by_pos"] = [s / c for s, c in zip(vtot(i, "exit_emb_by_pos_sum"), vtot(i, "exit_emb_by_pos_cnt"))]
    for nm in ("attn", "mlp"):
        rec[f"{nm}_write_rms"] = tot(i, f"{nm}_write_rms_sum") / n_tok
    for nm in ("attn", "mlp"):
        rec[f"{nm}_write_cos_out"] = tot(i, f"{nm}_write_cos_out_sum") / n_tok
    rec["source_rms"] = [s / n_tok for s in vtot(i, "source_rms_sum")]
    assert len(rec["exit_probs"]) == rec["n_sources"] == len(rec["source_rms"]) == len(rec["mlp_in_probs"])
    layers.append(rec)

ablations = {}
if all("ablations" in r for r in ranks):
    for v in ranks[0]["ablations"]:
        a = [r["ablations"][v] for r in ranks]
        ablations[v] = {"ce": sum(x["ce_sum"] for x in a) / sum(x["n"] for x in a), "acc": sum(x["acc_sum"] for x in a) / sum(x["n"] for x in a)}
else:
    print("WARNING: some rank has no final ablations (run not finished)")

merged = {"model": "Kimi-K3", "text": text, "tokens": n_pred, "n_layers": ranks[0]["n_layers"], "block_layers": ranks[0]["block_layers"],
          "layers": layers, "ablations": ablations, "queries": ranks[0]["queries"]}
json.dump(merged, open(os.path.join(out, "merged.json"), "w"), indent=1)
timing = {"ranks": len(ranks), "variants": ranks[0]["variants"],
          "layers": [{"layer": i, "load_s": max(r["layers"][i]["load_s"] for r in ranks), "run_s": max(r["layers"][i]["run_s"] for r in ranks),
                      "wait_s": max(r["layers"][i]["wait_s"] for r in ranks), "peak_gb": max(r["layers"][i]["peak_gb"] for r in ranks),
                      "active": ranks[0]["layers"][i]["active"]} for i in range(n)]}
json.dump(timing, open(os.path.join(out, "timing.json"), "w"), indent=1)
for rec in layers:
    print(f"L{rec['layer']:2d} next {rec['next_ce']:.4f} exit {rec['exit_ce']:.4f} sum {rec['sum_ce']:.4f} acc {rec['exit_acc']:.3f} | "
          f"H(attn/mlp/exit) {rec['attn_in_entropy']:.2f}/{rec['mlp_in_entropy']:.2f}/{rec['exit_entropy']:.2f} | "
          f"emb@exit {[round(p, 3) for p in rec['exit_emb_by_pos']]} | rms a/m {rec['attn_write_rms']:.3f}/{rec['mlp_write_rms']:.3f}")
for v, a in ablations.items():
    print(f"final {v:16s} ce {a['ce']:.4f} acc {a['acc']:.4f}")
