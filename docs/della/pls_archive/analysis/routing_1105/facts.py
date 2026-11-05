"""Fact sheet of the post "Deeper Models Gain from Residual Paths, Not from Learned Routing". Every number in the post comes from this file's output (facts.json, facts.md). Source: tmp/pls_blog/d48g4.npz (W&B reself/marin-della; final = mean of the last 50 logged steps of train/pls/L47; eval = eval/L47/loss at the last step; ordinary = -sep-bbfrozen runs: the backbone trains on the last layer's loss only), and measured files named below.
Noise of the measure: pairs of repeated runs of the routing-free designs and of Pre-LN."""
import json, numpy as np
from pathlib import Path
z = np.load("/scratch/gpfs/GROUP/USER/tmp/pls_blog/d48g4.npz", allow_pickle=True)
def run(t):
    k = f"-sep-bbfrozen-{t}"
    return dict(tag=t, heldout=round(float(z[k + "/eval"][47]), 3), train=round(float(z[k + "/final"][47]), 3), params_M=round(float(z[k + "/params"]) / 1e6, 1) if k + "/params" in z.files else None)
G = dict(
    pre_ln=["apreln-vs-d48-g4", "apreln-vs-d48-s1", "apreln-fa4thd-vs-d48", "apreln-vs-d48-rerun", "apreln-fa4thd-vs-d48-rerun"],
    one_path_other=["vs-d48-g4", "alns-vs-d48-g4", "akeel-vs-d48-g4"],
    hc_mixing=["ahc-vs-d48-g4", "ahc-vs-d48-s1", "ahc-fa4thd-vs-d48"],
    hc_mixing_overflowed=["ahc-fa4thd-vs-d48-s1"],
    hc_mixing_variants=["ahc_pos-vs-d48-g4", "ahc-hcrwsig-vs-d48", "ahc-hcrwj-fa4thd-vs-d48", "ahc-hcwsh-fa4thd-vs-d48"],
    hc_static_mixing=["ahc-rstatic-vs-d48"],
    hc_no_mixing=["ahc-hcid-fa4thd-vs-d48", "ahc-hcid-fa4thd-vs-d48-s1"],
    hc_scalar=["ahc-hcscalar-fa4thd-vs-d48"],
    hc_moves_keeps_sum=["ahc-hcmove-fa4thd-vs-d48"],
    hc_keeps_sum_signed=["ahc-hcsum-hcf32-fa4thd-vs-d48"], hc_only_shrinks=["ahc-hcshrink-fa4thd-vs-d48"], hc_half_dose_shrink=["ahc-hcshrink-sp0.5-fa4thd-vs-d48"],
    hc_streams_with_mixing=["ahc-hcs1-vs-d48", "ahc-hcs2-vs-d48", "ahc-hcs8-vs-d48"],
    mhc=["amhc-vs-d48-g4"], mhc_linear=["amhc-mhcrwlin-vs-d48"],
    mhc_no_mixing=["amhc-mhcid-vs-d48"], mhc_no_mixing_linear=["amhc-mhcid-mhcrwlin-fa4thd-vs-d48", "amhc-mhcid-mhcrwlin-fa4thd-vs-d48-s1"],
    attnres=["aattnres-vs-d48-g4", "aattnres-vs-d48-s1-g4"], attnres_unit_rms=["aattnres-vnorm-vs-d48"], attnres_static=["aattnres-rstatic-vs-d48"],
    attnres_block=["aattnres_block-vs-d48-g4"], moda=["amoda-vs-d48-g4"])
F = {g: [run(t) for t in ts] for g, ts in G.items()}
ho = lambda g: [r["heldout"] for r in F[g]]
tr = lambda g: [r["train"] for r in F[g]]
D = dict(
    hc_remove_mixing_heldout=[round(a - b, 3) for b in ho("hc_no_mixing") for a in ho("hc_mixing")],
    mhc_remove_mixing_heldout=round(ho("mhc_no_mixing")[0] - ho("mhc")[0], 3),
    mhc_linear_remove_mixing_heldout=[round(a - ho("mhc_linear")[0], 3) for a in ho("mhc_no_mixing_linear")],
    hc_remove_token_dependence_heldout=[round(a - ho("hc_static_mixing")[0], 3) for a in ho("hc_mixing")],
    attnres_remove_token_dependence_heldout=[round(a - ho("attnres_static")[0], 3) for a in ho("attnres")],
    attnres_unit_rms_vs_static=round(ho("attnres_unit_rms")[0] - ho("attnres_static")[0], 3),
    one_path_range=[min(ho("pre_ln") + ho("one_path_other")[:2]), max(ho("pre_ln") + ho("one_path_other")[:2])],
    several_paths_no_mixing_range=[min(ho("hc_no_mixing") + ho("mhc_no_mixing") + ho("mhc_no_mixing_linear") + ho("attnres_static")), max(ho("hc_no_mixing") + ho("mhc_no_mixing") + ho("mhc_no_mixing_linear") + ho("attnres_static"))],
    hc_streams_with_mixing_heldout={"1": ho("hc_streams_with_mixing")[0], "2": ho("hc_streams_with_mixing")[1], "4": ho("hc_mixing"), "8": ho("hc_streams_with_mixing")[2]},
    repeat_spread=dict(hc_no_mixing=round(abs(ho("hc_no_mixing")[0] - ho("hc_no_mixing")[1]), 3), mhc_no_mixing_linear=round(abs(ho("mhc_no_mixing_linear")[0] - ho("mhc_no_mixing_linear")[1]), 3), pre_ln=round(max(ho("pre_ln")) - min(ho("pre_ln")), 3)))
# Failures: excess of the last layer's train loss over its 101-step running median, step 200 on
def spikes(t):
    k = f"-sep-bbfrozen-{t}"; s, c = z[k + "/step"], z[k + "/ce"][:, 47]
    m = np.array([np.median(c[max(0, i - 50):i + 51]) for i in range(len(c))]); ex = c - m; ex[s < 200] = 0
    ev, last = 0, -100
    for i in np.where(ex > 0.3)[0]:
        ev += int(s[i] - last > 50); last = s[i]
    return dict(events=ev, largest_excess=round(float(ex.max()), 2), step_of_largest=int(s[np.argmax(ex)]))
S = {t: spikes(t) for g in ("attnres", "attnres_static", "attnres_unit_rms", "hc_mixing", "hc_mixing_overflowed", "hc_no_mixing", "mhc", "mhc_no_mixing", "pre_ln") for t in G[g]}
M = dict(
    mhc_mixing_on_tokens=dict(mean_offdiag_per_row=0.029, p99=0.333, max=0.807, loss_as_trained=2.873, loss_every_A_replaced_by_I=3.480, source="tmp/pls_hcid/mhc_mix_full.py, 16,380 FineWeb-Edu tokens"),
    hc_overflow=dict(step_first_rise=4408, step_5_nats=4420, final_train=9.318, final_heldout=9.885, median_state_norm_L48=1.5e20, float32_sum_of_squares_overflow_at=1.8e19, share_tokens_at_uniform_loss=0.701, ln_V=11.76, grad_norm_median_steps_4400_4599=0.97, source="docs pls.md item 24, tmp/pls_hcid/state_margin.py, ce_by_target.py"),
    hc_state_norm_median_L48=dict(hc_mixing_other_run=1.2e7, hc_no_mixing=667, source="res_gain tables tmp/pls_hcid/res_gain_all.txt, res_gain_b.txt (no mixing: 6.674e+02)"),
    hc_layer12_mixing_row_sums_mlp=[0.81, 0.10, -0.47, 0.86], source_row_sums="tmp/pls_hcid/row_sums.py, HC ordinary final checkpoint")
# Depth test (2026-11-05): tmp/pls_blog/depth_final.json, tmp/pls_hcid/res_gain_depth*.npz through tmp/pls_hcid/scale_depth.json
DF = json.load(open("/scratch/gpfs/GROUP/USER/tmp/pls_blog/depth_final.json"))
SC = json.load(open("/scratch/gpfs/GROUP/USER/tmp/pls_hcid/scale_depth.json"))
end = lambda runs: [r[-1][1] for r in runs]
mean = lambda v: float(np.mean(v))
DEPTH = {}
for l in ("12", "24", "48"):
    nm, lm, pl = end(DF["no_mixing"][l]), end(DF["learned"][l]), DF["preln"][l]
    st = [q[0] for q in DF["no_mixing"][l][0]]
    by = [round(mean([r[j][1] for r in DF["no_mixing"][l]]) - mean([r[j][1] for r in DF["learned"][l]]), 3) for j in range(len(st))]
    a, b = SC[f"HC/{l}"], SC[f"I/{l}"]
    cum = np.array(a["cum"])
    DEPTH[l] = dict(heldout_one_path_preln=[round(v, 3) for v in pl], heldout_no_mixing=[round(v, 3) for v in nm], heldout_learned_mixing=[round(v, 3) for v in lm],
                    gain_from_learned_mixing=round(mean(nm) - mean(lm), 3), gain_from_paths=round(mean(pl) - mean(nm), 3), gain_from_learned_mixing_by_step=dict(zip(map(str, st), by)),
                    median_stream_sum_norm_last_layer=dict(learned_mixing=round(a["s"][-1], 0), no_mixing=round(b["s"][-1], 0)),
                    orders_of_magnitude_of_the_mixing_scale_change=round(float(max(cum.max(), 0) - min(cum.min(), 0)), 1),
                    last_write_over_new_sum=dict(learned_mixing=float(f"{a['share'][-1]:.2g}"), no_mixing=float(f"{b['share'][-1]:.2g}")))
PAPERS = dict(hc_table3=dict(freeze_width_connections_loss_rise=0.021, layers=16, tokens="500B", model="OLMo-1B"), mhc_table1=dict(identity_mixing_loss_rise=0.022, layers="not stated"), attnres_table4=dict(input_independent_loss_rise=0.012, layers=16, tokens="87.9B"), muddformer_table5=dict(static_perplexity_rise=0.35, layers=24, tokens="7B"), source="related.md")
ARMS = {k: dict(heldout=SC[k]["heldout"], orders_of_magnitude_of_the_mixing_scale_change=round(SC[k]["orders"], 1), median_stream_sum_norm_last_layer=round(SC[k]["s"][-1], 0), last_write_over_new_sum=float(f"{SC[k]['share'][-1]:.2g}")) for k in ("I/48", "HC moves/48", "HC keeps sum/48", "HC scalar/48", "HC static/48", "HC half dose/48", "HC only shrinks/48", "HC/48")}
out = dict(runs=F, differences=D, failures=S, measured=M, depth=DEPTH, four_stream_arms_48_layers=ARMS, papers=PAPERS, derived=dict(hc_layer12_mean_row_sum=round(float(np.mean(M["hc_layer12_mixing_row_sums_mlp"])), 2), hc_scalar_stream_sum_norm_last_layer_48=round(SC["HC scalar/48"]["s"][-1], 0), params_M_without_probe_heads={"12": 177, "24": 223, "48": 314}, width=512, mlp=1792, tokens="2.6B", steps=4959, parameter_spread_across_designs_48_layers_percent=1.5, parameter_spread_within_a_comparison_48_layers_percent=0.1, learned_mixing_48_product_of_gains=dict(lowest_log10=-4.46, at_layer=22.5, at_last_layer_log10=0.14), gain_over_no_mixing_same_depth_48=dict(one_learned_number=0.015, constant_mixing=-0.029, half_dose_shrink=-0.096, only_shrinks=-0.107, moves_content_nonnegative=0.002, moves_content_signed=-0.011)))
json.dump(out, open("facts.json", "w"), indent=1)
L = ["# Fact sheet, Deeper Models Gain from Residual Paths, Not from Learned Routing", "", "Held-out = eval/L47/loss, Paloma c4_en, last step. Train = mean of the last 50 steps of the last layer. Runs below: 48 layers, width 512, 2.6B tokens, ordinary models. Depth section: 12, 24, 48 layers at the same budget. Gain from learned mixing = no mixing minus learned mixing, held-out. Gain from paths = Pre-LN minus no mixing, held-out. A positive gain means the thing helps. Stream sum norm and write share: median over 8,192 FineWeb-Edu tokens, final checkpoint.", ""]
for g, rs in F.items():
    L.append(f"- {g}: " + "; ".join(f"{r['tag']}: held-out {r['heldout']}, train {r['train']}" for r in rs))
L += ["", "## Differences (held-out, positive = the design with routing is worse)", ""] + [f"- {k}: {v}" for k, v in D.items()]
L += ["", "## Failures", ""] + [f"- {k}: {v}" for k, v in S.items()]
L += ["", "## Measured elsewhere", ""] + [f"- {k}: {v}" for k, v in M.items()]
L += ["", "## Depth", ""] + [f"- {k} layers: {v}" for k, v in DEPTH.items()]
L += ["", "## Four-stream arms at 48 layers: scale of the stream sum", ""] + [f"- {k}: {v}" for k, v in ARMS.items()]
L += ["", "## Papers (full detail in related.md)", ""] + [f"- {k}: {v}" for k, v in PAPERS.items()]
L += ["", "## Derived", ""] + [f"- {k}: {v}" for k, v in out["derived"].items()]
open("facts.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
