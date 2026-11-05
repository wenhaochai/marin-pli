"""Fact sheet of the post "Residual Paths Help, Learned Routing Does Not". Every number in the post comes from this file's output (facts.json, facts.md). Source: tmp/pls_blog/d48g4.npz (W&B reself/marin-della; final = mean of the last 50 logged steps of train/pls/L47; eval = eval/L47/loss at the last step; ordinary = -sep-bbfrozen runs: the backbone trains on the last layer's loss only), and measured files named below.
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
out = dict(runs=F, differences=D, failures=S, measured=M)
json.dump(out, open("facts.json", "w"), indent=1)
L = ["# Fact sheet, Residual Paths Help, Learned Routing Does Not", "", "Held-out = eval/L47/loss, Paloma c4_en, last step. Train = mean of the last 50 steps of the last layer. All 48 layers, width 512, 2.6B tokens, ordinary models.", ""]
for g, rs in F.items():
    L.append(f"- {g}: " + "; ".join(f"{r['tag']}: held-out {r['heldout']}, train {r['train']}" for r in rs))
L += ["", "## Differences (held-out, positive = the design with routing is worse)", ""] + [f"- {k}: {v}" for k, v in D.items()]
L += ["", "## Failures", ""] + [f"- {k}: {v}" for k, v in S.items()]
L += ["", "## Measured elsewhere", ""] + [f"- {k}: {v}" for k, v in M.items()]
open("facts.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
