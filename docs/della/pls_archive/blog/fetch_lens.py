"""Logit lens against each layer's own readout (Figure 16, fig-lens; owner popups 2026-10-17). Per model, two held-out
(Paloma) losses by layer from W&B reself/marin-della:
  own:  eval/L{k}/loss of the source run (its separate head or probe at layer k; the last layer is the model's own head);
  lens: eval/L{k}/loss of the source's -lensat<step> run (pls_lens_group.sbatch, launcher LENS_FROM: the frozen checkpoint
        read at every layer through the model's final norm and lm_head).
A run counts only when W&B says finished and it logged every eval/L{k}/loss. Panels: 130m (6 layers; separate heads are
the 8-GPU rerun 15720165 that kept its checkpoint) and 48 layers of width 512, Sandwich-LN.
Panels d48-<arch>: the other nine 48-layer designs (Figure 17). Panel depth: the ordinary models of 12, 24 and 36
layers, lens only (they have no readouts).
W&B -> lens.json {panel: {setup: {"own": [...], "lens": [...]}}} ("lens" alone under depth); a setup without its runs is left out.
Usage: python fetch_lens.py. Exits 2 on a W&B error."""
import json, sys
from pathlib import Path
import wandb

OUT = Path(__file__).resolve().parent / "lens.json"
P = "reself/marin-della/muonh-qwen3-"
# panel -> setup -> (source run id, lens run id, layers)
SPECS = {
    "130m": {"sep": ("130m-della8xh100-pls1-sep", "130m-della8xh100-pls1-sep-lensat4958", 6),
             "probes": ("130m-della4xh100-pls1-sep-bbfrozen", "130m-della8xh100-pls1-sep-bbfrozen-lensat4958", 6)},
    "d48": {"sep": ("130m-della4xh100-pls1-sep-vs-d48-g4", "130m-della8xh100-pls1-sep-vs-d48-lensat4958-g4", 48),
            "probes": ("130m-della4xh100-pls1-sep-bbfrozen-vs-d48-g4", "130m-della8xh100-pls1-sep-bbfrozen-vs-d48-lensat4958-g4", 48)},
}
# 48 layers, the other nine designs (Figure 17; owner popup 2026-10-18): panel "d48-<arch>"
for a in ("preln", "lns", "deepnorm", "keel", "hc", "mhc", "attnres", "attnres_block", "moda"):
    SPECS[f"d48-{a}"] = {"sep": (f"130m-della4xh100-pls1-sep-a{a}-vs-d48-g4", f"130m-della8xh100-pls1-sep-a{a}-vs-d48-lensat4958-g4", 48),
                         "probes": (f"130m-della4xh100-pls1-sep-bbfrozen-a{a}-vs-d48-g4", f"130m-della8xh100-pls1-sep-bbfrozen-a{a}-vs-d48-lensat4958-g4", 48)}
# ordinary models of 12, 24, 36 layers (Figure 16, third panel): the logit lens alone, no readout of their own
DEPTH = {f"d{n}": (f"130m-della8xh100-pls0-d{n}-lensat4958", n) for n in (12, 24, 36)}
api = wandb.Api(timeout=300)
errors = 0


def per_layer(rid, L):
    """eval/L{k}/loss, k = 0..L-1, of a finished run; None if the run is missing, unfinished or incomplete."""
    global errors
    try:
        r = api.run(P + rid)
    except Exception as e:
        if "Could not find run" not in str(e) and "404" not in str(e):
            print(rid, "W&B ERROR", str(e)[:200]); errors += 1
        return None
    v = [r.summary.get(f"eval/L{k}/loss") for k in range(L)]
    if r.state != "finished" or any(x is None for x in v):
        print(rid, r.state, "missing", sum(x is None for x in v), "of", L, "layers: skipped")
        return None
    return [round(float(x), 5) for x in v]


out = {}
for panel, setups in SPECS.items():
    for setup, (src, lens, L) in setups.items():
        own, ln = per_layer(src, L), per_layer(lens, L)
        if own and ln:
            out.setdefault(panel, {})[setup] = {"own": own, "lens": ln}
            print(panel, setup, "own", own[:3], "...", own[-2:], "| lens", ln[:3], "...", ln[-2:])
for setup, (lens, L) in DEPTH.items():
    ln = per_layer(lens, L)
    if ln:
        out.setdefault("depth", {})[setup] = {"lens": ln}
        print("depth", setup, "lens", ln[:2], "...", ln[-2:])
OUT.write_text(json.dumps(out, indent=1) + "\n")
print("wrote", OUT, {p: sorted(s) for p, s in out.items()})
sys.exit(2 if errors else 0)
