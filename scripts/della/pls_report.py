"""Per-layer report for the pls runs (VARIANT=pls): the last per-layer eval of each run side by side, the in-run check that
the last layer's readout equals the main eval, and per-layer curves (train/pls/L{k} and eval/L{k}/...).

    python scripts/della/pls_report.py --size=130m --suffixes=-pls1,-pls0 [--out=docs/della/pls_figs] [--smoke=40]

Final-layer comparisons against the baseline pool go through scripts/della/objective_compare.py (its guards: finished runs
only, same budget, Welch with df). This script reads per-layer keys only, which the baseline runs do not have.
"""
import argparse
import os
import re

import numpy as np
import wandb

PROJECT = "reself/marin-della"
C4_BPB = "paloma/c4_en-marin-tokenizer/bpb"
MACRO = "paloma/macro_loss"
LAYER_RE = re.compile(r"^eval/L(\d+)/loss$")


def layers(summary) -> list[int]:
    return sorted({int(m.group(1)) for k in summary.keys() if (m := LAYER_RE.match(k))})


def fmt(x) -> str:
    return "     -" if x is None else f"{x:.4f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size", default="130m")
    p.add_argument("--suffixes", default="-pls1,-pls0")
    p.add_argument("--smoke", default="", help="SMOKE_STEPS of smoke runs to read (run ids end in -smoke<N>)")
    p.add_argument("--out", default="", help="directory for per-layer curve PNGs (omit to skip plots)")
    a = p.parse_args()
    api = wandb.Api(timeout=120)
    sfxs = [s.strip() for s in a.suffixes.split(",") if s.strip()]
    runs = {}
    for sfx in sfxs:
        rid = f"muonh-qwen3-{a.size}-della4xh100{sfx}" + (f"-smoke{a.smoke}" if a.smoke else "")
        run = api.run(f"{PROJECT}/{rid}")
        runs[sfx] = run
        s = run.summary
        print(f"{rid}: state={run.state} step={s.get('_step')} global_step={s.get('global_step')} "
              f"num_train_steps={run.config.get('trainer', {}).get('num_train_steps')}")

    # 1. last per-layer eval, side by side
    cols = [("loss", "loss"), ("macro", MACRO), ("c4_bpb", C4_BPB), ("bpb", "bpb")]
    Ls = {sfx: layers(r.summary) for sfx, r in runs.items()}
    all_layers = sorted(set().union(*Ls.values())) if Ls else []
    print("\nlast eval, per layer (eval/L{k}/...):")
    head = "layer " + " ".join(f"{sfx + ':' + c:>18}" for sfx in sfxs for c, _ in cols)
    print(head)
    for k in all_layers:
        vals = []
        for sfx in sfxs:
            s = runs[sfx].summary
            vals += [fmt(s.get(f"eval/L{k}/{key}")) for _, key in cols]
        print(f"L{k:<4} " + " ".join(f"{v:>18}" for v in vals))
    if len(sfxs) == 2:
        s1, s0 = runs[sfxs[0]].summary, runs[sfxs[1]].summary
        print(f"\ndelta {sfxs[0]} - {sfxs[1]} per layer (loss / macro / c4_bpb):")
        for k in all_layers:
            d = [s1.get(f"eval/L{k}/{key}", np.nan) - s0.get(f"eval/L{k}/{key}", np.nan) for key in ("loss", MACRO, C4_BPB)]
            print(f"L{k:<4} " + " ".join(f"{x:+.4f}" for x in d))

    # 2. the last layer's readout must equal the main eval (same data, same loss path)
    print("\nin-run check: eval/L{last}/* vs eval/*")
    for sfx, r in runs.items():
        s = r.summary
        if not Ls[sfx]:
            print(f"{sfx}: no per-layer eval keys")
            continue
        last = max(Ls[sfx])
        pre = f"eval/L{last}/"
        diffs = [abs(s[k] - s["eval/" + k[len(pre):]]) for k in s.keys()
                 if k.startswith(pre) and "time" not in k and ("eval/" + k[len(pre):]) in s and isinstance(s[k], (int, float))]
        print(f"{sfx}: {len(diffs)} keys, max |diff| = {max(diffs) if diffs else float('nan'):.2e}")

    # 3. curves
    if a.out:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(a.out, exist_ok=True)
        L = max(all_layers) + 1 if all_layers else 0
        for metric, keyf, fname in (
            ("train CE", lambda k: f"train/pls/L{k}", "train_per_layer"),
            ("eval Paloma macro loss", lambda k: f"eval/L{k}/{MACRO}", "eval_macro_per_layer"),
            ("eval c4_en bpb", lambda k: f"eval/L{k}/{C4_BPB}", "eval_c4_bpb_per_layer"),
        ):
            fig, ax = plt.subplots(figsize=(7, 4.5))
            cmap = plt.get_cmap("viridis")
            for sfx, style in zip(sfxs, ("-", "--", ":")):
                keys = [keyf(k) for k in range(L)]
                hist = runs[sfx].scan_history(keys=["_step", *keys])
                rows = [(h["_step"], [h.get(k) for k in keys]) for h in hist]
                if not rows:
                    continue
                steps = np.array([r[0] for r in rows])
                for k in range(L):
                    y = np.array([np.nan if r[1][k] is None else r[1][k] for r in rows], dtype=float)
                    ax.plot(steps, y, style, color=cmap(k / max(L - 1, 1)), lw=1.2, label=f"{sfx} L{k}")
            ax.set_xlabel("step")
            ax.set_ylabel(metric)
            ax.set_title(f"{a.size}: {metric} by layer ({' solid, '.join(sfxs)} dashed)")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=6, ncol=2)
            path = os.path.join(a.out, f"{a.size}_{fname}.png")
            fig.tight_layout()
            fig.savefig(path, dpi=150)
            plt.close(fig)
            print("wrote", path)


if __name__ == "__main__":
    main()
