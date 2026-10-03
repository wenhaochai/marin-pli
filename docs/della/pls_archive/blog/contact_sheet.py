"""Numbered contact sheet of the earlier Epoch-style figures (writing:plot), for choosing which to keep."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
B = Path("/scratch/gpfs/GROUP/USER/tmp/pls_blog"); F = Path("/scratch/gpfs/GROUP/USER/project/marin-pls/docs/della/pls_figs")
items = [B / n for n in ["layer_probe_130m.png", "layer_backbone_130m.png", "heads_compare_130m.png", "layer_backbone_300m.png",
                         "layer_probe_300m_progress.png", "final_gap.png", "layer_paloma_130m.png", "layer_paloma_300m.png", "layer_train.png"]]
items += [F / n for n in ["130m_train_per_layer.png", "130m_eval_macro_per_layer.png", "130m_eval_c4_bpb_per_layer.png", "130m_dynamic_signals.png"]]
W, cols = 760, 3
font = ImageFont.truetype(str(B / "fonts" / "InstrumentSans-SemiBold.ttf"), 30) if (B / "fonts" / "InstrumentSans-SemiBold.ttf").exists() else ImageFont.load_default()
thumbs = []
for i, p in enumerate(items, 1):
    im = Image.open(p).convert("RGB"); im = im.resize((W, int(im.height * W / im.width)))
    lab = Image.new("RGB", (W, 54), "white"); ImageDraw.Draw(lab).text((10, 10), f"{i}. {p.name}", fill="black", font=font)
    t = Image.new("RGB", (W, im.height + 54), "white"); t.paste(lab, (0, 0)); t.paste(im, (0, 54)); thumbs.append(t)
rows = [thumbs[i:i + cols] for i in range(0, len(thumbs), cols)]
H = sum(max(t.height for t in r) + 20 for r in rows)
sheet = Image.new("RGB", (cols * (W + 20) + 20, H + 20), (235, 235, 235)); y = 20
for r in rows:
    for c, t in enumerate(r): sheet.paste(t, (20 + c * (W + 20), y))
    y += max(t.height for t in r) + 20
sheet.save(B / "earlier_figures_sheet.png"); print(sheet.size, len(items))
