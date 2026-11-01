"""My fixes to five Argon units after reading the units the merge script rejected (2026-11-01). Each fix is a small
string replacement on the polished text, asserted to apply once; writes my_overrides.json for apply_units.py."""
import json, re
from pathlib import Path
new = {}
for f in sorted(Path("units").glob("part_*.txt")):
    cur = None
    for line in f.read_text().split("\n"):
        if line.startswith("@@ "): cur = line[3:].split("|")[0].strip()
        elif cur and line.strip() and not line.startswith("##"): new[cur] = line.strip(); cur = None
FIX = {
    # the motivation sentence of Figure 34 was dropped; restore it
    "u110en": [("ran out of GPU memory for AttnRes and MoDA. Here ten panels",
                "ran out of GPU memory for AttnRes and MoDA. This figure tests the lens of Figures 16 and 17 on one text for all ten designs. Ten panels")],
    "u110zh": [("因 GPU 显存不足而未能完成。这张图的十个面板",
                "因 GPU 显存不足而未能完成。这张图在同一份文本上，对全部十种结构检验图 16 和图 17 的 lens。图中十个面板")],
    # keep "another cross-entropy implementation"; keep 折线 (the old text did not name the line's colour)
    "u123en": [("trained with bf16 logit gradients as in Pre-LN and the FA4 attention kernel.",
                "trained with another cross-entropy implementation, whose logit gradients are bf16 as in Pre-LN, and with the FA4 attention kernel.")],
    "u123zh": [("蓝线按深度画出", "折线按深度画出")],
    # "pays the full separate-heads cost" is not "the same as most designs"
    "u133zh": [("AttnRes (Block) 的独立头代价与多数结构相同，", "AttnRes (Block) 承受全部的独立头代价，")],
}
out = {}
for i, reps in FIX.items():
    t = new[i]
    for a, b in reps:
        assert t.count(a) == 1, (i, a, t[:200]); t = t.replace(a, b)
    out[i] = t
json.dump(out, open("my_overrides.json", "w"), ensure_ascii=False, indent=1)
print("wrote", len(out), "overrides")
