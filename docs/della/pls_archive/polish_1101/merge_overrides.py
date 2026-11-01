"""Combine my_overrides.json with the auditors' overrides_<k>.json into all_overrides.json; an auditor's text wins
(it was made from the NEW text, which already carried my fix). Prints any unit both touched."""
import json
from pathlib import Path
out = json.load(open("my_overrides.json"))
for k in range(1, 6):
    d = json.load(open(f"overrides_{k}.json"))
    for i, t in d.items():
        if i in out: print("both:", i)
        out[i] = t
json.dump(out, open("all_overrides.json", "w"), ensure_ascii=False, indent=1)
print(len(out), "overrides")
