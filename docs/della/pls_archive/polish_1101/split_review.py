"""Split review.md (apply_units.py) into review_<k>.md by the part named in each unit header."""
import re
from pathlib import Path
blocks = re.split(r"(?m)^(?=## u\d{3})", Path("review.md").read_text())
head, parts = blocks[0], {}
for b in blocks[1:]:
    k = re.match(r"## u\d{3}\w\w \| part (\d)", b).group(1)
    parts.setdefault(k, []).append(b)
for k, bs in sorted(parts.items()):
    Path(f"review_{k}.md").write_text(head + "".join(bs)); print(k, len(bs))
