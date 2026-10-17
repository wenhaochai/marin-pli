"""Token file for the Kimi K3 logit lens (k3_lens.py): wikitext-103 raw test articles, concatenated with the K3 tokenizer
(no special tokens), cut into N sequences of T tokens. Runs on a vis node (needs the internet for the dataset once).
Usage: python k3_tokens.py MODEL_DIR OUT.npy [N=64] [T=1024]"""
import sys
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

model, out = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 64
T = int(sys.argv[4]) if len(sys.argv) > 4 else 1024
tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
ids = []
for line in ds["text"]:
    if line.strip():
        ids.extend(tok.encode(line, add_special_tokens=False))
    if len(ids) >= N * T:
        break
arr = np.asarray(ids[: N * T], dtype=np.int64).reshape(N, T)
np.save(out, arr)
print("tokens", arr.shape, "vocab max", int(arr.max()), "first", tok.decode(arr[0, :40].tolist())[:200])
