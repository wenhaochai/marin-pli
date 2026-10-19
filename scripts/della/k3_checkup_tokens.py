"""Token file for the Kimi K3 check-up (k3_checkup.py): Paloma c4_en validation, re-tokenized with the K3 tokenizer.
The Marin store keeps Paloma c4_en tokenized with the Llama 3 tokenizer (zarr3, read with tensorstore; eos 128001 ends
each document). Each document is decoded with the Llama 3 tokenizer, encoded again with K3's tokenizer exactly as
k3_tokens.py encoded WikiText (tok.encode(text, add_special_tokens=False)), and preceded by K3's [BOS] (163584); the
documents are concatenated in store order and cut into N sequences of T tokens. Runs on a vis node, in two steps,
because tensorstore lives in the jax env and the K3 tokenizer in the k3lens env:
  envs/jax/bin/python    k3_checkup_tokens.py dump  IDS.npz
  envs/k3lens/bin/python k3_checkup_tokens.py build IDS.npz MODEL_DIR LLAMA_TOKENIZER_JSON OUT.npy [N=64] [T=1024]
build also writes OUT.npy with .json in place of .npy: a note of how the file was made.
"""
import json
import sys

import numpy as np

STORE = "/scratch/gpfs/GROUP/USER/marin_store_big/tokenized/paloma/c4_en-cf1f79/validation/part-00000-of-00001/input_ids"
LLAMA_BOS, LLAMA_EOS, K3_BOS = 128000, 128001, 163584

if sys.argv[1] == "dump":
    import tensorstore as ts
    ledger = json.load(open(STORE.rsplit("/", 2)[0] + "/shard_ledger.json"))
    n_tok = ledger["field_counts"]["input_ids"]
    spec = lambda f: {"driver": "zarr3", "kvstore": {"driver": "file", "path": f"{STORE}/{f}"}}
    data = ts.open(spec("data")).result()[:n_tok].read().result()
    rows = ledger["total_num_rows"]
    offsets = ts.open(spec("offsets")).result()[: rows + 1].read().result()
    np.savez(sys.argv[2], ids=np.asarray(data), offsets=np.asarray(offsets), n_tok=n_tok, rows=rows)
    print("tokens", n_tok, "rows", rows, "eos count", int((data == LLAMA_EOS).sum()), "offsets head", offsets[:4], "tail", offsets[-2:])
    sys.exit(0)

assert sys.argv[1] == "build"
src, model, llama_json, out = sys.argv[2:6]
N = int(sys.argv[6]) if len(sys.argv) > 6 else 64
T = int(sys.argv[7]) if len(sys.argv) > 7 else 1024
from tokenizers import Tokenizer  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

z = np.load(src)
ids = z["ids"]
llama = Tokenizer.from_file(llama_json)
k3 = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
ends = np.flatnonzero(ids == LLAMA_EOS)
docs, start = [], 0
for e in ends:
    d = ids[start:e]
    if len(d) and d[0] == LLAMA_BOS:
        d = d[1:]
    if len(d):
        docs.append(d)
    start = e + 1
out_ids, used = [], 0
for d in docs:
    text = llama.decode(d.tolist(), skip_special_tokens=False)
    out_ids.append(K3_BOS)
    out_ids.extend(k3.encode(text, add_special_tokens=False))
    used += 1
    if len(out_ids) >= N * T:
        break
assert len(out_ids) >= N * T, f"only {len(out_ids)} K3 tokens from {used} documents"
arr = np.asarray(out_ids[: N * T], dtype=np.int64).reshape(N, T)
np.save(out, arr)
note = dict(
    text="Paloma c4_en validation (Marin store, Llama-3 tokenized), re-tokenized with the Kimi K3 tokenizer",
    store=STORE, llama_tokenizer=llama_json, k3_tokenizer=model,
    method="split Llama-3 ids at eos 128001 (drop a leading bos 128000), decode each document with the Llama 3 tokenizer, "
           "encode it with K3's tokenizer (tok.encode(text, add_special_tokens=False), as k3_tokens.py did for WikiText), "
           "put K3 [BOS]=163584 before every document, concatenate in store order, cut into N sequences of T tokens",
    shape=[N, T], documents_in_store=len(docs), documents_used=used, k3_bos_count=int((arr == K3_BOS).sum()),
    script="marin-pls-auto10/scripts/della/k3_checkup_tokens.py",
    first_text=k3.decode(arr[0, :60].tolist())[:300],
)
json.dump(note, open(out[:-4] + ".json", "w"), indent=1)
print(json.dumps(note, indent=1))
