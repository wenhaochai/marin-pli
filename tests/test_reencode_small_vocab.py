"""CPU check of the through-text conversion (convert_small_vocab_cache.reencode_docs, used for a fresh BPE): with the 8K
truncation as the target tokenizer, converting through the text must give exactly the id-level expansion (convert_docs)
on every document whose Marin ids are canonical, keep special tokens in place (moved to K + id - 128000), decode to the
same text, and handle a special token in the middle of a document and an empty document. Run on a vis node:
    K=8000 CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu nice -n 19 .venv/bin/python tests/test_reencode_small_vocab.py"""
import os

os.environ.setdefault("K", "8000")
import numpy as np
from tokenizers import Tokenizer

from levanter.store.cache import TreeCache

from experiments.references import convert_small_vocab_cache as C
from experiments.references.small_vocab_tokenizer import N_ORDINARY, SRC, expansion

K = 8000
R = "/scratch/gpfs/GROUP/USER/marin_store_big"
full = Tokenizer.from_file(os.path.join(SRC, "tokenizer.json"))
small = Tokenizer.from_file(f"{R}/tokenizers/marin-small/v{K}/tokenizer.json")
flat, offs = expansion(K, f"{R}/tokenizers/marin-small/v{K}")
BOS, EOS = full.token_to_id("<|begin_of_text|>"), full.token_to_id("<|end_of_text|>")

# real documents: two Paloma subsets (code and web), the first 200 each
docs = []
for sub in ("dolma_100_programing_languages", "c4_en"):
    c = TreeCache.load(f"{R}/tokenized/paloma/{sub}-{[d for d in os.listdir(f'{R}/tokenized/paloma') if d.startswith(sub + '-')][0].rsplit('-', 1)[1]}/validation", C.EX)
    docs += [np.asarray(x["input_ids"]) for x in c.get_batch_sync(list(range(200)))]
# synthetic: a special token in the middle, specials only, an empty document, unicode and digits
texts = ["Hello world, 12345 apples.\n\n  indented", "naïve café — 東京 🙂", "def f(x):\n    return x**2  # comment"]
enc = [full.encode(t, add_special_tokens=False).ids for t in texts]
docs += [np.asarray([BOS] + enc[0] + [EOS] + enc[1] + [EOS]), np.asarray([BOS, EOS]), np.zeros(0, np.int64), np.asarray([BOS] + enc[2])]

got, n_in, n_out = C.reencode_docs(docs, full, small, K)
ref, r_in, r_out = C.convert_docs(docs, flat, offs)
assert n_in == r_in == sum(len(d) for d in docs), (n_in, r_in)
assert len(got) == len(ref) == len(docs)
canon = same = 0
for d, g, r in zip(docs, got, ref):
    d = d.tolist(); g = np.asarray(g).tolist(); r = np.asarray(r).tolist()
    assert [t - N_ORDINARY + K for t in d if t >= N_ORDINARY] == [t for t in g if t >= K], "special tokens moved or lost"
    assert full.decode([t for t in d if t < N_ORDINARY], skip_special_tokens=False) == small.decode([t for t in g if t < K], skip_special_tokens=False), "text differs"
    runs = [[]]
    for t in d:
        if t >= N_ORDINARY:
            runs.append([])
        else:
            runs[-1].append(t)
    if all(full.encode(full.decode(x), add_special_tokens=False).ids == x for x in runs if x):
        canon += 1
        assert g == r, "through-text conversion differs from the id-level expansion on a canonical document"
    same += g == r
print(f"{len(docs)} documents: {canon} canonical, all equal to the id expansion; {same} equal overall; tokens {n_in} -> {n_out} (expansion {r_out})")
print("REENCODE OK")
