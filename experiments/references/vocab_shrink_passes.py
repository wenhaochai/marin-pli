# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Builds the training caches and the Paloma caches of the runs with a vocabulary that shrinks as data repeats (owner 2026-11-04,
a friend's two ideas; experiments/references/vocab_shrink_qwen3.py is the model).

Every arm reads the same text: the Marin tokenizer's repeated eighth at 300m (183,040 windows of 4096 tokens, pass 1 of
materialized/300m-rep8-rs, in the order the plain 8-pass run reads them). For pass e of 8 each source window takes a view, a K,
from that pass's shares, and goes through the exact id-level expansion to K tokens (small_vocab_tokenizer.expansion; a truncation's
BPE on a Marin token's bytes), kept in Marin ids. Each view's stream is cut into windows of the arm's SEQ_LEN, a partial window
at the end of a pass is dropped, and one written window holds one view. A pass with one view keeps its order, so
equal views in later passes are equal windows, as in a plain 8-pass run; a pass with several views shuffles each chunk of
8192 source windows (seed 30000 + pass), so batches mix them.

Arms (ARMS below): vsfix<K> K on all 8 passes (the control, on the same text as the other arms); vscut<K> path 2, one view per
pass, cut when the epoch count doubles; vslate<K> one late cut, 128K for passes 1-4 and K for passes 5-8; vsmix<e> path 1, the friend's shifting shares (read off their figure) on all five
vocabularies, with the eighth pass at the figure's epoch <e>. vsconst, vsgrow, vsfrozen, vs128k16k, vs16k8k and vsuniq ask what in path 1 does the work (comments in ARMS).

    python -m experiments.references.vocab_shrink_passes check                 (expansion against the K caches of Paloma)
    python -m experiments.references.vocab_shrink_passes paloma 64000 32000 16000 8000  (Paloma caches of K in Marin ids: tokenized/paloma-v<K>m)
    python -m experiments.references.vocab_shrink_passes <arm> [...]           (MARIN_PREFIX/materialized/300m-rep8-<arm>/{train,meta.json})
meta.json: schedule (shares per pass), seq_len, eval_k, pass_sequences, sequences, view_windows per pass. Checks while writing:
every window of a K view holds ids below K or specials only; a pass of one view holds the source text, byte for byte, on a
sample of windows.
"""

import json
import math
import os
import sys
import time

import numpy as np

from levanter.store.cache import CacheMetadata, SerialCacheWriter, TreeCache

from experiments.references.small_vocab_tokenizer import N_ORDINARY, convert, expansion

PREFIX = os.environ.get("MARIN_PREFIX", "/scratch/gpfs/GROUP/USER/marin_store_big")
SOURCE = f"{PREFIX}/materialized/300m-rep8-rs"
EX = {"input_ids": np.zeros((0,), dtype=np.int32)}
EQW = {128000: 4096, 64000: 4224, 32000: 4480, 16000: 4992, 8000: 5504}   # each K's equal-text window (the page's fig-eqwin)
VIEWS = (128000, 64000, 32000, 16000, 8000)
# The friend's Path 1 figure, shares of sequences at its epoch marks, read off the plot (rounded to sum to 1)
MARKS = {1: (0.60, 0.18, 0.10, 0.07, 0.05), 2: (0.35, 0.27, 0.19, 0.11, 0.08), 4: (0.13, 0.19, 0.25, 0.22, 0.21), 8: (0.00, 0.05, 0.16, 0.29, 0.50)}


def shares_at(epoch: float):
    """The figure's shares at an epoch, linear in log2(epoch) between its marks; epochs below 1 take the first mark's."""
    x = math.log2(max(epoch, 1.0))
    lo, hi = min(math.floor(x), 2), min(math.floor(x), 2) + 1
    f = min(x - lo, 1.0)
    a, b = MARKS[2 ** lo], MARKS[2 ** hi]
    s = [(1 - f) * u + f * v for u, v in zip(a, b)]
    t = sum(s)
    return {k: round(v / t, 4) for k, v in zip(VIEWS, s) if v > 0}


def _mix(last_epoch: float):
    """Pass e of 8 at the figure's epoch 1 + (e - 1) x (last_epoch - 1) / 7 on the log scale: e^(log(last_epoch)/log(8))."""
    return [shares_at(e ** (math.log(last_epoch) / math.log(8))) for e in range(1, 9)]


def _one(ks):
    return [{k: 1.0} for k in ks]


def _mean(views):
    """The shares of a schedule over its passes, as one pass."""
    return {k: round(sum(p.get(k, 0.0) for p in views) / len(views), 4) for k in VIEWS if any(k in p for p in views)}


ARMS = {
    # name: (views per pass, SEQ_LEN, the K that evaluation in training uses)
    "vsfix32k": (_one([32000] * 8), 4480, 32000),
    "vsfix16k": (_one([16000] * 8), 4992, 16000),
    "vsfix64k": (_one([64000] * 8), 4224, 64000),    # with vsfix8k, the same text at the other K a mixed run is scored at
    "vsfix8k": (_one([8000] * 8), 5504, 8000),
    # the figure's rule on our 8 passes: cut at epochs 2 and 4 (8 ends the run)
    "vscut32k": (_one([128000, 64000, 64000, 32000, 32000, 32000, 32000, 32000]), 4480, 32000),
    # the same rule one doubling earlier, so that the last half runs at 16K, as in the figure's 16 epochs
    "vscut16k": (_one([128000, 64000, 32000, 32000, 16000, 16000, 16000, 16000]), 4992, 16000),
    # 128K while the regret is near 0 (it opens from about pass 5 at every K, tmp_plots/vocab/hnet_diag/gap_timing.py), then one cut
    "vslate32k": (_one([128000] * 4 + [32000] * 4), 4480, 32000),
    "vslate16k": (_one([128000] * 4 + [16000] * 4), 4992, 16000),
    "vsmix8": (_mix(8.0), 4992, 16000),    # the figure's real epoch count
    "vsmix4": (_mix(4.0), 4992, 16000),    # a slower shift: the eighth pass at the figure's epoch 4
    # What in path 1 does the work? The two mixed runs end 0.04 below 16K on the same text (w1067, w1068, w1070).
    "vsconst": ([_mean(_mix(8.0))] * 8, 4992, 16000),    # vsmix8's overall shares with no shift: does the shift matter?
    "vsgrow": (list(reversed(_mix(8.0))), 4992, 16000),    # vsmix8's shift reversed: toward large vocabularies as data repeats
    "vsfrozen": ([_mean(_mix(8.0))] * 8, 4992, 16000),    # vsconst, but a window keeps its view on every pass, so its tokens repeat exactly: does the redraw matter?
    "vs128k16k": ([{128000: 0.5, 16000: 0.5}] * 8, 4992, 16000),    # two views, one of them large
    "vs16k8k": ([{16000: 0.5, 8000: 0.5}] * 8, 4992, 16000),    # two views, no large one: do the large views matter?
    "vsuniq": ([_mean(_mix(8.0))] * 8, 4992, 16000),    # vsconst on 8 times the text, no window twice: is the gain a lower regret or a better level?
    # Follow-ups of w1086 (the mix's regret stays near 0.02 at every K with a new view on each pass, and its score on unique tokens is about flat over K).
    "vs16kw": (_one([16000] * 8), 4992, 16000),    # vsfix16k's windows with every row in the softmax and the optimizer step, as in a mix: does that width change a small K's score?
    "vs16kwu": (_one([16000] * 8), 4992, 16000),    # vs16kw on 8 times the text
    "vs128k16ku": ([{128000: 0.5, 16000: 0.5}] * 8, 4992, 16000),    # vs128k16k on 8 times the text: do large views lower a small K's score on unique tokens?
    "vs16k8ku": ([{16000: 0.5, 8000: 0.5}] * 8, 4992, 16000),    # vs16k8k on 8 times the text: the same without a large view
    "vsp50": ([{128000: 0.5, 64000: 0.125, 32000: 0.125, 16000: 0.125, 8000: 0.125}] * 8, 4992, 16000),    # half the windows at 128K, the rest spread: how much of a mix does 128K need?
    "vsp80": ([{128000: 0.8, 64000: 0.05, 32000: 0.05, 16000: 0.05, 8000: 0.05}] * 8, 4992, 16000),
    "vsl128": (_one([128000] * 4) + [_mean(_mix(8.0))] * 4, 4992, 16000),    # 128K on passes 1-4, while the regret is near 0 (gap_timing.py), then the no-shift mix: can 128K keep its rows' training and lose the regret?
    "vsp50u": ([{128000: 0.5, 64000: 0.125, 32000: 0.125, 16000: 0.125, 8000: 0.125}] * 8, 4992, 16000),    # vsp50 on 8 times the text
    "vsp80u": ([{128000: 0.8, 64000: 0.05, 32000: 0.05, 16000: 0.05, 8000: 0.05}] * 8, 4992, 16000),    # vsp80 on 8 times the text
    # The regret of Qian Liu's two proposals themselves (owner popup 2026-11-05): each schedule on 8 times the text, no window twice.
    "vsmix8u": (_mix(8.0), 4992, 16000),    # vsmix8 on 8 times the text
    "vscut32ku": (_one([128000, 64000, 64000, 32000, 32000, 32000, 32000, 32000]), 4480, 32000),    # vscut32k on 8 times the text
    "vsgrowu": (list(reversed(_mix(8.0))), 4992, 16000),    # vsgrow (the shift reversed, Path 1c) on 8 times the text (owner 2026-11-05: 「是不是可以加一个1c，就是反过来，先小词表再大词表」)
}
FROZEN = ("vsfrozen",)   # one draw of views, for all passes
UNIQUE = ("vsuniq", "vs16kwu", "vs128k16ku", "vs16k8ku", "vsp50u", "vsp80u", "vsmix8u", "vscut32ku", "vsgrowu")   # each pass reads the next 183,040 windows of the plain unique-token run's order
WIDE = ("vs16kw", "vs16kwu")   # one view, but every row stays in the softmax and the optimizer step (della_muonh_qwen3_scaling._vs_schedule)


def expansions(keys):
    """K -> (flat, offs), every Marin id to its K-token pieces, in Marin ids (K's specials back at N_ORDINARY + j)."""
    out = {}
    for k in keys:
        if k == N_ORDINARY:
            continue
        flat, offs = expansion(k, f"{PREFIX}/tokenizers/marin-small/v{k}")
        out[k] = (np.where(flat >= k, flat + (N_ORDINARY - k), flat).astype(np.int32), offs)
    return out


def expand(ids, k, ex):
    if k == N_ORDINARY:
        return np.asarray(ids, np.int32).ravel()
    return np.asarray(convert(np.asarray(ids, np.int64).ravel(), *ex[k]), np.int32).ravel()


def read_docs(path, a, b):
    c = TreeCache.load(path, EX)
    return [np.asarray(x["input_ids"], np.int64) for x in c.get_batch_sync(list(range(a, b)))]


def read_all(path):
    c = TreeCache.load(path, EX)
    return [np.asarray(x["input_ids"], np.int64) for a in range(0, len(c), 4096) for x in c.get_batch_sync(list(range(a, min(len(c), a + 4096))))]


def paloma_subsets():
    return sorted(d for d in os.listdir(f"{PREFIX}/tokenized/paloma-v8000") if os.path.isdir(f"{PREFIX}/tokenized/paloma-v8000/{d}/validation"))


def marin_paloma(sub):
    import glob  # noqa: PLC0415
    (path,) = glob.glob(f"{PREFIX}/paloma/{sub}-marin-tokenizer/*/validation")
    return path


def check_expansion(keys):
    """The expansion of the Marin Paloma docs against the K caches of Paloma, document by document, on 4 subsets."""
    ex = expansions(keys)
    for sub in ("ptb", "gab", "c4_en", "dolma_100_programing_languages"):
        src = read_all(marin_paloma(sub))
        for k in keys:
            got = read_all(f"{PREFIX}/tokenized/paloma-v{k}/{sub}/validation")
            assert len(got) == len(src), (sub, k, len(got), len(src))
            bad = 0
            for d, g in zip(src, got):
                e = expand(d, k, ex)
                g2 = np.where(g >= k, g + (N_ORDINARY - k), g)
                bad += not np.array_equal(e, g2)
            print(f"{sub} K={k}: {len(src)} documents, {bad} differ", flush=True)
            assert bad == 0, (sub, k)
    print("expansion check ok", flush=True)


def paloma(keys):
    """tokenized/paloma-v<K>m/<subset>/validation: the K caches of Paloma with their specials at the Marin ids."""
    for k in keys:
        for sub in paloma_subsets():
            docs = read_all(f"{PREFIX}/tokenized/paloma-v{k}/{sub}/validation")
            out = f"{PREFIX}/tokenized/paloma-v{k}m/{sub}/validation"
            os.makedirs(out, exist_ok=True)
            with SerialCacheWriter(out, EX, metadata=CacheMetadata.empty()) as w:
                for a in range(0, len(docs), 4096):
                    w.write_batch([{"input_ids": np.where(d >= k, d + (N_ORDINARY - k), d).astype(np.int32)} for d in docs[a:a + 4096]])
            print(f"K={k} {sub}: {len(docs)} documents", flush=True)


def source_subset():
    m = json.load(open(f"{SOURCE}/meta.json"))
    n = m["subset_sequences"]
    assert m["mode"] == "rs" and m["seq_len"] == 4096 and m["vocab_k"] == 0, m
    rows = np.zeros((n, 4096), np.int32)
    c = TreeCache.load(f"{SOURCE}/train", EX)
    for a in range(0, n, 8192):
        for i, x in enumerate(c.get_batch_sync(list(range(a, min(n, a + 8192))))):
            rows[a + i] = np.asarray(x["input_ids"], np.int32)
    return rows


def unique_reader(first_block):
    """Reads windows of 4096 Marin tokens of the plain unique-token run, in its order (vocab_materialize_passes.subset, with no repeats). Its first windows are checked against the repeated eighth."""
    import asyncio  # noqa: PLC0415
    import dataclasses  # noqa: PLC0415
    import glob  # noqa: PLC0415
    from unittest import mock  # noqa: PLC0415

    import jax  # noqa: PLC0415
    from haliax import Axis  # noqa: PLC0415

    for k in ("PASS_DATA", "VARIANT", "SMOKE_STEPS", "TOTAL_STEPS", "SEQ_LEN", "BATCH", "TPP", "SEED"):
        os.environ.pop(k, None)
    os.environ.update(SIZE="300m", DATA_EPOCHS="1", DRY_RUN="1", MEMSTATS_EVERY="0")
    from experiments.references import della_muonh_qwen3_scaling as scaling  # noqa: PLC0415

    tc = scaling.muonh_qwen3_run("300m").build_config(mock.MagicMock()).train_config
    batch, data = tc.trainer.train_batch_size, tc.data
    (name, _), = data.max_train_batches.items()
    tok = os.path.dirname(glob.glob("/scratch/gpfs/GROUP/USER/cache/huggingface/hub/models--marin-community--marin-tokenizer/snapshots/*/tokenizer.json")[0])
    data = dataclasses.replace(data, tokenizer=tok, components={**data.components, name: scaling._cache_component(f"{PREFIX}/fineweb-edu-10B/2026.06.28/train")})
    _, shuffle_key = jax.random.split(jax.random.PRNGKey(tc.data_seed))
    ds = data.train_sets(Axis("position", 4096), key=shuffle_key, initial_batch_size=batch)[name]

    def read(a, b):
        out = np.zeros((b - a, 4096), np.int32)

        async def go():
            for i in range(a, b, 2048):
                got = await ds.get_batch(list(range(i, min(b, i + 2048))))
                out[i - a:i - a + len(got)] = np.stack([np.asarray(x["input_ids"] if isinstance(x, dict) else x.tokens) for x in got])
        asyncio.run(go())
        return out

    same = int((read(0, 2048) == first_block[:2048]).all(axis=1).sum())
    print(f"unique source: {same} of the first 2048 windows equal the repeated eighth's", flush=True)
    return read


def build(arm, sub, ex):
    t0 = time.time()
    views, seq, eval_k = ARMS[arm]
    n = len(sub)
    read = unique_reader(sub) if arm in UNIQUE else None
    first = None
    out = f"{PREFIX}/materialized/300m-rep8-{arm}"
    os.makedirs(out, exist_ok=True)
    meta = dict(arm=arm, source=SOURCE, subset_sequences=n, seq_len=seq, eval_k=eval_k, wide=arm in WIDE, schedule=[{str(k): v for k, v in p.items()} for p in views], pass_sequences=[], view_windows=[])
    with SerialCacheWriter(f"{out}/train", EX, metadata=CacheMetadata.empty()) as w:
        for e, shares in enumerate(views, 1):
            ks = list(shares)
            rng = np.random.default_rng(30000 + e)
            if read:
                sub = read((e - 1) * n, e * n)
            draw_rng = np.random.default_rng(30000) if arm in FROZEN else rng
            draw = draw_rng.choice(ks, size=n, p=np.asarray([shares[k] for k in ks]) / sum(shares.values())) if len(ks) > 1 else np.full(n, ks[0])
            carry = {k: np.zeros(0, np.int32) for k in ks}
            counts = {k: 0 for k in ks}
            for a in range(0, n, 8192):
                rows, d = sub[a:a + 8192], draw[a:a + 8192]
                parts = []
                for k in ks:
                    sel = d == k
                    if not sel.any():
                        continue
                    s = np.concatenate([carry[k], expand(rows[sel], k, ex)])
                    m = len(s) // seq
                    if m:
                        win = s[:m * seq].reshape(m, seq)
                        assert ((win < k) | (win >= N_ORDINARY)).all(), (arm, e, k)
                        parts.append(win)
                        counts[k] += m
                    carry[k] = s[m * seq:]
                if not parts:
                    continue
                win = np.concatenate(parts)
                if arm in FROZEN and a == 0:   # one draw of views: the first chunk's windows repeat exactly on every pass
                    got = sorted(r.tobytes() for r in win)
                    first = first or got
                    assert got == first, (arm, e)
                if len(ks) > 1:
                    win = win[rng.permutation(len(win))]
                w.write_batch([{"input_ids": r} for r in win])
                if a == 0 and len(ks) == 1 and ks[0] != N_ORDINARY:
                    # the same text: expanding the first source windows again gives these windows
                    assert np.array_equal(win[0], expand(sub[:8192], ks[0], ex)[:seq]), (arm, e)
            meta["pass_sequences"].append(int(sum(counts.values())))
            meta["view_windows"].append({str(k): int(v) for k, v in counts.items()})
            print(f"{arm} pass {e}: {meta['view_windows'][-1]} ({time.time() - t0:.0f}s)", flush=True)
    meta["sequences"] = int(sum(meta["pass_sequences"]))
    meta["tokens"] = meta["sequences"] * seq
    meta["seconds"] = round(time.time() - t0)
    json.dump(meta, open(f"{out}/meta.json", "w"), indent=1)
    print(f"{arm}: {meta['sequences']} windows of {seq}, {meta['sequences'] // 128} steps at batch 128, {meta['seconds']} s", flush=True)


def main():
    a = sys.argv[1:]
    if a[0] == "check":
        check_expansion([64000, 32000, 16000, 8000])
    elif a[0] == "paloma":
        paloma([int(x) for x in a[1:]])
    elif a[0] == "show":
        for arm, (v, seq, ek) in ARMS.items():
            print(arm, seq, ek, [p for p in v])
    else:
        sub = source_subset()
        print("source", sub.shape, flush=True)
        ex = expansions([k for k in VIEWS if any(k in p for arm in a for p in ARMS[arm][0])])
        for arm in a:
            build(arm, sub, ex)


if __name__ == "__main__":
    main()
