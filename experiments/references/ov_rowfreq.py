# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""How often each OV n-gram table row is read in one pass over a run's training data (CPU).

Rebuilds the run's shuffled fineweb-edu-10B order as ov_memorization_eval does, takes the sequences the run cycles through
(the first S of the order; S = N without DATA_EPOCHS), and counts, for the order-2 and order-3 tables (k = 1, as at 300m),
how many token positions read each row: row = ngram_index([x_t, x_{t-1}, ...]) mod m with the model's rule for window
starts and document boundaries (previous tokens of an earlier document or before the window read as 0). Counts are per
pass; a run with P passes reads a row P times as often. Writes logs/ov_rowfreq_<size>_rep<epochs>.npz (count2, count3).

    SIZE=300m DATA_EPOCHS=2.4 [OV_M=12.8e6] [MAX_BLOCKS=n] python -m experiments.references.ov_rowfreq
"""

import os
import time

import jax
import numpy as np

from experiments.references import ov_memorization_eval as ome
from experiments.references.della_muonh_qwen3_scaling import SIZES

V = 128256  # the model's (padded) vocabulary size, as in ngram_index
SIZE = os.environ.get("SIZE", "300m")
EPOCHS = float(os.environ.get("DATA_EPOCHS", "0"))
OV_M = int(float(os.environ.get("OV_M", "12.8e6")))
BLOCK = 8192  # sequences per read
L = 4096


def find_permutation(ds):
    """The Permutation of the PermutationDataset inside ds (the shuffled component)."""
    from levanter.data.dataset import PermutationDataset
    seen, stack = set(), [ds]
    while stack:
        d = stack.pop()
        if id(d) in seen:
            continue
        seen.add(id(d))
        if isinstance(d, PermutationDataset):
            return d, _perm(d)
        stack += [v for v in vars(d).values() if hasattr(v, "__dict__")]
    raise RuntimeError("no PermutationDataset found")


def _perm(d):
    import asyncio
    return asyncio.run(d._get_permutation())


def main():
    s = SIZES[SIZE]
    B, steps = s["batch"], s["steps"]
    data = ome.data_config()
    tok = data.the_tokenizer
    eos = tok.eos_token_id
    from haliax import Axis
    _, shuffle_key = jax.random.split(jax.random.PRNGKey(ome.DATA_SEED))
    order = data.train_sets(Axis("position", L), key=shuffle_key, initial_batch_size=B)[ome.TRAIN_NAME]
    N = len(order.as_sync_dataset())
    S = round(steps / EPOCHS) * B if EPOCHS > 0 else N
    pds, perm = find_permutation(order)
    inner = pds.dataset
    # check: order[i] == inner[perm(i)] for a few i
    so, si = order.as_sync_dataset(), inner.as_sync_dataset()
    for i in (0, 7, S - 1):
        a, b = np.asarray(so[i].tokens), np.asarray(si[int(perm(np.array([i]))[0])].tokens)
        assert np.array_equal(a, b), i
    member = np.zeros(N, bool)
    member[np.asarray(perm(np.arange(S, dtype=np.int64)))] = True
    print(f"N={N} S={S} ({member.sum()} member sequences), eos={eos}", flush=True)

    from experiments.references.over_vocab_qwen3 import OverVocabQwen3Config
    cfg = OverVocabQwen3Config(hidden_dim=s["hidden"], intermediate_dim=s["inter"], num_layers=s["layers"], num_heads=s["heads"],
                               num_kv_heads=s["kv"], oe_m=OV_M)
    if cfg.k != 1:
        raise ValueError(f"{SIZE} has k = {cfg.k} tables per order; this counter handles k = 1 (one order-2 and one order-3 table)")
    (_, m2), (_, m3) = cfg.moduli()   # the order-2 and order-3 tables' moduli
    v2_3 = pow(V, 2, m3)
    c2 = np.zeros(m2, np.int64)
    c3 = np.zeros(m3, np.int64)
    nblocks = -(-N // BLOCK)
    nb = int(os.environ.get("MAX_BLOCKS", nblocks))
    t0 = time.time()
    for b in range(nb):
        lo, hi = b * BLOCK, min(N, (b + 1) * BLOCK)
        sel = np.nonzero(member[lo:hi])[0] + lo
        if len(sel) == 0:
            continue
        x = np.stack([np.asarray(si[int(j)].tokens) for j in sel]) if len(sel) < 64 else np.asarray(
            [ex.tokens for ex in si.get_batch(sel.tolist())])
        x = x.astype(np.int64)                                   # [n, L]
        eosm = np.zeros_like(x, bool); eosm[:, 1:] = x[:, :-1] == eos
        seg = np.cumsum(eosm, axis=1)
        p1 = np.zeros_like(x); p1[:, 1:] = x[:, :-1]; p1[:, 1:][seg[:, 1:] != seg[:, :-1]] = 0
        p2 = np.zeros_like(x); p2[:, 2:] = x[:, :-2]; p2[:, 2:][seg[:, 2:] != seg[:, :-2]] = 0
        i2 = (x % m2 + (p1 * (V % m2)) % m2) % m2
        i3 = (x % m3 + (p1 * (V % m3)) % m3 + (p2 * v2_3) % m3) % m3
        c2 += np.bincount(i2.ravel(), minlength=m2)
        c3 += np.bincount(i3.ravel(), minlength=m3)
        if b % 20 == 0:
            print(f"block {b}/{nb} {time.time() - t0:.0f}s", flush=True)
    out = f"logs/ov_rowfreq_{SIZE}_rep{EPOCHS:g}_m{OV_M / 1e6:g}m.npz"
    np.savez_compressed(out, count2=c2.astype(np.int32), count3=c3.astype(np.int32), N=N, S=S, blocks=nb)
    for name, c in (("2-gram", c2), ("3-gram", c3)):
        pos = c.sum()
        print(name, f"positions {pos}, rows used {np.count_nonzero(c)} of {len(c)}; share of positions on rows read <=1/<=3/<=10/<=100 times:",
              [round(float(c[(c > 0) & (c <= k)].sum() / pos), 4) for k in (1, 3, 10, 100)], flush=True)
    print("written", out)


if __name__ == "__main__":
    main()
