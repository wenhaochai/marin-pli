# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-position losses of one checkpoint on the repeated text itself: the first MAX_WINDOWS sequences of the 8-pass
run's subset (pass 1 of materialized/<size>[-v<K>k|-bytes]-rep8-rs, the subset in the run's order), for the vocabulary
page's question how much each tokenizer memorises the repeated text, and of which tokens (owner popup 2026-10-24, test
B). The 8-pass run saw these sequences 8 times, its unique-token partner once; their per-position loss difference is the
memorisation, to be grouped by target frequency next to the Paloma cost of the same pair (vocab_token_losses outputs).

vocab_token_losses.py with its data swapped: one component whose validation split is a symlink to the subset cache's
train split (SUBSET_VIEW/validation -> SUBSET/train), so the windows are built exactly as the Paloma ones (and as the
training examples: same cache format, SEQ_LEN, end-of-text blocking). Output: an npz with subset/loss, subset/weight,
subset/tokens.

    CKPT=<.../checkpoints/step-N> SIZE=300m [VOCAB_K=8000] SUBSET=<materialized dir> SUBSET_VIEW=<scratch dir>
    MAX_WINDOWS=<sequences> OUT=<file.npz> python -m experiments.references.vocab_subset_losses
"""

import os

import experiments.references.vocab_token_losses as vtk

SUBSET, VIEW = os.environ["SUBSET"].rstrip("/"), os.environ["SUBSET_VIEW"].rstrip("/")
assert vtk.MAX_WINDOWS > 0, "set MAX_WINDOWS: the subset holds ~10^5 sequences"


def data_config():
    os.makedirs(VIEW, exist_ok=True)
    link = os.path.join(VIEW, "validation")
    if not os.path.islink(link):
        os.symlink(os.path.join(SUBSET, "train"), link)
    assert os.path.realpath(link) == os.path.realpath(os.path.join(SUBSET, "train")), link
    if vtk.VOCAB_K:
        tok = f"{vtk.PREFIX}/tokenizers/marin-small/v{vtk.VOCAB_K}"
    else:
        import glob
        tok = os.path.dirname(glob.glob("/scratch/gpfs/GROUP/USER/cache/huggingface/hub/models--marin-community--marin-tokenizer/snapshots/*/tokenizer.json")[0])
    return vtk.LmDataConfig(components={"subset": vtk.component(VIEW)}, train_weights={"subset": 0.0}, tokenizer=tok, cache_dir=None, shuffle=False)


vtk.data_config = data_config

if __name__ == "__main__":
    vtk.main()
