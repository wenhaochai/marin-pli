# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the Delphi rung's data deps on a Della login node (compute nodes cannot reach huggingface.co).

    MARIN_PREFIX=/scratch/gpfs/KARTHIKN/wc9403/marin_store_big python -m scripts.della.prep_delphi_data
"""

from marin.execution.lazy import run

from experiments.references.della_delphi_ladder import _train_data

if __name__ == "__main__":
    train, validation = _train_data()
    run(*validation, max_concurrent=4)  # train caches are pinned to existing July-built dirs
    print("DELPHI DATA PREP DONE")
